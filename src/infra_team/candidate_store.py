"""Project-managed, pinned candidate store with atomic READY markers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .candidate_registry import ResolvedCandidate
from .compatibility import AutonomyPolicy

Downloader = Callable[[str, str, str], str]


class CandidateStoreError(RuntimeError):
    """Candidate preparation was refused or did not complete safely."""


@dataclass(frozen=True)
class PreparedCandidate:
    candidate_id: str
    manifest_id: str
    manifest_hash: str
    revision: str
    root: Path
    local_model_path: Path
    ready: bool
    reused: bool

    def apply(self, candidate: ResolvedCandidate) -> ResolvedCandidate:
        if candidate.manifest_hash != self.manifest_hash:
            raise CandidateStoreError("prepared candidate manifest hash mismatch")
        return candidate.with_local_model_path(str(self.local_model_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "manifest_id": self.manifest_id,
            "manifest_hash": self.manifest_hash,
            "revision": self.revision,
            "root": str(self.root),
            "local_model_path": str(self.local_model_path),
            "ready": self.ready,
            "reused": self.reused,
        }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def _default_downloader(repo_id: str, revision: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.tiktoken",
            "tokenizer*",
            "vocab*",
            "merges*",
        ],
    )


class CandidateStore:
    def __init__(
        self,
        root: str | Path,
        downloader: Downloader | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.candidates_dir = self.root / ".infra-team" / "candidates"
        self.downloader = downloader or _default_downloader

    def _store_size_gb(self) -> float:
        total = 0
        seen: set[tuple[int, int]] = set()
        if not self.candidates_dir.exists():
            return 0.0
        for path in self.candidates_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_size
        return total / (1024**3)

    @staticmethod
    def _hash_files(assets: Path) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(assets.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(assets)
            # huggingface_hub may maintain transport metadata under .cache.
            # It is not part of the runnable candidate and can change without
            # changing model bytes, so only hash the actual snapshot files.
            if relative.parts and relative.parts[0] == ".cache":
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(assets.resolve())
            except ValueError as exc:
                raise CandidateStoreError(
                    f"candidate file escapes store through a symlink: {path}"
                ) from exc
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                while chunk := fh.read(8 * 1024 * 1024):
                    digest.update(chunk)
            files.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        if not files:
            raise CandidateStoreError("download completed without candidate files")
        return {
            "schema_version": 1,
            "algorithm": "sha256",
            "files": files,
            "total_bytes": sum(item["size"] for item in files),
        }

    @staticmethod
    def _prepared_from(candidate: ResolvedCandidate, root: Path, reused: bool) -> PreparedCandidate:
        return PreparedCandidate(
            candidate_id=candidate.id,
            manifest_id=candidate.manifest_id,
            manifest_hash=candidate.manifest_hash,
            revision=str(candidate.source_revision),
            root=root,
            local_model_path=root / "assets",
            ready=True,
            reused=reused,
        )

    def get_ready(self, candidate: ResolvedCandidate) -> PreparedCandidate | None:
        final = self.candidates_dir / candidate.manifest_id
        if not (final / "READY").is_file():
            return None
        try:
            with (final / "resolved.json").open(encoding="utf-8") as fh:
                resolved = json.load(fh)
            with (final / "files.json").open(encoding="utf-8") as fh:
                expected_files = json.load(fh)
            ready_hash = (final / "READY").read_text(encoding="utf-8").strip()
        except (OSError, json.JSONDecodeError):
            return None
        if (
            ready_hash != candidate.manifest_hash
            or resolved.get("manifest_hash") != candidate.manifest_hash
            or resolved.get("revision") != candidate.source_revision
            or not (final / "assets").is_dir()
        ):
            return None
        try:
            actual_files = self._hash_files(final / "assets")
        except CandidateStoreError:
            return None
        if actual_files != expected_files:
            return None
        return self._prepared_from(candidate, final, reused=True)

    def apply_ready_candidates(
        self, candidates: dict[str, ResolvedCandidate]
    ) -> dict[str, ResolvedCandidate]:
        """Rewrite ready variants to the verified project-local asset path."""
        output: dict[str, ResolvedCandidate] = {}
        prepared_by_manifest: dict[str, PreparedCandidate | None] = {}
        for candidate_id, candidate in candidates.items():
            if not candidate.requires_prepare:
                output[candidate_id] = candidate
                continue
            if candidate.manifest_id not in prepared_by_manifest:
                prepared_by_manifest[candidate.manifest_id] = self.get_ready(candidate)
            prepared = prepared_by_manifest[candidate.manifest_id]
            output[candidate_id] = (
                prepared.apply(candidate) if prepared is not None else candidate
            )
        return output

    def prepare(
        self,
        candidate: ResolvedCandidate,
        policy: AutonomyPolicy,
    ) -> PreparedCandidate:
        if not candidate.requires_prepare or not candidate.source_repo_id:
            raise CandidateStoreError("baseline or source-less candidate cannot be prepared")
        if candidate.manifest is None or candidate.source_revision is None:
            raise CandidateStoreError("candidate has no validated manifest or pinned revision")
        if candidate.estimated_download_gb > policy.max_download_gb_per_candidate:
            raise CandidateStoreError("candidate exceeds per-candidate download budget")
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = self.candidates_dir / ".locks"
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_dir / f"{candidate.manifest_id}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return self._prepare_locked(candidate, policy)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _prepare_locked(
        self,
        candidate: ResolvedCandidate,
        policy: AutonomyPolicy,
    ) -> PreparedCandidate:
        ready = self.get_ready(candidate)
        if ready is not None:
            return ready
        if (
            self._store_size_gb() + candidate.estimated_download_gb
            > policy.max_total_candidate_store_gb
        ):
            raise CandidateStoreError("candidate exceeds total candidate store budget")
        free_gb = shutil.disk_usage(self.candidates_dir).free / (1024**3)
        if free_gb < candidate.estimated_download_gb + 2.0:
            raise CandidateStoreError("insufficient free disk for candidate preparation")

        final = self.candidates_dir / candidate.manifest_id
        if final.exists():
            raise CandidateStoreError(
                "candidate directory exists without a matching READY marker; "
                "its revision or file hashes do not match the trusted manifest"
            )
        staging = self.candidates_dir / (
            f".prepare-{candidate.manifest_id}-{uuid.uuid4().hex}"
        )
        assets = staging / "assets"
        started = time.time()
        try:
            staging.mkdir(parents=True, exist_ok=False)
            assets.mkdir()
            result = Path(
                self.downloader(
                    candidate.source_repo_id,
                    str(candidate.source_revision),
                    str(assets),
                )
            ).resolve()
            if result != assets.resolve():
                try:
                    result.relative_to(assets.resolve())
                except ValueError as exc:
                    raise CandidateStoreError(
                        "downloader returned a path outside the candidate staging directory"
                    ) from exc
            files = self._hash_files(assets)
            actual_gb = files["total_bytes"] / (1024**3)
            if actual_gb > policy.max_download_gb_per_candidate:
                raise CandidateStoreError("downloaded files exceed candidate download budget")
            _atomic_json(staging / "manifest.json", candidate.manifest.to_dict())
            _atomic_json(
                staging / "resolved.json",
                {
                    **candidate.to_dict(),
                    "candidate_id": candidate.id,
                    "revision": candidate.source_revision,
                    "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            _atomic_json(staging / "files.json", files)
            (staging / "prepare.log").write_text(
                (
                    f"repo_id={candidate.source_repo_id}\n"
                    f"revision={candidate.source_revision}\n"
                    f"manifest_hash={candidate.manifest_hash}\n"
                    f"elapsed_seconds={time.time() - started:.3f}\n"
                ),
                encoding="utf-8",
            )
            (staging / "READY").write_text(
                candidate.manifest_hash + "\n", encoding="utf-8"
            )
            os.replace(staging, final)
            return self._prepared_from(candidate, final, reused=False)
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging)
            if isinstance(exc, CandidateStoreError):
                raise
            raise CandidateStoreError(f"candidate preparation failed: {exc}") from exc
