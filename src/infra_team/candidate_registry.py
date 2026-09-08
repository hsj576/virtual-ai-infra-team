"""Trusted local candidate registry.

The first implementation intentionally supports local, project-maintained
manifests only. A registry entry describes a candidate but cannot define shell
commands; launch behavior remains in policy.py's code-owned templates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .candidate_manifest import CandidateManifest, ManifestError, parse_manifest
from .compatibility import AutonomyPolicy, PreflightError, preflight_candidate
from .runner import CandidateSpec

BUILTIN_REGISTRY_DIR = Path(__file__).with_name("registry")
TARGET_MODEL = "mlx-community/Qwen3.8-27B-4bit"


class RegistryError(ValueError):
    """A local registry could not be loaded or resolved safely."""


@dataclass(frozen=True)
class ResolvedCandidate:
    id: str
    description: str
    target_model: str
    target_revision: str | None
    draft_model: str | None
    draft_kind: str | None
    draft_block_size: int | None
    manifest_id: str
    manifest_hash: str
    kind: str
    source_provider: str | None
    source_repo_id: str | None
    source_revision: str | None
    launch_template: str
    estimated_download_gb: float
    max_peak_memory_gb: float
    quality_suite: str
    benchmark_suite: str
    minimum_speedup_percent: float
    manifest: CandidateManifest | None = None
    local_model_path: str | None = None

    @property
    def requires_prepare(self) -> bool:
        return self.id != "baseline" and self.source_repo_id is not None

    def with_local_model_path(self, path: str) -> "ResolvedCandidate":
        return replace(self, draft_model=path, local_model_path=path)

    def to_spec(self, repeats: int) -> CandidateSpec:
        return CandidateSpec(
            id=self.id,
            target_model=self.target_model,
            draft_model=self.draft_model,
            draft_kind=self.draft_kind,
            draft_block_size=self.draft_block_size,
            temperature=0.0,
            enable_thinking=False,
            repeats=repeats,
            target_revision=self.target_revision,
            candidate_manifest_id=(
                self.manifest_id if self.id != "baseline" else None
            ),
            candidate_manifest_hash=(
                self.manifest_hash if self.id != "baseline" else None
            ),
            runtime_version=(
                self.manifest.compatibility.runtime_min_version
                if self.manifest is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "target_model": self.target_model,
            "target_revision": self.target_revision,
            "draft_model": self.draft_model,
            "draft_kind": self.draft_kind,
            "draft_block_size": self.draft_block_size,
            "manifest_id": self.manifest_id,
            "manifest_hash": self.manifest_hash,
            "kind": self.kind,
            "source": {
                "provider": self.source_provider,
                "repo_id": self.source_repo_id,
                "revision": self.source_revision,
            },
            "launch_template": self.launch_template,
            "estimated_download_gb": self.estimated_download_gb,
            "max_peak_memory_gb": self.max_peak_memory_gb,
            "quality_suite": self.quality_suite,
            "benchmark_suite": self.benchmark_suite,
            "minimum_speedup_percent": self.minimum_speedup_percent,
            "local_model_path": self.local_model_path,
        }


class CandidateRegistry:
    def __init__(
        self,
        directory: str | Path = BUILTIN_REGISTRY_DIR,
        registry_name: str = "builtin",
        manifests: Iterable[CandidateManifest] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.registry_name = registry_name
        self._manifests = tuple(manifests) if manifests is not None else None

    @classmethod
    def builtin(cls) -> "CandidateRegistry":
        return cls(BUILTIN_REGISTRY_DIR, "builtin")

    @classmethod
    def from_manifests(
        cls,
        manifests: Iterable[CandidateManifest],
        registry_name: str = "builtin",
    ) -> "CandidateRegistry":
        return cls(BUILTIN_REGISTRY_DIR, registry_name, manifests)

    def load(self) -> list[CandidateManifest]:
        if self._manifests is not None:
            return list(self._manifests)
        if not self.directory.is_dir():
            raise RegistryError(f"registry directory does not exist: {self.directory}")
        manifests: list[CandidateManifest] = []
        for path in sorted(self.directory.glob("*.yaml")):
            try:
                with path.open(encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh)
                manifests.append(parse_manifest(raw))
            except (OSError, yaml.YAMLError, ManifestError) as exc:
                raise RegistryError(f"invalid registry manifest {path.name}: {exc}") from exc
        if not manifests:
            raise RegistryError(f"registry contains no manifests: {self.directory}")
        return manifests

    def snapshot(self) -> dict[str, Any]:
        manifests = self.load()
        entries = [
            {
                "id": manifest.id,
                "hash": manifest.manifest_hash,
                "source": {
                    "repo_id": manifest.source.repo_id,
                    "revision": manifest.source.revision,
                },
            }
            for manifest in manifests
        ]
        digest = hashlib.sha256(
            json.dumps(
                entries, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 1,
            "registry": self.registry_name,
            "snapshot_hash": digest,
            "manifests": entries,
        }

    def resolve_all(
        self,
        policy: AutonomyPolicy | None = None,
        environment: dict[str, Any] | None = None,
    ) -> dict[str, ResolvedCandidate]:
        policy = policy or AutonomyPolicy()
        environment = environment or {
            "arch": "arm64",
            "chip": "Apple Silicon",
            "packages": {"mlx-vlm": "999.0.0"},
        }
        resolved: dict[str, ResolvedCandidate] = {
            "baseline": ResolvedCandidate(
                id="baseline",
                description="Plain autoregressive decoding, no drafter.",
                target_model=TARGET_MODEL,
                target_revision=None,
                draft_model=None,
                draft_kind=None,
                draft_block_size=None,
                manifest_id="builtin-baseline-v1",
                manifest_hash="builtin-baseline-v1",
                kind="baseline",
                source_provider=None,
                source_repo_id=None,
                source_revision=None,
                launch_template="mlx_vlm_baseline",
                estimated_download_gb=0.0,
                max_peak_memory_gb=policy.max_peak_memory_gb,
                quality_suite="qwen38-smoke-v1",
                benchmark_suite="qwen38-speed-v1",
                minimum_speedup_percent=policy.minimum_speedup_percent,
            )
        }
        manifest_ids: set[str] = set()
        for manifest in self.load():
            if manifest.id in manifest_ids:
                raise RegistryError(f"duplicate manifest id: {manifest.id}")
            manifest_ids.add(manifest.id)
            try:
                preflight_candidate(
                    manifest,
                    policy,
                    environment,
                    registry_name=self.registry_name,
                )
            except PreflightError as exc:
                raise RegistryError(f"manifest {manifest.id} failed preflight: {exc}") from exc
            for variant in manifest.variants:
                if variant.id in resolved:
                    raise RegistryError(f"duplicate candidate id: {variant.id}")
                resolved[variant.id] = ResolvedCandidate(
                    id=variant.id,
                    description=variant.description,
                    target_model=manifest.compatibility.target_model,
                    target_revision=manifest.compatibility.target_revision,
                    draft_model=manifest.source.repo_id,
                    draft_kind=manifest.draft_kind,
                    draft_block_size=variant.draft_block_size,
                    manifest_id=manifest.id,
                    manifest_hash=manifest.manifest_hash,
                    kind=manifest.kind,
                    source_provider=manifest.source.provider,
                    source_repo_id=manifest.source.repo_id,
                    source_revision=manifest.source.revision,
                    launch_template=manifest.launch_template,
                    estimated_download_gb=manifest.estimated_download_gb,
                    max_peak_memory_gb=manifest.max_peak_memory_gb,
                    quality_suite=manifest.quality_suite,
                    benchmark_suite=manifest.benchmark_suite,
                    minimum_speedup_percent=manifest.minimum_speedup_percent,
                    manifest=manifest,
                )
        return resolved

    def inspect(self, manifest_id: str) -> dict[str, Any]:
        for manifest in self.load():
            if manifest.id == manifest_id:
                return {
                    "manifest": manifest.to_dict(),
                    "manifest_hash": manifest.manifest_hash,
                    "registry": self.registry_name,
                }
        raise RegistryError(f"candidate manifest not found: {manifest_id}")

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2)
