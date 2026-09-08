"""Strict, non-executable candidate manifest schema.

A manifest describes facts about a candidate. It never contains shell commands
or arbitrary executable hooks. All launch behavior is selected later from a
code-owned template whitelist.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

MANIFEST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{1,95}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")


class ManifestError(ValueError):
    """The candidate manifest is malformed or unsafe."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must be an object")
    return value


def _only(mapping: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ManifestError(f"{name} contains unknown field(s): {', '.join(sorted(unknown))}")


def _text(mapping: dict[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name}.{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class CandidateSource:
    provider: str
    repo_id: str
    revision: str


@dataclass(frozen=True)
class Compatibility:
    target_model: str
    target_revision: str | None
    runtime: str
    runtime_min_version: str
    platform: str


@dataclass(frozen=True)
class CandidateVariant:
    id: str
    description: str
    draft_block_size: int | None


@dataclass(frozen=True)
class CandidateManifest:
    schema_version: int
    id: str
    kind: str
    status: str
    description: str
    source: CandidateSource
    compatibility: Compatibility
    allow_remote_code: bool
    estimated_download_gb: float
    max_peak_memory_gb: float
    launch_template: str
    draft_kind: str
    variants: tuple[CandidateVariant, ...]
    quality_suite: str
    benchmark_suite: str
    minimum_speedup_percent: float
    max_error_rate: float
    require_all_quality_gates: bool
    license_name: str
    review_status: str
    manifest_hash: str
    canonical: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.canonical))


def parse_manifest(raw: dict[str, Any]) -> CandidateManifest:
    """Parse and validate one manifest from an already-safe YAML/JSON loader."""
    root = _mapping(raw, "manifest")
    _only(
        root,
        {
            "schema_version",
            "id",
            "kind",
            "status",
            "description",
            "source",
            "compatibility",
            "artifacts",
            "resources",
            "launch",
            "verification",
            "promotion",
            "license",
        },
        "manifest",
    )
    if root.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")

    manifest_id = _text(root, "id", "manifest")
    if not MANIFEST_ID_RE.fullmatch(manifest_id):
        raise ManifestError("invalid manifest id")

    source_raw = _mapping(root.get("source"), "source")
    _only(source_raw, {"provider", "repo_id", "revision"}, "source")
    revision = _text(source_raw, "revision", "source").lower()
    if not COMMIT_RE.fullmatch(revision):
        raise ManifestError("source.revision must be a fixed commit hash")
    repo_id = _text(source_raw, "repo_id", "source")
    if repo_id.count("/") != 1 or any(part in ("", ".", "..") for part in repo_id.split("/")):
        raise ManifestError("source.repo_id must be namespace/repository")

    compatibility_raw = _mapping(root.get("compatibility"), "compatibility")
    _only(
        compatibility_raw,
        {
            "target_model",
            "target_revision",
            "runtime",
            "runtime_min_version",
            "platform",
        },
        "compatibility",
    )
    target_revision = compatibility_raw.get("target_revision")
    if target_revision is not None:
        if not isinstance(target_revision, str) or not COMMIT_RE.fullmatch(target_revision.lower()):
            raise ManifestError("compatibility.target_revision must be a fixed commit hash")
        target_revision = target_revision.lower()

    artifacts_raw = _mapping(root.get("artifacts"), "artifacts")
    _only(artifacts_raw, {"allow_remote_code", "estimated_download_gb"}, "artifacts")
    allow_remote_code = artifacts_raw.get("allow_remote_code")
    if allow_remote_code is not False:
        raise ManifestError("remote code must be explicitly disabled")

    resources_raw = _mapping(root.get("resources"), "resources")
    _only(resources_raw, {"max_peak_memory_gb"}, "resources")

    launch_raw = _mapping(root.get("launch"), "launch")
    _only(launch_raw, {"template", "parameters"}, "launch")
    parameters = _mapping(launch_raw.get("parameters"), "launch.parameters")
    _only(parameters, {"draft_kind", "variants"}, "launch.parameters")
    variants_raw = parameters.get("variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ManifestError("launch.parameters.variants must be a non-empty list")
    variants: list[CandidateVariant] = []
    seen: set[str] = set()
    for index, item in enumerate(variants_raw):
        item = _mapping(item, f"launch.parameters.variants[{index}]")
        _only(item, {"id", "description", "draft_block_size"}, f"variant[{index}]")
        candidate_id = _text(item, "id", f"variant[{index}]")
        if not CANDIDATE_ID_RE.fullmatch(candidate_id):
            raise ManifestError(f"invalid candidate id: {candidate_id}")
        if candidate_id == "baseline" or candidate_id in seen:
            raise ManifestError(f"duplicate or reserved candidate id: {candidate_id}")
        seen.add(candidate_id)
        block = item.get("draft_block_size")
        if block is not None and (not isinstance(block, int) or isinstance(block, bool)):
            raise ManifestError(f"variant {candidate_id} draft_block_size must be an integer or null")
        variants.append(
            CandidateVariant(
                id=candidate_id,
                description=_text(item, "description", f"variant[{index}]"),
                draft_block_size=block,
            )
        )

    verification_raw = _mapping(root.get("verification"), "verification")
    _only(verification_raw, {"quality_suite", "benchmark_suite"}, "verification")
    promotion_raw = _mapping(root.get("promotion"), "promotion")
    _only(
        promotion_raw,
        {"minimum_speedup_percent", "max_error_rate", "require_all_quality_gates"},
        "promotion",
    )
    license_raw = _mapping(root.get("license"), "license")
    _only(license_raw, {"name", "review_status"}, "license")

    try:
        estimated_download_gb = float(artifacts_raw["estimated_download_gb"])
        max_peak_memory_gb = float(resources_raw["max_peak_memory_gb"])
        minimum_speedup_percent = float(promotion_raw["minimum_speedup_percent"])
        max_error_rate = float(promotion_raw["max_error_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("manifest numeric fields are missing or invalid") from exc
    if estimated_download_gb <= 0 or max_peak_memory_gb <= 0:
        raise ManifestError("download and memory estimates must be positive")
    if minimum_speedup_percent <= 0 or max_error_rate < 0:
        raise ManifestError("promotion thresholds are invalid")
    if promotion_raw.get("require_all_quality_gates") is not True:
        raise ManifestError("all quality gates must be required")

    canonical = json.loads(json.dumps(root, sort_keys=True))
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CandidateManifest(
        schema_version=1,
        id=manifest_id,
        kind=_text(root, "kind", "manifest"),
        status=_text(root, "status", "manifest"),
        description=_text(root, "description", "manifest"),
        source=CandidateSource(
            provider=_text(source_raw, "provider", "source"),
            repo_id=repo_id,
            revision=revision,
        ),
        compatibility=Compatibility(
            target_model=_text(compatibility_raw, "target_model", "compatibility"),
            target_revision=target_revision,
            runtime=_text(compatibility_raw, "runtime", "compatibility"),
            runtime_min_version=_text(
                compatibility_raw, "runtime_min_version", "compatibility"
            ),
            platform=_text(compatibility_raw, "platform", "compatibility"),
        ),
        allow_remote_code=False,
        estimated_download_gb=estimated_download_gb,
        max_peak_memory_gb=max_peak_memory_gb,
        launch_template=_text(launch_raw, "template", "launch"),
        draft_kind=_text(parameters, "draft_kind", "launch.parameters"),
        variants=tuple(variants),
        quality_suite=_text(verification_raw, "quality_suite", "verification"),
        benchmark_suite=_text(verification_raw, "benchmark_suite", "verification"),
        minimum_speedup_percent=minimum_speedup_percent,
        max_error_rate=max_error_rate,
        require_all_quality_gates=True,
        license_name=_text(license_raw, "name", "license"),
        review_status=_text(license_raw, "review_status", "license"),
        manifest_hash=digest,
        canonical=canonical,
    )
