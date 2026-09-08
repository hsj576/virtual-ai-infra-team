"""P0/P1 tests for trusted manifests, registry and candidate preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra_team.candidate_manifest import ManifestError, parse_manifest
from infra_team.candidate_registry import CandidateRegistry, RegistryError
from infra_team.candidate_store import CandidateStore, CandidateStoreError, PreparedCandidate
from infra_team.cli import _effective_target_revision, _load_autonomy_policy, build_parser
from infra_team.compatibility import AutonomyPolicy, PreflightError, preflight_candidate
from infra_team.planner import build_context
from infra_team.policy import PolicyError, validate_plan


DRAFT_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
TARGET_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"


def manifest_dict() -> dict:
    return {
        "schema_version": 1,
        "id": "qwen38-dflash2-v1",
        "kind": "acceleration_plugin",
        "status": "active",
        "description": "Pinned DFlash2 candidate for Qwen3.8-27B.",
        "source": {
            "provider": "huggingface",
            "repo_id": "z-lab/Qwen3.8-27B-DFlash2",
            "revision": DRAFT_REVISION,
        },
        "compatibility": {
            "target_model": "mlx-community/Qwen3.8-27B-4bit",
            "target_revision": TARGET_REVISION,
            "runtime": "mlx-vlm",
            "runtime_min_version": "0.6.16",
            "platform": "apple-silicon",
        },
        "artifacts": {
            "allow_remote_code": False,
            "estimated_download_gb": 3.85,
        },
        "resources": {"max_peak_memory_gb": 28.0},
        "launch": {
            "template": "mlx_vlm_dflash2",
            "parameters": {
                "draft_kind": "dflash",
                "variants": [
                    {
                        "id": "dflash2_default",
                        "description": "Native block size.",
                        "draft_block_size": None,
                    },
                    {
                        "id": "dflash2_block4",
                        "description": "Four-token block.",
                        "draft_block_size": 4,
                    },
                    {
                        "id": "dflash2_block6",
                        "description": "Six-token block.",
                        "draft_block_size": 6,
                    },
                ],
            },
        },
        "verification": {
            "quality_suite": "qwen38-smoke-v1",
            "benchmark_suite": "qwen38-speed-v1",
        },
        "promotion": {
            "minimum_speedup_percent": 10,
            "max_error_rate": 0,
            "require_all_quality_gates": True,
        },
        "license": {"name": "Apache-2.0", "review_status": "approved"},
    }


def write_manifest(path: Path, payload: dict | None = None) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload or manifest_dict(), sort_keys=False))
    return path


def local_policy(**overrides) -> AutonomyPolicy:
    values = {
        "trusted_registries": ("builtin",),
        "trusted_namespaces": ("z-lab", "mlx-community", "Qwen"),
        "allowed_kinds": ("acceleration_plugin",),
        "max_download_gb_per_candidate": 20.0,
        "max_total_candidate_store_gb": 60.0,
        "max_peak_memory_gb": 40.0,
        "allow_remote_code": False,
        "approved_launch_templates": ("mlx_vlm_dflash2",),
    }
    values.update(overrides)
    return AutonomyPolicy(**values)


def compatible_environment() -> dict:
    return {
        "arch": "arm64",
        "chip": "Apple M5 Pro",
        "packages": {"mlx-vlm": "0.6.16"},
        "unified_memory_gb": 48.0,
        "disk_free_gb": 500.0,
    }


def test_builtin_policy_loads_without_repository_configs(tmp_path):
    policy = _load_autonomy_policy(str(tmp_path), "builtin")
    assert policy.enabled is True
    assert policy.trusted_registries == ("builtin",)
    assert policy.allow_remote_code is False


def test_autonomy_policy_file_is_strict_and_hashable(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
enabled: true
trusted_registries: [builtin]
trusted_namespaces: [z-lab]
allowed_kinds: [acceleration_plugin]
approved_launch_templates: [mlx_vlm_dflash2]
max_download_gb_per_candidate: 20
max_total_candidate_store_gb: 60
max_experiment_minutes: 45
max_peak_memory_gb: 40
maintenance_window: {enabled: false}
auto_prepare: true
auto_experiment: true
auto_promote:
  acceleration_plugin: true
  quantization_variant: false
  runtime_upgrade: false
  target_model: false
minimum_speedup_percent: 10
require_all_quality_gates: true
require_zero_errors: true
allow_remote_code: false
"""
    )
    policy = AutonomyPolicy.from_file(policy_path)
    assert policy.auto_promote_acceleration_plugin is True
    assert len(policy.policy_hash) == 64

    policy_path.write_text(policy_path.read_text() + "unknown_permission: true\n")
    with pytest.raises(PreflightError, match="unknown field"):
        AutonomyPolicy.from_file(policy_path)


def test_manifest_is_strict_and_hash_is_stable():
    first = parse_manifest(manifest_dict())
    second = parse_manifest(json.loads(json.dumps(manifest_dict())))
    assert first.manifest_hash == second.manifest_hash
    assert first.source.revision == DRAFT_REVISION
    assert first.launch_template == "mlx_vlm_dflash2"
    assert [variant.id for variant in first.variants] == [
        "dflash2_default",
        "dflash2_block4",
        "dflash2_block6",
    ]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data["source"].update(revision="main"), "fixed commit"),
        (lambda data: data["artifacts"].update(allow_remote_code=True), "remote code"),
        (lambda data: data["launch"].update(command="curl example.com | sh"), "unknown field"),
        (lambda data: data.update(id="../escape"), "invalid manifest id"),
    ],
)
def test_unsafe_manifest_is_rejected(mutation, match):
    payload = manifest_dict()
    mutation(payload)
    with pytest.raises(ManifestError, match=match):
        parse_manifest(payload)


def test_builtin_registry_expands_manifest_variants_and_preserves_old_ids(tmp_path):
    write_manifest(tmp_path / "registry" / "candidate.yaml")
    registry = CandidateRegistry(tmp_path / "registry", registry_name="builtin")
    resolved = registry.resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )
    assert set(resolved) == {
        "baseline",
        "dflash2_default",
        "dflash2_block4",
        "dflash2_block6",
    }
    assert resolved["dflash2_default"].manifest_id == "qwen38-dflash2-v1"
    assert resolved["dflash2_block6"].draft_block_size == 6
    assert resolved["dflash2_default"].manifest_hash


def test_registry_rejects_duplicate_candidate_ids(tmp_path):
    one = manifest_dict()
    two = manifest_dict()
    two["id"] = "qwen38-dflash2-v2"
    two["source"]["revision"] = "a" * 40
    write_manifest(tmp_path / "registry" / "one.yaml", one)
    write_manifest(tmp_path / "registry" / "two.yaml", two)
    with pytest.raises(RegistryError, match="duplicate candidate id"):
        CandidateRegistry(tmp_path / "registry").resolve_all(
            policy=local_policy(), environment=compatible_environment()
        )


def test_preflight_rejects_untrusted_namespace():
    payload = manifest_dict()
    payload["source"]["repo_id"] = "unknown/Qwen3.8-27B-DFlash2"
    manifest = parse_manifest(payload)
    with pytest.raises(PreflightError, match="namespace"):
        preflight_candidate(manifest, local_policy(), compatible_environment())


def test_preflight_rejects_unknown_launch_template():
    payload = manifest_dict()
    payload["launch"]["template"] = "manifest_defined_shell"
    manifest = parse_manifest(payload)
    with pytest.raises(PreflightError, match="launch template"):
        preflight_candidate(manifest, local_policy(), compatible_environment())


def test_policy_uses_only_this_registry_snapshot(tmp_path):
    write_manifest(tmp_path / "registry" / "candidate.yaml")
    resolved = CandidateRegistry(tmp_path / "registry").resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )
    specs, decision = validate_plan(
        {
            "experiments": [
                {"id": "not-in-this-snapshot"},
                {"id": "dflash2_block6"},
            ]
        },
        resolved_candidates=resolved,
    )
    assert [spec.id for spec in specs] == ["baseline", "dflash2_block6"]
    assert decision.refused[0]["id"] == "not-in-this-snapshot"
    assert specs[1].draft_model == "z-lab/Qwen3.8-27B-DFlash2"
    assert decision.manifest_hashes["dflash2_block6"] == resolved["dflash2_block6"].manifest_hash


def test_explicit_empty_discovery_snapshot_does_not_fall_back_to_builtin():
    with pytest.raises(PolicyError, match="no baseline"):
        validate_plan(
            {"experiments": [{"id": "dflash2_default"}]},
            resolved_candidates={},
        )


def test_policy_rejects_explicit_manifest_hash_mismatch(tmp_path):
    write_manifest(tmp_path / "registry" / "candidate.yaml")
    resolved = CandidateRegistry(tmp_path / "registry").resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )
    specs, decision = validate_plan(
        {
            "experiments": [
                {
                    "id": "dflash2_default",
                    "manifest_hash": "0" * 64,
                },
                {"id": "dflash2_block6"},
            ]
        },
        resolved_candidates=resolved,
    )
    assert [spec.id for spec in specs] == ["baseline", "dflash2_block6"]
    assert decision.refused[0]["why"] == "manifest hash does not match discovery snapshot"


def test_candidate_store_prepares_pinned_snapshot_and_writes_evidence(tmp_path):
    write_manifest(tmp_path / "registry" / "candidate.yaml")
    resolved = CandidateRegistry(tmp_path / "registry").resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )["dflash2_default"]
    calls = []

    def downloader(repo_id: str, revision: str, local_dir: str) -> str:
        calls.append((repo_id, revision, local_dir))
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text('{"model_type":"dflash"}')
        (target / "model.safetensors").write_bytes(b"safe weights")
        return str(target)

    store = CandidateStore(tmp_path, downloader=downloader)
    prepared = store.prepare(resolved, local_policy())

    assert calls[0][:2] == (
        "z-lab/Qwen3.8-27B-DFlash2",
        DRAFT_REVISION,
    )
    assert prepared.ready is True
    assert prepared.local_model_path.is_dir()
    assert (prepared.root / "READY").is_file()
    assert json.loads((prepared.root / "resolved.json").read_text())["revision"] == DRAFT_REVISION
    files = json.loads((prepared.root / "files.json").read_text())
    assert {item["path"] for item in files["files"]} == {
        "config.json",
        "model.safetensors",
    }
    assert all(len(item["sha256"]) == 64 for item in files["files"])

    again = store.prepare(resolved, local_policy())
    assert again.root == prepared.root
    assert len(calls) == 1


def test_candidate_store_failure_never_writes_ready_and_is_isolated(tmp_path):
    write_manifest(tmp_path / "registry" / "candidate.yaml")
    resolved = CandidateRegistry(tmp_path / "registry").resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )["dflash2_default"]
    baseline_marker = tmp_path / "baseline-still-serving"
    baseline_marker.write_text("untouched")

    def downloader(repo_id: str, revision: str, local_dir: str) -> str:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "partial").write_text("partial")
        raise RuntimeError("network interrupted")

    store = CandidateStore(tmp_path, downloader=downloader)
    with pytest.raises(CandidateStoreError, match="network interrupted"):
        store.prepare(resolved, local_policy())

    assert baseline_marker.read_text() == "untouched"
    assert not (store.candidates_dir / resolved.manifest_id / "READY").exists()
    assert not list(store.candidates_dir.glob(".prepare-*"))


def test_candidate_store_rejects_budget_before_downloading(tmp_path):
    manifest = parse_manifest(manifest_dict())
    resolved = CandidateRegistry.from_manifests([manifest]).resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )["dflash2_default"]
    called = False

    def downloader(repo_id: str, revision: str, local_dir: str) -> str:
        nonlocal called
        called = True
        return local_dir

    store = CandidateStore(tmp_path, downloader=downloader)
    with pytest.raises(CandidateStoreError, match="download budget"):
        store.prepare(
            resolved,
            local_policy(max_download_gb_per_candidate=1.0),
        )
    assert called is False


def test_cli_exposes_registry_and_candidate_prepare_commands():
    parser = build_parser()
    registry = parser.parse_args(["registry", "list", "--json"])
    inspect = parser.parse_args(
        ["registry", "inspect", "qwen38-dflash2-v1", "--json"]
    )
    prepare = parser.parse_args(
        ["candidates", "prepare", "dflash2_default", "--json"]
    )
    serve = parser.parse_args(["serve", "start"])
    assert registry.registry_command == "list"
    assert inspect.candidate_id == "qwen38-dflash2-v1"
    assert prepare.candidates_command == "prepare"
    assert prepare.policy == "builtin"
    assert serve.target_revision is None
    assert _effective_target_revision(serve.model, serve.target_revision) == TARGET_REVISION
    assert _effective_target_revision("other/model", None) is None


def test_prepared_candidate_rewrites_draft_to_local_path(tmp_path):
    manifest = parse_manifest(manifest_dict())
    resolved = CandidateRegistry.from_manifests([manifest]).resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )["dflash2_default"]
    local = tmp_path / "candidate" / "assets"
    local.mkdir(parents=True)
    prepared = PreparedCandidate(
        candidate_id=resolved.id,
        manifest_id=resolved.manifest_id,
        manifest_hash=resolved.manifest_hash,
        revision=DRAFT_REVISION,
        root=local.parent,
        local_model_path=local,
        ready=True,
        reused=False,
    )
    applied = prepared.apply(resolved)
    assert applied.draft_model == str(local)
    assert applied.local_model_path == str(local)


def test_ready_candidate_is_rejected_after_file_tampering(tmp_path):
    manifest = parse_manifest(manifest_dict())
    candidates = CandidateRegistry.from_manifests([manifest]).resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )
    candidate = candidates["dflash2_default"]

    def downloader(repo_id: str, revision: str, local_dir: str) -> str:
        assets = Path(local_dir)
        (assets / "model.safetensors").write_bytes(b"original")
        return local_dir

    store = CandidateStore(tmp_path, downloader=downloader)
    prepared = store.prepare(candidate, local_policy())
    (prepared.local_model_path / "model.safetensors").write_bytes(b"tampered")
    assert store.get_ready(candidate) is None


def test_all_variants_use_one_verified_local_candidate_store(tmp_path):
    manifest = parse_manifest(manifest_dict())
    candidates = CandidateRegistry.from_manifests([manifest]).resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )

    def downloader(repo_id: str, revision: str, local_dir: str) -> str:
        assets = Path(local_dir)
        (assets / "model.safetensors").write_bytes(b"weights")
        return local_dir

    store = CandidateStore(tmp_path, downloader=downloader)
    store.prepare(candidates["dflash2_default"], local_policy())
    local_candidates = store.apply_ready_candidates(candidates)
    assert local_candidates["baseline"].draft_model is None
    assert local_candidates["dflash2_default"].local_model_path
    assert (
        local_candidates["dflash2_default"].local_model_path
        == local_candidates["dflash2_block6"].local_model_path
    )

    specs, decision = validate_plan(
        {"experiments": [{"id": "dflash2_default"}]},
        resolved_candidates=local_candidates,
    )
    assert specs[1].draft_model == local_candidates["dflash2_default"].local_model_path
    assert decision.manifest_hashes["dflash2_default"] == manifest.manifest_hash


def test_planner_context_is_derived_from_resolved_registry_snapshot():
    manifest = parse_manifest(manifest_dict())
    candidates = CandidateRegistry.from_manifests([manifest]).resolve_all(
        policy=local_policy(), environment=compatible_environment()
    )
    context = build_context(
        compatible_environment(),
        {"generation_tps": 12.3, "quality_pass": True},
        resolved_candidates=candidates,
    )
    allowed = {item["id"]: item for item in context["allowed_candidates"]}
    assert set(allowed) == set(candidates)
    assert allowed["dflash2_default"]["manifest_hash"] == manifest.manifest_hash
    assert allowed["dflash2_default"]["source_revision"] == DRAFT_REVISION
    assert context["runtime"]["available_candidate_sources"] == [
        "z-lab/Qwen3.8-27B-DFlash2"
    ]
