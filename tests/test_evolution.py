"""P2 tests for the single-run self-evolution outer loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra_team.candidate_manifest import parse_manifest
from infra_team.candidate_registry import CandidateRegistry
from infra_team.cli import build_parser
from infra_team.compatibility import AutonomyPolicy
from infra_team.evolution import EvolutionEngine, EvolutionState
from infra_team.service_manager import ServiceSpec


REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
TARGET_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"


def manifest():
    return parse_manifest(
        {
            "schema_version": 1,
            "id": "qwen38-dflash2-v1",
            "kind": "acceleration_plugin",
            "status": "active",
            "description": "test",
            "source": {
                "provider": "huggingface",
                "repo_id": "z-lab/Qwen3.8-27B-DFlash2",
                "revision": REVISION,
            },
            "compatibility": {
                "target_model": "mlx-community/Qwen3.8-27B-4bit",
                "target_revision": TARGET_REVISION,
                "runtime": "mlx-vlm",
                "runtime_min_version": "0.6.16",
                "platform": "apple-silicon",
            },
            "artifacts": {"allow_remote_code": False, "estimated_download_gb": 3.85},
            "resources": {"max_peak_memory_gb": 28},
            "launch": {
                "template": "mlx_vlm_dflash2",
                "parameters": {
                    "draft_kind": "dflash",
                    "variants": [
                        {
                            "id": "dflash2_default",
                            "description": "native",
                            "draft_block_size": None,
                        },
                        {
                            "id": "dflash2_block6",
                            "description": "block6",
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
    )


def environment() -> dict:
    return {
        "arch": "arm64",
        "chip": "Apple M5 Pro",
        "gpu_cores": 20,
        "unified_memory_gb": 48,
        "macos_version": "26.5.2",
        "disk_free_gb": 500,
        "packages": {"mlx": "0.32.2", "mlx-vlm": "0.6.16"},
    }


class FakeServiceManager:
    def __init__(self):
        self.healthy = True
        self.spec = {
            "name": "default",
            "model": "mlx-community/Qwen3.8-27B-4bit",
            "host": "127.0.0.1",
            "port": 8000,
            "draft_model": None,
            "draft_kind": None,
            "draft_block_size": None,
            "max_tokens": None,
            "enable_thinking": False,
        }

    def status(self):
        return {
            "name": "default",
            "managed": self.healthy,
            "healthy": self.healthy,
            "status": "running" if self.healthy else "unhealthy",
            "base_url": "http://127.0.0.1:8000",
            "spec": dict(self.spec),
        }

    def start(self, spec, wait_timeout=900.0):
        self.spec = spec.to_dict()
        self.healthy = True
        return self.status()


class FakePrepared:
    def __init__(self, candidate, root):
        self.candidate_id = candidate.id
        self.manifest_id = candidate.manifest_id
        self.manifest_hash = candidate.manifest_hash
        self.revision = candidate.source_revision
        self.root = Path(root) / candidate.manifest_id
        self.local_model_path = self.root / "assets"
        self.local_model_path.mkdir(parents=True, exist_ok=True)
        self.ready = True
        self.reused = False

    def apply(self, candidate):
        return candidate.with_local_model_path(str(self.local_model_path))

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "manifest_id": self.manifest_id,
            "manifest_hash": self.manifest_hash,
            "revision": self.revision,
            "root": str(self.root),
            "local_model_path": str(self.local_model_path),
            "ready": True,
            "reused": False,
        }


class FakeStore:
    def __init__(self, root, service, fail=False):
        self.root = Path(root)
        self.service = service
        self.fail = fail
        self.calls = []

    def prepare(self, candidate, policy):
        assert self.service.status()["healthy"] is True
        self.calls.append(candidate.id)
        if self.fail:
            raise RuntimeError("download interrupted")
        return FakePrepared(candidate, self.root)

    def apply_ready_candidates(self, candidates):
        return candidates


class FakeMemory:
    def __init__(self):
        self.evolutions = []
        self.summaries = []

    def query_prior(self, environment, target_model, target_revision, resolved_candidates):
        return {
            "environment_fingerprint": "fp",
            "ranking": [
                {
                    "candidate_id": "dflash2_default",
                    "match": "none",
                    "recommendation": "experiment",
                }
            ],
            "requires_local_validation": True,
        }

    def record_supervisor_summary(self, summary, **kwargs):
        self.summaries.append((summary, kwargs))
        return {"candidate_runs_written": 2, "promotion_written": True}

    def record_evolution(self, **kwargs):
        self.evolutions.append(kwargs)


class FakeSupervisor:
    def __init__(
        self,
        report,
        resolved_candidates,
        accepted=True,
        fail=False,
        baseline_ok=True,
    ):
        self.report = report
        self.resolved_candidates = resolved_candidates
        self.accepted = accepted
        self.fail = fail
        self.baseline_ok = baseline_ok

    def run(self):
        assert self.resolved_candidates["dflash2_default"].local_model_path
        if not self.baseline_ok:
            return {
                "run_id": "run-baseline-failed",
                "path": "/tmp/run-baseline-failed",
                "environment": environment(),
                "baseline": {
                    "id": "baseline",
                    "ok": False,
                    "spec": {"id": "baseline", "target_model": "mlx-community/Qwen3.8-27B-4bit"},
                    "quality": {},
                    "performance": {},
                    "error": "baseline failed",
                },
                "candidates": [],
                "verdict": {
                    "selected_id": "baseline",
                    "accepted": False,
                    "speedup_percent": 0,
                    "evaluations": [],
                },
                "service_transition": None,
            }
        self.report("plan_start", {})
        self.report("plan_done", {"plan": {"experiments": []}, "meta": {}})
        if self.fail:
            raise RuntimeError("candidate crashed")
        self.report("service_switch_start", {"name": "default", "base_url": "http://127.0.0.1:8000"})
        self.report("candidate_start", {"id": "dflash2_default"})
        self.report("candidate_done", {"id": "dflash2_default", "ok": True})
        self.report("verdict", {"accepted": self.accepted})
        if self.accepted:
            self.report("promotion_start", {"candidate_id": "dflash2_default"})
            self.report("online_verify_start", {"candidate_id": "dflash2_default"})
            self.report("service_switch_done", {"status": "serving_selected"})
        return {
            "run_id": "run-1",
            "path": "/tmp/run-1",
            "environment": environment(),
            "plan": {"experiments": [{"id": "dflash2_default"}]},
            "plan_meta": {"source": "target_service"},
            "policy": {"approved": ["baseline", "dflash2_default"]},
            "baseline": {
                "id": "baseline",
                "ok": True,
                "spec": {"id": "baseline", "target_model": "mlx-community/Qwen3.8-27B-4bit"},
                "quality": {"quality_pass": True},
                "performance": {"generation_tps_median": 18, "peak_memory_gb": 16, "error_rate": 0},
            },
            "candidates": [
                {
                    "id": "dflash2_default",
                    "ok": True,
                    "spec": {
                        "id": "dflash2_default",
                        "target_model": "mlx-community/Qwen3.8-27B-4bit",
                        "draft_model": self.resolved_candidates["dflash2_default"].draft_model,
                    },
                    "quality": {"quality_pass": True},
                    "performance": {"generation_tps_median": 36, "peak_memory_gb": 22, "error_rate": 0},
                }
            ],
            "verdict": {
                "selected_id": "dflash2_default" if self.accepted else "baseline",
                "accepted": self.accepted,
                "speedup_percent": 100 if self.accepted else 0,
                "evaluations": [],
            },
            "service_transition": {"status": "serving_selected"} if self.accepted else None,
            "elapsed_seconds": 1,
        }


def make_engine(
    tmp_path,
    accepted=True,
    prepare_fail=False,
    supervisor_fail=False,
    baseline_ok=True,
):
    service = FakeServiceManager()
    store = FakeStore(tmp_path / "store", service, fail=prepare_fail)
    memory = FakeMemory()
    registry = CandidateRegistry.from_manifests([manifest()])

    def supervisor_factory(**kwargs):
        return FakeSupervisor(
            kwargs["report"],
            kwargs["resolved_candidates"],
            accepted=accepted,
            fail=supervisor_fail,
            baseline_ok=baseline_ok,
        )

    engine = EvolutionEngine(
        root=tmp_path,
        service_name="default",
        policy=AutonomyPolicy(),
        registry=registry,
        store=store,
        memory=memory,
        service_manager=service,
        probe_fn=environment,
        supervisor_factory=supervisor_factory,
        repeats=3,
    )
    return engine, service, store, memory


def read_events(path: str):
    return [json.loads(line) for line in Path(path, "events.ndjson").read_text().splitlines()]


def test_evolve_once_runs_full_outer_loop_and_records_memory(tmp_path):
    engine, service, store, memory = make_engine(tmp_path)
    result = engine.run_once()
    assert result["status"] == EvolutionState.COMPLETED.value
    assert result["supervisor_run_id"] == "run-1"
    assert store.calls == ["dflash2_default"]
    assert service.status()["healthy"] is True
    assert len(memory.summaries) == 1
    assert memory.evolutions[-1]["status"] == EvolutionState.COMPLETED.value

    events = read_events(result["path"])
    states = [event["state"] for event in events]
    for state in (
        "TRIGGERED",
        "DISCOVERING",
        "PREFLIGHTING",
        "PREPARING",
        "READY",
        "PLANNING",
        "PLAN_FROZEN",
        "MAINTENANCE",
        "EXPERIMENTING",
        "SELECTING",
        "PROMOTING",
        "ONLINE_VERIFYING",
        "REMEMBERING",
        "COMPLETED",
    ):
        assert state in states
    for name in (
        "trigger.json",
        "registry_snapshot.json",
        "candidate_manifest.json",
        "candidate_resolution.json",
        "candidate_prepare.json",
        "memory_prior.json",
        "active_recipe_before.json",
        "active_recipe_after.json",
        "experience_update.json",
        "summary.json",
    ):
        assert Path(result["path"], name).is_file(), name


def test_evolution_freezes_running_spec_when_stored_recipe_is_stale(tmp_path):
    engine, service, _store, _memory = make_engine(tmp_path)
    stale_spec = dict(service.spec)
    stale_spec["draft_model"] = "stale-draft"
    service.active_recipe = lambda: {
        "schema_version": 1,
        "service_name": "default",
        "candidate_id": "stale",
        "service_spec": stale_spec,
    }

    result = engine.run_once()
    before = json.loads(
        Path(result["path"], "active_recipe_before.json").read_text()
    )

    assert before["source"] == "running_service_snapshot"
    assert before["spec"] == service.spec

    stale = tmp_path / ".infra-team" / "evolution" / "99999999-235959-stale"
    stale.mkdir(parents=True)
    (stale / "events.ndjson").write_text(
        json.dumps({"state": "MAINTENANCE", "message": "interrupted"}) + "\n"
    )
    (stale / "active_recipe_before.json").write_text(json.dumps(before))
    service.spec = stale_spec
    service.healthy = False

    recovered = engine.run_once()

    assert recovered["status"] == EvolutionState.BASELINE_RESTORED.value
    assert ServiceSpec(**service.spec) == ServiceSpec(**before["spec"])
    assert service.healthy is True


def test_evolution_preserves_matching_stored_recipe_metadata(tmp_path):
    engine, service, _store, _memory = make_engine(tmp_path)
    stored = {
        "schema_version": 1,
        "service_name": "default",
        "candidate_id": "baseline",
        "service_spec": dict(service.spec),
        "promoted_from_run": "trusted-run",
        "online_quality": {"quality_pass": True},
    }
    service.active_recipe = lambda: stored

    result = engine.run_once()
    before = json.loads(
        Path(result["path"], "active_recipe_before.json").read_text()
    )

    assert before == stored


def test_no_improvement_is_completed_and_remembered(tmp_path):
    engine, _service, _store, memory = make_engine(tmp_path, accepted=False)
    result = engine.run_once()
    assert result["status"] == EvolutionState.NO_IMPROVEMENT.value
    assert memory.evolutions[-1]["status"] == EvolutionState.NO_IMPROVEMENT.value
    assert len(memory.summaries) == 1


def test_supervisor_baseline_failure_is_not_mislabeled_no_improvement(tmp_path):
    engine, service, _store, memory = make_engine(tmp_path, baseline_ok=False)
    result = engine.run_once()
    assert result["status"] == EvolutionState.BASELINE_FAILED.value
    assert service.status()["healthy"] is True
    assert memory.evolutions[-1]["status"] == EvolutionState.BASELINE_FAILED.value
    assert len(memory.summaries) == 1


def test_prepare_failure_does_not_stop_baseline_or_run_supervisor(tmp_path):
    engine, service, _store, memory = make_engine(tmp_path, prepare_fail=True)
    result = engine.run_once()
    assert result["status"] == EvolutionState.PREPARE_FAILED.value
    assert service.status()["healthy"] is True
    assert memory.evolutions[-1]["status"] == EvolutionState.PREPARE_FAILED.value
    assert memory.summaries == []
    states = [event["state"] for event in read_events(result["path"])]
    assert "PLAN_FROZEN" not in states
    assert "MAINTENANCE" not in states


def test_evolve_refuses_to_prepare_without_healthy_managed_baseline(tmp_path):
    engine, service, store, memory = make_engine(tmp_path)
    service.healthy = False
    result = engine.run_once()
    assert result["status"] == EvolutionState.BASELINE_FAILED.value
    assert store.calls == []
    assert memory.evolutions[-1]["status"] == EvolutionState.BASELINE_FAILED.value


def test_supervisor_failure_after_plan_freeze_is_audited(tmp_path):
    engine, service, _store, memory = make_engine(tmp_path, supervisor_fail=True)
    result = engine.run_once()
    assert result["status"] == EvolutionState.CANDIDATE_FAILED.value
    assert service.status()["healthy"] is True
    states = [event["state"] for event in read_events(result["path"])]
    assert "PLAN_FROZEN" in states
    assert states[-1] == "CANDIDATE_FAILED"
    assert memory.evolutions[-1]["error"] == "candidate crashed"


def test_evolution_lock_refuses_parallel_run(tmp_path):
    engine, *_ = make_engine(tmp_path)
    engine.lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with engine.lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another evolution"):
            engine.run_once()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_interrupted_pre_freeze_run_is_closed_then_new_cycle_runs(tmp_path):
    engine, _service, _store, memory = make_engine(tmp_path)
    stale = tmp_path / ".infra-team" / "evolution" / "20260828-180000-stale"
    stale.mkdir(parents=True)
    (stale / "events.ndjson").write_text(
        json.dumps({"state": "PREPARING", "message": "interrupted"}) + "\n"
    )
    result = engine.run_once()
    assert result["status"] == EvolutionState.COMPLETED.value
    stale_events = read_events(str(stale))
    assert stale_events[-1]["state"] == EvolutionState.DISCOVERY_REJECTED.value
    assert any(
        row["evolution_id"] == "20260828-180000-stale"
        and row["status"] == EvolutionState.DISCOVERY_REJECTED.value
        for row in memory.evolutions
    )


def test_interrupted_post_freeze_run_restores_previous_recipe_without_replanning(tmp_path):
    engine, service, store, memory = make_engine(tmp_path)
    stale = tmp_path / ".infra-team" / "evolution" / "20260828-190000-stale"
    stale.mkdir(parents=True)
    (stale / "events.ndjson").write_text(
        json.dumps({"state": "MAINTENANCE", "message": "interrupted"}) + "\n"
    )
    (stale / "active_recipe_before.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "service_name": "default",
                "candidate_id": "baseline",
                "service_spec": service.spec,
            }
        )
    )
    service.healthy = False
    result = engine.run_once()
    assert result["status"] == EvolutionState.BASELINE_RESTORED.value
    assert result["evolution_id"] == "20260828-190000-stale"
    assert service.status()["healthy"] is True
    assert store.calls == []
    assert memory.summaries == []
    assert read_events(result["path"])[-1]["state"] == "BASELINE_RESTORED"


def test_cli_exposes_evolve_once_and_memory_commands():
    parser = build_parser()
    evolve = parser.parse_args(
        ["evolve", "--once", "--service-name", "default", "--repeats", "3"]
    )
    show = parser.parse_args(["memory", "show", "--json"])
    query = parser.parse_args(
        [
            "memory",
            "query",
            "--target",
            "mlx-community/Qwen3.8-27B-4bit",
            "--target-revision",
            TARGET_REVISION,
            "--json",
        ]
    )
    assert evolve.once is True
    assert evolve.service_name == "default"
    assert show.memory_command == "show"
    assert query.target_revision == TARGET_REVISION
