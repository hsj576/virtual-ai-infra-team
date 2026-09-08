"""Planner routing and managed-service safety tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from infra_team.candidate_registry import CandidateRegistry
from infra_team.compatibility import AutonomyPolicy
from infra_team.planner import (
    OpenAICompatiblePlanner,
    PlannerError,
    PlannerRouter,
    RuleBasedPlanner,
    _chat_endpoint,
    _extract_json,
)
from infra_team.policy import PolicyError
from infra_team.service_benchmark import benchmark_service
from infra_team.runner import CandidateResult, CandidateSpec
from infra_team.selector import AcceptancePolicy
from infra_team.service_manager import (
    ManagedService,
    ServiceError,
    ServiceSpec,
    build_mlx_server_command,
)
from infra_team.supervisor import Supervisor


VALID_PLAN = {
    "hypothesis": "test",
    "experiments": [
        {"id": "baseline", "reason": "reference"},
        {"id": "dflash2_default", "reason": "candidate"},
    ],
    "acceptance_policy": {
        "quality_pass": True,
        "minimum_speedup_percent": 5,
        "max_memory_gb": 40,
    },
}


def test_supervisor_refuses_policy_without_frozen_candidate_snapshot(tmp_path):
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)

    with pytest.raises(PolicyError, match="candidate snapshot is not initialized"):
        supervisor.enforce_policy(VALID_PLAN)


def test_supervisor_fallback_cannot_reuse_weakened_planner_thresholds(tmp_path):
    resolved = CandidateRegistry.builtin().resolve_all()
    supervisor = Supervisor(
        str(tmp_path),
        managed_service_name=None,
        resolved_candidates=resolved,
        autonomy_policy=AutonomyPolicy(),
    )
    malformed = {
        "experiments": [{"id": "baseline"}],
        "acceptance_policy": {
            "minimum_speedup_percent": 1,
            "max_memory_gb": 44,
        },
    }

    _, payload, policy = supervisor.enforce_policy(malformed)

    assert payload["policy_fallback_used"] is True
    assert policy.minimum_speedup_percent == 10.0
    assert policy.max_memory_gb == 28.0
    assert payload["acceptance_policy_sources"]["planner"] == {
        "max_memory_gb": 40.0,
        "minimum_speedup_percent": 5.0,
    }


class FakePlanner:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = 0

    def create_plan(self, context):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

    def describe(self):
        return {"type": "fake", "name": self.name}


def test_router_uses_target_first_without_touching_fallback():
    target = FakePlanner("target_service", json.dumps(VALID_PLAN))
    fallback = FakePlanner("local_fallback", json.dumps(VALID_PLAN))
    router = PlannerRouter([target, fallback, RuleBasedPlanner()])

    plan, meta = router.create_plan({"machine": "test"})

    assert plan["source"] == "target_service"
    assert meta["source"] == "target_service"
    assert meta["failover_used"] is False
    assert target.calls == 1
    assert fallback.calls == 0


def test_router_fails_over_to_local_only_when_target_fails():
    target = FakePlanner("target_service", error=PlannerError("offline"))
    fallback = FakePlanner("local_fallback", json.dumps(VALID_PLAN))
    router = PlannerRouter([target, fallback, RuleBasedPlanner()])

    plan, meta = router.create_plan({})

    assert plan["source"] == "local_fallback"
    assert meta["failover_used"] is True
    assert [a["backend"] for a in meta["attempts"]] == [
        "target_service",
        "local_fallback",
    ]
    assert meta["attempts"][0]["ok"] is False
    assert target.calls == fallback.calls == 1


def test_router_skips_malformed_plan_and_uses_next_backend():
    target = FakePlanner("target_service", "not json")
    external = FakePlanner("external_api", json.dumps(VALID_PLAN))
    plan, meta = PlannerRouter([target, external, RuleBasedPlanner()]).create_plan({})
    assert plan["source"] == "external_api"
    assert len(meta["attempts"]) == 2


def test_router_always_has_rule_based_safety_net():
    target = FakePlanner("target_service", error=PlannerError("offline"))
    local = FakePlanner("local_fallback", error=MemoryError("cannot load"))
    plan, meta = PlannerRouter([target, local, RuleBasedPlanner()]).create_plan({})
    assert plan["source"] == "rule_based"
    assert meta["failover_used"] is True
    assert len(plan["experiments"]) >= 2


def test_openai_planner_description_never_leaks_api_key():
    planner = OpenAICompatiblePlanner(
        name="external_api",
        base_url="https://example.invalid/v1",
        model="some-model",
        api_key="top-secret",
        locality="external",
    )
    assert "top-secret" not in json.dumps(planner.describe())
    assert planner.describe()["authenticated"] is True


def test_chat_endpoint_accepts_base_with_or_without_v1():
    assert _chat_endpoint("http://localhost:8000") == (
        "http://localhost:8000/v1/chat/completions"
    )
    assert _chat_endpoint("http://localhost:8000/v1") == (
        "http://localhost:8000/v1/chat/completions"
    )


def test_json_extractor_handles_braces_inside_strings():
    text = json.dumps(
        {
            "hypothesis": "a { brace } inside a string",
            "experiments": [{"id": "baseline"}, {"id": "dflash2_default"}],
        }
    )
    assert _extract_json(text)["hypothesis"] == "a { brace } inside a string"


def test_service_command_is_a_fixed_argv_template():
    spec = ServiceSpec(
        name="qwen",
        model="model/path",
        port=8123,
        draft_model="draft/path",
        draft_kind="dflash",
        draft_block_size=4,
    )
    command = build_mlx_server_command(spec, "/safe/python")
    assert command[:3] == ["/safe/python", "-m", "mlx_vlm.server"]
    assert command[command.index("--model") + 1] == "model/path"
    assert command[command.index("--draft-model") + 1] == "draft/path"
    assert "shell" not in command


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 0},
        {"port": 70000},
        {"draft_kind": "invented"},
        {"draft_block_size": 99},
    ],
)
def test_service_command_rejects_unbounded_parameters(kwargs):
    raw = {"name": "x", "model": "m"}
    raw.update(kwargs)
    with pytest.raises(ServiceError):
        build_mlx_server_command(ServiceSpec(**raw), "/safe/python")


def test_service_spec_keeps_recipe_provenance_and_launches_resolved_snapshot():
    spec = ServiceSpec(
        name="qwen",
        model="org/model",
        target_revision="a" * 40,
        resolved_model_path="/cache/snapshots/" + "a" * 40,
        candidate_manifest_id="candidate-v1",
        candidate_manifest_hash="b" * 64,
        runtime_version="0.6.16",
    )
    command = build_mlx_server_command(spec, "/safe/python")
    assert command[command.index("--model") + 1] == spec.resolved_model_path
    assert "candidate-v1" not in command
    assert "b" * 64 not in command
    assert spec.to_dict()["target_revision"] == "a" * 40


def test_target_revision_is_resolved_to_fixed_huggingface_snapshot(monkeypatch, tmp_path):
    resolved = tmp_path / "models" / "snapshots" / ("a" * 40)
    resolved.mkdir(parents=True)
    calls = []

    def fake_snapshot_download(repo_id, revision):
        calls.append((repo_id, revision))
        return str(resolved)

    monkeypatch.setattr(
        "infra_team.service_manager.snapshot_download", fake_snapshot_download
    )
    spec = ServiceSpec(
        name="default", model="org/model", target_revision="a" * 40
    )
    pinned = ManagedService._resolve_target_snapshot(spec)
    assert calls == [("org/model", "a" * 40)]
    assert pinned.model == "org/model"
    assert pinned.resolved_model_path == str(resolved)
    command = build_mlx_server_command(pinned, "/safe/python")
    assert command[command.index("--model") + 1] == str(resolved)


def test_active_recipe_promotion_is_atomic_and_tracks_previous(tmp_path, monkeypatch):
    manager = ManagedService(str(tmp_path), name="default")
    first = ServiceSpec(name="default", model="target")
    second = ServiceSpec(
        name="default",
        model="target",
        draft_model="/local/draft",
        draft_kind="dflash",
        candidate_manifest_id="qwen38-dflash2-v1",
        candidate_manifest_hash="c" * 64,
    )
    status = {"healthy": True, "spec": first.to_dict()}
    monkeypatch.setattr(manager, "status", lambda: status)
    manager.atomically_promote(
        manager.recipe_from_spec(first, source="manual", run_id="run-1")
    )
    assert manager.active_recipe()["service_spec"]["draft_model"] is None
    assert manager.previous_recipe() is None

    status["spec"] = second.to_dict()
    manager.atomically_promote(
        manager.recipe_from_spec(
            second,
            source="evolution",
            run_id="run-2",
            candidate_id="dflash2_default",
            online_quality={"quality_pass": True, "passed": 4, "total": 4},
        )
    )
    active = manager.active_recipe()
    previous = manager.previous_recipe()
    assert active["candidate_id"] == "dflash2_default"
    assert active["service_spec"]["draft_model"] == "/local/draft"
    assert active["online_quality"]["quality_pass"] is True
    assert previous["promoted_from_run"] == "run-1"
    assert len(list((tmp_path / ".infra-team/services/default/recipe_history").glob("*.json"))) == 2


def test_active_recipe_refuses_unhealthy_or_mismatched_service(tmp_path, monkeypatch):
    manager = ManagedService(str(tmp_path), name="default")
    spec = ServiceSpec(name="default", model="target")
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {"healthy": False, "spec": spec.to_dict()},
    )
    with pytest.raises(ServiceError, match="healthy"):
        manager.atomically_promote(manager.recipe_from_spec(spec, source="test"))

    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "healthy": True,
            "spec": ServiceSpec(name="default", model="other").to_dict(),
        },
    )
    with pytest.raises(ServiceError, match="does not match"):
        manager.atomically_promote(manager.recipe_from_spec(spec, source="test"))


def test_status_without_state_is_safe(tmp_path):
    manager = ManagedService(str(tmp_path), name="missing")
    assert manager.status() == {
        "name": "missing",
        "status": "not_deployed",
        "managed": False,
    }


def test_public_state_redacts_control_capability():
    public = ManagedService._public_state(
        {
            "pid": 123,
            "signature": "secret",
            "control_host": "127.0.0.1",
            "control_port": 9999,
            "spec": {"model": "x"},
        }
    )
    assert public["pid"] == 123
    assert public["spec"] == {"model": "x"}
    assert "signature" not in public
    assert "control_host" not in public
    assert "control_port" not in public


def test_stop_without_state_is_idempotent(tmp_path):
    manager = ManagedService(str(tmp_path), name="missing")
    result = manager.stop()
    assert result["stopped"] is False


def test_stop_refuses_unowned_pid(tmp_path):
    manager = ManagedService(str(tmp_path), name="x")
    manager._write_state(
        {
            "name": "x",
            "pid": 1,
            "signature": "not-ours",
            "spec": {"host": "127.0.0.1", "port": 9999},
        }
    )
    with pytest.raises(ServiceError, match="refusing to stop"):
        manager.stop()


class FakeMLXHandler(BaseHTTPRequestHandler):
    latest = None
    request_count = 0

    def log_message(self, _format, *args):
        pass

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._json({"status": "healthy", "loaded_model": "fake"})
        elif self.path in ("/metrics", "/v1/metrics"):
            self._json({"latest": type(self).latest, "recent": []})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = payload["messages"][-1]["content"]
        if "Return the ExperimentPlan JSON" in prompt:
            text = json.dumps(VALID_PLAN)
        elif "17 multiplied" in prompt:
            text = "391"
        elif "capital of France" in prompt:
            text = '{"city":"Paris","country":"France"}'
        elif "midday sky" in prompt:
            text = "blue"
        elif "function named add" in prompt:
            text = "```python\ndef add(a, b):\n    return a + b\n```"
        else:
            text = "A sufficiently long deterministic benchmark response."
        type(self).request_count += 1
        type(self).latest = {
            "timestamp_unix": __import__("time").time(),
            "endpoint": "/chat/completions",
            "model": payload["model"],
            "prompt_tokens": 20,
            "generated_tokens": 40,
            "decode_tok_s": 20.0 + type(self).request_count,
            "prefill_tok_s": 100.0,
            "ttft_s": 0.05,
            "request_elapsed_s": 2.0,
            "peak_memory_gb": 21.5,
            "finish_reason": "stop",
        }
        self._json(
            {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 40},
            }
        )


def test_real_openai_backend_can_plan_through_target_service_api():
    FakeMLXHandler.latest = None
    FakeMLXHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMLXHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        planner = OpenAICompatiblePlanner(
            name="target_service",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-model",
        )
        plan, meta = PlannerRouter([planner, RuleBasedPlanner()]).create_plan(
            {"hardware": {"chip": "test"}}
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert plan["source"] == "target_service"
    assert meta["source"] == "target_service"
    assert meta["failover_used"] is False


def test_running_service_is_benchmarked_through_its_api():
    FakeMLXHandler.latest = None
    FakeMLXHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMLXHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = benchmark_service(
            f"http://127.0.0.1:{server.server_port}", "fake-model", repeats=1
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.ok is True
    assert result.quality_pass is True
    assert result.quality["passed"] == result.quality["total"] == 4
    assert result.performance["source"] == "deployed_openai_compatible_service"
    assert result.performance["samples"] == 3
    assert result.peak_memory_gb == 21.5
    assert result.generation_tps > 0


class RecordingManager:
    def __init__(self, failures=0):
        self.failures = failures
        self.started = []
        self.stop_calls = 0

    def start(self, spec, wait_timeout=900.0):
        self.started.append(spec)
        if self.failures:
            self.failures -= 1
            raise ServiceError("candidate boot failed")
        return {"healthy": True, "spec": spec.to_dict()}

    def stop(self, timeout=30.0):
        self.stop_calls += 1
        return {"stopped": True}


def _measured(cid, tps, draft_model=None):
    return CandidateResult(
        id=cid,
        ok=True,
        spec={
            "id": cid,
            "target_model": "target",
            "draft_model": draft_model,
            "draft_kind": "dflash" if draft_model else None,
            "draft_block_size": 4 if draft_model else None,
            "enable_thinking": False,
        },
        quality={"quality_pass": True},
        performance={
            "generation_tps_median": tps,
            "peak_memory_gb": 22,
            "error_rate": 0,
        },
    )


def test_successful_promotion_reuses_the_stable_api_address(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "infra_team.supervisor.quality_gate_service",
        lambda *args, **kwargs: {"quality_pass": True, "passed": 4, "total": 4},
    )
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)
    manager = RecordingManager()
    supervisor.service_manager = manager
    supervisor.original_service_spec = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    baseline = _measured("baseline", 10)
    candidate = _measured("dflash2_default", 20, draft_model="draft")
    verdict = {"selected_id": candidate.id, "accepted": True}

    transition = supervisor.restore_or_promote_service(
        verdict, baseline, [candidate]
    )

    assert transition["status"] == "serving_selected"
    assert len(manager.started) == 1
    assert manager.started[0].port == 8000
    assert manager.started[0].draft_model == "draft"


def test_autonomy_policy_can_block_promotion_and_restore_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "infra_team.supervisor.quality_gate_service",
        lambda *args, **kwargs: {"quality_pass": True, "passed": 4, "total": 4},
    )
    supervisor = Supervisor(
        str(tmp_path), managed_service_name=None, allow_promotion=False
    )
    manager = RecordingManager()
    supervisor.service_manager = manager
    supervisor.original_service_spec = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    baseline = _measured("baseline", 10)
    candidate = _measured("dflash2_default", 20, draft_model="draft")
    verdict = {
        "selected_id": candidate.id,
        "accepted": True,
        "selected_tps": 20,
        "speedup_percent": 100,
    }

    transition = supervisor.restore_or_promote_service(
        verdict, baseline, [candidate]
    )

    assert transition["status"] == "baseline_retained_policy"
    assert manager.started[0].draft_model is None
    assert verdict["selected_id"] == "baseline"
    assert verdict["accepted"] is False
    assert "disabled by autonomy policy" in verdict["reason"]


def test_post_restart_quality_failure_restores_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "infra_team.supervisor.quality_gate_service",
        lambda *args, **kwargs: {"quality_pass": False, "passed": 3, "total": 4},
    )
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)
    manager = RecordingManager()
    supervisor.service_manager = manager
    baseline_spec = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    supervisor.original_service_spec = baseline_spec
    baseline = _measured("baseline", 10)
    candidate = _measured("dflash2_default", 20, draft_model="draft")
    verdict = {
        "selected_id": candidate.id,
        "accepted": True,
        "selected_tps": 20,
        "speedup_percent": 100,
    }

    transition = supervisor.restore_or_promote_service(
        verdict, baseline, [candidate]
    )

    assert transition["status"] == "baseline_restored"
    assert manager.stop_calls == 1
    assert len(manager.started) == 2
    assert verdict["accepted"] is False
    assert verdict["selected_id"] == "baseline"


def test_failed_promotion_restores_baseline_and_overrides_verdict(tmp_path):
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)
    manager = RecordingManager(failures=1)
    supervisor.service_manager = manager
    baseline_spec = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    supervisor.original_service_spec = baseline_spec
    baseline = _measured("baseline", 10)
    candidate = _measured("dflash2_default", 20, draft_model="draft")
    verdict = {
        "selected_id": candidate.id,
        "accepted": True,
        "selected_tps": 20,
        "speedup_percent": 100,
    }

    transition = supervisor.restore_or_promote_service(
        verdict, baseline, [candidate]
    )

    assert transition["status"] == "baseline_restored"
    assert len(manager.started) == 2
    assert manager.started[1] == baseline_spec
    assert verdict["accepted"] is False
    assert verdict["selected_id"] == "baseline"
    assert "baseline restored" in verdict["reason"]


def test_keyboard_interrupt_during_maintenance_restores_baseline(monkeypatch, tmp_path):
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)
    manager = RecordingManager()
    original = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    baseline = _measured("baseline", 10)
    supervisor.service_manager = manager
    supervisor.original_service_spec = original
    monkeypatch.setattr(supervisor, "probe", lambda: {"chip": "test"})
    monkeypatch.setattr(supervisor, "measure_baseline", lambda: baseline)
    monkeypatch.setattr(
        supervisor,
        "make_plan",
        lambda env, measured: (VALID_PLAN, {"source": "test"}),
    )
    monkeypatch.setattr(
        supervisor,
        "enforce_policy",
        lambda plan: (
            [CandidateSpec(id="baseline", target_model="target")],
            {},
            AcceptancePolicy(),
        ),
    )
    monkeypatch.setattr(supervisor, "stop_service_for_experiments", lambda: True)

    def interrupt(specs, measured):
        raise KeyboardInterrupt()

    monkeypatch.setattr(supervisor, "run_experiments", interrupt)

    with pytest.raises(KeyboardInterrupt):
        supervisor._run_impl()
    assert manager.started[-1] == original
    with open(supervisor.art.path + "/interrupt_recovery.json") as fh:
        assert json.load(fh)["status"] == "baseline_restored"


def test_keyboard_interrupt_reports_manual_recovery_when_rollback_fails(
    monkeypatch, tmp_path
):
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)
    manager = RecordingManager(failures=1)
    original = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    baseline = _measured("baseline", 10)
    supervisor.service_manager = manager
    supervisor.original_service_spec = original
    monkeypatch.setattr(supervisor, "probe", lambda: {"chip": "test"})
    monkeypatch.setattr(supervisor, "measure_baseline", lambda: baseline)
    monkeypatch.setattr(
        supervisor,
        "make_plan",
        lambda env, measured: (VALID_PLAN, {"source": "test"}),
    )
    monkeypatch.setattr(
        supervisor,
        "enforce_policy",
        lambda plan: (
            [CandidateSpec(id="baseline", target_model="target")],
            {},
            AcceptancePolicy(),
        ),
    )
    monkeypatch.setattr(supervisor, "stop_service_for_experiments", lambda: True)

    def interrupt(specs, measured):
        raise KeyboardInterrupt()

    monkeypatch.setattr(supervisor, "run_experiments", interrupt)
    with pytest.raises(RuntimeError, match="manual recovery required"):
        supervisor._run_impl()
    with open(supervisor.art.path + "/interrupt_recovery.json") as fh:
        assert json.load(fh)["status"] == "rollback_failed"


def test_service_candidates_use_same_api_benchmark_path(monkeypatch, tmp_path):
    supervisor = Supervisor(str(tmp_path), managed_service_name=None)
    manager = RecordingManager()
    supervisor.service_manager = manager
    supervisor.original_service_spec = ServiceSpec(
        name="default", model="target", host="127.0.0.1", port=8000
    )
    baseline = _measured("baseline", 10)
    candidate_spec = CandidateSpec(
        id="dflash2_default",
        target_model="target",
        draft_model="draft",
        draft_kind="dflash",
        repeats=1,
    )

    def fake_benchmark(base_url, model, repeats, timeout):
        return _measured("baseline", 20, draft_model="draft")

    monkeypatch.setattr("infra_team.supervisor.benchmark_service", fake_benchmark)
    results = supervisor.run_experiments([candidate_spec], baseline)

    assert results[0].id == "dflash2_default"
    assert results[0].generation_tps == 20
    assert manager.started[0].draft_model == "draft"
    assert manager.stop_calls == 1


def test_supervisor_can_reuse_evolution_workspace_lock(monkeypatch, tmp_path):
    supervisor = Supervisor(
        str(tmp_path),
        managed_service_name=None,
        optimization_lock_held=True,
    )
    monkeypatch.setattr(supervisor, "_run_impl", lambda: {"ok": True})
    lock_path = tmp_path / ".infra-team" / "optimize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert supervisor.run() == {"ok": True}
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_plan_artifact_is_marked_frozen_before_execution(tmp_path):
    supervisor = Supervisor(
        str(tmp_path),
        use_planner=False,
        managed_service_name=None,
        evolution_context={
            "trigger": {"source": "cli_once"},
            "manifest_hashes": {"qwen38-dflash2-v1": "abc"},
            "memory_prior": {"requires_local_validation": True},
            "autonomy_policy_hash": "policy-hash",
        },
    )
    baseline = _measured("baseline", 10)
    env = {
        "chip": "test",
        "gpu_cores": 1,
        "unified_memory_gb": 8,
        "cpu_cores": {"total": 1},
        "packages": {},
    }
    supervisor.make_plan(env, baseline)
    with open(
        supervisor.art.path + "/agent_plan.json", encoding="utf-8"
    ) as fh:
        artifact = json.load(fh)
    assert artifact["status"] == "frozen_before_execution"
    assert artifact["context"]["evolution"]["trigger"]["source"] == "cli_once"
    assert artifact["context"]["evolution"]["memory_prior"]["requires_local_validation"] is True
    assert artifact["context"]["evolution"]["autonomy_policy_hash"] == "policy-hash"
