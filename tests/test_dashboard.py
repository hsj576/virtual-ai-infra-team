"""Tests for the local deployment, control and chat dashboard."""

from __future__ import annotations

import fcntl
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from infra_team.dashboard import (
    DashboardController,
    DashboardData,
    _benchmark_evidence,
    _stream_openai_chat,
    create_dashboard_server,
)
from infra_team.recipe_memory import RecipeMemory
from infra_team.service_manager import ServiceSpec


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(
    root: Path, run_id: str, speedup: float = 100.0, repeats: int = 1
) -> Path:
    run = root / ".infra-team" / "runs" / run_id
    baseline = {
        "id": "baseline",
        "ok": True,
        "spec": {"id": "baseline", "target_model": "target", "repeats": repeats},
        "quality": {"quality_pass": True, "passed": 4, "total": 4},
        "performance": {
            "generation_tps_median": 10.0,
            "peak_memory_gb": 16.0,
            "error_rate": 0,
            "samples": 3,
        },
    }
    candidate = {
        "id": "dflash2_default",
        "ok": True,
        "spec": {
            "id": "dflash2_default",
            "target_model": "target",
            "draft_model": "draft",
            "repeats": repeats,
        },
        "quality": {"quality_pass": True, "passed": 4, "total": 4},
        "performance": {
            "generation_tps_median": 20.0,
            "peak_memory_gb": 21.0,
            "error_rate": 0,
            "samples": 3,
        },
    }
    verdict = {
        "selected_id": "dflash2_default",
        "accepted": True,
        "baseline_tps": 10.0,
        "selected_tps": 20.0,
        "speedup_percent": speedup,
        "reason": "faster and valid",
        "evaluations": [
            {
                "id": "baseline",
                "qualified": True,
                "speedup_percent": 0,
            },
            {
                "id": "dflash2_default",
                "qualified": True,
                "accepted": True,
                "speedup_percent": speedup,
            },
        ],
    }
    _write_json(
        run / "summary.json",
        {
            "run_id": run_id,
            "environment": {
                "chip": "Apple Test",
                "gpu_cores": 20,
                "unified_memory_gb": 48,
                "packages": {"mlx": "1", "mlx-vlm": "2"},
            },
            "plan": {"source": "target_service", "hypothesis": "test"},
            "plan_meta": {"source": "target_service"},
            "baseline": baseline,
            "candidates": [candidate],
            "verdict": verdict,
            "elapsed_seconds": 12.3,
        },
    )
    _write_json(run / "verdict.json", verdict)
    _write_json(
        run / "quality.json",
        {
            "dflash2_default": {
                "quality_pass": True,
                "checks": [
                    {"id": "arithmetic", "passed": True, "detail": "ok"},
                    {"id": "json_schema", "passed": True, "detail": "ok"},
                    {"id": "instruction_following", "passed": True, "detail": "ok"},
                    {"id": "code_generation", "passed": True, "detail": "ok"},
                ],
            }
        },
    )
    _write_json(
        run / "service_transition.json",
        {
            "status": "serving_selected",
            "post_restart_quality": {
                "quality_pass": True,
                "passed": 4,
                "total": 4,
                "checks": [
                    {"id": "arithmetic", "passed": True, "detail": "ok"},
                    {"id": "json_schema", "passed": True, "detail": "ok"},
                    {"id": "instruction_following", "passed": True, "detail": "ok"},
                    {"id": "code_generation", "passed": True, "detail": "ok"},
                ],
            },
        },
    )
    return run


def _make_evolution(
    root: Path,
    evolution_id: str = "20260902-120000-demo001",
    run_id: str = "20260902-120001",
    status: str = "COMPLETED",
) -> Path:
    _make_run(root, run_id, speedup=106.84, repeats=3)
    path = root / ".infra-team" / "evolution" / evolution_id
    path.mkdir(parents=True, exist_ok=True)
    events = [
        {"time": "2026-09-02T12:00:00+0800", "state": "DISCOVERING", "message": "Reading trusted registry"},
        {"time": "2026-09-02T12:00:05+0800", "state": "PREPARING", "message": "Candidate preparation complete"},
        {"time": "2026-09-02T12:00:10+0800", "state": "PLAN_FROZEN", "message": "ExperimentPlan frozen before maintenance"},
        {"time": "2026-09-02T12:01:00+0800", "state": "SELECTING", "message": "Independent selector produced a verdict"},
        {"time": "2026-09-02T12:01:10+0800", "state": status, "message": "Evolution completed and experience saved"},
    ]
    (path / "events.ndjson").write_text(
        "\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8"
    )
    _write_json(
        path / "summary.json",
        {
            "evolution_id": evolution_id,
            "status": status,
            "path": str(path),
            "supervisor_run_id": run_id,
            "supervisor_path": str(root / ".infra-team" / "runs" / run_id),
            "selected_id": "dflash2_default",
            "accepted": True,
            "speedup_percent": 106.84,
        },
    )
    _write_json(
        path / "active_recipe_before.json",
        {
            "schema_version": 1,
            "service_name": "default",
            "candidate_id": "baseline",
            "service_spec": {
                "name": "default",
                "model": "target",
                "host": "127.0.0.1",
                "port": 8000,
            },
        },
    )
    _write_json(
        path / "active_recipe_after.json",
        {
            "schema_version": 1,
            "service_name": "default",
            "candidate_id": "dflash2_default",
            "service_spec": {
                "name": "default",
                "model": "target",
                "host": "127.0.0.1",
                "port": 8000,
                "draft_model": str(root / ".infra-team" / "candidates" / "draft"),
            },
            "target": {"model": "target", "revision": "a" * 40},
            "runtime": {"name": "mlx-vlm", "version": "0.6.16"},
            "accelerator": {
                "manifest_id": "qwen38-dflash2-v1",
                "manifest_hash": "b" * 64,
                "draft_model": str(root / ".infra-team" / "candidates" / "draft"),
            },
            "online_quality": {"quality_pass": True, "passed": 4, "total": 4},
        },
    )
    _write_json(
        path / "memory_prior.json",
        {
            "environment_fingerprint": "c" * 64,
            "ranking": [
                {
                    "candidate_id": "dflash2_default",
                    "match": "exact",
                    "recommendation": "prioritize_revalidation",
                }
            ],
            "requires_local_validation": True,
        },
    )
    _write_json(
        path / "supervisor_link.json",
        {
            "run_id": run_id,
            "path": str(root / ".infra-team" / "runs" / run_id),
            "plan_artifact": str(root / ".infra-team" / "runs" / run_id / "agent_plan.json"),
        },
    )
    return path


def test_latest_valid_run_skips_explicitly_invalidated_run(tmp_path):
    valid = _make_run(tmp_path, "20260826-120000")
    invalid = _make_run(tmp_path, "20260826-130000")
    _write_json(invalid / "INVALIDATED.json", {"valid": False, "reason": "unfair"})

    data = DashboardData(str(tmp_path)).latest_valid_run()

    assert data["id"] == valid.name
    assert data["accepted"] is True
    assert data["quality"]["post_restart_verified"] is True
    assert data["benchmark_mode"] == "smoke"
    assert data["repeats"] == 1
    assert data["publishable_performance"] is False
    assert len(data["candidates"]) == 2


def test_repeated_run_is_marked_as_formal_performance_evidence(tmp_path):
    _make_run(tmp_path, "20260826-120000", repeats=3)

    data = DashboardData(str(tmp_path)).latest_valid_run()

    assert data["benchmark_mode"] == "formal"
    assert data["repeats"] == 3
    assert data["publishable_performance"] is True


def test_incomplete_or_mixed_repeat_evidence_is_never_marked_formal():
    missing = _benchmark_evidence(
        [{"spec": {"repeats": 3}}, {"spec": {}}]
    )
    mixed = _benchmark_evidence(
        [{"spec": {"repeats": 3}}, {"spec": {"repeats": 5}}]
    )
    invalid = [
        _benchmark_evidence(
            [{"spec": {"repeats": value}}, {"spec": {"repeats": value}}]
        )
        for value in (3.5, "3", True, 0)
    ]
    absent_baseline = _benchmark_evidence(
        [{}, {"spec": {"repeats": 3}}]
    )
    candidate_only = _benchmark_evidence([{"spec": {"repeats": 3}}])

    assert missing["benchmark_mode"] == "unclassified"
    assert missing["publishable_performance"] is False
    assert mixed["benchmark_mode"] == "unclassified"
    assert mixed["publishable_performance"] is False
    assert all(item["benchmark_mode"] == "unclassified" for item in invalid)
    assert all(item["publishable_performance"] is False for item in invalid)
    assert absent_baseline["benchmark_mode"] == "unclassified"
    assert candidate_only["benchmark_mode"] == "unclassified"


def test_recent_runs_keeps_invalid_run_visible_but_flagged(tmp_path):
    _make_run(tmp_path, "20260826-120000")
    invalid = _make_run(tmp_path, "20260826-130000")
    _write_json(invalid / "INVALIDATED.json", {"valid": False, "reason": "unfair"})

    runs = DashboardData(str(tmp_path)).recent_runs()

    assert runs[0]["id"] == invalid.name
    assert runs[0]["valid"] is False
    assert runs[0]["invalid_reason"] == "unfair"


def test_snapshot_does_not_expose_service_control_capability(tmp_path):
    manager = DashboardData(str(tmp_path)).service
    manager._write_state(
        {
            "name": "default",
            "pid": 123,
            "signature": "secret-capability",
            "control_host": "127.0.0.1",
            "control_port": 9999,
            "spec": {"host": "127.0.0.1", "port": 8000, "model": "target"},
        }
    )

    service = DashboardData(str(tmp_path)).snapshot()["service"]

    assert "signature" not in service
    assert "control_host" not in service
    assert "control_port" not in service
    assert "server_log" not in service
    assert str(tmp_path) not in json.dumps(DashboardData(str(tmp_path)).snapshot())


def test_fresh_workspace_uses_packaged_historical_evidence(tmp_path):
    data = DashboardData(str(tmp_path))
    snapshot = data.evolution_snapshot()
    replay = snapshot["latest"]
    assert replay["id"] == "20260831-212200-2b79db"
    assert replay["label"] == "内置历史真实运行回放"
    assert replay["mode"] == "historical_replay"
    assert replay["run"]["benchmark_mode"] == "formal"
    assert replay["run"]["selected_tps"] == 37.52
    assert data.evolution_detail(replay["id"]) == replay


def test_dashboard_snapshot_exposes_plain_language_product_state(tmp_path):
    evolution = _make_evolution(tmp_path)
    memory = RecipeMemory(tmp_path)
    memory.record_evolution(
        evolution_id=evolution.name,
        status="COMPLETED",
        trigger={"source": "cli_once"},
        artifact_path=str(evolution),
        supervisor_run_id="20260902-120001",
    )
    data = DashboardData(str(tmp_path))
    data.service.active_recipe = lambda: _read_json_for_test(
        evolution / "active_recipe_after.json"
    )
    data.service.previous_recipe = lambda: _read_json_for_test(
        evolution / "active_recipe_before.json"
    )

    snapshot = data.snapshot()

    assert snapshot["product"]["promise"] == "安装一次，以后本地 AI 自己测试和升级。"
    assert snapshot["stable_api"]["unchanged"] is True
    assert snapshot["recipes"]["active"]["candidate_id"] == "dflash2_default"
    assert snapshot["recipes"]["previous"]["candidate_id"] == "baseline"
    assert snapshot["recipes"]["recoverable"] is True
    assert snapshot["evolution"]["latest"]["id"] == evolution.name
    assert snapshot["evolution"]["latest"]["mode"] == "historical_replay"
    assert snapshot["memory"]["schema_version"] == 2
    assert snapshot["memory"]["evolution_runs"] == 1
    assert "path" not in snapshot["memory"]


def _read_json_for_test(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_evolution_replay_reads_immutable_artifacts_and_redacts_absolute_paths(tmp_path):
    path = _make_evolution(tmp_path)
    data = DashboardData(str(tmp_path))
    first = data.evolution_detail(path.name)
    data.service.status = lambda: {"healthy": False, "status": "changed"}
    second = data.evolution_detail(path.name)

    assert first == second
    assert first["mode"] == "historical_replay"
    assert first["label"] == "历史真实运行回放"
    assert first["terminal"] is True
    assert first["events"][-1]["state"] == "COMPLETED"
    assert first["run"]["benchmark_mode"] == "formal"
    assert first["run"]["speedup_percent"] == 106.84
    serialized = json.dumps(first, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert first["artifact_ref"].startswith(".infra-team/evolution/")
    assert first["technical"]["manifest_id"] == "qwen38-dflash2-v1"


def test_evolution_detail_refuses_invalid_id_and_degrades_corrupt_events(tmp_path):
    path = _make_evolution(tmp_path)
    assert DashboardData(str(tmp_path)).evolution_detail("../secret") is None
    with (path / "events.ndjson").open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
    detail = DashboardData(str(tmp_path)).evolution_detail(path.name)
    assert detail["events"][-1]["state"] == "COMPLETED"
    assert detail["event_errors"] == 1


def test_watch_and_memory_summary_are_sanitized(tmp_path):
    memory = RecipeMemory(tmp_path)
    memory.record_watch_state(
        context_key="context",
        service_name="default",
        manifest_id="qwen38-dflash2-v1",
        manifest_hash="a" * 64,
        environment_fingerprint="b" * 64,
        policy_hash="c" * 64,
        snapshot_hash="d" * 64,
        status="WAITING_WINDOW",
        attempt_count=2,
        next_attempt_at="2026-09-03T02:00:00+08:00",
        now="2026-09-02T17:00:00+08:00",
        last_error="private local path /Users/example/model",
    )
    data = DashboardData(str(tmp_path))
    watch = data.watch_status()
    summary = data.memory_summary()

    assert watch["status"] == "WAITING_WINDOW"
    assert watch["next_attempt_at"] == "2026-09-03T02:00:00+08:00"
    assert "last_error" not in watch
    assert "manifest_hash" not in watch
    assert summary["watch_states"] == 1
    assert "path" not in summary
    assert "/Users/" not in json.dumps(summary)


def test_dashboard_http_routes_serve_ui_and_json(tmp_path):
    _make_run(tmp_path, "20260826-120000")
    evolution = _make_evolution(tmp_path)
    server = create_dashboard_server(str(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
            csp = response.headers["Content-Security-Policy"]
        with urllib.request.urlopen(base + "/static/dashboard.css", timeout=3) as response:
            css = response.read().decode("utf-8")
        with urllib.request.urlopen(base + "/static/dashboard.js", timeout=3) as response:
            javascript = response.read().decode("utf-8")
        with urllib.request.urlopen(base + "/api/dashboard", timeout=3) as response:
            payload = json.load(response)
        with urllib.request.urlopen(
            base + "/api/runs/20260826-120000", timeout=3
        ) as response:
            run = json.load(response)
        with urllib.request.urlopen(base + "/api/evolution/current", timeout=3) as response:
            current_evolution = json.load(response)
        with urllib.request.urlopen(
            base + f"/api/evolutions/{evolution.name}", timeout=3
        ) as response:
            replay = json.load(response)
        with urllib.request.urlopen(base + "/api/watch/status", timeout=3) as response:
            watch = json.load(response)
        with urllib.request.urlopen(base + "/api/memory/summary", timeout=3) as response:
            memory = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "Local AI Upgrade" in html
    assert "安装一次，以后本地 AI 自己测试和升级" in html
    assert "检查并升级本地 AI" in html
    assert "历史真实运行回放" in html
    assert "/static/dashboard.css" in html
    assert "/static/dashboard.js" in html
    assert ":root" in css
    assert "streamControlPost" in javascript
    assert "renderEvolution" in javascript
    assert "script-src 'self'" in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp
    assert payload["latest_run"]["selected_id"] == "dflash2_default"
    assert payload["product"]["human_decisions_after_start"] == 0
    assert run["quality"]["passed"] == 4
    assert current_evolution["id"] == evolution.name
    assert replay["mode"] == "historical_replay"
    assert watch["enabled"] is True
    assert memory["schema_version"] == 2


def test_dashboard_is_local_only(tmp_path):
    try:
        create_dashboard_server(str(tmp_path), host="0.0.0.0", port=0)
    except ValueError as exc:
        assert "local-only" in str(exc)
    else:
        raise AssertionError("dashboard should not bind to non-loopback addresses")


def _post(url: str, body: dict, headers: dict | None = None):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=3)


def test_write_api_requires_same_origin_and_control_token(tmp_path):
    server = create_dashboard_server(str(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/session") as response:
            session = json.load(response)
        try:
            _post(base + "/api/chat", {"mode": "planner", "messages": []})
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("write without token must be refused")

        try:
            _post(
                base + "/api/chat",
                {"mode": "planner", "messages": [{"role": "user", "content": "/help"}]},
                {
                    "Origin": "http://evil.example",
                    "X-Infra-Control-Token": session["token"],
                },
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("cross-origin write must be refused")

        headers = {
            "Origin": base,
            "X-Infra-Control-Token": session["token"],
        }
        with _post(
            base + "/api/chat",
            {
                "mode": "planner",
                "messages": [{"role": "user", "content": "/help"}],
            },
            headers,
        ) as response:
            payload = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["ok"] is True
    assert "/optimize" in payload["content"]


def test_evolve_control_endpoint_requires_confirmation_token_and_calls_outer_loop(tmp_path):
    server = create_dashboard_server(str(tmp_path), port=0)
    controller = server.RequestHandlerClass.controller
    calls = []
    controller.start_evolution = lambda repeats=1: calls.append(repeats) or {
        "ok": True,
        "message": "完整演进已启动",
        "evolution": {"running": True},
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/session") as response:
            token = json.load(response)["token"]
        headers = {"Origin": base, "X-Infra-Control-Token": token}
        with _post(
            base + "/api/control/evolve", {"repeats": 3}, headers
        ) as response:
            result = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert calls == [3]
    assert result["evolution"]["running"] is True


class FakeManagedService:
    def __init__(self, healthy=False):
        self.healthy = healthy
        self.start_calls = 0
        self.stop_calls = 0
        self.spec = ServiceSpec(name="default", model="target", port=8000)

    def status(self):
        return {
            "managed": self.healthy,
            "healthy": self.healthy,
            "status": "running" if self.healthy else "stopped",
            "base_url": "http://127.0.0.1:8000",
            "spec": self.spec.to_dict(),
        }

    def saved_spec(self):
        return self.spec

    def start(self, spec, wait_timeout=900):
        self.start_calls += 1
        self.healthy = True
        self.spec = spec
        return self.status()

    def stop(self, timeout=30):
        self.stop_calls += 1
        self.healthy = False
        return {"stopped": True, "status": "stopped", "spec": self.spec.to_dict()}


def test_controller_start_and_stop_use_saved_managed_spec(tmp_path):
    data = DashboardData(str(tmp_path))
    controller = DashboardController(data)
    fake = FakeManagedService()
    controller.service = fake

    started = controller.start_service()
    stopped = controller.stop_service()

    assert started["ok"] is True
    assert fake.start_calls == 1
    assert stopped["ok"] is True
    assert fake.stop_calls == 1


def test_planner_slash_command_requests_confirmation_without_execution(tmp_path):
    controller = DashboardController(DashboardData(str(tmp_path)))
    response = controller.chat(
        "planner", [{"role": "user", "content": "/optimize"}]
    )
    assert response["requested_action"] == "evolve"
    assert response["deterministic"] is True


def test_model_chat_and_planner_chat_use_bounded_api(monkeypatch, tmp_path):
    data = DashboardData(str(tmp_path))
    controller = DashboardController(data)
    fake = FakeManagedService(healthy=True)
    controller.service = fake
    data.service = fake
    captured = []

    def fake_post(url, payload, timeout=600):
        captured.append(payload)
        return {"choices": [{"message": {"content": "response"}}]}

    monkeypatch.setattr("infra_team.dashboard._post_json", fake_post)
    deployed = controller.chat(
        "deployed", [{"role": "user", "content": "hello"}]
    )
    planner = controller.chat(
        "planner", [{"role": "user", "content": "analyze this service"}]
    )

    assert deployed["content"] == "response"
    assert planner["content"] == "response"
    assert captured[0]["messages"][0]["role"] == "user"
    assert captured[1]["messages"][0]["role"] == "system"
    assert "cannot execute shell" in captured[1]["messages"][0]["content"]


def test_chat_rejects_unbounded_history(tmp_path):
    controller = DashboardController(DashboardData(str(tmp_path)))
    too_many = [{"role": "user", "content": "x"}] * 25
    try:
        controller.chat("deployed", too_many)
    except ValueError as exc:
        assert "1 到 24" in str(exc)
    else:
        raise AssertionError("unbounded history should be rejected")


def test_evolution_is_single_flight_and_uses_fixed_cli(monkeypatch, tmp_path):
    data = DashboardData(str(tmp_path))
    controller = DashboardController(data)
    fake_service = FakeManagedService(healthy=True)
    controller.service = fake_service

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = None

        def poll(self):
            return self.returncode

    created = []

    def fake_popen(command, **kwargs):
        process = FakeProcess(command, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr("infra_team.dashboard.subprocess.Popen", fake_popen)
    first = controller.start_evolution(repeats=3)
    assert first["evolution"]["running"] is True
    assert "evolve" in created[0].command
    assert "--once" in created[0].command
    assert "optimize" not in created[0].command
    assert created[0].command[created[0].command.index("--repeats") + 1] == "3"
    assert created[0].command[created[0].command.index("--policy") + 1] == "builtin"

    try:
        controller.start_evolution(repeats=3)
    except RuntimeError as exc:
        assert "正在运行" in str(exc)
    else:
        raise AssertionError("a second optimization must be refused")


def test_evolution_rejects_unapproved_repeat_count(tmp_path):
    data = DashboardData(str(tmp_path))
    controller = DashboardController(data)
    controller.service = FakeManagedService(healthy=True)
    try:
        controller.start_evolution(repeats=2)
    except ValueError as exc:
        assert "1、3 或 5" in str(exc)
    else:
        raise AssertionError("unapproved repeat count should be rejected")


def test_dashboard_detects_cross_process_optimization_lock(tmp_path):
    controller = DashboardController(DashboardData(str(tmp_path)))
    controller._optimization_lock.parent.mkdir(parents=True, exist_ok=True)
    with controller._optimization_lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert controller.status()["optimization"]["running"] is True
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class FakeSSEHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        assert payload["stream"] is True
        body = (
            'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_openai_sse_parser_yields_text_deltas():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        chunks = list(
            _stream_openai_chat(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                {"model": "fake", "messages": [{"role": "user", "content": "x"}]},
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert chunks == ["hello ", "world"]


def test_controller_stream_chat_emits_start_deltas_and_done(monkeypatch, tmp_path):
    data = DashboardData(str(tmp_path))
    controller = DashboardController(data)
    fake = FakeManagedService(healthy=True)
    controller.service = fake
    data.service = fake
    monkeypatch.setattr(
        "infra_team.dashboard._stream_openai_chat",
        lambda *args, **kwargs: iter(["first", " second"]),
    )

    events = list(
        controller.stream_chat(
            "deployed", [{"role": "user", "content": "hello"}]
        )
    )

    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
    assert "".join(event.get("content", "") for event in events) == "first second"


def test_deterministic_command_streams_immediately(tmp_path):
    controller = DashboardController(DashboardData(str(tmp_path)))
    events = list(
        controller.stream_chat(
            "planner", [{"role": "user", "content": "/help"}]
        )
    )
    assert events[0]["type"] == "delta"
    assert "/optimize" in events[0]["content"]
    assert events[-1] == {"type": "done", "mode": "planner", "deterministic": True}


def test_dashboard_stream_endpoint_uses_ndjson(tmp_path):
    server = create_dashboard_server(str(tmp_path), port=0)
    controller = server.RequestHandlerClass.controller
    controller.stream_chat = lambda mode, messages: iter(
        [
            {"type": "start", "mode": mode},
            {"type": "delta", "content": "A"},
            {"type": "delta", "content": "B"},
            {"type": "done", "mode": mode},
        ]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/session") as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(
            base + "/api/chat/stream",
            data=json.dumps(
                {"mode": "deployed", "messages": [{"role": "user", "content": "x"}]}
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": base,
                "X-Infra-Control-Token": token,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.headers["Content-Type"].startswith("application/x-ndjson")
            events = [json.loads(line) for line in response if line.strip()]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
