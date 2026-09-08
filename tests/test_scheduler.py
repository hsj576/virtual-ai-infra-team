"""P4 tests for continuous registry watch, maintenance windows and launchd."""

from __future__ import annotations

import json
import plistlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infra_team.candidate_manifest import parse_manifest
from infra_team.candidate_registry import CandidateRegistry
from infra_team.cli import build_parser
from infra_team.compatibility import AutonomyPolicy, MaintenanceWindow
from infra_team.recipe_memory import RecipeMemory, environment_fingerprint
from infra_team.scheduler import (
    EvolutionWatcher,
    NotificationCenterNotifier,
    render_launchd_plist,
    watch_context_key,
)


REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
TARGET_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"


def manifest(revision: str = REVISION, manifest_id: str = "qwen38-dflash2-v1"):
    return parse_manifest(
        {
            "schema_version": 1,
            "id": manifest_id,
            "kind": "acceleration_plugin",
            "status": "active",
            "description": "test",
            "source": {
                "provider": "huggingface",
                "repo_id": "z-lab/Qwen3.8-27B-DFlash2",
                "revision": revision,
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
                            "id": f"{manifest_id}:native",
                            "description": "native",
                            "draft_block_size": None,
                        }
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


def env() -> dict:
    return {
        "arch": "arm64",
        "chip": "Apple M5 Pro",
        "gpu_cores": 20,
        "unified_memory_gb": 48,
        "macos_version": "26.5.2",
        "disk_free_gb": 500,
        "packages": {"mlx": "0.32.2", "mlx-vlm": "0.6.16"},
    }


def policy(window: MaintenanceWindow | None = None) -> AutonomyPolicy:
    return AutonomyPolicy(
        maintenance_window=window
        or MaintenanceWindow(
            enabled=True,
            timezone="Asia/Shanghai",
            start="02:00",
            end="05:00",
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        ),
        watch_poll_interval_seconds=60,
        watch_backoff_base_seconds=30,
        watch_backoff_cap_seconds=600,
        watch_no_improvement_recheck_seconds=3600,
        notifications_enabled=True,
    )


class FakeService:
    def __init__(self):
        self.recipe = {
            "schema_version": 1,
            "service_name": "default",
            "candidate_id": "baseline",
            "service_spec": {
                "name": "default",
                "model": "mlx-community/Qwen3.8-27B-4bit",
                "host": "127.0.0.1",
                "port": 8000,
            },
        }

    def status(self):
        return {
            "healthy": True,
            "managed": True,
            "name": "default",
            "base_url": "http://127.0.0.1:8000",
            "spec": self.recipe["service_spec"],
        }

    def active_recipe(self):
        return self.recipe


class FakeEngine:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run_once(self, trigger_source="cli_once", trigger_context=None):
        self.calls.append(
            {"trigger_source": trigger_source, "trigger_context": trigger_context}
        )
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeNotifier:
    def __init__(self):
        self.events = []

    def send(self, template_id, fields):
        self.events.append((template_id, fields))
        return {"sent": True, "template_id": template_id}


class MutableClock:
    def __init__(self, current: datetime):
        self.current = current
        self.sleeps = []

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


def make_watcher(
    tmp_path,
    clock,
    results,
    watcher_policy=None,
    manifests=None,
):
    memory = RecipeMemory(tmp_path)
    engine = FakeEngine(results)
    notifier = FakeNotifier()
    registry = CandidateRegistry.from_manifests(manifests or [manifest()])
    watcher = EvolutionWatcher(
        root=tmp_path,
        service_name="default",
        policy=watcher_policy or policy(),
        registry=registry,
        memory=memory,
        service_manager=FakeService(),
        engine_factory=lambda: engine,
        notifier=notifier,
        probe_fn=env,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    return watcher, memory, engine, notifier


def test_maintenance_window_handles_same_day_and_cross_midnight():
    daytime = MaintenanceWindow(
        enabled=True,
        timezone="Asia/Shanghai",
        start="02:00",
        end="05:00",
        weekdays=(0,),
    )
    assert daytime.allows(datetime(2026, 8, 31, 3, 0, tzinfo=UTC)) is False
    assert daytime.allows(datetime(2026, 8, 30, 19, 0, tzinfo=UTC)) is True

    overnight = MaintenanceWindow(
        enabled=True,
        timezone="Asia/Shanghai",
        start="23:00",
        end="02:00",
        weekdays=(0,),
    )
    assert overnight.allows(datetime(2026, 8, 31, 15, 30, tzinfo=UTC)) is True
    # Tuesday 01:00 belongs to Monday's 23:00 window.
    assert overnight.allows(datetime(2026, 8, 31, 17, 0, tzinfo=UTC)) is True
    assert overnight.allows(datetime(2026, 8, 31, 19, 0, tzinfo=UTC)) is False


def test_policy_roundtrip_includes_watch_window_backoff_and_notifications(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
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
maintenance_window:
  enabled: true
  timezone: Asia/Shanghai
  start: '02:00'
  end: '05:00'
  weekdays: [0, 1, 2, 3, 4, 5, 6]
watch:
  poll_interval_seconds: 60
  backoff_base_seconds: 30
  backoff_cap_seconds: 600
  no_improvement_recheck_seconds: 3600
notifications:
  enabled: true
  macos_notification_center: true
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
    loaded = AutonomyPolicy.from_file(path)
    assert loaded.maintenance_window.enabled is True
    assert loaded.maintenance_window.timezone == "Asia/Shanghai"
    assert loaded.watch_backoff_cap_seconds == 600
    assert loaded.notifications_enabled is True
    assert loaded.policy_hash == AutonomyPolicy.from_file(path).policy_hash
    cadence_changed = AutonomyPolicy(
        maintenance_window=loaded.maintenance_window,
        watch_poll_interval_seconds=999,
        watch_backoff_base_seconds=loaded.watch_backoff_base_seconds,
        watch_backoff_cap_seconds=loaded.watch_backoff_cap_seconds,
        watch_no_improvement_recheck_seconds=(
            loaded.watch_no_improvement_recheck_seconds
        ),
        notifications_enabled=loaded.notifications_enabled,
        notifications_macos_notification_center=(
            loaded.notifications_macos_notification_center
        ),
        trusted_namespaces=loaded.trusted_namespaces,
    )
    assert cadence_changed.policy_hash != loaded.policy_hash
    assert cadence_changed.evaluation_policy_hash == loaded.evaluation_policy_hash


def test_watch_context_changes_for_manifest_environment_or_policy():
    base = watch_context_key(
        manifest_hash="a" * 64,
        environment_fingerprint="b" * 64,
        policy_hash="c" * 64,
        service_name="default",
    )
    assert base != watch_context_key(
        manifest_hash="d" * 64,
        environment_fingerprint="b" * 64,
        policy_hash="c" * 64,
        service_name="default",
    )
    assert base != watch_context_key(
        manifest_hash="a" * 64,
        environment_fingerprint="e" * 64,
        policy_hash="c" * 64,
        service_name="default",
    )


def test_watch_bootstraps_completed_state_from_existing_active_evidence(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[],
    )
    manifest_hash = manifest().manifest_hash
    watcher.service_manager.recipe["accelerator"] = {
        "manifest_hash": manifest_hash
    }
    memory.latest_manifest_outcome = lambda *args: {
        "accepted": True,
        "all_quality_pass": True,
        "any_quality_pass": True,
        "last_seen_at": "2026-08-28T20:00:00+00:00",
        "result_count": 2,
    }
    result = watcher.run_cycle()
    assert result["action"] == "duplicate_suppressed"
    assert engine.calls == []
    assert memory.list_watch_states(limit=1)[0]["status"] == "COMPLETED"


def test_existing_waiting_state_can_bootstrap_from_completed_p3_evidence(tmp_path):
    outside = MutableClock(datetime(2026, 8, 31, 9, 0, tzinfo=UTC))
    watcher, memory, engine, _notifier = make_watcher(
        tmp_path,
        outside,
        results=[],
    )
    assert watcher.run_cycle()["action"] == "outside_maintenance_window"
    watcher.service_manager.recipe["accelerator"] = {
        "manifest_hash": manifest().manifest_hash
    }
    memory.latest_manifest_outcome = lambda *args: {
        "accepted": True,
        "all_quality_pass": True,
        "any_quality_pass": True,
        "last_seen_at": "2026-08-28T20:00:00+00:00",
        "result_count": 2,
    }
    outside.current = datetime(2026, 8, 31, 18, 0, tzinfo=UTC)
    assert watcher.run_cycle()["action"] == "duplicate_suppressed"
    assert memory.list_watch_states(limit=1)[0]["status"] == "COMPLETED"
    assert engine.calls == []


def test_watch_runs_new_manifest_once_and_suppresses_duplicate(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, memory, engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {
                "evolution_id": "evo-1",
                "status": "COMPLETED",
                "path": "/tmp/evo-1",
                "accepted": True,
                "selected_id": "qwen38-dflash2-v1:native",
            }
        ],
    )
    first = watcher.run_cycle()
    second = watcher.run_cycle()
    assert first["action"] == "evolved"
    assert second["action"] == "duplicate_suppressed"
    assert len(engine.calls) == 1
    assert engine.calls[0]["trigger_source"] == "watch"
    assert engine.calls[0]["trigger_context"]["manifest_ids"] == [
        "qwen38-dflash2-v1"
    ]
    assert memory.list_watch_states(limit=5)[0]["status"] == "COMPLETED"
    assert [item[0] for item in notifier.events] == [
        "candidate_detected",
        "evolution_started",
        "evolution_completed",
    ]


def test_watch_waits_outside_window_without_calling_engine(tmp_path):
    clock = MutableClock(datetime(2026, 8, 31, 9, 0, tzinfo=UTC))
    watcher, memory, engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[],
    )
    result = watcher.run_cycle()
    assert result["action"] == "outside_maintenance_window"
    assert engine.calls == []
    states = memory.list_watch_states(limit=5)
    assert states[0]["status"] == "WAITING_WINDOW"
    assert states[0]["next_attempt_at"]
    assert notifier.events[-1][0] == "waiting_window"
    assert watcher.run_cycle()["action"] == "backoff"
    assert len(notifier.events) == 1


def test_transient_failure_uses_persistent_exponential_backoff(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, memory, engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {
                "evolution_id": "evo-fail-1",
                "status": "PREPARE_FAILED",
                "path": "/tmp/evo-fail-1",
                "error": "network interrupted",
                "error_type": "CandidateStoreError",
            },
            {
                "evolution_id": "evo-fail-2",
                "status": "BASELINE_FAILED",
                "path": "/tmp/evo-fail-2",
                "error": "service unavailable",
                "error_type": "RuntimeError",
            },
        ],
    )
    first = watcher.run_cycle()
    assert first["action"] == "evolution_failed"
    state = memory.list_watch_states(limit=1)[0]
    assert state["attempt_count"] == 1
    assert state["next_attempt_at"] == (
        clock.current + timedelta(seconds=30)
    ).isoformat()

    blocked = watcher.run_cycle()
    assert blocked["action"] == "backoff"
    assert len(engine.calls) == 1
    clock.current += timedelta(seconds=30)
    second = watcher.run_cycle()
    assert second["action"] == "evolution_failed"
    state = memory.list_watch_states(limit=1)[0]
    assert state["attempt_count"] == 2
    assert state["next_attempt_at"] == (
        clock.current + timedelta(seconds=60)
    ).isoformat()
    assert notifier.events[-1][0] == "evolution_failed"


def test_engine_exception_is_audited_and_retried_with_backoff(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, memory, engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[RuntimeError("optimization lock busy")],
    )
    result = watcher.run_cycle()
    assert result["action"] == "evolution_failed"
    assert result["result"]["status"] == "CANDIDATE_FAILED"
    state = memory.list_watch_states(limit=1)[0]
    assert state["attempt_count"] == 1
    assert state["error_class"] == "RuntimeError"
    assert notifier.events[-1][0] == "evolution_failed"
    assert len(engine.calls) == 1


def test_no_improvement_uses_long_recheck_cooldown(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {
                "evolution_id": "evo-none",
                "status": "NO_IMPROVEMENT",
                "path": "/tmp/evo-none",
                "accepted": False,
            },
            {
                "evolution_id": "evo-recheck",
                "status": "COMPLETED",
                "path": "/tmp/evo-recheck",
                "accepted": True,
            },
        ],
    )
    assert watcher.run_cycle()["action"] == "no_improvement"
    assert watcher.run_cycle()["action"] == "backoff"
    clock.current += timedelta(seconds=3600)
    assert watcher.run_cycle()["action"] == "evolved"
    assert len(engine.calls) == 2


def test_same_manifest_quality_failure_is_permanently_suppressed(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {
                "evolution_id": "evo-quality-fail",
                "status": "NO_IMPROVEMENT",
                "path": "/tmp/evo-quality-fail",
                "accepted": False,
            }
        ],
    )
    evidence_calls = 0

    def latest_outcome(*args):
        nonlocal evidence_calls
        evidence_calls += 1
        if evidence_calls == 1:
            return None
        return {
            "accepted": False,
            "all_quality_pass": False,
            "any_quality_pass": False,
            "last_seen_at": clock.current.isoformat(),
            "result_count": 1,
        }

    memory.latest_manifest_outcome = latest_outcome
    assert watcher.run_cycle()["action"] == "no_improvement"
    state = memory.list_watch_states(limit=1)[0]
    assert state["status"] == "DISCOVERY_REJECTED"
    assert state["next_attempt_at"] is None
    assert watcher.run_cycle()["action"] == "duplicate_suppressed"
    assert len(engine.calls) == 1


def test_multi_manifest_run_records_each_manifest_outcome_independently(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    first = manifest()
    second = manifest(revision="a" * 40, manifest_id="qwen38-dflash2-v2")
    watcher, memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {
                "evolution_id": "evo-multi",
                "status": "COMPLETED",
                "path": "/tmp/evo-multi",
                "selected_id": "qwen38-dflash2-v1:native",
            }
        ],
        manifests=[first, second],
    )
    assert watcher.run_cycle()["action"] == "evolved"
    states = {
        item["manifest_id"]: item for item in memory.list_watch_states(limit=10)
    }
    assert states["qwen38-dflash2-v1"]["status"] == "COMPLETED"
    assert states["qwen38-dflash2-v2"]["status"] == "NO_IMPROVEMENT"
    assert states["qwen38-dflash2-v2"]["next_attempt_at"]
    assert len(engine.calls) == 1


def test_manifest_revision_change_creates_new_watch_context(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    registry = CandidateRegistry.from_manifests([manifest()])
    watcher, _memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {"evolution_id": "evo-1", "status": "COMPLETED", "path": "/tmp/evo-1"},
            {"evolution_id": "evo-2", "status": "COMPLETED", "path": "/tmp/evo-2"},
        ],
        manifests=[manifest()],
    )
    watcher.registry = registry
    assert watcher.run_cycle()["action"] == "evolved"
    watcher.registry = CandidateRegistry.from_manifests(
        [manifest(revision="a" * 40)]
    )
    assert watcher.run_cycle()["action"] == "evolved"
    assert len(engine.calls) == 2


def test_rollback_failed_blocks_same_context_until_manifest_changes(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, _memory, engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {
                "evolution_id": "evo-bad",
                "status": "ROLLBACK_FAILED",
                "path": "/tmp/evo-bad",
                "error": "manual recovery required",
            }
        ],
    )
    assert watcher.run_cycle()["action"] == "blocked_manual_recovery"
    assert watcher.run_cycle()["action"] == "blocked_manual_recovery"
    assert len(engine.calls) == 1
    assert notifier.events[-1][0] == "manual_recovery_required"


def test_watch_loop_persists_registry_error_backoff_and_keeps_running(tmp_path):
    clock = MutableClock(datetime(2026, 8, 31, 9, 0, tzinfo=UTC))
    watcher, memory, _engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[],
    )

    class BrokenRegistry:
        registry_name = "builtin"

        def load(self):
            raise RuntimeError("registry temporarily unavailable")

    watcher.registry = BrokenRegistry()
    summary = watcher.watch(max_cycles=2)
    assert summary["actions"] == {"scheduler_error": 2}
    assert clock.sleeps == [30]
    state = memory.get_watch_state(watcher.scheduler_context_key)
    assert state["attempt_count"] == 2
    assert state["error_class"] == "RuntimeError"
    assert notifier.events[-1][0] == "evolution_failed"


def test_scheduler_error_backoff_survives_process_restart(tmp_path):
    clock = MutableClock(datetime(2026, 8, 31, 9, 0, tzinfo=UTC))
    watcher, memory, _engine, notifier = make_watcher(
        tmp_path,
        clock,
        results=[],
    )

    class BrokenRegistry:
        registry_name = "builtin"

        def __init__(self):
            self.calls = 0

        def load(self):
            self.calls += 1
            raise RuntimeError("registry temporarily unavailable")

    first_registry = BrokenRegistry()
    watcher.registry = first_registry
    assert watcher.watch(max_cycles=1)["actions"] == {"scheduler_error": 1}
    assert first_registry.calls == 1

    second_registry = BrokenRegistry()
    restarted = EvolutionWatcher(
        root=tmp_path,
        service_name="default",
        policy=watcher.policy,
        registry=second_registry,
        memory=RecipeMemory(tmp_path),
        service_manager=FakeService(),
        engine_factory=lambda: FakeEngine([]),
        notifier=notifier,
        probe_fn=env,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    assert restarted.watch(max_cycles=1)["actions"] == {"scheduler_backoff": 1}
    assert second_registry.calls == 0


def test_watch_loop_sleeps_between_cycles_and_does_not_hold_optimize_lock(tmp_path):
    clock = MutableClock(datetime(2026, 8, 31, 9, 0, tzinfo=UTC))
    watcher, _memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[],
    )
    summary = watcher.watch(max_cycles=3)
    assert summary["cycles"] == 3
    assert clock.sleeps == [60, 60]
    assert engine.calls == []
    # The scheduler itself does not create or hold the maintenance lock.
    assert not (tmp_path / ".infra-team" / "optimize.lock").exists()


def test_notification_failure_never_aborts_evolution(tmp_path):
    clock = MutableClock(datetime(2026, 8, 30, 19, 0, tzinfo=UTC))
    watcher, _memory, engine, _notifier = make_watcher(
        tmp_path,
        clock,
        results=[
            {"evolution_id": "evo-1", "status": "COMPLETED", "path": "/tmp/evo-1"}
        ],
    )

    class RaisingNotifier:
        def send(self, template_id, fields):
            raise OSError("notifications unavailable")

    watcher.notifier = RaisingNotifier()
    assert watcher.run_cycle()["action"] == "evolved"
    assert len(engine.calls) == 1


def test_notification_uses_fixed_argv_and_audits_without_script_injection(tmp_path):
    calls = []

    def sender(argv, **kwargs):
        calls.append((argv, kwargs))
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    memory = RecipeMemory(tmp_path)
    notifier = NotificationCenterNotifier(
        tmp_path,
        memory=memory,
        enabled=True,
        sender=sender,
    )
    result = notifier.send(
        "evolution_failed",
        {
            "evolution_id": 'x" & do shell script "touch /tmp/pwn" & "',
            "status": "PREPARE_FAILED",
            "artifact_path": "/tmp/evo",
        },
    )
    assert result["sent"] is True
    argv, kwargs = calls[0]
    assert argv[:2] == ["/usr/bin/osascript", "-e"]
    assert kwargs.get("shell") is not True
    assert "touch /tmp/pwn" not in argv[2]
    audit = Path(tmp_path, ".infra-team/notifications/events.ndjson")
    event = json.loads(audit.read_text().splitlines()[-1])
    assert event["template_id"] == "evolution_failed"
    assert "fields_hash" in event
    assert memory.show()["notification_events"] == 1


def test_launchd_renderer_uses_fixed_arguments_and_never_installs(tmp_path):
    output = tmp_path / "watch.plist"
    payload = render_launchd_plist(
        workspace_root=tmp_path,
        python_executable=Path("/safe/python"),
        output_path=output,
        policy_path="configs/autonomy-local.yaml",
        service_name="default",
        poll_interval_seconds=300,
    )
    parsed = plistlib.loads(output.read_bytes())
    args = parsed["ProgramArguments"]
    assert args[:3] == ["/safe/python", "-m", "infra_team.cli"]
    assert "--watch" in args
    assert parsed["WorkingDirectory"] == str(tmp_path.resolve())
    assert "launchctl" not in json.dumps(parsed)
    assert payload["installed"] is False


def test_cli_exposes_watch_and_launchd_render():
    parser = build_parser()
    watch = parser.parse_args(
        ["evolve", "--watch", "--max-cycles", "2", "--poll-interval", "30"]
    )
    launchd = parser.parse_args(
        ["launchd", "render", "--output", "configs/watch.plist"]
    )
    assert watch.watch is True
    assert watch.max_cycles == 2
    assert watch.poll_interval == 30
    assert launchd.launchd_command == "render"
