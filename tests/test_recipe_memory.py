"""P3 tests for durable local Recipe Memory."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from infra_team.candidate_manifest import parse_manifest
from infra_team.candidate_registry import CandidateRegistry
from infra_team.compatibility import AutonomyPolicy
from infra_team.recipe_memory import RecipeMemory, environment_fingerprint


DRAFT_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
TARGET_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"


def environment(chip: str = "Apple M5 Pro", runtime: str = "0.6.16") -> dict:
    return {
        "chip": chip,
        "gpu_cores": 20,
        "unified_memory_gb": 48,
        "macos_version": "26.5.2",
        "packages": {"mlx": "0.32.2", "mlx-vlm": runtime},
    }


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
                "revision": DRAFT_REVISION,
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


def candidates():
    return CandidateRegistry.from_manifests([manifest()]).resolve_all(
        policy=AutonomyPolicy(),
        environment={
            "arch": "arm64",
            "chip": "Apple M5 Pro",
            "disk_free_gb": 100,
            "packages": {"mlx-vlm": "0.6.16"},
        },
    )


def successful_summary() -> dict:
    return {
        "run_id": "20260828-190000",
        "path": "/tmp/run",
        "environment": environment(),
        "baseline": {
            "id": "baseline",
            "ok": True,
            "spec": {"id": "baseline", "target_model": "mlx-community/Qwen3.8-27B-4bit"},
            "quality": {"quality_pass": True, "passed": 4, "total": 4},
            "performance": {
                "generation_tps_median": 18.0,
                "ttft_seconds_median": 0.4,
                "peak_memory_gb": 16.5,
                "error_rate": 0,
            },
        },
        "candidates": [
            {
                "id": "dflash2_default",
                "ok": True,
                "spec": {
                    "id": "dflash2_default",
                    "target_model": "mlx-community/Qwen3.8-27B-4bit",
                    "draft_model": "/local/draft",
                    "draft_kind": "dflash",
                },
                "quality": {"quality_pass": True, "passed": 4, "total": 4},
                "performance": {
                    "generation_tps_median": 36.0,
                    "generation_tps_stdev": 0.7,
                    "ttft_seconds_median": 0.3,
                    "peak_memory_gb": 22.0,
                    "error_rate": 0,
                },
            }
        ],
        "verdict": {
            "selected_id": "dflash2_default",
            "accepted": True,
            "speedup_percent": 100.0,
            "evaluations": [
                {
                    "id": "baseline",
                    "qualified": True,
                    "quality_pass": True,
                    "accepted": False,
                },
                {
                    "id": "dflash2_default",
                    "qualified": True,
                    "quality_pass": True,
                    "accepted": True,
                },
            ],
        },
        "service_transition": {
            "status": "serving_selected",
            "post_restart_quality": {"quality_pass": True, "passed": 4, "total": 4},
        },
    }


def test_recipe_memory_migrates_v1_to_v2_without_reset(tmp_path):
    database = tmp_path / ".infra-team" / "memory" / "recipes.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE legacy_marker(value TEXT)")
        conn.execute("INSERT INTO legacy_marker(value) VALUES ('preserved')")
        conn.execute("PRAGMA user_version = 1")
    memory = RecipeMemory(tmp_path)
    with sqlite3.connect(database) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT value FROM legacy_marker").fetchone()[0] == "preserved"
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"watch_states", "notification_events"} <= names
    assert memory.list_watch_states() == []


def test_environment_fingerprint_is_stable_and_runtime_sensitive():
    first = environment_fingerprint(environment(), "target", TARGET_REVISION)
    second = environment_fingerprint(dict(environment()), "target", TARGET_REVISION)
    changed = environment_fingerprint(environment(runtime="0.7.0"), "target", TARGET_REVISION)
    assert first == second
    assert first != changed
    assert len(first) == 64


def test_record_summary_creates_exact_prior_and_promotion(tmp_path):
    memory = RecipeMemory(tmp_path)
    summary = successful_summary()
    resolved = candidates()
    update = memory.record_supervisor_summary(
        summary,
        resolved_candidates=resolved,
        evolution_id="evo-1",
        previous_recipe={"candidate_id": "baseline"},
        selected_recipe={"candidate_id": "dflash2_default"},
    )
    assert update["candidate_runs_written"] == 2
    assert update["promotion_written"] is True

    prior = memory.query_prior(
        environment(),
        target_model="mlx-community/Qwen3.8-27B-4bit",
        target_revision=TARGET_REVISION,
        resolved_candidates=resolved,
    )
    best = next(item for item in prior["ranking"] if item["candidate_id"] == "dflash2_default")
    assert prior["ranking"][0]["candidate_id"] == "dflash2_default"
    assert best["match"] == "exact"
    assert best["recommendation"] == "prioritize_revalidation"
    assert best["accepted"] is True
    assert best["speedup_percent"] == 100.0

    stats = memory.show()
    assert stats["environments"] == 1
    assert stats["candidate_runs"] == 2
    assert stats["promotions"] == 1
    assert stats["evolution_runs"] == 0


def test_similar_environment_is_only_a_prior(tmp_path):
    memory = RecipeMemory(tmp_path)
    resolved = candidates()
    memory.record_supervisor_summary(successful_summary(), resolved_candidates=resolved)
    prior = memory.query_prior(
        environment(chip="Apple M6 Pro"),
        target_model="mlx-community/Qwen3.8-27B-4bit",
        target_revision=TARGET_REVISION,
        resolved_candidates=resolved,
    )
    item = next(row for row in prior["ranking"] if row["candidate_id"] == "dflash2_default")
    assert item["match"] == "similar"
    assert item["recommendation"] == "use_as_prior_and_revalidate"
    assert prior["requires_local_validation"] is True


def test_quality_failure_is_remembered_as_skip_same_version(tmp_path):
    memory = RecipeMemory(tmp_path)
    resolved = candidates()
    summary = successful_summary()
    candidate = summary["candidates"][0]
    candidate["quality"] = {"quality_pass": False, "passed": 3, "total": 4}
    summary["verdict"]["accepted"] = False
    summary["verdict"]["selected_id"] = "baseline"
    summary["verdict"]["speedup_percent"] = 0
    summary["verdict"]["evaluations"][1].update(
        quality_pass=False,
        qualified=False,
        accepted=False,
        disqualifications=["quality gate failed"],
    )
    summary["service_transition"] = None
    memory.record_supervisor_summary(summary, resolved_candidates=resolved)

    prior = memory.query_prior(
        environment(),
        target_model="mlx-community/Qwen3.8-27B-4bit",
        target_revision=TARGET_REVISION,
        resolved_candidates=resolved,
    )
    item = next(row for row in prior["ranking"] if row["candidate_id"] == "dflash2_default")
    assert item["recommendation"] == "skip_same_version_quality_failure"
    assert item["quality_pass"] is False


def test_evolution_status_and_failure_persist_after_reopen(tmp_path):
    memory = RecipeMemory(tmp_path)
    memory.record_evolution(
        evolution_id="evo-failed",
        status="PREPARE_FAILED",
        trigger={"source": "cli_once"},
        artifact_path="/tmp/evo-failed",
        error="network interrupted",
    )
    reopened = RecipeMemory(tmp_path)
    rows = reopened.list_evolutions(limit=5)
    assert rows[0]["evolution_id"] == "evo-failed"
    assert rows[0]["status"] == "PREPARE_FAILED"
    assert rows[0]["error"] == "network interrupted"


def test_promotion_can_be_marked_rolled_back(tmp_path):
    memory = RecipeMemory(tmp_path)
    memory.record_supervisor_summary(
        successful_summary(),
        resolved_candidates=candidates(),
        evolution_id="evo-1",
        previous_recipe={"candidate_id": "baseline"},
        selected_recipe={"candidate_id": "dflash2_default"},
    )
    memory.mark_promotion_rolled_back(
        run_id="20260828-190000",
        reason="online regression",
        active_seconds=120.0,
    )
    promotion = memory.list_promotions(limit=1)[0]
    assert promotion["rolled_back"] is True
    assert promotion["rollback_reason"] == "online regression"
    assert promotion["active_seconds"] == 120.0
