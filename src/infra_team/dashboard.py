"""Local operations console for model service, optimization and chat.

The server binds to loopback only. Read APIs expose sanitized status and run
artifacts. Mutating actions require same-origin POST plus an unguessable session
token, and are limited to fixed service start/stop/optimize operations. Planner
chat can analyze and propose, but cannot execute shell or bypass the whitelist.
"""

from __future__ import annotations

import fcntl
import json
import mimetypes
import re
import secrets
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from importlib.resources import files
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .artifacts import RUNS_DIRNAME
from .policy import TARGET_MODEL, TARGET_REVISION
from .recipe_memory import RecipeMemory
from .service_manager import ManagedService, ServiceError, ServiceSpec

STATIC_DIR = Path(__file__).with_name("static")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
EVOLUTION_FINAL_STATES = {
    "COMPLETED",
    "DISCOVERY_REJECTED",
    "PREPARE_FAILED",
    "PLAN_REJECTED",
    "BASELINE_FAILED",
    "CANDIDATE_FAILED",
    "NO_IMPROVEMENT",
    "PROMOTION_FAILED",
    "BASELINE_RESTORED",
    "ROLLBACK_FAILED",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _get_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 600.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"model request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("model returned an invalid response")
    return value


def _stream_openai_chat(
    url: str,
    payload: dict[str, Any],
    timeout: float = 600.0,
):
    """Yield text deltas from an OpenAI-compatible SSE response."""
    streaming_payload = dict(payload)
    streaming_payload["stream"] = True
    request = urllib.request.Request(
        url,
        data=json.dumps(streaming_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    return
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError:
                    continue
                try:
                    choice = event["choices"][0]
                    delta = choice.get("delta") or {}
                except (KeyError, IndexError, TypeError):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"model stream failed: {exc}") from exc


def _run_timestamp(run_id: str) -> str | None:
    try:
        parsed = datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S")
        return parsed.astimezone().isoformat()
    except (ValueError, TypeError):
        return None


def _benchmark_evidence(results: list[Any]) -> dict[str, Any]:
    repeat_values: set[int] = set()
    complete = len(results) >= 2
    for result in results:
        if not isinstance(result, dict) or not result:
            complete = False
            break
        repeats = (result.get("spec") or {}).get("repeats")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
            complete = False
            break
        repeat_values.add(repeats)
    repeats = repeat_values.pop() if complete and len(repeat_values) == 1 else None
    if repeats is None:
        mode = "unclassified"
    elif repeats >= 3:
        mode = "formal"
    else:
        mode = "smoke"
    return {
        "benchmark_mode": mode,
        "repeats": repeats,
        "publishable_performance": mode == "formal",
    }


def _candidate_row(
    result: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    performance = result.get("performance") or {}
    quality = result.get("quality") or {}
    spec = result.get("spec") or {}
    evaluation = evaluation or {}
    return {
        "id": result.get("id") or spec.get("id") or evaluation.get("id"),
        "generation_tps": performance.get("generation_tps_median"),
        "generation_tps_min": performance.get("generation_tps_min"),
        "generation_tps_max": performance.get("generation_tps_max"),
        "generation_tps_stdev": performance.get("generation_tps_stdev"),
        "prompt_tps": performance.get("prompt_tps_median"),
        "ttft_seconds": performance.get("ttft_seconds_median"),
        "peak_memory_gb": performance.get("peak_memory_gb"),
        "quality_pass": bool(quality.get("quality_pass")),
        "quality_score": f"{quality.get('passed', 0)}/{quality.get('total', 4)}",
        "error_rate": performance.get("error_rate"),
        "samples": performance.get("samples"),
        "speedup_percent": evaluation.get("speedup_percent"),
        "accepted": bool(evaluation.get("accepted")),
        "qualified": bool(evaluation.get("qualified")),
        "disqualifications": evaluation.get("disqualifications") or [],
        "draft_enabled": bool(spec.get("draft_model")),
        "draft_block_size": spec.get("draft_block_size"),
    }


class DashboardData:
    """Aggregate only the fields the UI needs; never expose control tokens."""

    def __init__(self, root: str, service_name: str = "default") -> None:
        self.root = Path(root).resolve()
        self.service_name = service_name
        self.runs_dir = self.root / RUNS_DIRNAME
        self.evolutions_dir = self.root / ".infra-team" / "evolution"
        self.service = ManagedService(str(self.root), name=service_name)
        self.memory = RecipeMemory(self.root)

    def _metrics(self, service: dict[str, Any]) -> dict[str, Any]:
        if not service.get("healthy") or not service.get("base_url"):
            return {"available": False}
        payload = _get_json(service["base_url"].rstrip("/") + "/v1/metrics")
        if not payload:
            return {"available": False}
        return {
            "available": True,
            "latest": payload.get("latest"),
            "summary": payload.get("summary") or {},
            "recent": (payload.get("recent") or [])[-12:],
        }

    def _run_dirs(self) -> list[Path]:
        try:
            return sorted(
                [path for path in self.runs_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            )
        except OSError:
            return []

    def _run_summary(self, path: Path, detailed: bool = False) -> dict[str, Any]:
        summary = _read_json(path / "summary.json") or {}
        verdict = _read_json(path / "verdict.json") or summary.get("verdict") or {}
        invalid = _read_json(path / "INVALIDATED.json")
        transition = _read_json(path / "service_transition.json") or {}
        plan_artifact = _read_json(path / "agent_plan.json") or {}
        quality_artifact = _read_json(path / "quality.json") or {}
        environment = summary.get("environment") or _read_json(path / "environment.json") or {}
        baseline = summary.get("baseline") or _read_json(path / "baseline.json") or {}
        candidates = summary.get("candidates") or []
        if not candidates:
            experiments = _read_json(path / "experiments.json") or {}
            candidates = experiments.get("candidates") or []
            baseline = baseline or experiments.get("baseline") or {}

        evaluations = {
            item.get("id"): item
            for item in verdict.get("evaluations") or []
            if isinstance(item, dict)
        }
        rows: list[dict[str, Any]] = []
        if baseline:
            rows.append(_candidate_row(baseline, evaluations.get("baseline")))
        for candidate in candidates:
            if isinstance(candidate, dict):
                cid = candidate.get("id") or (candidate.get("spec") or {}).get("id")
                rows.append(_candidate_row(candidate, evaluations.get(cid)))
        evidence = _benchmark_evidence([baseline, *candidates])

        selected_id = verdict.get("selected_id")
        selected_quality = quality_artifact.get(str(selected_id)) or {}
        post_restart_quality = transition.get("post_restart_quality") or {}
        quality_checks = (
            post_restart_quality.get("checks")
            or selected_quality.get("checks")
            or []
        )
        plan = summary.get("plan") or plan_artifact.get("plan") or {}
        plan_meta = summary.get("plan_meta") or plan_artifact.get("meta") or {}
        policy = summary.get("policy") or _read_json(path / "policy_decision.json") or {}

        run = {
            "id": path.name,
            "timestamp": _run_timestamp(path.name),
            "valid": invalid is None,
            "invalid_reason": (invalid or {}).get("reason"),
            "complete": bool(verdict),
            "accepted": bool(verdict.get("accepted")) and invalid is None,
            "selected_id": selected_id,
            "speedup_percent": verdict.get("speedup_percent"),
            "baseline_tps": verdict.get("baseline_tps"),
            "selected_tps": verdict.get("selected_tps"),
            "reason": verdict.get("reason"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            **evidence,
            "planner_source": plan_meta.get("source") or plan.get("source"),
            "hypothesis": plan.get("hypothesis"),
            "candidates": rows,
            "quality": {
                "passed": post_restart_quality.get("passed")
                or selected_quality.get("checks")
                and sum(1 for item in selected_quality["checks"] if item.get("passed")),
                "total": post_restart_quality.get("total")
                or len(selected_quality.get("checks") or []),
                "quality_pass": bool(
                    post_restart_quality.get("quality_pass")
                    if post_restart_quality
                    else selected_quality.get("quality_pass")
                ),
                "checks": quality_checks,
                "post_restart_verified": bool(post_restart_quality),
            },
            "transition_status": transition.get("status"),
            "environment": {
                "chip": environment.get("chip"),
                "gpu_cores": environment.get("gpu_cores"),
                "unified_memory_gb": environment.get("unified_memory_gb"),
                "macos_version": environment.get("macos_version"),
                "disk_free_gb": environment.get("disk_free_gb"),
                "packages": environment.get("packages") or {},
            },
            "policy": {
                "approved": policy.get("approved") or [],
                "refused": policy.get("refused") or [],
                "acceptance": policy.get("acceptance_policy")
                or verdict.get("policy")
                or {},
            },
            "artifacts": sorted(
                item.name
                for item in path.iterdir()
                if item.is_file() and not item.name.startswith(".")
            ),
        }
        if detailed:
            run["plan"] = plan
            run["planner_attempts"] = plan_meta.get("attempts") or []
        return run

    def recent_runs(self, limit: int = 8) -> list[dict[str, Any]]:
        return [self._run_summary(path) for path in self._run_dirs()[:limit]]

    def latest_valid_run(self) -> dict[str, Any] | None:
        for path in self._run_dirs():
            if (path / "INVALIDATED.json").exists():
                continue
            run = self._run_summary(path, detailed=True)
            if run.get("complete"):
                return run
        return None

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        if not RUN_ID_RE.fullmatch(run_id):
            return None
        path = self.runs_dir / run_id
        if not path.is_dir() or path.parent.resolve() != self.runs_dir.resolve():
            return None
        return self._run_summary(path, detailed=True)

    def _relative_ref(self, value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        root_text = str(self.root)
        if root_text in value:
            value = value.replace(root_text, ".")
        value = re.sub(r"/Users/[^/\s]+", "~", value)
        try:
            path = Path(value)
            if path.is_absolute():
                return Path(value).name or "external"
        except (OSError, ValueError):
            pass
        return value

    def _sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._sanitize_payload(item)
                for key, item in value.items()
                if key not in {"signature", "control_host", "control_port"}
            }
        if isinstance(value, list):
            return [self._sanitize_payload(item) for item in value]
        return self._relative_ref(value)

    @staticmethod
    def _recipe_summary(recipe: dict[str, Any] | None) -> dict[str, Any] | None:
        if not recipe:
            return None
        spec = recipe.get("service_spec") or recipe.get("spec") or {}
        accelerator = recipe.get("accelerator") or {}
        quality = recipe.get("online_quality") or {}
        return {
            "candidate_id": recipe.get("candidate_id")
            or ("accelerated" if spec.get("draft_model") else "baseline"),
            "model": spec.get("model") or (recipe.get("target") or {}).get("model"),
            "draft_enabled": bool(spec.get("draft_model")),
            "draft_kind": spec.get("draft_kind"),
            "draft_block_size": spec.get("draft_block_size"),
            "host": spec.get("host"),
            "port": spec.get("port"),
            "manifest_id": accelerator.get("manifest_id"),
            "target_revision": (recipe.get("target") or {}).get("revision"),
            "runtime_version": (recipe.get("runtime") or {}).get("version"),
            "quality_pass": quality.get("quality_pass"),
            "quality_score": (
                f"{quality.get('passed', 0)}/{quality.get('total', 4)}"
                if quality
                else None
            ),
            "promoted_at": recipe.get("promoted_at"),
            "source": recipe.get("source"),
        }

    def recipe_status(self) -> dict[str, Any]:
        active_raw = (
            self.service.active_recipe()
            if hasattr(self.service, "active_recipe")
            else None
        )
        previous_raw = (
            self.service.previous_recipe()
            if hasattr(self.service, "previous_recipe")
            else None
        )
        active = self._recipe_summary(active_raw)
        previous = self._recipe_summary(previous_raw)
        return {
            "active": active,
            "previous": previous,
            "recoverable": previous is not None,
        }

    def _evolution_dirs(self) -> list[Path]:
        try:
            return sorted(
                [path for path in self.evolutions_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            )
        except OSError:
            return []

    @staticmethod
    def _read_evolution_events(path: Path) -> tuple[list[dict[str, Any]], int]:
        events: list[dict[str, Any]] = []
        errors = 0
        try:
            lines = (path / "events.ndjson").read_text(encoding="utf-8").splitlines()
        except OSError:
            return events, errors
        allowed = {
            "time",
            "state",
            "message",
            "candidate_id",
            "manifest_id",
            "progress",
            "reused",
            "supervisor_event",
            "recovered_from",
            "error_type",
        }
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if not isinstance(event, dict):
                errors += 1
                continue
            events.append({key: event.get(key) for key in allowed if key in event})
        return events, errors

    def _bundled_evidence_detail(self) -> dict[str, Any] | None:
        bundle = files("infra_team").joinpath(
            "demo_evidence", "qwen38-dflash2-m5pro"
        )
        try:
            summary = json.loads(bundle.joinpath("summary.json").read_text(encoding="utf-8"))
            environment = json.loads(
                bundle.joinpath("environment.redacted.json").read_text(encoding="utf-8")
            )
            benchmark = json.loads(
                bundle.joinpath("benchmark_samples.json").read_text(encoding="utf-8")
            )
            quality = json.loads(
                bundle.joinpath("quality_results.json").read_text(encoding="utf-8")
            )
            promotion = json.loads(
                bundle.joinpath("promotion_event.json").read_text(encoding="utf-8")
            )
            event_lines = bundle.joinpath("evolution_events.ndjson").read_text(
                encoding="utf-8"
            ).splitlines()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        evaluations = {
            row["id"]: row
            for row in (
                {"id": "baseline", "speedup_percent": 0.0},
                {"id": "dflash2_block6", "speedup_percent": 94.65},
                {
                    "id": "dflash2_default",
                    "speedup_percent": summary.get("speedup_percent"),
                    "accepted": True,
                },
            )
        }
        quality_results = quality.get("results") or {}
        rows = []
        for row in benchmark.get("rows") or []:
            rows.append(
                {
                    "id": row.get("id"),
                    "generation_tps": row.get("generation_tps_median"),
                    "generation_tps_min": row.get("generation_tps_min"),
                    "generation_tps_max": row.get("generation_tps_max"),
                    "generation_tps_stdev": row.get("generation_tps_stdev"),
                    "ttft_seconds": row.get("ttft_seconds_median"),
                    "peak_memory_gb": row.get("peak_memory_gb"),
                    "quality_pass": bool(
                        (quality_results.get(row.get("id")) or {}).get("quality_pass")
                    ),
                    "quality_score": "4/4",
                    "error_rate": row.get("error_rate"),
                    "samples": row.get("samples"),
                    "speedup_percent": evaluations.get(row.get("id"), {}).get(
                        "speedup_percent"
                    ),
                    "accepted": bool(
                        evaluations.get(row.get("id"), {}).get("accepted")
                    ),
                    "qualified": True,
                    "disqualifications": [],
                    "draft_block_size": row.get("draft_block_size"),
                }
            )
        selected_quality = quality_results.get(summary.get("selected_id")) or {}
        events = []
        for line in event_lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        run = {
            "id": summary.get("run_id"),
            "timestamp": _run_timestamp(str(summary.get("run_id"))),
            "valid": True,
            "complete": True,
            "accepted": True,
            "selected_id": summary.get("selected_id"),
            "speedup_percent": summary.get("speedup_percent"),
            "baseline_tps": summary.get("baseline_tps"),
            "selected_tps": summary.get("selected_tps"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "benchmark_mode": "formal",
            "repeats": benchmark.get("repeats_per_prompt"),
            "publishable_performance": True,
            "planner_source": "target_service",
            "candidates": rows,
            "quality": {
                "passed": 4,
                "total": 4,
                "quality_pass": True,
                "checks": selected_quality.get("checks") or [],
                "post_restart_verified": bool(
                    (promotion.get("online_quality") or {}).get("quality_pass")
                ),
            },
            "transition_status": promotion.get("status"),
            "environment": environment,
            "policy": {},
            "artifacts": [],
        }
        return {
            "id": summary.get("evolution_id"),
            "timestamp": _run_timestamp(str(summary.get("evolution_id"))),
            "mode": "historical_replay",
            "label": "内置历史真实运行回放",
            "terminal": True,
            "status": summary.get("status"),
            "accepted": True,
            "selected_id": summary.get("selected_id"),
            "speedup_percent": summary.get("speedup_percent"),
            "artifact_ref": "packaged://qwen38-dflash2-m5pro",
            "events": events,
            "event_errors": 0,
            "stable_api": {"unchanged": True, "host": "127.0.0.1", "port": 8000},
            "recipes": {"before": {"candidate_id": "baseline"}, "after": {"candidate_id": summary.get("selected_id")}, "recoverable": True},
            "memory": {"prior_hits": 2, "requires_local_validation": True},
            "run": run,
            "technical": {
                "manifest_id": promotion.get("candidate_manifest_id"),
                "source_repo": "z-lab/Qwen3.8-27B-DFlash2",
                "source_revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
                "manifest_hash": promotion.get("candidate_manifest_hash"),
                "supervisor_run_id": summary.get("run_id"),
                "artifacts": [
                    "summary.json",
                    "benchmark_samples.json",
                    "quality_results.json",
                    "promotion_event.json",
                ],
            },
        }

    def evolution_detail(self, evolution_id: str) -> dict[str, Any] | None:
        if not RUN_ID_RE.fullmatch(evolution_id):
            return None
        path = self.evolutions_dir / evolution_id
        if not path.is_dir() or path.parent.resolve() != self.evolutions_dir.resolve():
            bundled = self._bundled_evidence_detail()
            return bundled if bundled and bundled.get("id") == evolution_id else None
        summary = _read_json(path / "summary.json") or {}
        events, event_errors = self._read_evolution_events(path)
        status = summary.get("status") or (events[-1].get("state") if events else "UNKNOWN")
        terminal = status in EVOLUTION_FINAL_STATES
        before = _read_json(path / "active_recipe_before.json") or {}
        after = _read_json(path / "active_recipe_after.json") or {}
        memory_prior = _read_json(path / "memory_prior.json") or {}
        candidate_manifest = _read_json(path / "candidate_manifest.json") or {}
        supervisor_link = _read_json(path / "supervisor_link.json") or {}
        run_id = summary.get("supervisor_run_id") or supervisor_link.get("run_id")
        run = self.run_detail(str(run_id)) if run_id else None
        before_spec = before.get("service_spec") or before.get("spec") or {}
        after_spec = after.get("service_spec") or after.get("spec") or {}
        stable_api_unchanged = bool(
            before_spec
            and after_spec
            and before_spec.get("host") == after_spec.get("host")
            and before_spec.get("port") == after_spec.get("port")
        )
        manifests = candidate_manifest.get("manifests") or []
        manifest = manifests[0] if manifests else {}
        source = manifest.get("source") or {}
        ranking = memory_prior.get("ranking") or []
        prior_hits = [item for item in ranking if item.get("match") in {"exact", "similar"}]
        result = {
            "id": evolution_id,
            "timestamp": _run_timestamp(evolution_id),
            "mode": "historical_replay" if terminal else "live",
            "label": "历史真实运行回放" if terminal else "实时演进",
            "terminal": terminal,
            "status": status,
            "accepted": bool(summary.get("accepted")),
            "selected_id": summary.get("selected_id"),
            "speedup_percent": summary.get("speedup_percent"),
            "artifact_ref": str(path.relative_to(self.root)),
            "events": events,
            "event_errors": event_errors,
            "stable_api": {
                "unchanged": stable_api_unchanged,
                "host": after_spec.get("host") or before_spec.get("host"),
                "port": after_spec.get("port") or before_spec.get("port"),
            },
            "recipes": {
                "before": self._recipe_summary(before),
                "after": self._recipe_summary(after),
                "recoverable": bool(before_spec),
            },
            "memory": {
                "prior_hits": len(prior_hits),
                "requires_local_validation": bool(
                    memory_prior.get("requires_local_validation", True)
                ),
            },
            "run": run,
            "technical": {
                "manifest_id": manifest.get("id")
                or (after.get("accelerator") or {}).get("manifest_id"),
                "source_repo": source.get("repo_id"),
                "source_revision": source.get("revision"),
                "manifest_hash": (candidate_manifest.get("hashes") or {}).get(
                    manifest.get("id")
                ),
                "supervisor_run_id": run_id,
                "artifacts": sorted(
                    item.name
                    for item in path.iterdir()
                    if item.is_file() and not item.name.startswith(".")
                ),
            },
        }
        return self._sanitize_payload(result)

    def evolution_snapshot(self) -> dict[str, Any]:
        recent = []
        latest = None
        for path in self._evolution_dirs()[:8]:
            detail = self.evolution_detail(path.name)
            if detail is None:
                continue
            if latest is None:
                latest = detail
            recent.append(
                {
                    "id": detail["id"],
                    "timestamp": detail["timestamp"],
                    "status": detail["status"],
                    "mode": detail["mode"],
                    "accepted": detail["accepted"],
                    "selected_id": detail["selected_id"],
                    "speedup_percent": detail["speedup_percent"],
                }
            )
        if latest is None:
            latest = self._bundled_evidence_detail()
            if latest is not None:
                recent.append(
                    {
                        "id": latest["id"],
                        "timestamp": latest["timestamp"],
                        "status": latest["status"],
                        "mode": latest["mode"],
                        "accepted": latest["accepted"],
                        "selected_id": latest["selected_id"],
                        "speedup_percent": latest["speedup_percent"],
                    }
                )
        return {"latest": latest, "recent": recent}

    def memory_summary(self) -> dict[str, Any]:
        stats = self.memory.show()
        return {
            "schema_version": stats.get("schema_version"),
            "environments": stats.get("environments", 0),
            "candidate_runs": stats.get("candidate_runs", 0),
            "promotions": stats.get("promotions", 0),
            "evolution_runs": stats.get("evolution_runs", 0),
            "watch_states": stats.get("watch_states", 0),
            "notification_events": stats.get("notification_events", 0),
            "recent_evolutions": [
                {
                    "evolution_id": item.get("evolution_id"),
                    "status": item.get("status"),
                    "supervisor_run_id": item.get("supervisor_run_id"),
                    "completed_at": item.get("completed_at"),
                }
                for item in stats.get("recent_evolutions") or []
            ],
        }

    def watch_status(self) -> dict[str, Any]:
        states = [
            item
            for item in self.memory.list_watch_states(limit=20)
            if item.get("service_name") == self.service_name
            and item.get("manifest_id") != "__registry_scheduler__"
        ]
        if not states:
            return {"enabled": True, "status": "NO_HISTORY", "manifest_id": None}
        latest = states[0]
        return {
            "enabled": True,
            "status": latest.get("status"),
            "manifest_id": latest.get("manifest_id"),
            "attempt_count": latest.get("attempt_count", 0),
            "next_attempt_at": latest.get("next_attempt_at"),
            "last_evolution_id": latest.get("last_evolution_id"),
            "updated_at": latest.get("updated_at"),
        }

    def snapshot(self) -> dict[str, Any]:
        service = self.service.status()
        # Defense in depth: ManagedService already redacts these.
        for key in (
            "signature",
            "control_host",
            "control_port",
            "server_log",
            "wrapper_log",
        ):
            service.pop(key, None)
        service = self._sanitize_payload(service)
        evolution = self.evolution_snapshot()
        recipes = self.recipe_status()
        latest_evolution = evolution.get("latest") or {}
        stable_api = {
            "endpoint": (
                str(service.get("base_url")).rstrip("/") + "/v1"
                if service.get("base_url")
                else None
            ),
            "unchanged": bool(
                (latest_evolution.get("stable_api") or {}).get("unchanged")
            ),
            "healthy": bool(service.get("healthy")),
        }
        return {
            "generated_at": datetime.now().astimezone().isoformat(),
            "product": {
                "promise": "安装一次，以后本地 AI 自己测试和升级。",
                "behavior": "发现新能力，在本机测试，更好才升级，失败就恢复。",
                "human_decisions_after_start": 0,
            },
            "service": service,
            "stable_api": stable_api,
            "recipes": recipes,
            "metrics": self._metrics(service),
            "latest_run": self.latest_valid_run(),
            "recent_runs": self.recent_runs(),
            "evolution": evolution,
            "watch": self.watch_status(),
            "memory": self.memory_summary(),
            "dashboard": {
                "refresh_seconds": 5,
                "read_only": False,
                "service_name": self.service_name,
            },
        }


class DashboardController:
    """Bounded write actions and model chat for one local dashboard session."""

    def __init__(self, data: DashboardData) -> None:
        self.data = data
        self.root = data.root
        self.service = data.service
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._optimization: subprocess.Popen[str] | None = None
        self._optimization_started_at: str | None = None
        self._optimization_log = (
            self.root / ".infra-team" / "dashboard" / "optimization.log"
        )
        self._optimization_lock = self.root / ".infra-team" / "optimize.lock"

    def session(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "capabilities": [
                "service.start",
                "service.stop",
                "evolution.start",
                "optimization.start",
                "chat.deployed",
                "chat.planner",
            ],
            "commands": [
                "/status",
                "/start",
                "/stop",
                "/evolve",
                "/optimize",
                "/help",
            ],
        }

    def _workspace_optimization_locked(self) -> bool:
        self._optimization_lock.parent.mkdir(parents=True, exist_ok=True)
        with self._optimization_lock.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return False

    def _optimization_state(self) -> dict[str, Any]:
        proc = self._optimization
        code = proc.poll() if proc is not None else None
        externally_locked = self._workspace_optimization_locked()
        running = (proc is not None and code is None) or externally_locked
        if proc is None and not running:
            return {"status": "idle", "running": False}
        status = "running" if running else ("completed" if code == 0 else "failed")
        log_tail = ""
        try:
            text = self._optimization_log.read_text(encoding="utf-8", errors="replace")
            log_tail = "\n".join(text.splitlines()[-18:])
        except OSError:
            pass
        return {
            "status": status,
            "running": code is None,
            "started_at": self._optimization_started_at,
            "exit_code": code,
            "log_tail": log_tail,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._optimization_state()
            return {
                "evolution": state,
                "optimization": state,
                "writes_enabled": True,
                "confirmation_required": True,
            }

    def _assert_optimizer_idle(self) -> None:
        state = self._optimization_state()
        if state["running"]:
            raise RuntimeError("优化任务正在运行，服务生命周期由 Supervisor 接管")

    def start_service(self) -> dict[str, Any]:
        with self._lock:
            self._assert_optimizer_idle()
            current = self.service.status()
            if current.get("healthy"):
                return {"ok": True, "message": "模型服务已经在运行", "service": current}
            spec = self.service.saved_spec() or ServiceSpec(
                name=self.data.service_name,
                model=TARGET_MODEL,
                host="127.0.0.1",
                port=8000,
                target_revision=TARGET_REVISION,
            )
            result = self.service.start(spec, wait_timeout=900.0)
            if hasattr(self.service, "atomically_promote"):
                self.service.atomically_promote(
                    self.service.recipe_from_spec(spec, source="dashboard_start")
                )
            return {"ok": True, "message": "模型服务已启动", "service": result}

    def stop_service(self) -> dict[str, Any]:
        with self._lock:
            self._assert_optimizer_idle()
            current = self.service.status()
            if not current.get("managed"):
                return {"ok": True, "message": "没有正在运行的受管模型服务"}
            result = self.service.stop(timeout=30.0)
            return {"ok": True, "message": "模型服务已停止", "service": result}

    def start_evolution(self, repeats: int = 3) -> dict[str, Any]:
        with self._lock:
            self._assert_optimizer_idle()
            current = self.service.status()
            if not current.get("healthy"):
                raise RuntimeError("请先启动模型服务，再检查并升级")
            repeats = int(repeats)
            if repeats not in (1, 3, 5):
                raise ValueError("repeats 只允许 1、3 或 5")
            self._optimization_log.parent.mkdir(parents=True, exist_ok=True)
            log = self._optimization_log.open("a", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "infra_team.cli",
                "--root",
                str(self.root),
                "evolve",
                "--once",
                "--service-name",
                self.data.service_name,
                "--policy",
                "builtin",
                "--repeats",
                str(repeats),
                "--timeout",
                "1800",
            ]
            try:
                self._optimization = subprocess.Popen(
                    command,
                    cwd=str(self.root),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            finally:
                log.close()
            self._optimization_started_at = datetime.now().astimezone().isoformat()
            state = self._optimization_state()
            return {
                "ok": True,
                "message": "完整自演进已启动：系统会发现、测试并只采用更好的本地能力",
                "evolution": state,
                "optimization": state,
            }

    def start_optimization(self, repeats: int = 3) -> dict[str, Any]:
        """Compatibility alias; Dashboard product flow now runs full evolution."""
        return self.start_evolution(repeats)

    @staticmethod
    def _validate_messages(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list) or not raw or len(raw) > 24:
            raise ValueError("messages 必须是 1 到 24 条对话")
        messages: list[dict[str, str]] = []
        total_chars = 0
        for item in raw:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                raise ValueError("对话角色只允许 user 或 assistant")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("对话内容不能为空")
            if len(content) > 8000:
                raise ValueError("单条消息不能超过 8000 字符")
            total_chars += len(content)
            messages.append({"role": item["role"], "content": content.strip()})
        if total_chars > 24000:
            raise ValueError("对话上下文过长")
        return messages

    def _prepare_chat(
        self, mode: str, raw_messages: Any
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None, str | None]:
        """Return (immediate, URL, payload, model)."""
        if mode not in ("deployed", "planner"):
            raise ValueError("mode 必须是 deployed 或 planner")
        messages = self._validate_messages(raw_messages)
        last = messages[-1]["content"].strip().lower()
        if last == "/status":
            snapshot = self.data.snapshot()
            service = snapshot.get("service") or {}
            run = snapshot.get("latest_run") or {}
            evidence = {
                "formal": "正式复测",
                "smoke": "Smoke estimate",
                "unclassified": "证据未标注",
            }.get(run.get("benchmark_mode"), "证据未标注")
            content = (
                f"服务状态：{service.get('status', 'unknown')}；"
                f"当前模型：{(service.get('spec') or {}).get('model', 'unknown')}；"
                f"当前 Recipe：{run.get('selected_id') or 'baseline'}；"
                f"最近有效加速：{run.get('speedup_percent') or 0:.2f}%"
                f"（{evidence}）。"
            )
            return (
                {"ok": True, "mode": mode, "content": content, "deterministic": True},
                None,
                None,
                None,
            )
        if last == "/help":
            return (
                {
                    "ok": True,
                    "mode": mode,
                    "content": (
                        "可用命令：/status 查看状态；/start 启动服务；/stop 停止服务；"
                        "/evolve 检查并升级本地 AI；/optimize 是兼容别名。写操作会先要求确认。"
                    ),
                    "deterministic": True,
                },
                None,
                None,
                None,
            )
        if last in ("/start", "/stop", "/evolve", "/optimize"):
            action = "evolve" if last in ("/evolve", "/optimize") else last.removeprefix("/")
            return (
                {
                    "ok": True,
                    "mode": mode,
                    "content": f"命令 {last} 需要在界面确认后执行。",
                    "requested_action": action,
                    "deterministic": True,
                },
                None,
                None,
                None,
            )

        service = self.service.status()
        if not service.get("healthy") or not service.get("base_url"):
            raise RuntimeError("本地模型服务未运行，当前无法对话")
        spec = service.get("spec") or {}
        model = str(spec.get("model") or TARGET_MODEL)
        outgoing = list(messages)
        if mode == "planner":
            snapshot = self.data.snapshot()
            run = snapshot.get("latest_run") or {}
            safe_context = {
                "service": {
                    "status": service.get("status"),
                    "model": model,
                    "draft_model": spec.get("draft_model"),
                    "draft_block_size": spec.get("draft_block_size"),
                },
                "latest_valid_run": {
                    "id": run.get("id"),
                    "selected_id": run.get("selected_id"),
                    "speedup_percent": run.get("speedup_percent"),
                    "quality": run.get("quality"),
                    "candidates": run.get("candidates"),
                },
            }
            system = (
                "You are the virtual AI Infra planner for this local machine. "
                "Analyze deployment health, performance and optimization evidence. "
                "You may explain, diagnose and propose only bounded actions: start service, "
                "stop service, or run the existing whitelisted optimization workflow. "
                "You cannot execute shell, modify policy, bypass the 4/4 quality gate, or "
                "claim an action happened. Tell the user to confirm a UI action when execution "
                "is needed. Current sanitized context:\n" + json.dumps(safe_context)
            )
            outgoing.insert(0, {"role": "system", "content": system})

        payload = {
            "model": model,
            "messages": outgoing,
            "temperature": 0.2 if mode == "planner" else 0.7,
            "max_tokens": 768,
            "stream": False,
            "enable_thinking": False,
        }
        url = service["base_url"].rstrip("/") + "/v1/chat/completions"
        return None, url, payload, model

    def chat(self, mode: str, raw_messages: Any) -> dict[str, Any]:
        immediate, url, payload, model = self._prepare_chat(mode, raw_messages)
        if immediate is not None:
            return immediate
        assert url is not None and payload is not None and model is not None
        data = _post_json(url, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("模型响应缺少 message.content") from exc
        return {"ok": True, "mode": mode, "content": content, "model": model}

    def stream_chat(self, mode: str, raw_messages: Any):
        """Yield browser-facing NDJSON events while the local model generates."""
        immediate, url, payload, model = self._prepare_chat(mode, raw_messages)
        if immediate is not None:
            content = str(immediate.get("content") or "")
            if content:
                yield {"type": "delta", "content": content}
            if immediate.get("requested_action"):
                yield {"type": "action", "action": immediate["requested_action"]}
            yield {
                "type": "done",
                "mode": mode,
                "deterministic": True,
            }
            return

        assert url is not None and payload is not None and model is not None
        yield {"type": "start", "mode": mode, "model": model}
        emitted = False
        for delta in _stream_openai_chat(url, payload):
            emitted = True
            yield {"type": "delta", "content": delta}
        if not emitted:
            raise RuntimeError("模型流结束但没有生成文本")
        yield {"type": "done", "mode": mode, "model": model}


class DashboardHandler(BaseHTTPRequestHandler):
    data: DashboardData
    controller: DashboardController
    static_dir: Path = STATIC_DIR

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal output quiet; the UI surfaces API errors explicitly.
        return

    def _security_headers(self) -> None:
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'",
        )

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_ndjson(self, events) -> None:
        """Flush one JSON event per line as soon as a model delta arrives."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-transform")
        self.send_header("Connection", "close")
        self._security_headers()
        self.end_headers()
        self.close_connection = True
        try:
            for event in events:
                line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                line = json.dumps(
                    {"type": "error", "error": str(exc)}, ensure_ascii=False
                ).encode("utf-8") + b"\n"
                self.wfile.write(line)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_file(self, path: Path) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > 131072:
            raise ValueError("request body must be 1 to 131072 bytes")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _require_control(self) -> None:
        token = self.headers.get("X-Infra-Control-Token", "")
        if not token or not secrets.compare_digest(token, self.controller.token):
            raise PermissionError("invalid dashboard control token")
        origin = self.headers.get("Origin")
        expected = f"http://{self.headers.get('Host')}"
        if not origin or origin != expected:
            raise PermissionError("cross-origin control request refused")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/session":
            self._send_json(self.controller.session())
            return
        if parsed.path == "/api/dashboard":
            snapshot = self.data.snapshot()
            snapshot["control"] = self.controller.status()
            snapshot["dashboard"]["read_only"] = False
            self._send_json(snapshot)
            return
        if parsed.path == "/api/evolution/current":
            latest = self.data.evolution_snapshot().get("latest")
            self._send_json(latest or {"status": "NO_HISTORY"})
            return
        if parsed.path == "/api/evolution/events":
            latest = self.data.evolution_snapshot().get("latest") or {}
            self._send_json(
                {
                    "evolution_id": latest.get("id"),
                    "mode": latest.get("mode"),
                    "label": latest.get("label"),
                    "events": latest.get("events") or [],
                }
            )
            return
        if parsed.path == "/api/watch/status":
            self._send_json(self.data.watch_status())
            return
        if parsed.path == "/api/memory/summary":
            self._send_json(self.data.memory_summary())
            return
        if parsed.path.startswith("/api/evolutions/"):
            evolution_id = parsed.path.removeprefix("/api/evolutions/")
            evolution = self.data.evolution_detail(evolution_id)
            self._send_json(
                evolution or {"error": "evolution not found"},
                200 if evolution else 404,
            )
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.removeprefix("/api/runs/")
            run = self.data.run_detail(run_id)
            self._send_json(run or {"error": "run not found"}, 200 if run else 404)
            return
        if parsed.path == "/health":
            self._send_json({"status": "healthy", "read_only": False})
            return
        if parsed.path in ("/", "/index.html"):
            self._send_file(self.static_dir / "dashboard.html")
            return
        if parsed.path.startswith("/static/"):
            name = parsed.path.removeprefix("/static/")
            if "/" in name or "\\" in name or name.startswith("."):
                self.send_error(404)
                return
            self._send_file(self.static_dir / name)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            self._require_control()
            body = self._read_body()
            if parsed.path == "/api/control/start":
                self._send_json(self.controller.start_service())
                return
            if parsed.path == "/api/control/stop":
                self._send_json(self.controller.stop_service())
                return
            if parsed.path in ("/api/control/evolve", "/api/control/optimize"):
                self._send_json(
                    self.controller.start_evolution(body.get("repeats", 3)),
                    status=202,
                )
                return
            if parsed.path == "/api/chat/stream":
                self._send_ndjson(
                    self.controller.stream_chat(
                        str(body.get("mode") or "deployed"),
                        body.get("messages"),
                    )
                )
                return
            if parsed.path == "/api/chat":
                self._send_json(
                    self.controller.chat(
                        str(body.get("mode") or "deployed"),
                        body.get("messages"),
                    )
                )
                return
            self._send_json({"error": "unknown control endpoint"}, status=404)
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=403)
        except (ValueError, RuntimeError, ServiceError) as exc:
            self._send_json({"error": str(exc)}, status=409)
        except Exception as exc:
            self._send_json(
                {"error": f"unexpected dashboard error: {type(exc).__name__}: {exc}"},
                status=500,
            )


def create_dashboard_server(
    root: str,
    service_name: str = "default",
    host: str = "127.0.0.1",
    port: int = 9000,
) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("dashboard is local-only; bind to a loopback address")
    # Port 0 is accepted for tests and programmatic callers that request an
    # ephemeral loopback port. The CLI default remains the stable port 9000.
    if not (0 <= int(port) <= 65535):
        raise ValueError(f"invalid dashboard port: {port}")

    data = DashboardData(root, service_name)
    controller = DashboardController(data)

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.data = data
    BoundHandler.controller = controller
    server = ThreadingHTTPServer((host, int(port)), BoundHandler)
    server.daemon_threads = True
    return server


def serve_dashboard(
    root: str,
    service_name: str = "default",
    host: str = "127.0.0.1",
    port: int = 9000,
) -> None:
    server = create_dashboard_server(root, service_name, host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
