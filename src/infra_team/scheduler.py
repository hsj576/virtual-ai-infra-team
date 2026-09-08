"""P4 continuous registry watcher, notifications and launchd rendering.

The watcher never holds the optimization lock while polling or sleeping. Each
actual evolution delegates to ``EvolutionEngine.run_once()``, which owns the
shared maintenance lock only for the bounded run.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from . import probe_macos
from .candidate_registry import CandidateRegistry
from .compatibility import AutonomyPolicy
from .policy import TARGET_MODEL
from .recipe_memory import RecipeMemory, environment_fingerprint
from .service_manager import ManagedService


TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_NO_IMPROVEMENT = {"NO_IMPROVEMENT"}
PERMANENT_BLOCK = {"ROLLBACK_FAILED", "DISCOVERY_REJECTED", "PLAN_REJECTED"}
TRANSIENT_FAILURE = {
    "PREPARE_FAILED",
    "BASELINE_FAILED",
    "CANDIDATE_FAILED",
    "PROMOTION_FAILED",
    "BASELINE_RESTORED",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def watch_context_key(
    *,
    manifest_hash: str,
    environment_fingerprint: str,
    policy_hash: str,
    service_name: str,
) -> str:
    payload = {
        "manifest_hash": manifest_hash,
        "environment_fingerprint": environment_fingerprint,
        "policy_hash": policy_hash,
        "service_name": service_name,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_field(value: Any, limit: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._:/+% -]", "?", str(value or ""))
    return text[:limit]


NOTIFICATION_TEMPLATES = {
    "candidate_detected": (
        "AI Infra Self-Evolution",
        "Detected trusted candidate: {manifest_id}",
    ),
    "waiting_window": (
        "AI Infra Self-Evolution",
        "Waiting for maintenance window: {next_window}",
    ),
    "evolution_started": (
        "AI Infra Self-Evolution",
        "Evolution started for {manifest_id}",
    ),
    "evolution_completed": (
        "AI Infra Self-Evolution",
        "Evolution completed: {status} / {selected_id}",
    ),
    "evolution_failed": (
        "AI Infra Self-Evolution",
        "Evolution failed: {status}; retry scheduled",
    ),
    "manual_recovery_required": (
        "AI Infra Self-Evolution",
        "Automatic rollback failed; manual recovery required",
    ),
}


class NotificationCenterNotifier:
    """Fixed-template macOS notifications plus append-only audit."""

    def __init__(
        self,
        root: str | Path,
        *,
        memory: RecipeMemory,
        enabled: bool,
        sender: Callable[..., Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.memory = memory
        self.enabled = enabled
        self.sender = sender or subprocess.run
        self.audit_path = (
            self.root / ".infra-team" / "notifications" / "events.ndjson"
        )

    @staticmethod
    def _applescript(title: str, message: str) -> str:
        # Values are sanitized to an intentionally small character set before
        # interpolation. Candidate-provided prose or commands are never used.
        return f'display notification "{message}" with title "{title}"'

    def send(self, template_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        if template_id not in NOTIFICATION_TEMPLATES:
            raise ValueError(f"unknown notification template: {template_id}")
        safe_fields = {key: _safe_field(value) for key, value in fields.items()}
        title_template, message_template = NOTIFICATION_TEMPLATES[template_id]
        title = title_template.format_map(_DefaultFields(safe_fields))
        message = message_template.format_map(_DefaultFields(safe_fields))
        fields_hash = hashlib.sha256(
            _canonical(safe_fields).encode("utf-8")
        ).hexdigest()
        sent = False
        detail = "notifications disabled"
        if self.enabled:
            try:
                result = self.sender(
                    [
                        "/usr/bin/osascript",
                        "-e",
                        self._applescript(title, message),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                sent = int(result.returncode) == 0
                detail = (getattr(result, "stderr", "") or "").strip()[:300]
            except (OSError, subprocess.SubprocessError) as exc:
                detail = str(exc)[:300]
        event = {
            "time": _iso(datetime.now(UTC)),
            "template_id": template_id,
            "fields_hash": fields_hash,
            "sent": sent,
            "detail": detail,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.memory.record_notification(
            template_id=template_id,
            fields_hash=fields_hash,
            sent=sent,
            detail=detail,
            created_at=event["time"],
        )
        return event


class _DefaultFields(dict):
    def __missing__(self, key: str) -> str:
        return "-"


class NullNotifier:
    def send(self, template_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        return {"sent": False, "template_id": template_id}


class EvolutionWatcher:
    """Poll trusted Registry and run bounded P2 cycles when due."""

    def __init__(
        self,
        *,
        root: str | Path,
        service_name: str,
        policy: AutonomyPolicy,
        registry: CandidateRegistry,
        memory: RecipeMemory,
        service_manager: ManagedService,
        engine_factory: Callable[[], Any],
        notifier: Any | None = None,
        probe_fn: Callable[[], dict[str, Any]] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        poll_interval_seconds: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.service_name = service_name
        self.policy = policy
        self.registry = registry
        self.memory = memory
        self.service_manager = service_manager
        self.engine_factory = engine_factory
        self.notifier = notifier or NullNotifier()
        self.probe_fn = probe_fn
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.sleep_fn = sleep_fn or time.sleep
        self.poll_interval_seconds = int(
            poll_interval_seconds or policy.watch_poll_interval_seconds
        )
        if self.poll_interval_seconds <= 0:
            raise ValueError("watch poll interval must be positive")
        self.scheduler_context_key = hashlib.sha256(
            _canonical(
                {
                    "kind": "registry_scheduler",
                    "service_name": self.service_name,
                    "policy_hash": self.policy.evaluation_policy_hash,
                }
            ).encode("utf-8")
        ).hexdigest()

    def _notify(self, template_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.notifier.send(template_id, fields)
        except Exception as exc:
            return {
                "sent": False,
                "template_id": template_id,
                "detail": f"notification failed: {exc}",
            }

    def _probe(self, manifests) -> dict[str, Any]:
        if self.probe_fn is not None:
            return self.probe_fn()
        return probe_macos.probe(
            [TARGET_MODEL, *(manifest.source.repo_id for manifest in manifests)]
        ).to_dict()

    def _contexts(self) -> tuple[list[dict[str, Any]], str]:
        manifests = self.registry.load()
        snapshot = self.registry.snapshot()
        environment = self._probe(manifests)
        target_revision = next(
            (
                manifest.compatibility.target_revision
                for manifest in manifests
                if manifest.compatibility.target_revision
            ),
            None,
        )
        fingerprint = environment_fingerprint(
            environment, TARGET_MODEL, target_revision
        )
        contexts = []
        for manifest in manifests:
            contexts.append(
                {
                    "context_key": watch_context_key(
                        manifest_hash=manifest.manifest_hash,
                        environment_fingerprint=fingerprint,
                        policy_hash=self.policy.evaluation_policy_hash,
                        service_name=self.service_name,
                    ),
                    "manifest_id": manifest.id,
                    "manifest_hash": manifest.manifest_hash,
                    "environment_fingerprint": fingerprint,
                    "policy_hash": self.policy.evaluation_policy_hash,
                    "snapshot_hash": snapshot.get("snapshot_hash"),
                    "candidate_ids": [variant.id for variant in manifest.variants],
                }
            )
        contexts.sort(key=lambda item: item["manifest_id"])
        return contexts, str(snapshot.get("snapshot_hash") or "")

    def _bootstrap_from_history(
        self, context: dict[str, Any], now: datetime
    ) -> dict[str, Any] | None:
        evidence = self.memory.latest_manifest_outcome(
            context["manifest_hash"], context["environment_fingerprint"]
        )
        if evidence is None:
            return None
        active = self.service_manager.active_recipe() or {}
        active_manifest_hash = ((active.get("accelerator") or {}).get("manifest_hash"))
        if evidence["accepted"] and active_manifest_hash == context["manifest_hash"]:
            status = "COMPLETED"
            next_attempt_at = None
        elif not evidence["any_quality_pass"]:
            status = "DISCOVERY_REJECTED"
            next_attempt_at = None
        elif not evidence["accepted"]:
            status = "NO_IMPROVEMENT"
            next_attempt_at = _iso(
                now
                + timedelta(
                    seconds=self.policy.watch_no_improvement_recheck_seconds
                )
            )
        else:
            return None
        self.memory.record_watch_state(
            context_key=context["context_key"],
            service_name=self.service_name,
            manifest_id=context["manifest_id"],
            manifest_hash=context["manifest_hash"],
            environment_fingerprint=context["environment_fingerprint"],
            policy_hash=context["policy_hash"],
            snapshot_hash=context["snapshot_hash"],
            status=status,
            attempt_count=0,
            next_attempt_at=next_attempt_at,
            completed_at=_iso(now) if status in TERMINAL_SUCCESS | PERMANENT_BLOCK else None,
            now=_iso(now),
        )
        return self.memory.get_watch_state(context["context_key"])

    def _due_contexts(
        self, contexts: list[dict[str, Any]], now: datetime
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        due = []
        suppressed = []
        for context in contexts:
            state = self.memory.get_watch_state(context["context_key"])
            if state is None:
                state = self._bootstrap_from_history(context, now)
            elif state.get("status") in {"DETECTED", "WAITING_WINDOW"}:
                bootstrapped = self._bootstrap_from_history(context, now)
                if bootstrapped is not None:
                    state = bootstrapped
            if state is None:
                due.append(context)
                continue
            status = str(state.get("status") or "")
            next_attempt = _parse_iso(state.get("next_attempt_at"))
            if status == "COMPLETED":
                suppressed.append({**context, "reason": "completed"})
            elif status == "ROLLBACK_FAILED":
                suppressed.append({**context, "reason": "manual_recovery"})
            elif status in PERMANENT_BLOCK:
                suppressed.append({**context, "reason": "permanent_block"})
            elif next_attempt is not None and now.astimezone(UTC) < next_attempt:
                suppressed.append(
                    {
                        **context,
                        "reason": "backoff",
                        "next_attempt_at": state.get("next_attempt_at"),
                    }
                )
            else:
                due.append(context)
        return due, suppressed

    def _record_detected(self, context: dict[str, Any], now: datetime) -> None:
        self.memory.record_watch_state(
            context_key=context["context_key"],
            service_name=self.service_name,
            manifest_id=context["manifest_id"],
            manifest_hash=context["manifest_hash"],
            environment_fingerprint=context["environment_fingerprint"],
            policy_hash=context["policy_hash"],
            snapshot_hash=context["snapshot_hash"],
            status="DETECTED",
            attempt_count=0,
            now=_iso(now),
        )

    def _complete_contexts(
        self,
        contexts: list[dict[str, Any]],
        result: dict[str, Any],
        now: datetime,
    ) -> None:
        global_status = str(result.get("status") or "UNKNOWN")
        selected_id = result.get("selected_id")
        for context in contexts:
            status = global_status
            existing = self.memory.get_watch_state(context["context_key"])
            attempts = int((existing or {}).get("attempt_count") or 0)
            next_attempt = None
            evidence = self.memory.latest_manifest_outcome(
                context["manifest_hash"], context["environment_fingerprint"]
            )
            if global_status in TERMINAL_SUCCESS | TERMINAL_NO_IMPROVEMENT:
                if evidence is not None and not evidence["any_quality_pass"]:
                    status = "DISCOVERY_REJECTED"
                elif evidence is not None and evidence["accepted"]:
                    status = "COMPLETED"
                elif global_status in TERMINAL_SUCCESS and (
                    selected_id in context.get("candidate_ids", ())
                    or (selected_id is None and len(contexts) == 1)
                ):
                    status = "COMPLETED"
                else:
                    status = "NO_IMPROVEMENT"
                if status == "NO_IMPROVEMENT":
                    next_attempt = _iso(
                        now
                        + timedelta(
                            seconds=self.policy.watch_no_improvement_recheck_seconds
                        )
                    )
                attempts = 0
            elif global_status in TRANSIENT_FAILURE:
                attempts += 1
                delay = min(
                    self.policy.watch_backoff_cap_seconds,
                    self.policy.watch_backoff_base_seconds * (2 ** (attempts - 1)),
                )
                next_attempt = _iso(now + timedelta(seconds=delay))
            elif global_status in PERMANENT_BLOCK:
                attempts = 0
            self.memory.record_watch_state(
                context_key=context["context_key"],
                service_name=self.service_name,
                manifest_id=context["manifest_id"],
                manifest_hash=context["manifest_hash"],
                environment_fingerprint=context["environment_fingerprint"],
                policy_hash=context["policy_hash"],
                snapshot_hash=context["snapshot_hash"],
                status=status,
                attempt_count=attempts,
                next_attempt_at=next_attempt,
                last_evolution_id=result.get("evolution_id"),
                error_class=result.get("error_type"),
                last_error=result.get("error"),
                completed_at=(
                    _iso(now)
                    if status in TERMINAL_SUCCESS | PERMANENT_BLOCK
                    else None
                ),
                now=_iso(now),
            )

    def run_cycle(self) -> dict[str, Any]:
        now = self.now_fn()
        contexts, snapshot_hash = self._contexts()
        due, suppressed = self._due_contexts(contexts, now)
        manual = [item for item in suppressed if item["reason"] == "manual_recovery"]
        if manual:
            self._notify(
                "manual_recovery_required",
                {
                    "manifest_id": manual[0]["manifest_id"],
                    "status": "ROLLBACK_FAILED",
                },
            )
            return {
                "action": "blocked_manual_recovery",
                "manifest_ids": [item["manifest_id"] for item in manual],
            }
        if not due:
            backoff = [item for item in suppressed if item["reason"] == "backoff"]
            if backoff:
                return {
                    "action": "backoff",
                    "next_attempt_at": min(
                        item["next_attempt_at"] for item in backoff
                    ),
                }
            return {"action": "duplicate_suppressed", "snapshot_hash": snapshot_hash}

        if not self.policy.maintenance_window.allows(now):
            next_window = self.policy.maintenance_window.next_start(now)
            next_attempt_at = _iso(next_window)
            for context in due:
                existing = self.memory.get_watch_state(context["context_key"])
                self.memory.record_watch_state(
                    context_key=context["context_key"],
                    service_name=self.service_name,
                    manifest_id=context["manifest_id"],
                    manifest_hash=context["manifest_hash"],
                    environment_fingerprint=context["environment_fingerprint"],
                    policy_hash=context["policy_hash"],
                    snapshot_hash=context["snapshot_hash"],
                    status="WAITING_WINDOW",
                    attempt_count=int((existing or {}).get("attempt_count") or 0),
                    next_attempt_at=next_attempt_at,
                    now=_iso(now),
                )
            self._notify(
                "waiting_window", {"next_window": next_window.isoformat()}
            )
            return {
                "action": "outside_maintenance_window",
                "next_window": next_window.isoformat(),
                "manifest_ids": [item["manifest_id"] for item in due],
            }

        for context in due:
            if self.memory.get_watch_state(context["context_key"]) is None:
                self._record_detected(context, now)
                self._notify(
                    "candidate_detected", {"manifest_id": context["manifest_id"]}
                )
        manifest_ids = [item["manifest_id"] for item in due]
        self._notify(
            "evolution_started", {"manifest_id": ",".join(manifest_ids)}
        )
        engine = self.engine_factory()
        try:
            result = engine.run_once(
                trigger_source="watch",
                trigger_context={
                    "manifest_ids": manifest_ids,
                    "manifest_hashes": {
                        item["manifest_id"]: item["manifest_hash"] for item in due
                    },
                    "snapshot_hash": snapshot_hash,
                },
            )
        except Exception as exc:
            result = {
                "evolution_id": None,
                "status": "CANDIDATE_FAILED",
                "path": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        completed_at = self.now_fn()
        self._complete_contexts(due, result, completed_at)
        status = str(result.get("status") or "UNKNOWN")
        fields = {
            "evolution_id": result.get("evolution_id"),
            "status": status,
            "selected_id": result.get("selected_id"),
            "artifact_path": result.get("path"),
        }
        if status in TERMINAL_SUCCESS:
            self._notify("evolution_completed", fields)
            return {"action": "evolved", "result": result}
        if status in TERMINAL_NO_IMPROVEMENT:
            self._notify("evolution_completed", fields)
            return {"action": "no_improvement", "result": result}
        if status == "ROLLBACK_FAILED":
            self._notify("manual_recovery_required", fields)
            return {"action": "blocked_manual_recovery", "result": result}
        self._notify("evolution_failed", fields)
        return {"action": "evolution_failed", "result": result}

    def _scheduler_error(self, exc: Exception, now: datetime) -> tuple[dict[str, Any], int]:
        existing = self.memory.get_watch_state(self.scheduler_context_key)
        attempts = int((existing or {}).get("attempt_count") or 0) + 1
        delay = min(
            self.policy.watch_backoff_cap_seconds,
            self.policy.watch_backoff_base_seconds * (2 ** (attempts - 1)),
        )
        next_attempt = _iso(now + timedelta(seconds=delay))
        self.memory.record_watch_state(
            context_key=self.scheduler_context_key,
            service_name=self.service_name,
            manifest_id="__registry_scheduler__",
            manifest_hash="unavailable",
            environment_fingerprint="unknown",
            policy_hash=self.policy.evaluation_policy_hash,
            snapshot_hash=None,
            status="SCHEDULER_ERROR",
            attempt_count=attempts,
            next_attempt_at=next_attempt,
            error_class=type(exc).__name__,
            last_error=str(exc),
            now=_iso(now),
        )
        self._notify(
            "evolution_failed",
            {
                "status": "SCHEDULER_ERROR",
                "evolution_id": "-",
                "artifact_path": "-",
            },
        )
        return (
            {
                "action": "scheduler_error",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "next_attempt_at": next_attempt,
            },
            delay,
        )

    def watch(self, max_cycles: int | None = None) -> dict[str, Any]:
        cycles = 0
        actions: dict[str, int] = {}
        while max_cycles is None or cycles < max_cycles:
            sleep_seconds = self.poll_interval_seconds
            scheduler_state = self.memory.get_watch_state(self.scheduler_context_key)
            scheduler_due = _parse_iso(
                (scheduler_state or {}).get("next_attempt_at")
            )
            now = self.now_fn()
            if (
                scheduler_state
                and scheduler_state.get("status") == "SCHEDULER_ERROR"
                and scheduler_due is not None
                and now.astimezone(UTC) < scheduler_due
            ):
                remaining = max(
                    1,
                    int((scheduler_due - now.astimezone(UTC)).total_seconds()),
                )
                outcome = {
                    "action": "scheduler_backoff",
                    "next_attempt_at": scheduler_due.isoformat(),
                }
                sleep_seconds = remaining
            else:
                try:
                    outcome = self.run_cycle()
                    scheduler_state = self.memory.get_watch_state(
                        self.scheduler_context_key
                    )
                    if scheduler_state and scheduler_state.get("status") == "SCHEDULER_ERROR":
                        self.memory.record_watch_state(
                            context_key=self.scheduler_context_key,
                            service_name=self.service_name,
                            manifest_id="__registry_scheduler__",
                            manifest_hash="available",
                            environment_fingerprint="unknown",
                            policy_hash=self.policy.evaluation_policy_hash,
                            snapshot_hash=None,
                            status="HEALTHY",
                            attempt_count=0,
                            now=_iso(self.now_fn()),
                        )
                except Exception as exc:
                    outcome, sleep_seconds = self._scheduler_error(exc, self.now_fn())
            cycles += 1
            action = str(outcome.get("action") or "unknown")
            actions[action] = actions.get(action, 0) + 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            self.sleep_fn(sleep_seconds)
        return {"cycles": cycles, "actions": actions}


def render_launchd_plist(
    *,
    workspace_root: str | Path,
    python_executable: str | Path,
    output_path: str | Path,
    policy_path: str,
    service_name: str,
    poll_interval_seconds: int,
    label: str = "cn.workbuddy.infra-team.watch",
) -> dict[str, Any]:
    """Render but never install a launchd agent definition."""
    root = Path(workspace_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    logs = root / ".infra-team" / "scheduler"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": label,
        "ProgramArguments": [
            str(Path(python_executable)),
            "-m",
            "infra_team.cli",
            "--root",
            str(root),
            "evolve",
            "--watch",
            "--service-name",
            service_name,
            "--policy",
            policy_path,
            "--poll-interval",
            str(int(poll_interval_seconds)),
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": max(10, min(int(poll_interval_seconds), 3600)),
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "launchd.stdout.log"),
        "StandardErrorPath": str(logs / "launchd.stderr.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True))
    os.replace(temporary, output)
    return {"path": str(output), "label": label, "installed": False, "plist": payload}
