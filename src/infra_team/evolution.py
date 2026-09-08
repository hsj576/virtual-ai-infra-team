"""Policy-bounded self-evolution outer loop.

P2 deliberately reuses the existing Supervisor execution inner loop. This
module owns discovery, deterministic preflight, candidate preparation, memory
prior lookup, evolution events and experience updates.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from . import probe_macos
from .candidate_registry import CandidateRegistry, RegistryError, ResolvedCandidate
from .candidate_store import CandidateStore, CandidateStoreError
from .compatibility import AutonomyPolicy
from .policy import TARGET_MODEL
from .recipe_memory import RecipeMemory
from .service_manager import ManagedService, ServiceError, ServiceSpec
from .supervisor import Supervisor


class EvolutionState(str, Enum):
    IDLE = "IDLE"
    TRIGGERED = "TRIGGERED"
    DISCOVERING = "DISCOVERING"
    PREFLIGHTING = "PREFLIGHTING"
    PREPARING = "PREPARING"
    READY = "READY"
    PLANNING = "PLANNING"
    PLAN_FROZEN = "PLAN_FROZEN"
    MAINTENANCE = "MAINTENANCE"
    EXPERIMENTING = "EXPERIMENTING"
    SELECTING = "SELECTING"
    PROMOTING = "PROMOTING"
    ONLINE_VERIFYING = "ONLINE_VERIFYING"
    REMEMBERING = "REMEMBERING"
    COMPLETED = "COMPLETED"
    DISCOVERY_REJECTED = "DISCOVERY_REJECTED"
    PREPARE_FAILED = "PREPARE_FAILED"
    PLAN_REJECTED = "PLAN_REJECTED"
    BASELINE_FAILED = "BASELINE_FAILED"
    CANDIDATE_FAILED = "CANDIDATE_FAILED"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    PROMOTION_FAILED = "PROMOTION_FAILED"
    BASELINE_RESTORED = "BASELINE_RESTORED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"


FINAL_STATES = {
    EvolutionState.COMPLETED,
    EvolutionState.DISCOVERY_REJECTED,
    EvolutionState.PREPARE_FAILED,
    EvolutionState.PLAN_REJECTED,
    EvolutionState.BASELINE_FAILED,
    EvolutionState.CANDIDATE_FAILED,
    EvolutionState.NO_IMPROVEMENT,
    EvolutionState.PROMOTION_FAILED,
    EvolutionState.BASELINE_RESTORED,
    EvolutionState.ROLLBACK_FAILED,
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


@dataclass
class EvolutionArtifacts:
    root: Path
    evolution_id: str

    @property
    def path(self) -> Path:
        return self.root / ".infra-team" / "evolution" / self.evolution_id

    @property
    def events_path(self) -> Path:
        return self.path / "events.ndjson"

    def ensure(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        self.ensure()
        target = self.path / name
        _atomic_json(target, payload)
        return target

    def emit(
        self,
        state: EvolutionState,
        message: str,
        **fields: Any,
    ) -> dict[str, Any]:
        self.ensure()
        event = {
            "time": _now(),
            "state": state.value,
            "message": message,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return event


SupervisorFactory = Callable[..., Any]
ProbeFunction = Callable[[], dict[str, Any]]
Reporter = Callable[[str, dict[str, Any]], None]


class EvolutionEngine:
    """One autonomous, auditable discovery-to-memory run."""

    def __init__(
        self,
        root: str | Path,
        service_name: str = "default",
        policy: AutonomyPolicy | None = None,
        registry: CandidateRegistry | None = None,
        store: CandidateStore | None = None,
        memory: RecipeMemory | None = None,
        service_manager: ManagedService | None = None,
        probe_fn: ProbeFunction | None = None,
        supervisor_factory: SupervisorFactory | None = None,
        report: Reporter | None = None,
        repeats: int = 3,
        candidate_timeout: int = 3600,
        use_planner: bool = True,
        target_planner_endpoint: str | None = None,
        disable_target_planner: bool = False,
        local_fallback_model: str | None = None,
        external_planner_base_url: str | None = None,
        external_planner_model: str | None = None,
        external_planner_api_key: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.service_name = service_name
        self.policy = policy or AutonomyPolicy()
        self.registry = registry or CandidateRegistry.builtin()
        self.store = store or CandidateStore(self.root)
        self.memory = memory or RecipeMemory(self.root)
        self.service_manager = service_manager or ManagedService(
            str(self.root), service_name
        )
        self.probe_fn = probe_fn
        self.supervisor_factory = supervisor_factory
        self.report = report or (lambda event, data: None)
        self.repeats = repeats
        self.candidate_timeout = min(
            int(candidate_timeout), int(self.policy.max_experiment_minutes) * 60
        )
        self.use_planner = use_planner
        self.target_planner_endpoint = target_planner_endpoint
        self.disable_target_planner = disable_target_planner
        self.local_fallback_model = local_fallback_model
        self.external_planner_base_url = external_planner_base_url
        self.external_planner_model = external_planner_model
        self.external_planner_api_key = external_planner_api_key
        # Share the same workspace lock as manual optimize and Dashboard. One
        # maintenance-capable workflow may own the workspace at a time.
        self.lock_path = self.root / ".infra-team" / "optimize.lock"

    def _emit(
        self,
        artifacts: EvolutionArtifacts,
        state: EvolutionState,
        message: str,
        **fields: Any,
    ) -> dict[str, Any]:
        event = artifacts.emit(state, message, **fields)
        self.report("evolution_state", event)
        return event

    def _new_artifacts(self) -> EvolutionArtifacts:
        evolution_id = (
            time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        )
        artifacts = EvolutionArtifacts(self.root, evolution_id)
        artifacts.ensure()
        return artifacts

    def _probe(self, manifests) -> dict[str, Any]:
        if self.probe_fn is not None:
            return self.probe_fn()
        models = [TARGET_MODEL, *(manifest.source.repo_id for manifest in manifests)]
        return probe_macos.probe(models).to_dict()

    @staticmethod
    def _recipe_from_status(status: dict[str, Any]) -> dict[str, Any]:
        return {
            "service_name": status.get("name"),
            "status": status.get("status"),
            "healthy": bool(status.get("healthy")),
            "base_url": status.get("base_url"),
            "spec": status.get("spec") or {},
            "source": "running_service_snapshot",
            "captured_at": _now(),
        }

    @staticmethod
    def _recipe_matches_status(
        recipe: dict[str, Any] | None, status: dict[str, Any]
    ) -> bool:
        if not isinstance(recipe, dict):
            return False
        recipe_spec = recipe.get("service_spec") or recipe.get("spec")
        status_spec = status.get("spec")
        if not isinstance(recipe_spec, dict) or not isinstance(status_spec, dict):
            return False
        try:
            return ServiceSpec(**recipe_spec) == ServiceSpec(**status_spec)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _prepare_representatives(
        resolved: dict[str, ResolvedCandidate],
    ) -> list[ResolvedCandidate]:
        by_manifest: dict[str, list[ResolvedCandidate]] = {}
        for candidate in resolved.values():
            if not candidate.requires_prepare:
                continue
            by_manifest.setdefault(candidate.manifest_id, []).append(candidate)
        representatives = []
        for candidates in by_manifest.values():
            representatives.append(
                next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.draft_block_size is None
                    ),
                    candidates[0],
                )
            )
        return sorted(representatives, key=lambda item: item.manifest_id)

    @staticmethod
    def _failure_state(current: EvolutionState) -> EvolutionState:
        if current in {EvolutionState.TRIGGERED, EvolutionState.READY}:
            return EvolutionState.BASELINE_FAILED
        if current in {EvolutionState.DISCOVERING, EvolutionState.PREFLIGHTING}:
            return EvolutionState.DISCOVERY_REJECTED
        if current == EvolutionState.PREPARING:
            return EvolutionState.PREPARE_FAILED
        if current == EvolutionState.PLANNING:
            return EvolutionState.PLAN_REJECTED
        if current == EvolutionState.PROMOTING:
            return EvolutionState.PROMOTION_FAILED
        if current == EvolutionState.ONLINE_VERIFYING:
            return EvolutionState.BASELINE_RESTORED
        return EvolutionState.CANDIDATE_FAILED

    def _supervisor_reporter(
        self, artifacts: EvolutionArtifacts, state_holder: dict[str, EvolutionState]
    ) -> Reporter:
        mapping = {
            "plan_start": (EvolutionState.PLANNING, "Target is generating an ExperimentPlan"),
            "plan_done": (EvolutionState.PLAN_FROZEN, "ExperimentPlan frozen before maintenance"),
            "service_switch_start": (EvolutionState.MAINTENANCE, "Entering controlled maintenance window"),
            "candidate_start": (EvolutionState.EXPERIMENTING, "Running candidate experiment"),
            "verdict": (EvolutionState.SELECTING, "Independent selector produced a verdict"),
            "promotion_start": (EvolutionState.PROMOTING, "Starting selected recipe at stable API"),
            "online_verify_start": (EvolutionState.ONLINE_VERIFYING, "Running post-restart online quality gate"),
        }

        def reporter(event: str, data: dict[str, Any]) -> None:
            if event in mapping:
                state, message = mapping[event]
                state_holder["state"] = state
                self._emit(artifacts, state, message, supervisor_event=event, detail=data)
            if event == "service_switch_done":
                status = data.get("status")
                if status == "baseline_restored":
                    state_holder["state"] = EvolutionState.BASELINE_RESTORED
                    self._emit(artifacts, 
                        EvolutionState.BASELINE_RESTORED,
                        "Selected recipe failed; baseline restored",
                        detail=data,
                    )
                elif status == "serving_selected":
                    self._emit(artifacts, 
                        EvolutionState.ONLINE_VERIFYING,
                        "Selected recipe is serving and online verification passed",
                        detail=data,
                    )
            elif event == "service_switch_failed":
                state_holder["state"] = EvolutionState.ROLLBACK_FAILED
                self._emit(artifacts, 
                    EvolutionState.ROLLBACK_FAILED,
                    "Selected recipe and rollback both failed",
                    detail=data,
                )
            self.report(event, data)

        return reporter

    def _make_supervisor(
        self,
        resolved: dict[str, ResolvedCandidate],
        environment: dict[str, Any],
        memory_prior: dict[str, Any],
        trigger: dict[str, Any],
        registry: CandidateRegistry,
        report: Reporter,
    ):
        target_endpoint = None if self.disable_target_planner else self.target_planner_endpoint
        if target_endpoint is None and not self.disable_target_planner:
            service = self.service_manager.status()
            if service.get("base_url"):
                target_endpoint = str(service["base_url"]).rstrip("/") + "/v1"
        kwargs = {
            "root": str(self.root),
            "repeats": self.repeats,
            "use_planner": self.use_planner,
            "report": report,
            "candidate_timeout": self.candidate_timeout,
            "managed_service_name": self.service_name,
            "resolved_candidates": resolved,
            "autonomy_policy": self.policy,
            "registry": registry,
            "environment_snapshot": environment,
            "target_planner_endpoint": target_endpoint,
            "local_fallback_model": self.local_fallback_model,
            "external_planner_base_url": self.external_planner_base_url,
            "external_planner_model": self.external_planner_model,
            "external_planner_api_key": self.external_planner_api_key,
            "optimization_lock_held": True,
            "allow_promotion": self.policy.auto_promote_acceleration_plugin,
            "evolution_context": {
                "trigger": trigger,
                "registry_snapshot": registry.snapshot(),
                "manifest_hashes": {
                    candidate.manifest_id: candidate.manifest_hash
                    for candidate in resolved.values()
                    if candidate.id != "baseline"
                },
                "resolved_revisions": {
                    candidate.manifest_id: candidate.source_revision
                    for candidate in resolved.values()
                    if candidate.source_revision
                },
                "local_candidate_paths": {
                    candidate.id: candidate.local_model_path
                    for candidate in resolved.values()
                    if candidate.local_model_path
                },
                "memory_prior": memory_prior,
                "autonomy_policy": self.policy.to_dict(),
                "autonomy_policy_hash": self.policy.policy_hash,
            },
        }
        if self.supervisor_factory is not None:
            return self.supervisor_factory(**kwargs)
        return Supervisor(**kwargs)

    def _latest_incomplete(self) -> tuple[EvolutionArtifacts, EvolutionState] | None:
        directory = self.root / ".infra-team" / "evolution"
        if not directory.is_dir():
            return None
        for path in sorted(
            (item for item in directory.iterdir() if item.is_dir()),
            key=lambda item: item.name,
            reverse=True,
        ):
            events_path = path / "events.ndjson"
            try:
                lines = [line for line in events_path.read_text().splitlines() if line]
                if not lines:
                    continue
                last = json.loads(lines[-1])
                state = EvolutionState(last["state"])
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
            if state not in FINAL_STATES:
                return EvolutionArtifacts(self.root, path.name), state
        return None

    def _recover_incomplete(
        self, artifacts: EvolutionArtifacts, state: EvolutionState
    ) -> dict[str, Any] | None:
        before_path = artifacts.path / "active_recipe_before.json"
        # Before a plan is frozen, no service mutation is authorized. Close the
        # stale attempt and safely begin a new idempotent discovery cycle.
        before_freeze = {
            EvolutionState.TRIGGERED,
            EvolutionState.DISCOVERING,
            EvolutionState.PREFLIGHTING,
            EvolutionState.PREPARING,
            EvolutionState.READY,
            EvolutionState.PLANNING,
        }
        if state in before_freeze:
            final = EvolutionState.DISCOVERY_REJECTED
            self._emit(
                artifacts,
                final,
                "Interrupted before PLAN_FROZEN; closing stale attempt before retry",
                recovered_from=state.value,
            )
            trigger = {"source": "recovery", "recovered_from": state.value}
            self.memory.record_evolution(
                evolution_id=artifacts.evolution_id,
                status=final.value,
                trigger=trigger,
                artifact_path=str(artifacts.path),
                error="interrupted before plan freeze",
            )
            return None

        # At or after PLAN_FROZEN, never ask a planner again. Recover the exact
        # pre-run service recipe and stop; a later explicit evolve --once may
        # begin a fresh cycle after the stable API is healthy.
        try:
            before = json.loads(before_path.read_text(encoding="utf-8"))
            spec_raw = before.get("service_spec") or before.get("spec") or {}
            if not spec_raw:
                raise ServiceError("active_recipe_before.json has no service spec")
            status = self.service_manager.status()
            desired_spec = ServiceSpec(**spec_raw)
            try:
                running_spec = ServiceSpec(**(status.get("spec") or {}))
            except (TypeError, ValueError):
                running_spec = None
            if not status.get("healthy") or running_spec != desired_spec:
                if status.get("managed"):
                    self.service_manager.stop(timeout=30.0)
                self.service_manager.start(desired_spec, wait_timeout=900.0)
            final = EvolutionState.BASELINE_RESTORED
            error = None
            message = "Interrupted frozen run recovered by restoring previous recipe"
        except (OSError, json.JSONDecodeError, TypeError, ServiceError) as exc:
            final = EvolutionState.ROLLBACK_FAILED
            error = str(exc)
            message = "Interrupted frozen run could not restore previous recipe"
        self._emit(
            artifacts,
            final,
            message,
            recovered_from=state.value,
            error=error,
        )
        result = {
            "evolution_id": artifacts.evolution_id,
            "status": final.value,
            "path": str(artifacts.path),
            "recovered_from": state.value,
            "error": error,
        }
        artifacts.write_json("summary.json", result)
        self.memory.record_evolution(
            evolution_id=artifacts.evolution_id,
            status=final.value,
            trigger={"source": "recovery", "recovered_from": state.value},
            artifact_path=str(artifacts.path),
            error=error,
        )
        return result

    def run_once(
        self,
        trigger_source: str = "cli_once",
        trigger_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if trigger_source not in {"cli_once", "watch"}:
            raise ValueError(f"unsupported evolution trigger source: {trigger_source}")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another evolution is already running in this workspace"
                ) from exc
            try:
                incomplete = self._latest_incomplete()
                if incomplete is not None:
                    recovered = self._recover_incomplete(*incomplete)
                    if recovered is not None:
                        return recovered
                return self._run_once_locked(
                    trigger_source=trigger_source,
                    trigger_context=trigger_context or {},
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _run_once_locked(
        self,
        *,
        trigger_source: str,
        trigger_context: dict[str, Any],
    ) -> dict[str, Any]:
        artifacts = self._new_artifacts()
        started_at = _now()
        trigger = {
            "schema_version": 1,
            "source": trigger_source,
            "requested_at": started_at,
            "service_name": self.service_name,
            "policy_hash": self.policy.policy_hash,
            "autonomy_level": "L3",
            "context": trigger_context,
        }
        artifacts.write_json("trigger.json", trigger)
        current = EvolutionState.TRIGGERED
        self._emit(artifacts, current, "One-shot evolution requested")
        supervisor_run_id = None
        supervisor_state_holder: dict[str, EvolutionState] | None = None

        try:
            before_status = self.service_manager.status()
            stored_recipe = (
                self.service_manager.active_recipe()
                if hasattr(self.service_manager, "active_recipe")
                else None
            )
            before_recipe = (
                stored_recipe
                if self._recipe_matches_status(stored_recipe, before_status)
                else self._recipe_from_status(before_status)
            )
            artifacts.write_json("active_recipe_before.json", before_recipe)
            if not before_status.get("healthy"):
                raise RuntimeError(
                    "managed baseline service must be healthy before autonomous preparation"
                )

            current = EvolutionState.DISCOVERING
            self._emit(artifacts, current, "Reading trusted local candidate registry")
            manifests = self.registry.load()
            selected_manifest_ids = set(trigger_context.get("manifest_ids") or ())
            if selected_manifest_ids:
                manifests = [
                    manifest
                    for manifest in manifests
                    if manifest.id in selected_manifest_ids
                ]
                missing = selected_manifest_ids - {manifest.id for manifest in manifests}
                if missing:
                    raise RegistryError(
                        "watch trigger references missing manifest(s): "
                        + ", ".join(sorted(missing))
                    )
            run_registry = CandidateRegistry.from_manifests(
                manifests,
                registry_name=self.registry.registry_name,
            )
            registry_snapshot = run_registry.snapshot()
            artifacts.write_json("registry_snapshot.json", registry_snapshot)
            artifacts.write_json(
                "candidate_manifest.json",
                {
                    "manifests": [manifest.to_dict() for manifest in manifests],
                    "hashes": {
                        manifest.id: manifest.manifest_hash for manifest in manifests
                    },
                },
            )

            current = EvolutionState.PREFLIGHTING
            self._emit(artifacts, current, "Checking trust, compatibility and resource policy")
            environment = self._probe(manifests)
            resolved = run_registry.resolve_all(
                policy=self.policy,
                environment=environment,
            )
            artifacts.write_json(
                "candidate_resolution.json",
                {
                    "policy_hash": self.policy.policy_hash,
                    "environment": environment,
                    "candidates": {
                        key: value.to_dict() for key, value in resolved.items()
                    },
                },
            )

            if not self.policy.auto_prepare:
                current = EvolutionState.DISCOVERY_REJECTED
                raise RuntimeError("autonomy policy does not allow automatic preparation")
            if not self.policy.auto_experiment:
                current = EvolutionState.DISCOVERY_REJECTED
                raise RuntimeError("autonomy policy does not allow automatic experiments")

            current = EvolutionState.PREPARING
            self._emit(
                artifacts,
                current,
                "Preparing pinned candidate while baseline remains online",
            )
            prepared_records = []
            prepared_by_manifest = {}
            for candidate in self._prepare_representatives(resolved):
                if not self.service_manager.status().get("healthy"):
                    current = EvolutionState.BASELINE_FAILED
                    raise RuntimeError("baseline became unhealthy during candidate preparation")
                prepared = self.store.prepare(candidate, self.policy)
                prepared_by_manifest[candidate.manifest_id] = prepared
                prepared_records.append(prepared.to_dict())
                self._emit(artifacts, 
                    EvolutionState.PREPARING,
                    "Candidate preparation complete",
                    candidate_id=candidate.id,
                    manifest_id=candidate.manifest_id,
                    progress=1.0,
                    reused=prepared.reused,
                )
            if not self.service_manager.status().get("healthy"):
                current = EvolutionState.BASELINE_FAILED
                raise RuntimeError("baseline is unhealthy after candidate preparation")
            resolved = {
                candidate_id: (
                    prepared_by_manifest[candidate.manifest_id].apply(candidate)
                    if candidate.manifest_id in prepared_by_manifest
                    else candidate
                )
                for candidate_id, candidate in resolved.items()
            }
            not_ready = [
                candidate.id
                for candidate in resolved.values()
                if candidate.requires_prepare and not candidate.local_model_path
            ]
            if not_ready:
                raise CandidateStoreError(
                    "candidate store did not produce READY local paths: "
                    + ", ".join(not_ready)
                )
            artifacts.write_json(
                "candidate_prepare.json",
                {
                    "prepared": prepared_records,
                    "baseline_healthy_during_prepare": True,
                },
            )
            artifacts.write_json(
                "candidate_resolution.json",
                {
                    "policy_hash": self.policy.policy_hash,
                    "environment": environment,
                    "candidates": {
                        key: value.to_dict() for key, value in resolved.items()
                    },
                    "all_candidates_ready": True,
                },
            )

            target_revision = next(
                (
                    candidate.target_revision
                    for candidate in resolved.values()
                    if candidate.target_revision
                ),
                None,
            )
            memory_prior = self.memory.query_prior(
                environment,
                target_model=TARGET_MODEL,
                target_revision=target_revision,
                resolved_candidates=resolved,
            )
            artifacts.write_json("memory_prior.json", memory_prior)
            current = EvolutionState.READY
            self._emit(artifacts, 
                current,
                "Candidates are verified and ready; local memory prior loaded",
                candidate_count=len(resolved) - 1,
            )

            supervisor_state_holder = {"state": current}
            supervisor = self._make_supervisor(
                resolved,
                environment,
                memory_prior,
                trigger,
                run_registry,
                self._supervisor_reporter(artifacts, supervisor_state_holder),
            )
            if hasattr(supervisor, "art"):
                artifacts.write_json(
                    "supervisor_link.json",
                    {
                        "run_id": supervisor.art.run_id,
                        "path": supervisor.art.path,
                        "plan_artifact": str(
                            Path(supervisor.art.path) / "agent_plan.json"
                        ),
                    },
                )
            summary = supervisor.run()
            current = supervisor_state_holder["state"]
            supervisor_run_id = summary.get("run_id")
            after_status = self.service_manager.status()
            after_recipe = (
                self.service_manager.active_recipe()
                if hasattr(self.service_manager, "active_recipe")
                else None
            ) or self._recipe_from_status(after_status)
            artifacts.write_json("active_recipe_after.json", after_recipe)

            current = EvolutionState.REMEMBERING
            self._emit(artifacts, current, "Writing success and failure evidence to Recipe Memory")
            experience = self.memory.record_supervisor_summary(
                summary,
                resolved_candidates=resolved,
                evolution_id=artifacts.evolution_id,
                previous_recipe=before_recipe,
                selected_recipe=after_recipe,
            )
            artifacts.write_json("experience_update.json", experience)

            transition = summary.get("service_transition") or {}
            verdict = summary.get("verdict") or {}
            baseline = summary.get("baseline") or {}
            if not baseline.get("ok", False):
                final = EvolutionState.BASELINE_FAILED
            elif transition.get("status") == "rollback_failed":
                final = EvolutionState.ROLLBACK_FAILED
            elif transition.get("status") == "baseline_restored":
                final = EvolutionState.BASELINE_RESTORED
            elif verdict.get("accepted"):
                final = EvolutionState.COMPLETED
            else:
                final = EvolutionState.NO_IMPROVEMENT
            self._emit(artifacts, 
                final,
                (
                    "Evolution completed and experience saved"
                    if final == EvolutionState.COMPLETED
                    else "Evolution produced no promotable improvement; evidence saved"
                ),
                supervisor_run_id=supervisor_run_id,
            )
            result = {
                "evolution_id": artifacts.evolution_id,
                "status": final.value,
                "path": str(artifacts.path),
                "supervisor_run_id": supervisor_run_id,
                "supervisor_path": summary.get("path"),
                "selected_id": verdict.get("selected_id"),
                "accepted": bool(verdict.get("accepted")),
                "speedup_percent": verdict.get("speedup_percent"),
                "memory_update": experience,
            }
            artifacts.write_json("summary.json", result)
            self.memory.record_evolution(
                evolution_id=artifacts.evolution_id,
                status=final.value,
                trigger=trigger,
                artifact_path=str(artifacts.path),
                supervisor_run_id=supervisor_run_id,
                started_at=started_at,
            )
            return result
        except Exception as exc:
            if supervisor_state_holder is not None:
                current = supervisor_state_holder["state"]
            final = current if current in FINAL_STATES else self._failure_state(current)
            self._emit(artifacts, final, str(exc), error_type=type(exc).__name__)
            result = {
                "evolution_id": artifacts.evolution_id,
                "status": final.value,
                "path": str(artifacts.path),
                "supervisor_run_id": supervisor_run_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            artifacts.write_json("summary.json", result)
            try:
                self.memory.record_evolution(
                    evolution_id=artifacts.evolution_id,
                    status=final.value,
                    trigger=trigger,
                    artifact_path=str(artifacts.path),
                    supervisor_run_id=supervisor_run_id,
                    error=str(exc),
                    started_at=started_at,
                )
            except Exception as memory_exc:
                self._emit(artifacts, 
                    final,
                    "Evolution failed and Recipe Memory update also failed",
                    original_error=str(exc),
                    memory_error=str(memory_exc),
                )
            return result
