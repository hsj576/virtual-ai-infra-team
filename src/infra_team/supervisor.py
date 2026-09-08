"""Supervisor: the always-alive orchestrator.

Owns the whole loop:

    probe -> baseline -> ask the model for a plan -> validate the plan
          -> run candidates -> verify -> select -> persist -> rollback record

Crucially this process never loads a model itself. Every model load happens
in a short-lived child process, so the supervisor stays alive even when a
27B candidate is being torn down or fails outright.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from typing import Any, Callable

from . import artifacts as artifacts_mod
from . import probe_macos
from .candidate_registry import CandidateRegistry, ResolvedCandidate
from .candidate_store import CandidateStore
from .compatibility import AutonomyPolicy
from .planner import build_context, request_plan
from .policy import (
    TARGET_MODEL,
    PolicyError,
    default_plan,
    resolve_acceptance_policy,
    validate_plan,
)
from .runner import CandidateResult, CandidateSpec, run_candidate
from .selector import AcceptancePolicy, select
from .service_benchmark import (
    ServiceBenchmarkError,
    benchmark_service,
    quality_gate_service,
)
from .service_manager import ManagedService, ServiceError, ServiceSpec

Reporter = Callable[[str, dict[str, Any]], None]


def _noop(event: str, data: dict[str, Any]) -> None:  # pragma: no cover
    pass


class Supervisor:
    """Runs one full optimization task and leaves an audit trail."""

    def __init__(
        self,
        root: str,
        python_executable: str | None = None,
        repeats: int = 3,
        use_planner: bool = True,
        report: Reporter | None = None,
        candidate_timeout: int = 3600,
        target_planner_endpoint: str | None = "http://127.0.0.1:8000/v1",
        local_fallback_model: str | None = None,
        external_planner_base_url: str | None = None,
        external_planner_model: str | None = None,
        external_planner_api_key: str | None = None,
        managed_service_name: str | None = "default",
        resolved_candidates: dict[str, ResolvedCandidate] | None = None,
        autonomy_policy: AutonomyPolicy | None = None,
        registry: CandidateRegistry | None = None,
        environment_snapshot: dict[str, Any] | None = None,
        evolution_context: dict[str, Any] | None = None,
        optimization_lock_held: bool = False,
        allow_promotion: bool = True,
    ) -> None:
        self.root = root
        self.python_executable = python_executable
        self.repeats = repeats
        self.use_planner = use_planner
        self.report = report or _noop
        self.candidate_timeout = candidate_timeout
        self.target_planner_endpoint = target_planner_endpoint
        self.local_fallback_model = local_fallback_model
        self.external_planner_base_url = external_planner_base_url
        self.external_planner_model = external_planner_model
        self.external_planner_api_key = external_planner_api_key
        self.managed_service_name = managed_service_name
        self.autonomy_policy = autonomy_policy or AutonomyPolicy()
        self.registry = registry or CandidateRegistry.builtin()
        self.resolved_candidates = dict(resolved_candidates or {})
        self.environment_snapshot = dict(environment_snapshot or {})
        self.evolution_context = dict(evolution_context or {})
        self.optimization_lock_held = optimization_lock_held
        self.allow_promotion = allow_promotion
        self.service_manager = (
            ManagedService(root, managed_service_name, python_executable)
            if managed_service_name
            else None
        )
        self.service_state: dict[str, Any] | None = None
        self.original_service_spec: ServiceSpec | None = None
        self.art = artifacts_mod.new_run(root)

    # -- helpers ---------------------------------------------------------
    def _log(self, text: str) -> None:
        self.art.append_text(
            "transcript.log", f"[{time.strftime('%H:%M:%S')}] {text}\n"
        )

    # -- phases ----------------------------------------------------------
    def probe(self) -> dict[str, Any]:
        self.report("probe_start", {})
        model_ids = {TARGET_MODEL}
        if self.resolved_candidates:
            model_ids.update(
                candidate.source_repo_id
                for candidate in self.resolved_candidates.values()
                if candidate.source_repo_id
            )
        else:
            model_ids.update(manifest.source.repo_id for manifest in self.registry.load())
        env = (
            dict(self.environment_snapshot)
            if self.environment_snapshot
            else probe_macos.probe(sorted(model_ids)).to_dict()
        )
        if not self.resolved_candidates:
            self.resolved_candidates = self.registry.resolve_all(
                policy=self.autonomy_policy,
                environment=env,
            )
        self.resolved_candidates = CandidateStore(self.root).apply_ready_candidates(
            self.resolved_candidates
        )
        self.art.write_json("environment.json", env)
        self.art.write_json("registry_snapshot.json", self.registry.snapshot())
        self._log(f"environment probed: {env['chip']}, {env['unified_memory_gb']}GB")
        self.report("probe_done", env)
        return env

    def measure_baseline(self) -> CandidateResult:
        # Product-shaped path: if a managed service is already healthy, use
        # its public API as the baseline. This avoids loading a second 27B
        # target merely to measure the first one.
        if self.service_manager:
            state = self.service_manager.status()
            self.service_state = state
            if state.get("healthy") and isinstance(state.get("spec"), dict):
                service_spec = ServiceSpec(**state["spec"])
                self.original_service_spec = service_spec
                self.target_planner_endpoint = service_spec.base_url + "/v1"
                self.report(
                    "baseline_start",
                    {"mode": "service", "base_url": service_spec.base_url + "/v1"},
                )
                self._log(
                    f"measuring managed service baseline at {service_spec.base_url}/v1"
                )
                result = benchmark_service(
                    service_spec.base_url,
                    service_spec.model,
                    repeats=self.repeats,
                    timeout=self.candidate_timeout,
                )
                # Preserve the exact serving recipe for rollback.
                result.spec = CandidateSpec(
                    id="baseline",
                    target_model=service_spec.model,
                    draft_model=service_spec.draft_model,
                    draft_kind=service_spec.draft_kind,
                    draft_block_size=service_spec.draft_block_size,
                    enable_thinking=service_spec.enable_thinking,
                    repeats=self.repeats,
                    target_revision=service_spec.target_revision,
                    candidate_manifest_id=service_spec.candidate_manifest_id,
                    candidate_manifest_hash=service_spec.candidate_manifest_hash,
                    runtime_version=service_spec.runtime_version,
                ).to_dict()
                self.art.write_json("baseline.json", result.to_dict())
                self._log(
                    f"service baseline: ok={result.ok} tps={result.generation_tps} "
                    f"mem={result.peak_memory_gb} quality={result.quality_pass}"
                )
                self.report("baseline_done", result.to_dict())
                return result

        # Research/offline compatibility path when no service is deployed.
        self.report("baseline_start", {"mode": "offline"})
        spec = CandidateSpec(
            id="baseline",
            target_model=TARGET_MODEL,
            repeats=self.repeats,
        )
        self._log("running offline baseline candidate")
        result = run_candidate(
            spec,
            python_executable=self.python_executable,
            timeout=self.candidate_timeout,
            log_path=self.art.log_path(),
        )
        self.art.write_json("baseline.json", result.to_dict())
        self._log(
            f"baseline: ok={result.ok} tps={result.generation_tps} "
            f"mem={result.peak_memory_gb} quality={result.quality_pass}"
        )
        self.report("baseline_done", result.to_dict())
        return result

    def make_plan(
        self, env: dict[str, Any], baseline: CandidateResult
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        baseline_summary = {
            "generation_tps": baseline.generation_tps,
            "peak_memory_gb": baseline.peak_memory_gb,
            "quality_pass": baseline.quality_pass,
        }
        context = build_context(
            env,
            baseline_summary,
            resolved_candidates=self.resolved_candidates or None,
        )
        if self.evolution_context:
            context["evolution"] = self.evolution_context

        if not self.use_planner:
            plan = default_plan()
            meta = {"source": "default_rule_based_plan", "planner_disabled": True}
        else:
            self.report("plan_start", {})
            self._log("requesting experiment plan from the model")
            plan, meta = request_plan(
                context,
                target_model=TARGET_MODEL,
                python_executable=self.python_executable,
                log_path=self.art.log_path(),
                target_endpoint=self.target_planner_endpoint,
                local_fallback_model=self.local_fallback_model,
                external_base_url=self.external_planner_base_url,
                external_model=self.external_planner_model,
                external_api_key=self.external_planner_api_key,
            )

        # This is the handoff boundary: after this atomic artifact exists,
        # candidate execution and rollback no longer depend on any planner
        # remaining online.
        self.art.write_json(
            "agent_plan.json",
            {
                "status": "frozen_before_execution",
                "context": context,
                "plan": plan,
                "meta": meta,
            },
        )
        self._log(f"plan source: {meta.get('source')}")
        self.report("plan_done", {"plan": plan, "meta": meta})
        return plan, meta

    def enforce_policy(
        self, plan: dict[str, Any]
    ) -> tuple[list[CandidateSpec], dict[str, Any], AcceptancePolicy]:
        if not self.resolved_candidates:
            raise PolicyError("candidate snapshot is not initialized; probe must run first")

        fallback_used = False
        effective_plan = plan
        try:
            specs, decision = validate_plan(
                effective_plan,
                repeats=self.repeats,
                resolved_candidates=self.resolved_candidates,
            )
        except PolicyError as exc:
            self._log(f"plan refused by policy ({exc}); using default plan")
            fallback_used = True
            effective_plan = default_plan()
            specs, decision = validate_plan(
                effective_plan,
                repeats=self.repeats,
                resolved_candidates=self.resolved_candidates,
            )

        policy_values, policy_sources = resolve_acceptance_policy(
            effective_plan,
            self.autonomy_policy,
            self.resolved_candidates,
            decision.approved,
        )
        policy_obj = AcceptancePolicy(**policy_values)

        payload = decision.to_dict()
        payload["policy_fallback_used"] = fallback_used
        payload["acceptance_policy"] = policy_obj.to_dict()
        payload["acceptance_policy_sources"] = policy_sources
        payload["executable_specs"] = [s.to_dict() for s in specs]
        self.art.write_json("policy_decision.json", payload)
        self._log(
            f"policy approved {payload['approved']}, refused "
            f"{[r['id'] for r in payload['refused']]}"
        )
        self.report("policy_done", payload)
        return specs, payload, policy_obj

    def stop_service_for_experiments(self) -> bool:
        """Stop the user's managed service only after the plan is frozen."""
        if not self.service_manager or not self.original_service_spec:
            return False
        self.report(
            "service_switch_start",
            {
                "name": self.original_service_spec.name,
                "base_url": self.original_service_spec.base_url,
            },
        )
        self._log("plan frozen; entering maintenance window for candidate tests")
        stopped = self.service_manager.stop(timeout=30.0)
        self.art.write_json(
            "service_stop.json",
            {
                "status": "maintenance_window",
                "original_service": self.original_service_spec.to_dict(),
                "stop_result": stopped,
            },
        )
        return bool(stopped.get("stopped"))

    def restore_or_promote_service(
        self,
        verdict: dict[str, Any],
        baseline: CandidateResult,
        candidates: list[CandidateResult],
    ) -> dict[str, Any] | None:
        """Start the selected recipe at the same API address.

        If promotion fails, attempt the original baseline recipe. A failed
        deployment is never reported as an accepted optimization.
        """
        if not self.service_manager or not self.original_service_spec:
            return None

        by_id = {r.id: r for r in [baseline, *candidates]}
        selected = by_id.get(verdict.get("selected_id"), baseline)
        original = self.original_service_spec
        promotion_blocked = bool(
            selected.id != "baseline" and verdict.get("accepted") and not self.allow_promotion
        )
        if promotion_blocked:
            self._log(
                f"candidate {selected.id} qualified but autonomy policy forbids promotion"
            )
            verdict.update(
                selected_id="baseline",
                accepted=False,
                selected_tps=baseline.generation_tps,
                speedup_percent=0.0,
                reason=(
                    f"candidate {selected.id} qualified, but automatic promotion "
                    "is disabled by autonomy policy; baseline retained"
                ),
            )
            selected = baseline
        selected_spec = selected.spec or baseline.spec
        desired = ServiceSpec(
            name=original.name,
            model=selected_spec.get("target_model") or original.model,
            host=original.host,
            port=original.port,
            draft_model=selected_spec.get("draft_model"),
            draft_kind=selected_spec.get("draft_kind"),
            draft_block_size=selected_spec.get("draft_block_size"),
            max_tokens=original.max_tokens,
            enable_thinking=bool(selected_spec.get("enable_thinking", False)),
            target_revision=selected_spec.get("target_revision") or original.target_revision,
            candidate_manifest_id=selected_spec.get("candidate_manifest_id"),
            candidate_manifest_hash=selected_spec.get("candidate_manifest_hash"),
            runtime_version=selected_spec.get("runtime_version") or original.runtime_version,
        )
        transition: dict[str, Any] = {
            "status": "starting_selected",
            "selected_candidate": selected.id,
            "desired_service": desired.to_dict(),
            "rollback_service": original.to_dict(),
        }
        self.art.write_json("service_transition.json", transition)
        self.report(
            "promotion_start",
            {"candidate_id": selected.id, "desired_service": desired.to_dict()},
        )

        try:
            status = self.service_manager.start(desired, wait_timeout=900.0)
            self.report(
                "online_verify_start",
                {"candidate_id": selected.id, "base_url": desired.base_url},
            )
            online_quality = quality_gate_service(
                desired.base_url, desired.model, timeout=self.candidate_timeout
            )
            if not online_quality.get("quality_pass"):
                # Stop the unhealthy promoted instance before restoring baseline.
                self.service_manager.stop(timeout=30.0)
                raise ServiceError(
                    "promoted service failed the post-restart 4/4 quality gate"
                )
            transition.update(
                status=(
                    "baseline_retained_policy"
                    if promotion_blocked
                    else "serving_selected"
                ),
                service=status,
                post_restart_quality=online_quality,
            )
            if hasattr(self.service_manager, "atomically_promote"):
                recipe = self.service_manager.recipe_from_spec(
                    desired,
                    source=(
                        "evolution_policy_retained"
                        if promotion_blocked
                        else "supervisor_promotion"
                    ),
                    run_id=self.art.run_id,
                    candidate_id=selected.id,
                    online_quality=online_quality,
                )
                self.service_manager.atomically_promote(recipe)
                transition["active_recipe"] = recipe
            self.report("service_switch_done", transition)
        except (ServiceError, ServiceBenchmarkError) as exc:
            transition.update(
                status="selected_start_failed",
                selected_error=str(exc),
            )
            self._log(f"selected service failed to start: {exc}; restoring baseline")
            try:
                rollback_status = self.service_manager.start(
                    original, wait_timeout=900.0
                )
                if hasattr(self.service_manager, "atomically_promote"):
                    rollback_recipe = self.service_manager.recipe_from_spec(
                        original,
                        source="supervisor_rollback",
                        run_id=self.art.run_id,
                        candidate_id="baseline",
                    )
                    self.service_manager.atomically_promote(rollback_recipe)
                transition.update(
                    status="baseline_restored",
                    rollback_service_status=rollback_status,
                )
                verdict.update(
                    selected_id="baseline",
                    accepted=False,
                    selected_tps=baseline.generation_tps,
                    speedup_percent=0.0,
                    reason=(
                        "candidate passed offline gates but could not start as "
                        f"the managed service; baseline restored: {exc}"
                    ),
                )
                self.report("service_switch_done", transition)
            except ServiceError as rollback_exc:
                transition.update(
                    status="rollback_failed",
                    rollback_error=str(rollback_exc),
                )
                verdict.update(
                    accepted=False,
                    reason=(
                        "selected service and baseline rollback both failed; "
                        f"manual recovery required: {rollback_exc}"
                    ),
                )
                self.report("service_switch_failed", transition)

        self.art.write_json("service_transition.json", transition)
        self.art.write_json("verdict.json", verdict)
        return transition

    def run_experiments(
        self, specs: list[CandidateSpec], baseline: CandidateResult
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for spec in specs:
            if spec.id == "baseline":
                continue  # already measured; do not pay for it twice
            self.report("candidate_start", {"id": spec.id})
            self._log(f"running candidate {spec.id}")
            if self.service_manager and self.original_service_spec:
                # Fair comparison: baseline and candidates must use the same
                # MLX-VLM HTTP serving path. Comparing a continuous-batching
                # server baseline with direct `generate()` would misattribute
                # execution-path overhead to speculative decoding.
                original = self.original_service_spec
                service_spec = ServiceSpec(
                    name=original.name,
                    model=spec.target_model,
                    host=original.host,
                    port=original.port,
                    draft_model=spec.draft_model,
                    draft_kind=spec.draft_kind,
                    draft_block_size=spec.draft_block_size,
                    max_tokens=original.max_tokens,
                    enable_thinking=spec.enable_thinking,
                    target_revision=spec.target_revision or original.target_revision,
                    candidate_manifest_id=spec.candidate_manifest_id,
                    candidate_manifest_hash=spec.candidate_manifest_hash,
                    runtime_version=spec.runtime_version or original.runtime_version,
                )
                started = False
                try:
                    self.service_manager.start(service_spec, wait_timeout=900.0)
                    started = True
                    res = benchmark_service(
                        service_spec.base_url,
                        service_spec.model,
                        repeats=spec.repeats,
                        timeout=self.candidate_timeout,
                    )
                    res.id = spec.id
                    res.spec = spec.to_dict()
                except (ServiceError, ServiceBenchmarkError) as exc:
                    res = CandidateResult(
                        id=spec.id,
                        ok=False,
                        spec=spec.to_dict(),
                        error=f"candidate service failed: {exc}",
                    )
                finally:
                    if started:
                        try:
                            self.service_manager.stop(timeout=30.0)
                        except ServiceError as exc:
                            if "res" in locals():
                                res.ok = False
                                res.error = f"candidate service teardown failed: {exc}"
            else:
                res = run_candidate(
                    spec,
                    python_executable=self.python_executable,
                    timeout=self.candidate_timeout,
                    log_path=self.art.log_path(),
                )
            self._log(
                f"candidate {spec.id}: ok={res.ok} tps={res.generation_tps} "
                f"mem={res.peak_memory_gb} quality={res.quality_pass} "
                f"err={res.error or '-'}"
            )
            results.append(res)
            self.report("candidate_done", res.to_dict())

        self.art.write_json(
            "experiments.json",
            {
                "baseline": baseline.to_dict(),
                "candidates": [r.to_dict() for r in results],
            },
        )
        return results

    def decide(
        self,
        env: dict[str, Any],
        baseline: CandidateResult,
        candidates: list[CandidateResult],
        policy_obj: AcceptancePolicy,
    ) -> dict[str, Any]:
        verdict = select(baseline, candidates, policy_obj)
        vd = verdict.to_dict()
        self.art.write_json("verdict.json", vd)

        # quality evidence, separated so it can be reviewed on its own
        self.art.write_json(
            "quality.json",
            {
                r.id: {
                    "quality_pass": r.quality.get("quality_pass"),
                    "checks": r.quality.get("checks"),
                    "raw_outputs": r.quality.get("raw_outputs"),
                }
                for r in [baseline, *candidates]
            },
        )

        by_id = {r.id: r for r in [baseline, *candidates]}
        winner = by_id.get(verdict.selected_id, baseline)
        artifacts_mod.write_recipe(self.art, vd, winner.spec or {}, env)
        artifacts_mod.write_rollback(self.art, baseline.spec or {}, vd)

        self._log(
            f"verdict: selected={vd['selected_id']} accepted={vd['accepted']} "
            f"speedup={vd['speedup_percent']}%"
        )
        self.report("verdict", vd)
        return vd

    # -- entry point -----------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Run once under a cross-process workspace lock.

        The lock prevents a dashboard restart, another terminal, or a second
        browser tab from launching concurrent maintenance windows. Evolution
        may hold the same lock across discovery and preparation, in which case
        this Supervisor reuses that ownership rather than reacquiring it.
        """
        if self.optimization_lock_held:
            return self._run_impl()
        lock_path = os.path.join(self.root, ".infra-team", "optimize.lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another optimization is already running in this workspace"
                ) from exc
            try:
                return self._run_impl()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _run_impl(self) -> dict[str, Any]:
        started = time.time()
        env = self.probe()
        baseline = self.measure_baseline()

        if not baseline.ok:
            # No trustworthy reference means no honest comparison.
            vd = {
                "selected_id": "baseline",
                "accepted": False,
                "reason": f"baseline failed to run: {baseline.error}",
                "baseline_tps": 0.0,
                "selected_tps": 0.0,
                "speedup_percent": 0.0,
                "policy": AcceptancePolicy().to_dict(),
                "evaluations": [],
            }
            self.art.write_json("verdict.json", vd)
            artifacts_mod.write_rollback(self.art, baseline.spec or {}, vd)
            self._log(f"aborting: baseline failed ({baseline.error})")
            self.report("verdict", vd)
            return {
                "run_id": self.art.run_id,
                "path": self.art.path,
                "verdict": vd,
                "environment": env,
                "elapsed_seconds": round(time.time() - started, 1),
            }

        plan, plan_meta = self.make_plan(env, baseline)
        specs, policy_payload, policy_obj = self.enforce_policy(plan)
        service_stopped = self.stop_service_for_experiments()
        candidates: list[CandidateResult] = []
        service_transition = None
        try:
            candidates = self.run_experiments(specs, baseline)
            vd = self.decide(env, baseline, candidates, policy_obj)
            if service_stopped:
                service_transition = self.restore_or_promote_service(
                    vd, baseline, candidates
                )
        except BaseException as controller_exc:
            # Unexpected errors and user interrupts must not strand a previously
            # healthy service. This rollback path contains no model judgement.
            recovery = {
                "status": "controller_interrupted",
                "error_type": type(controller_exc).__name__,
                "error": str(controller_exc),
            }
            if service_stopped and self.service_manager and self.original_service_spec:
                try:
                    rollback_status = self.service_manager.start(
                        self.original_service_spec, wait_timeout=900.0
                    )
                    recovery.update(
                        status="baseline_restored",
                        rollback_service_status=rollback_status,
                    )
                    self._log("unexpected failure: baseline service restored")
                    self.report("service_switch_done", recovery)
                except ServiceError as rollback_exc:
                    recovery.update(
                        status="rollback_failed",
                        rollback_error=str(rollback_exc),
                    )
                    self._log(
                        f"unexpected failure and baseline restore failed: {rollback_exc}"
                    )
                    self.report("service_switch_failed", recovery)
                    self.art.write_json("service_transition.json", recovery)
                    self.art.write_json("interrupt_recovery.json", recovery)
                    raise RuntimeError(
                        "controller interrupted and automatic baseline restore failed; "
                        f"manual recovery required: {rollback_exc}"
                    ) from controller_exc
            self.art.write_json("interrupt_recovery.json", recovery)
            raise

        summary = {
            "run_id": self.art.run_id,
            "path": self.art.path,
            "environment": env,
            "plan": plan,
            "plan_meta": {
                k: v for k, v in plan_meta.items() if k != "raw_response"
            },
            "policy": policy_payload,
            "baseline": baseline.to_dict(),
            "candidates": [c.to_dict() for c in candidates],
            "verdict": vd,
            "service_transition": service_transition,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        self.art.write_json("summary.json", summary)
        return summary
