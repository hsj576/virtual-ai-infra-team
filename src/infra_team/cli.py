"""CLI for the virtual AI Infra team.

    infra-team evolve --once   run discovery-to-memory self-evolution
    infra-team optimize        run only the existing execution inner loop
    infra-team memory show     inspect the local Recipe Memory
    infra-team probe           print the environment snapshot only
    infra-team show <run-id>   replay a previous run's artifacts

Terminal output is deliberately plain and honest: every number printed
comes from a measurement performed in this run. There are no progress
bars standing in for work that did not happen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.resources import as_file, files
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra_team import __version__, probe_macos  # noqa: E402
from infra_team.artifacts import RUNS_DIRNAME  # noqa: E402
from infra_team.candidate_registry import (  # noqa: E402
    CandidateRegistry,
    RegistryError,
)
from infra_team.candidate_store import CandidateStore, CandidateStoreError  # noqa: E402
from infra_team.compatibility import AutonomyPolicy, PreflightError  # noqa: E402
from infra_team.dashboard import serve_dashboard  # noqa: E402
from infra_team.evolution import EvolutionEngine, EvolutionState  # noqa: E402
from infra_team.policy import TARGET_MODEL, TARGET_REVISION  # noqa: E402
from infra_team.recipe_memory import RecipeMemory  # noqa: E402
from infra_team.scheduler import (  # noqa: E402
    EvolutionWatcher,
    NotificationCenterNotifier,
    render_launchd_plist,
)
from infra_team.service_manager import (  # noqa: E402
    ManagedService,
    ServiceError,
    ServiceSpec,
)
from infra_team.supervisor import Supervisor  # noqa: E402

# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class Printer:
    """Minimal styled output."""

    def __init__(self) -> None:
        self.colour = _supports_colour()

    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.colour else text

    def rule(self, title: str = "") -> None:
        width = 68
        if title:
            pad = width - len(title) - 3
            print(self._c(f"── {title} " + "─" * max(pad - 1, 0), DIM))
        else:
            print(self._c("─" * width, DIM))

    def header(self, text: str) -> None:
        print()
        print(self._c(text, BOLD))

    def kv(self, key: str, value: Any) -> None:
        print(f"  {key:<26} {value}")

    def step(self, text: str) -> None:
        print(self._c(f"▸ {text}", CYAN))

    def ok(self, text: str) -> None:
        print(self._c(f"  PASS  {text}", GREEN))

    def warn(self, text: str) -> None:
        print(self._c(f"  WARN  {text}", YELLOW))

    def fail(self, text: str) -> None:
        print(self._c(f"  FAIL  {text}", RED))

    def plain(self, text: str = "") -> None:
        print(text)


P = Printer()


def _fmt(value: Any, suffix: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    return f"{value}{suffix}"


# --------------------------------------------------------------------------
# Event reporting during a live run
# --------------------------------------------------------------------------


def make_reporter(started: float):
    def report(event: str, data: dict[str, Any]) -> None:
        elapsed = time.time() - started

        if event == "probe_start":
            P.step("Probing environment")

        elif event == "probe_done":
            cores = (data.get("cpu_cores") or {}).get("total")
            P.kv(
                "Machine",
                f"{data.get('chip')} / {cores}-core CPU / "
                f"{data.get('gpu_cores')}-core GPU",
            )
            P.kv("Unified memory", _fmt(data.get("unified_memory_gb"), " GB"))
            P.kv("macOS", f"{data.get('macos_version')} ({data.get('macos_build')})")
            pkgs = data.get("packages") or {}
            P.kv("Runtime", f"mlx {pkgs.get('mlx')} / mlx-vlm {pkgs.get('mlx-vlm')}")
            power = data.get("power") or {}
            P.kv(
                "Power",
                "AC" if power.get("on_ac_power") else "battery (results may vary)",
            )
            cache = data.get("model_cache") or {}
            for name, size in cache.items():
                P.kv("Cached weights", f"{name}  {_fmt(size, ' GB', 'not cached')}")

        elif event == "baseline_start":
            P.header("Baseline")
            if data.get("mode") == "service":
                P.step(f"Measuring the running service at {data.get('base_url')}")
            else:
                P.step("Loading target and measuring offline reference performance")

        elif event == "baseline_done":
            perf = data.get("performance") or {}
            qual = data.get("quality") or {}
            if data.get("ok"):
                P.kv("Generation speed", f"{perf.get('generation_tps_median')} tok/s")
                P.kv("Prompt speed", _fmt(perf.get("prompt_tps_median"), " tok/s"))
                P.kv("Peak memory", _fmt(perf.get("peak_memory_gb"), " GB"))
                P.kv("Load time", _fmt(data.get("load_seconds"), " s"))
                P.kv(
                    "Quality gate",
                    f"{qual.get('passed')}/{qual.get('total')} checks",
                )
            else:
                P.fail(f"baseline failed: {data.get('error')}")

        elif event == "plan_start":
            P.header("Experiment plan")
            P.step("Asking the target model to propose experiments")

        elif event == "plan_done":
            plan = data.get("plan") or {}
            meta = data.get("meta") or {}
            source = meta.get("source")
            if source == "target_service":
                P.ok("deployed target service returned a schema-valid plan")
            elif source in ("local_fallback", "external_api"):
                P.warn(f"planner failover used: {source}")
            elif source in ("rule_based", "default_rule_based_plan"):
                P.warn("using the deterministic safe plan")
            else:
                P.warn(f"planner source: {source or 'disabled'}")
            for attempt in meta.get("attempts") or []:
                if not attempt.get("ok"):
                    P.warn(
                        f"{attempt.get('backend')} unavailable: "
                        f"{attempt.get('detail', '')[:100]}"
                    )
            hyp = plan.get("hypothesis")
            if hyp:
                P.kv("Hypothesis", hyp[:100])
            for exp in plan.get("experiments", []):
                if isinstance(exp, dict):
                    P.kv("Proposed", f"{exp.get('id')} — {exp.get('reason', '')[:60]}")

        elif event == "policy_done":
            P.header("Policy check")
            P.kv("Approved", ", ".join(data.get("approved", [])))
            refused = data.get("refused") or []
            if refused:
                for r in refused:
                    P.fail(f"refused '{r['id']}': {r['why']}")
            else:
                P.ok("no unsafe or unknown action requested")
            for note in data.get("notes") or []:
                P.warn(note)

        elif event == "service_switch_start":
            P.header("Service handoff")
            P.step(
                f"Plan frozen; entering maintenance window for {data.get('name')}"
            )
            P.kv("Stable API", data.get("base_url") + "/v1")

        elif event == "service_switch_done":
            if data.get("status") == "serving_selected":
                P.ok("selected recipe is serving at the original API address")
            elif data.get("status") == "baseline_restored":
                P.warn("candidate deployment failed; baseline service restored")

        elif event == "service_switch_failed":
            P.fail("automatic rollback failed; manual recovery is required")

        elif event == "candidate_start":
            P.plain()
            P.step(f"Running candidate: {data.get('id')}")

        elif event == "candidate_done":
            perf = data.get("performance") or {}
            qual = data.get("quality") or {}
            if data.get("ok"):
                P.kv("Generation speed", f"{perf.get('generation_tps_median')} tok/s")
                P.kv("Peak memory", _fmt(perf.get("peak_memory_gb"), " GB"))
                P.kv(
                    "Quality gate",
                    f"{qual.get('passed')}/{qual.get('total')} checks",
                )
            else:
                P.fail(f"{data.get('id')} failed: {data.get('error')}")

        elif event == "verdict":
            P.header("Result")
            rows = data.get("evaluations") or []
            if rows:
                P.plain(
                    f"  {'candidate':<22}{'tok/s':>9}{'memory':>10}"
                    f"{'quality':>10}{'speedup':>10}"
                )
                for row in rows:
                    q = "PASS" if row.get("quality_pass") else "FAIL"
                    mem = _fmt(row.get("peak_memory_gb"), "GB")
                    tps = row.get("generation_tps") or 0
                    spd = row.get("speedup_percent")
                    spd_s = "reference" if row.get("role") == "baseline" else (
                        f"{spd:+.1f}%" if spd is not None else "-"
                    )
                    P.plain(
                        f"  {row.get('id', ''):<22}{tps:>9.2f}{mem:>10}"
                        f"{q:>10}{spd_s:>10}"
                    )
            P.plain()
            if data.get("accepted"):
                P.plain(P._c("  OPTIMIZATION ACCEPTED", BOLD + GREEN))
                P.kv("Selected", data.get("selected_id"))
                P.kv("Speedup", f"{data.get('speedup_percent'):+.2f}%")
            else:
                P.plain(P._c("  OPTIMIZATION REJECTED — baseline retained", BOLD + YELLOW))
                P.kv("Reason", data.get("reason"))
            for row in rows:
                for why in row.get("disqualifications") or []:
                    P.plain(P._c(f"    {row.get('id')}: {why}", DIM))
            P.kv("Elapsed", f"{elapsed:.1f} s")

    return report


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


DEFAULT_POLICY_SENTINEL = "builtin"


def _default_policy_resource():
    return files("infra_team").joinpath("defaults", "autonomy-local.yaml")


def _load_autonomy_policy(root: str, value: str) -> AutonomyPolicy:
    if value == DEFAULT_POLICY_SENTINEL:
        with as_file(_default_policy_resource()) as path:
            return AutonomyPolicy.from_file(path)
    path = value if os.path.isabs(value) else os.path.join(root, value)
    return AutonomyPolicy.from_file(path)


def _policy_cli_value(value: str) -> str:
    return DEFAULT_POLICY_SENTINEL if value == DEFAULT_POLICY_SENTINEL else value


def _effective_target_revision(model: str, revision: str | None) -> str | None:
    if revision is not None:
        return revision
    return TARGET_REVISION if model == TARGET_MODEL else None


def _resolve_builtin_registry(
    root: str, policy_path: str
) -> tuple[CandidateRegistry, AutonomyPolicy, dict[str, Any], dict[str, Any]]:
    registry = CandidateRegistry.builtin()
    policy = _load_autonomy_policy(root, policy_path)
    manifests = registry.load()
    models = [TARGET_MODEL, *(manifest.source.repo_id for manifest in manifests)]
    environment = probe_macos.probe(models).to_dict()
    resolved = registry.resolve_all(policy=policy, environment=environment)
    resolved = CandidateStore(root).apply_ready_candidates(resolved)
    return registry, policy, environment, resolved


def cmd_registry(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    registry = CandidateRegistry.builtin()
    try:
        if args.registry_command == "inspect":
            try:
                payload = registry.inspect(args.candidate_id)
            except RegistryError:
                policy = _load_autonomy_policy(root, args.policy)
                environment = probe_macos.probe(
                    [TARGET_MODEL, *(m.source.repo_id for m in registry.load())]
                ).to_dict()
                resolved = registry.resolve_all(policy=policy, environment=environment)
                candidate = resolved.get(args.candidate_id)
                if candidate is None:
                    raise
                payload = {
                    "registry": registry.registry_name,
                    "candidate": candidate.to_dict(),
                    "manifest": candidate.manifest.to_dict() if candidate.manifest else None,
                }
        else:
            registry, policy, environment, resolved = _resolve_builtin_registry(
                root, args.policy
            )
            payload = {
                **registry.snapshot(),
                "policy_hash": policy.policy_hash,
                "environment": {
                    "chip": environment.get("chip"),
                    "arch": environment.get("arch"),
                    "mlx_vlm": (environment.get("packages") or {}).get("mlx-vlm"),
                },
                "candidates": [candidate.to_dict() for candidate in resolved.values()],
            }
    except (OSError, RegistryError, PreflightError, ValueError) as exc:
        P.fail(str(exc))
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    P.rule("trusted candidate registry")
    if args.registry_command == "inspect":
        manifest = payload.get("manifest") or {}
        candidate = payload.get("candidate") or {}
        P.kv("Manifest", manifest.get("id") or candidate.get("manifest_id"))
        P.kv("Candidate", candidate.get("id") or "all manifest variants")
        source = manifest.get("source") or candidate.get("source") or {}
        P.kv("Source", source.get("repo_id"))
        P.kv("Revision", source.get("revision"))
        P.kv("Manifest hash", payload.get("manifest_hash") or candidate.get("manifest_hash"))
    else:
        for candidate in payload["candidates"]:
            P.kv(candidate["id"], candidate["description"])
        P.kv("Policy hash", payload["policy_hash"])
    P.rule()
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    try:
        _registry, policy, _environment, resolved = _resolve_builtin_registry(
            root, args.policy
        )
        candidate = resolved.get(args.candidate_id)
        if candidate is None:
            matches = [
                item
                for item in resolved.values()
                if item.manifest_id == args.candidate_id and item.id != "baseline"
            ]
            if matches:
                candidate = next(
                    (item for item in matches if item.draft_block_size is None), matches[0]
                )
        if candidate is None:
            raise CandidateStoreError(f"candidate not found: {args.candidate_id}")
        prepared = CandidateStore(root).prepare(candidate, policy)
        payload = {
            **prepared.to_dict(),
            "resolved_candidate": prepared.apply(candidate).to_dict(),
        }
    except (OSError, RegistryError, PreflightError, CandidateStoreError, ValueError) as exc:
        P.fail(str(exc))
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    P.rule("candidate store")
    P.ok("candidate is READY")
    P.kv("Manifest", payload["manifest_id"])
    P.kv("Revision", payload["revision"])
    P.kv("Local model", payload["local_model_path"])
    P.kv("Reused", payload["reused"])
    P.rule()
    return 0


def make_evolution_reporter(started: float):
    """Terminal timeline for both outer-loop and reused Supervisor events."""
    inner = make_reporter(started)

    def report(event: str, data: dict[str, Any]) -> None:
        if event == "evolution_state":
            state = data.get("state") or "UNKNOWN"
            message = data.get("message") or ""
            elapsed = time.time() - started
            P.plain(f"[{elapsed:7.1f}s] {state:<20} {message}")
            return
        inner(event, data)

    return report


def _make_evolution_engine(
    args: argparse.Namespace,
    root: str,
    policy: AutonomyPolicy,
    registry: CandidateRegistry,
    memory: RecipeMemory,
    *,
    reporter,
) -> EvolutionEngine:
    return EvolutionEngine(
        root=root,
        service_name=args.service_name,
        policy=policy,
        registry=registry,
        memory=memory,
        report=reporter,
        repeats=args.repeats,
        candidate_timeout=args.timeout,
        use_planner=not args.no_planner,
        target_planner_endpoint=args.target_endpoint,
        disable_target_planner=args.no_target_planner,
        local_fallback_model=(
            args.local_fallback_model if args.enable_local_fallback else None
        ),
        external_planner_base_url=args.external_planner_base_url,
        external_planner_model=args.external_planner_model,
        external_planner_api_key=os.getenv(args.external_planner_api_key_env),
    )


def cmd_evolve(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    started = time.time()
    try:
        policy = _load_autonomy_policy(root, args.policy)
        registry = CandidateRegistry.builtin()
        memory = RecipeMemory(root)
        reporter = (
            (lambda event, data: None)
            if args.json
            else make_evolution_reporter(started)
        )
        if args.once:
            result = _make_evolution_engine(
                args, root, policy, registry, memory, reporter=reporter
            ).run_once()
        else:
            service = ManagedService(root, args.service_name, sys.executable)
            notifier = NotificationCenterNotifier(
                root,
                memory=memory,
                enabled=(
                    policy.notifications_enabled
                    and policy.notifications_macos_notification_center
                ),
            )
            watcher = EvolutionWatcher(
                root=root,
                service_name=args.service_name,
                policy=policy,
                registry=registry,
                memory=memory,
                service_manager=service,
                engine_factory=lambda: _make_evolution_engine(
                    args, root, policy, registry, memory, reporter=reporter
                ),
                notifier=notifier,
                poll_interval_seconds=args.poll_interval,
            )
            result = watcher.watch(max_cycles=args.max_cycles)
            result["mode"] = "watch"
    except (OSError, ValueError, RuntimeError) as exc:
        P.fail(str(exc))
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.watch:
        P.rule("self-evolution watcher")
        P.kv("Cycles", result.get("cycles"))
        P.kv("Actions", result.get("actions"))
        P.kv("Poll interval", f"{args.poll_interval or policy.watch_poll_interval_seconds}s")
        P.rule()
    else:
        P.plain()
        P.rule("self-evolution result")
        P.kv("Evolution", result.get("evolution_id"))
        P.kv("Status", result.get("status"))
        P.kv("Supervisor run", result.get("supervisor_run_id") or "-")
        P.kv("Selected", result.get("selected_id") or "-")
        P.kv("Accepted", result.get("accepted", False))
        P.kv("Artifacts", os.path.relpath(result["path"], root))
        if result.get("error"):
            P.fail(result["error"])
        P.rule()
    if args.watch:
        return 0
    successful = {
        EvolutionState.COMPLETED.value,
        EvolutionState.NO_IMPROVEMENT.value,
        EvolutionState.BASELINE_RESTORED.value,
    }
    return 0 if result.get("status") in successful else 1


def cmd_launchd(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    output = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    try:
        policy = _load_autonomy_policy(root, args.policy)
        payload = render_launchd_plist(
            workspace_root=root,
            python_executable=sys.executable,
            output_path=output,
            policy_path=_policy_cli_value(args.policy),
            service_name=args.service_name,
            poll_interval_seconds=(
                args.poll_interval or policy.watch_poll_interval_seconds
            ),
            label=args.label,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        P.fail(str(exc))
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        P.rule("launchd agent definition")
        P.ok("plist rendered; it was not installed")
        P.kv("Path", os.path.relpath(payload["path"], root))
        P.kv("Label", payload["label"])
        P.warn("Review the plist before explicitly installing it with launchctl.")
        P.rule()
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    memory = RecipeMemory(root)
    try:
        if args.memory_command == "show":
            payload = memory.show()
        else:
            registry, policy, environment, resolved = _resolve_builtin_registry(
                root, args.policy
            )
            target_revision = args.target_revision or next(
                (
                    candidate.target_revision
                    for candidate in resolved.values()
                    if candidate.target_revision
                ),
                None,
            )
            payload = memory.query_prior(
                environment,
                target_model=args.target,
                target_revision=target_revision,
                resolved_candidates=resolved,
            )
            payload["registry_snapshot"] = registry.snapshot()
            payload["policy_hash"] = policy.policy_hash
    except (OSError, ValueError, RuntimeError) as exc:
        P.fail(str(exc))
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    P.rule("recipe memory")
    if args.memory_command == "show":
        P.kv("Database", payload["path"])
        P.kv("Environments", payload["environments"])
        P.kv("Candidate runs", payload["candidate_runs"])
        P.kv("Promotions", payload["promotions"])
        P.kv("Evolution runs", payload["evolution_runs"])
        P.kv("Watch states", payload.get("watch_states", 0))
        P.kv("Notifications", payload.get("notification_events", 0))
    else:
        P.kv("Environment", payload["environment_fingerprint"])
        for item in payload["ranking"]:
            P.kv(
                item["candidate_id"],
                f"{item['match']} / {item['recommendation']}",
            )
        P.warn("Memory is a search prior; every candidate must pass local validation.")
    P.rule()
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    started = time.time()

    P.rule("virtual AI Infra team")
    P.plain("  Goal        accelerate local Qwen3.8-27B inference")
    P.plain("  Constraint  quality gate must pass, peak memory <= 40 GB")
    P.plain("  Method      measure baseline, test candidates, keep only a")
    P.plain("              verified improvement")
    P.rule()

    target_endpoint = None if args.no_target_planner else args.target_endpoint
    local_fallback = args.local_fallback_model if args.enable_local_fallback else None
    sup = Supervisor(
        root=root,
        python_executable=sys.executable,
        repeats=args.repeats,
        use_planner=not args.no_planner,
        report=make_reporter(started),
        candidate_timeout=args.timeout,
        target_planner_endpoint=target_endpoint,
        local_fallback_model=local_fallback,
        external_planner_base_url=args.external_planner_base_url,
        external_planner_model=args.external_planner_model,
        external_planner_api_key=os.getenv(args.external_planner_api_key_env),
        managed_service_name=args.service_name,
    )
    summary = sup.run()

    P.plain()
    P.rule("artifacts")
    rel = os.path.relpath(summary["path"], root)
    P.plain(f"  {rel}")
    for name in sorted(os.listdir(summary["path"])):
        P.plain(f"    {name}")
    P.rule()

    return 0 if summary["verdict"].get("selected_id") else 1


def cmd_serve(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    manager = ManagedService(root, name=args.name, python_executable=sys.executable)

    if args.serve_command == "start":
        spec = ServiceSpec(
            name=args.name,
            model=args.model,
            host=args.host,
            port=args.port,
            draft_model=args.draft_model,
            draft_kind=args.draft_kind,
            draft_block_size=args.draft_block_size,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            target_revision=_effective_target_revision(
                args.model, args.target_revision
            ),
        )
        try:
            status = manager.start(spec, wait_timeout=args.wait_timeout)
            manager.atomically_promote(
                manager.recipe_from_spec(spec, source="manual_cli")
            )
        except ServiceError as exc:
            P.fail(str(exc))
            return 1
        P.rule("local model service")
        P.ok("service is healthy")
        P.kv("Name", status.get("name"))
        P.kv("API", status.get("base_url") + "/v1")
        P.kv("Model", (status.get("spec") or {}).get("model"))
        P.kv("PID", status.get("pid"))
        P.kv("Log", status.get("server_log"))
        P.rule()
        return 0

    if args.serve_command == "stop":
        try:
            status = manager.stop(timeout=args.stop_timeout)
        except ServiceError as exc:
            P.fail(str(exc))
            return 1
        if status.get("stopped"):
            P.ok(f"service '{args.name}' stopped")
        else:
            P.warn(f"service '{args.name}' was not deployed")
        return 0

    status = manager.status()
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    P.rule("local model service")
    P.kv("Name", status.get("name"))
    P.kv("Status", status.get("status"))
    P.kv("Managed", status.get("managed"))
    P.kv("Healthy", status.get("healthy"))
    if status.get("base_url"):
        P.kv("API", status.get("base_url") + "/v1")
    if status.get("spec"):
        P.kv("Model", status["spec"].get("model"))
        P.kv("Drafter", status["spec"].get("draft_model") or "none")
    P.rule()
    return 0 if status.get("healthy") else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    url = f"http://{args.host}:{args.port}"
    P.rule("virtual AI Infra dashboard")
    P.kv("URL", url)
    P.kv("Service", args.service_name)
    P.kv("Mode", "local control + chat; confirmation required")
    P.plain()
    P.plain("  Press Ctrl-C to stop the dashboard. The model service keeps running.")
    P.rule()
    try:
        serve_dashboard(
            root,
            service_name=args.service_name,
            host=args.host,
            port=args.port,
        )
    except OSError as exc:
        P.fail(f"dashboard could not start: {exc}")
        return 1
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    env = probe_macos.probe(
        ["mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2"]
    ).to_dict()
    if args.json:
        print(json.dumps(env, indent=2))
        return 0
    P.rule("environment")
    cores = (env.get("cpu_cores") or {}).get("total")
    P.kv("Chip", env.get("chip"))
    P.kv("CPU / GPU cores", f"{cores} / {env.get('gpu_cores')}")
    P.kv("Unified memory", _fmt(env.get("unified_memory_gb"), " GB"))
    P.kv("macOS", f"{env.get('macos_version')} ({env.get('macos_build')})")
    P.kv("Disk free", _fmt(env.get("disk_free_gb"), " GB"))
    for k, v in (env.get("packages") or {}).items():
        P.kv(k, _fmt(v, "", "not installed"))
    for k, v in (env.get("model_cache") or {}).items():
        P.kv(k, _fmt(v, " GB", "not cached"))
    power = env.get("power") or {}
    P.kv("On AC power", power.get("on_ac_power"))
    P.kv("Memory free", _fmt(power.get("memory_free_pct"), "%"))
    P.rule()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    runs_dir = os.path.join(root, RUNS_DIRNAME)
    if not os.path.isdir(runs_dir):
        P.fail("no runs found")
        return 1

    run_id = args.run_id
    if not run_id:
        runs = sorted(os.listdir(runs_dir))
        if not runs:
            P.fail("no runs found")
            return 1
        run_id = runs[-1]

    path = os.path.join(runs_dir, run_id)
    if not os.path.isdir(path):
        P.fail(f"run '{run_id}' not found")
        return 1

    P.rule(f"run {run_id}")
    verdict_file = os.path.join(path, "verdict.json")
    if os.path.exists(verdict_file):
        with open(verdict_file, encoding="utf-8") as fh:
            vd = json.load(fh)
        for row in vd.get("evaluations", []):
            q = "PASS" if row.get("quality_pass") else "FAIL"
            P.plain(
                f"  {row.get('id'):<22}{row.get('generation_tps') or 0:>9.2f} tok/s"
                f"   quality {q}   {row.get('speedup_percent'):+.1f}%"
            )
        P.plain()
        P.kv("Selected", vd.get("selected_id"))
        P.kv("Accepted", vd.get("accepted"))
        P.kv("Reason", vd.get("reason"))

    recipe = os.path.join(path, "selected_recipe.yaml")
    if os.path.exists(recipe):
        P.plain()
        P.rule("selected_recipe.yaml")
        with open(recipe, encoding="utf-8") as fh:
            print(fh.read())
    P.rule()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-team",
        description="A local virtual AI Infra team for Apple Silicon.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="workspace root for artifacts (default: cwd)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_registry = sub.add_parser(
        "registry", help="inspect the trusted local candidate registry"
    )
    registry_sub = p_registry.add_subparsers(dest="registry_command", required=True)
    p_registry_list = registry_sub.add_parser(
        "list", help="list compatible candidates from trusted manifests"
    )
    p_registry_list.add_argument(
        "--policy", default=DEFAULT_POLICY_SENTINEL
    )
    p_registry_list.add_argument("--json", action="store_true")
    p_registry_list.set_defaults(func=cmd_registry)
    p_registry_inspect = registry_sub.add_parser(
        "inspect", help="inspect a manifest or resolved candidate"
    )
    p_registry_inspect.add_argument("candidate_id")
    p_registry_inspect.add_argument(
        "--policy", default=DEFAULT_POLICY_SENTINEL
    )
    p_registry_inspect.add_argument("--json", action="store_true")
    p_registry_inspect.set_defaults(func=cmd_registry)

    p_candidates = sub.add_parser(
        "candidates", help="manage the project candidate store"
    )
    candidates_sub = p_candidates.add_subparsers(
        dest="candidates_command", required=True
    )
    p_prepare = candidates_sub.add_parser(
        "prepare", help="download and verify a pinned registry candidate"
    )
    p_prepare.add_argument("candidate_id")
    p_prepare.add_argument("--policy", default=DEFAULT_POLICY_SENTINEL)
    p_prepare.add_argument("--json", action="store_true")
    p_prepare.set_defaults(func=cmd_candidates)

    p_evolve = sub.add_parser(
        "evolve", help="run trusted discovery, preparation, optimization and memory"
    )
    evolve_mode = p_evolve.add_mutually_exclusive_group(required=True)
    evolve_mode.add_argument(
        "--once",
        action="store_true",
        help="run one complete evolution cycle",
    )
    evolve_mode.add_argument(
        "--watch",
        action="store_true",
        help="continuously watch trusted Registry for due candidates",
    )
    p_evolve.add_argument(
        "--max-cycles",
        type=int,
        help="stop watch after N cycles (test/smoke use; default runs continuously)",
    )
    p_evolve.add_argument(
        "--poll-interval",
        type=int,
        help="override policy watch polling interval in seconds",
    )
    p_evolve.add_argument("--service-name", default="default")
    p_evolve.add_argument("--policy", default=DEFAULT_POLICY_SENTINEL)
    p_evolve.add_argument("--repeats", type=int, default=3)
    p_evolve.add_argument("--timeout", type=int, default=3600)
    p_evolve.add_argument("--json", action="store_true")
    p_evolve.add_argument("--no-planner", action="store_true")
    p_evolve.add_argument(
        "--target-endpoint",
        help="override the managed service OpenAI endpoint used by Target planner",
    )
    p_evolve.add_argument("--no-target-planner", action="store_true")
    p_evolve.add_argument("--enable-local-fallback", action="store_true")
    p_evolve.add_argument(
        "--local-fallback-model",
        default="mlx-community/Qwen3.5-4B-MLX-4bit",
    )
    p_evolve.add_argument(
        "--external-planner-base-url",
        default=os.getenv("INFRA_PLANNER_BASE_URL"),
    )
    p_evolve.add_argument(
        "--external-planner-model", default=os.getenv("INFRA_PLANNER_MODEL")
    )
    p_evolve.add_argument(
        "--external-planner-api-key-env", default="INFRA_PLANNER_API_KEY"
    )
    p_evolve.set_defaults(func=cmd_evolve)

    p_launchd = sub.add_parser(
        "launchd", help="render a launchd agent definition without installing it"
    )
    launchd_sub = p_launchd.add_subparsers(dest="launchd_command", required=True)
    p_launchd_render = launchd_sub.add_parser(
        "render", help="render a reviewed launchd plist for evolve --watch"
    )
    p_launchd_render.add_argument(
        "--output", default=".infra-team/launchd/cn.workbuddy.infra-team.watch.plist"
    )
    p_launchd_render.add_argument("--policy", default=DEFAULT_POLICY_SENTINEL)
    p_launchd_render.add_argument("--service-name", default="default")
    p_launchd_render.add_argument("--poll-interval", type=int)
    p_launchd_render.add_argument(
        "--label", default="cn.workbuddy.infra-team.watch"
    )
    p_launchd_render.add_argument("--json", action="store_true")
    p_launchd_render.set_defaults(func=cmd_launchd)

    p_memory = sub.add_parser("memory", help="inspect local Recipe Memory")
    memory_sub = p_memory.add_subparsers(dest="memory_command", required=True)
    p_memory_show = memory_sub.add_parser("show", help="show Recipe Memory summary")
    p_memory_show.add_argument("--json", action="store_true")
    p_memory_show.set_defaults(func=cmd_memory)
    p_memory_query = memory_sub.add_parser(
        "query", help="query prior outcomes for the current environment"
    )
    p_memory_query.add_argument("--target", default=TARGET_MODEL)
    p_memory_query.add_argument("--target-revision")
    p_memory_query.add_argument("--policy", default=DEFAULT_POLICY_SENTINEL)
    p_memory_query.add_argument("--json", action="store_true")
    p_memory_query.set_defaults(func=cmd_memory)

    p_opt = sub.add_parser("optimize", help="run the full optimization loop")
    p_opt.add_argument(
        "--repeats", type=int, default=3, help="benchmark repeats per prompt"
    )
    p_opt.add_argument(
        "--no-planner",
        action="store_true",
        help="skip the model planner and use the default rule-based plan",
    )
    p_opt.add_argument(
        "--timeout", type=int, default=3600, help="per-candidate timeout in seconds"
    )
    p_opt.add_argument(
        "--service-name",
        default="default",
        help="managed service to measure, switch and restore",
    )
    p_opt.add_argument(
        "--target-endpoint",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible endpoint of the deployed target service",
    )
    p_opt.add_argument(
        "--no-target-planner",
        action="store_true",
        help="do not ask the deployed target service to plan",
    )
    p_opt.add_argument(
        "--enable-local-fallback",
        action="store_true",
        help="load the small local fallback planner only if target planning fails",
    )
    p_opt.add_argument(
        "--local-fallback-model",
        default="mlx-community/Qwen3.5-4B-MLX-4bit",
        help="small MLX planner loaded on demand during failover",
    )
    p_opt.add_argument(
        "--external-planner-base-url",
        default=os.getenv("INFRA_PLANNER_BASE_URL"),
        help="optional external OpenAI-compatible planner base URL",
    )
    p_opt.add_argument(
        "--external-planner-model",
        default=os.getenv("INFRA_PLANNER_MODEL"),
        help="model name for the external planner",
    )
    p_opt.add_argument(
        "--external-planner-api-key-env",
        default="INFRA_PLANNER_API_KEY",
        help="environment variable containing the external planner API key",
    )
    p_opt.set_defaults(func=cmd_optimize)

    p_serve = sub.add_parser("serve", help="manage the stable local model API")
    serve_sub = p_serve.add_subparsers(dest="serve_command", required=True)

    p_start = serve_sub.add_parser("start", help="start a managed MLX-VLM service")
    p_start.add_argument("--name", default="default")
    p_start.add_argument("--model", default=TARGET_MODEL)
    p_start.add_argument(
        "--target-revision",
        help=(
            "fixed Hugging Face commit used to resolve the target snapshot; "
            "the built-in Qwen target defaults to its verified commit"
        ),
    )
    p_start.add_argument("--host", default="127.0.0.1")
    p_start.add_argument("--port", type=int, default=8000)
    p_start.add_argument("--draft-model")
    p_start.add_argument("--draft-kind", choices=("dflash", "eagle3", "mtp"))
    p_start.add_argument("--draft-block-size", type=int)
    p_start.add_argument("--max-tokens", type=int)
    p_start.add_argument("--enable-thinking", action="store_true")
    p_start.add_argument("--wait-timeout", type=float, default=900.0)
    p_start.set_defaults(func=cmd_serve)

    p_stop = serve_sub.add_parser("stop", help="stop only this workspace's service")
    p_stop.add_argument("--name", default="default")
    p_stop.add_argument("--stop-timeout", type=float, default=30.0)
    p_stop.set_defaults(func=cmd_serve)

    p_status = serve_sub.add_parser("status", help="show managed service health")
    p_status.add_argument("--name", default="default")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_serve)

    p_dashboard = sub.add_parser(
        "dashboard", help="show deployment and optimization in a local web UI"
    )
    p_dashboard.add_argument("--service-name", default="default")
    p_dashboard.add_argument("--host", default="127.0.0.1")
    p_dashboard.add_argument("--port", type=int, default=9000)
    p_dashboard.set_defaults(func=cmd_dashboard)

    p_probe = sub.add_parser("probe", help="print the environment snapshot")
    p_probe.add_argument("--json", action="store_true", help="raw JSON output")
    p_probe.set_defaults(func=cmd_probe)

    p_show = sub.add_parser("show", help="show a previous run")
    p_show.add_argument("run_id", nargs="?", help="run id (default: latest)")
    p_show.set_defaults(func=cmd_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        P.plain()
        P.warn("interrupted; baseline was restored or left unchanged")
        return 130
    except RuntimeError as exc:
        P.fail(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
