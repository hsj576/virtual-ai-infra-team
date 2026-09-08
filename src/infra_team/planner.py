"""Pluggable experiment planners with target-model-first failover.

Normal operation asks the already-running local target service to propose an
ExperimentPlan. The plan is persisted before any model switch. If the target
service is unavailable *and replanning is genuinely required*, the router can
fall through to an on-demand small local model, an external OpenAI-compatible
API, and finally the deterministic safe plan.

No planner has execution authority. Every returned plan is still untrusted
input for policy.py and selector.py.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from .policy import TARGET_MODEL, candidate_summaries, default_plan

if TYPE_CHECKING:
    from .candidate_registry import ResolvedCandidate

PLANNER_WORKER = os.path.join(os.path.dirname(__file__), "planner_worker.py")
DEFAULT_TARGET_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_FALLBACK_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

SYSTEM_PROMPT = """\
You are the inference-optimization engineer on a local AI Infra team.
You are running on the very machine you are asked to optimize.

Given a hardware profile, a runtime description and a measured baseline,
choose which optimization candidates are worth testing.

You must reply with ONLY a JSON object, no prose and no code fence:

{
  "hypothesis": "<one sentence on why these candidates may help>",
  "experiments": [
    {"id": "<candidate id>", "reason": "<why this one>"}
  ],
  "acceptance_policy": {
    "quality_pass": true,
    "minimum_speedup_percent": <number between 1 and 100>,
    "max_memory_gb": <number between 8 and 44>
  }
}

Rules:
- Use only candidate ids from the allowed list. Never invent an id.
- Always include "baseline" as the first experiment.
- Choose between two and three candidates in total.
- You cannot run shell commands. You only propose; a separate controller
  validates and executes, and an independent verifier judges the result.
"""


class PlannerError(RuntimeError):
    """A planner could not return a usable ExperimentPlan."""


class PlannerBackend(Protocol):
    """Common contract for target, local fallback and external planners."""

    name: str

    def create_plan(self, context: dict[str, Any]) -> str | dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class PlannerAttempt:
    backend: str
    ok: bool
    detail: str
    description: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_context(
    environment: dict[str, Any],
    baseline: dict[str, Any] | None,
    resolved_candidates: dict[str, "ResolvedCandidate"] | None = None,
) -> dict[str, Any]:
    """Assemble non-sensitive facts from the exact discovery snapshot."""
    packages = environment.get("packages") or {}
    summaries = candidate_summaries(resolved_candidates)
    target = next((item for item in summaries if item["id"] == "baseline"), None)
    drafters = sorted(
        {
            str(item["source_repo_id"])
            for item in summaries
            if item.get("source_repo_id")
        }
    )
    return {
        "hardware": {
            "chip": environment.get("chip"),
            "gpu_cores": environment.get("gpu_cores"),
            "unified_memory_gb": environment.get("unified_memory_gb"),
            "cpu_cores": (environment.get("cpu_cores") or {}).get("total"),
        },
        "runtime": {
            "engine": "mlx-vlm",
            "engine_version": packages.get("mlx-vlm"),
            "mlx_version": packages.get("mlx"),
            "target_model": TARGET_MODEL if target else None,
            "target_quantization": "4-bit",
            "available_candidate_sources": drafters,
        },
        "baseline": baseline
        or {
            "generation_tps": None,
            "peak_memory_gb": None,
            "quality_pass": None,
            "note": "baseline not yet measured",
        },
        "allowed_candidates": summaries,
        "constraints": {
            "temperature": 0,
            "quality_must_pass": True,
            "quality_checks_required": "4/4",
            "max_memory_gb": 40,
            "minimum_speedup_percent": 5,
        },
    }


def _planner_user_message(context: dict[str, Any]) -> str:
    return (
        "Here is the machine, runtime and measured baseline:\n\n"
        + json.dumps(context, indent=2)
        + "\n\nReturn the ExperimentPlan JSON now."
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first complete ExperimentPlan object out of model text."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    for chunk in fenced + [text]:
        start = chunk.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(chunk)):
                char = chunk[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(chunk[start : i + 1])
                            if isinstance(obj, dict) and "experiments" in obj:
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
            start = chunk.find("{", start + 1)
    return None


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _openai_chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None,
    timeout: float,
) -> str:
    """Small dependency-free OpenAI-compatible chat client."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 700,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _chat_endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PlannerError(f"OpenAI-compatible request failed: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlannerError("response has no choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise PlannerError("planner returned empty content")
    return content


@dataclass
class OpenAICompatiblePlanner:
    """Planner backed by an already-running local service or an external API."""

    name: str
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 120.0
    locality: str = "local"

    def create_plan(self, context: dict[str, Any]) -> str:
        return _openai_chat(
            self.base_url,
            self.model,
            SYSTEM_PROMPT,
            _planner_user_message(context),
            self.api_key,
            self.timeout,
        )

    def describe(self) -> dict[str, Any]:
        # Never serialize credentials into artifacts.
        return {
            "type": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model,
            "locality": self.locality,
            "authenticated": bool(self.api_key),
        }


@dataclass
class LocalModelPlanner:
    """On-demand small local model, loaded only during failover."""

    model: str = DEFAULT_FALLBACK_MODEL
    python_executable: str = sys.executable
    timeout: int = 900
    log_path: str | None = None
    name: str = "local_fallback"

    def create_plan(self, context: dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "target_model": self.model,
                "system_prompt": SYSTEM_PROMPT,
                "context": context,
            }
        )
        try:
            proc = subprocess.run(
                [self.python_executable, PLANNER_WORKER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PlannerError(f"local fallback timed out after {self.timeout}s") from exc

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== planner {self.name} =====\n")
                fh.write(proc.stderr or "")

        for line in (proc.stdout or "").splitlines():
            if line.startswith("PLAN_RAW:"):
                try:
                    text = json.loads(line[len("PLAN_RAW:") :]).get("text", "")
                except json.JSONDecodeError:
                    text = ""
                if text:
                    return text
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise PlannerError("local planner produced no response: " + " | ".join(tail))

    def describe(self) -> dict[str, Any]:
        return {"type": "local_model", "model": self.model, "load_policy": "on_demand"}


@dataclass
class RuleBasedPlanner:
    name: str = "rule_based"

    def create_plan(self, context: dict[str, Any]) -> dict[str, Any]:
        return default_plan()

    def describe(self) -> dict[str, Any]:
        return {"type": "deterministic_safe_fallback"}


class PlannerRouter:
    """Try planners in order, recording every handoff without leaking secrets."""

    def __init__(self, backends: list[PlannerBackend]) -> None:
        if not backends:
            raise ValueError("PlannerRouter requires at least one backend")
        self.backends = backends

    def create_plan(
        self, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempts: list[PlannerAttempt] = []
        for backend in self.backends:
            try:
                raw = backend.create_plan(context)
                plan = raw if isinstance(raw, dict) else _extract_json(raw)
                if not isinstance(plan, dict):
                    raise PlannerError("response was not valid ExperimentPlan JSON")
                experiments = plan.get("experiments")
                if not isinstance(experiments, list) or not experiments:
                    raise PlannerError("plan.experiments was empty or invalid")
                plan = dict(plan)
                plan["source"] = backend.name
                attempts.append(
                    PlannerAttempt(
                        backend=backend.name,
                        ok=True,
                        detail="schema-shaped plan returned",
                        description=backend.describe(),
                    )
                )
                return plan, {
                    "source": backend.name,
                    "planner": backend.describe(),
                    "attempts": [a.to_dict() for a in attempts],
                    "failover_used": len(attempts) > 1,
                }
            except Exception as exc:
                attempts.append(
                    PlannerAttempt(
                        backend=backend.name,
                        ok=False,
                        detail=f"{type(exc).__name__}: {exc}",
                        description=backend.describe(),
                    )
                )

        # Normally unreachable when RuleBasedPlanner is last, but preserves
        # the safety invariant even if a caller builds a custom router.
        plan = default_plan()
        plan["source"] = "emergency_rule_based"
        return plan, {
            "source": "emergency_rule_based",
            "planner": {"type": "deterministic_safe_fallback"},
            "attempts": [a.to_dict() for a in attempts],
            "failover_used": True,
        }


def build_default_router(
    *,
    target_endpoint: str | None = DEFAULT_TARGET_ENDPOINT,
    target_model: str = "mlx-community/Qwen3.8-27B-4bit",
    local_fallback_model: str | None = None,
    external_base_url: str | None = None,
    external_model: str | None = None,
    external_api_key: str | None = None,
    python_executable: str | None = None,
    log_path: str | None = None,
) -> PlannerRouter:
    """Build target → local fallback → external API → rule routing."""
    backends: list[PlannerBackend] = []
    if target_endpoint:
        backends.append(
            OpenAICompatiblePlanner(
                name="target_service",
                base_url=target_endpoint,
                model=target_model,
                timeout=120.0,
                locality="local",
            )
        )
    if local_fallback_model:
        backends.append(
            LocalModelPlanner(
                model=local_fallback_model,
                python_executable=python_executable or sys.executable,
                log_path=log_path,
            )
        )
    if external_base_url and external_model:
        backends.append(
            OpenAICompatiblePlanner(
                name="external_api",
                base_url=external_base_url,
                model=external_model,
                api_key=external_api_key,
                timeout=180.0,
                locality="external",
            )
        )
    backends.append(RuleBasedPlanner())
    return PlannerRouter(backends)


def request_plan(
    context: dict[str, Any],
    target_model: str = "mlx-community/Qwen3.8-27B-4bit",
    python_executable: str | None = None,
    timeout: int = 900,
    log_path: str | None = None,
    *,
    target_endpoint: str | None = DEFAULT_TARGET_ENDPOINT,
    local_fallback_model: str | None = None,
    external_base_url: str | None = None,
    external_model: str | None = None,
    external_api_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility entry point used by Supervisor.

    Unlike the original implementation, this never reloads the 27B target.
    It calls the deployed target service first and only loads a configured
    small local model when failover is required.
    """
    router = build_default_router(
        target_endpoint=target_endpoint,
        target_model=target_model,
        local_fallback_model=local_fallback_model,
        external_base_url=external_base_url or os.getenv("INFRA_PLANNER_BASE_URL"),
        external_model=external_model or os.getenv("INFRA_PLANNER_MODEL"),
        external_api_key=external_api_key or os.getenv("INFRA_PLANNER_API_KEY"),
        python_executable=python_executable,
        log_path=log_path,
    )
    return router.create_plan(context)
