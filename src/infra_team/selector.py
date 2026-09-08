"""Selector: decides which candidate wins, using published rules only.

The model never declares its own success. This module applies fixed,
inspectable arithmetic to measured numbers. If nothing qualifies, the
baseline is the correct answer -- that is a valid, honest outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runner import CandidateResult


@dataclass
class AcceptancePolicy:
    """Thresholds a candidate must clear to be accepted."""

    quality_must_pass: bool = True
    max_memory_gb: float = 40.0
    minimum_speedup_percent: float = 5.0
    max_error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_must_pass": self.quality_must_pass,
            "max_memory_gb": self.max_memory_gb,
            "minimum_speedup_percent": self.minimum_speedup_percent,
            "max_error_rate": self.max_error_rate,
        }


@dataclass
class Verdict:
    """The final, auditable decision."""

    selected_id: str
    accepted: bool
    reason: str
    baseline_tps: float
    selected_tps: float
    speedup_percent: float
    policy: dict[str, Any]
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_id": self.selected_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "baseline_tps": self.baseline_tps,
            "selected_tps": self.selected_tps,
            "speedup_percent": self.speedup_percent,
            "policy": self.policy,
            "evaluations": self.evaluations,
        }


def _qualify(
    result: CandidateResult, policy: AcceptancePolicy
) -> tuple[bool, list[str]]:
    """Hard gates. A candidate failing any of these cannot be selected."""
    reasons: list[str] = []

    if not result.ok:
        reasons.append(f"run failed: {result.error}")
        return False, reasons

    if policy.quality_must_pass and not result.quality_pass:
        failed = [
            c["id"]
            for c in result.quality.get("checks", [])
            if not c.get("passed")
        ]
        reasons.append(f"quality gate failed: {failed}")

    if result.error_rate > policy.max_error_rate:
        reasons.append(
            f"error rate {result.error_rate:.2%} > "
            f"{policy.max_error_rate:.2%}"
        )

    raw_mem = result.performance.get("peak_memory_gb")
    if raw_mem is None:
        reasons.append("peak memory was not measured")
    else:
        mem = float(raw_mem)
        if mem > policy.max_memory_gb:
            reasons.append(
                f"peak memory {mem:.2f}GB > {policy.max_memory_gb:.2f}GB"
            )

    if result.generation_tps <= 0:
        reasons.append("no measured generation speed")

    return (not reasons), reasons


def select(
    baseline: CandidateResult,
    candidates: list[CandidateResult],
    policy: AcceptancePolicy | None = None,
) -> Verdict:
    """Apply the published selection rules.

    Returns a Verdict naming the winner. Falling back to the baseline is a
    legitimate result, not a failure of the system.
    """
    policy = policy or AcceptancePolicy()

    baseline_ok, baseline_reasons = _qualify(baseline, policy)
    baseline_tps = baseline.generation_tps

    evaluations: list[dict[str, Any]] = [
        {
            "id": baseline.id,
            "role": "baseline",
            "qualified": baseline_ok,
            "disqualifications": baseline_reasons,
            "generation_tps": baseline_tps,
            "peak_memory_gb": baseline.peak_memory_gb,
            "quality_pass": baseline.quality_pass,
            "error_rate": baseline.error_rate,
            "speedup_percent": 0.0,
        }
    ]

    if not baseline_ok:
        # Without a trustworthy baseline there is nothing to compare against.
        return Verdict(
            selected_id=baseline.id,
            accepted=False,
            reason=(
                "baseline itself did not qualify: "
                + "; ".join(baseline_reasons)
            ),
            baseline_tps=baseline_tps,
            selected_tps=baseline_tps,
            speedup_percent=0.0,
            policy=policy.to_dict(),
            evaluations=evaluations,
        )

    accepted_pool: list[tuple[CandidateResult, float]] = []

    for cand in candidates:
        ok, reasons = _qualify(cand, policy)
        speedup = (
            round((cand.generation_tps - baseline_tps) / baseline_tps * 100, 2)
            if baseline_tps > 0
            else 0.0
        )
        meets_speedup = speedup >= policy.minimum_speedup_percent
        if ok and not meets_speedup:
            reasons.append(
                f"speedup {speedup:+.2f}% < "
                f"{policy.minimum_speedup_percent:.2f}% threshold"
            )

        evaluations.append(
            {
                "id": cand.id,
                "role": "candidate",
                "qualified": ok,
                "accepted": ok and meets_speedup,
                "disqualifications": reasons,
                "generation_tps": cand.generation_tps,
                "peak_memory_gb": cand.peak_memory_gb,
                "quality_pass": cand.quality_pass,
                "error_rate": cand.error_rate,
                "speedup_percent": speedup,
            }
        )

        if ok and meets_speedup:
            accepted_pool.append((cand, speedup))

    if not accepted_pool:
        return Verdict(
            selected_id=baseline.id,
            accepted=False,
            reason=(
                "no candidate cleared the quality and speedup thresholds; "
                "keeping baseline"
            ),
            baseline_tps=baseline_tps,
            selected_tps=baseline_tps,
            speedup_percent=0.0,
            policy=policy.to_dict(),
            evaluations=evaluations,
        )

    winner, winner_speedup = max(
        accepted_pool, key=lambda pair: pair[0].generation_tps
    )
    return Verdict(
        selected_id=winner.id,
        accepted=True,
        reason=(
            f"fastest qualifying candidate at {winner.generation_tps:.2f} tok/s "
            f"({winner_speedup:+.2f}% vs baseline)"
        ),
        baseline_tps=baseline_tps,
        selected_tps=winner.generation_tps,
        speedup_percent=winner_speedup,
        policy=policy.to_dict(),
        evaluations=evaluations,
    )
