"""Tests for the pieces that decide correctness.

The value of this project rests on the claim that the model cannot talk
its way into a "win". These tests defend that claim.
"""

from __future__ import annotations

import pytest

from infra_team.candidate_registry import CandidateRegistry
from infra_team.compatibility import AutonomyPolicy
from infra_team.policy import (
    PolicyError,
    default_plan,
    resolve_acceptance_policy,
    sanitise_acceptance,
    validate_plan,
)
from infra_team.runner import CandidateResult
from infra_team.selector import AcceptancePolicy, select
from infra_team.tasks import evaluate_quality

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_result(
    cid: str,
    tps: float,
    memory: float = 22.0,
    quality_pass: bool = True,
    ok: bool = True,
    error_rate: float = 0.0,
) -> CandidateResult:
    return CandidateResult(
        id=cid,
        ok=ok,
        spec={"id": cid, "target_model": "t"},
        quality={
            "quality_pass": quality_pass,
            "checks": [{"id": "arithmetic", "passed": quality_pass}],
        },
        performance={
            "generation_tps_median": tps,
            "peak_memory_gb": memory,
            "error_rate": error_rate,
        },
    )


GOOD_OUTPUTS = {
    "arithmetic": "391",
    "json_schema": '{"city": "Paris", "country": "France"}',
    "instruction_following": "blue",
    "code_generation": "```python\ndef add(a, b):\n    return a + b\n```",
}


# --------------------------------------------------------------------------
# quality gate
# --------------------------------------------------------------------------


def test_clean_output_passes_every_check():
    report = evaluate_quality(GOOD_OUTPUTS)
    assert report["quality_pass"] is True
    assert report["passed"] == report["total"] == 4


@pytest.mark.parametrize(
    "task_id,broken,label",
    [
        ("arithmetic", "390", "wrong arithmetic"),
        ("arithmetic", "", "empty output"),
        ("arithmetic", "39\ufffd1", "mojibake"),
        ("json_schema", "the capital is Paris", "no JSON"),
        ("json_schema", '{"city": "Berlin", "country": "DE"}', "wrong content"),
        ("json_schema", '{"city": "Paris"}', "missing key"),
        ("instruction_following", "I think it is a lovely azure shade", "verbose"),
        ("code_generation", "```python\ndef add(a, b)\n  return a+b\n```", "syntax"),
        ("code_generation", "```python\nx = 1\n```", "no function"),
        ("code_generation", "```python\ndef add(a, b):\n    return a - b\n```", "wrong operation"),
        ("code_generation", "```python\ndef add(a):\n    return a + a\n```", "wrong signature"),
        ("code_generation", "```python\ndef sum_values(a, b):\n    return a + b\n```", "wrong name"),
        ("code_generation", "```python\ndef add(a, b):\n    return 0\n    return a + b\n```", "unreachable valid return"),
        ("code_generation", "```python\nasync def add(a, b):\n    return a + b\n```", "async function"),
        ("code_generation", "```python\ndef add(a, b):\n    print(a)\n    return a + b\n```", "side effect"),
        ("code_generation", "```python\ndef add(a, a):\n    return a + a\n```", "duplicate arguments"),
        ("code_generation", "```python\ndef add(a, b):\n    return a + b\n```\nextra prose", "trailing prose"),
        ("code_generation", "```python\ndef add(a, b):\n    return a + b\n```\n```python\nx = 1\n```", "second code block"),
    ],
)
def test_broken_output_fails_the_gate(task_id, broken, label):
    outputs = dict(GOOD_OUTPUTS)
    outputs[task_id] = broken
    assert evaluate_quality(outputs)["quality_pass"] is False, label


def test_repetition_loop_is_caught():
    outputs = dict(GOOD_OUTPUTS)
    outputs["instruction_following"] = "blue " * 40
    assert evaluate_quality(outputs)["quality_pass"] is False


def test_thinking_block_is_tolerated():
    outputs = dict(GOOD_OUTPUTS)
    outputs["arithmetic"] = "<think>17*23, carry the 1</think>391"
    assert evaluate_quality(outputs)["quality_pass"] is True


# --------------------------------------------------------------------------
# policy / whitelist
# --------------------------------------------------------------------------


def test_default_plan_is_executable():
    specs, decision = validate_plan(default_plan())
    assert [s.id for s in specs][0] == "baseline"
    assert len(specs) >= 2
    assert decision.refused == []


def test_shell_command_is_refused_not_executed():
    specs, decision = validate_plan(
        {"experiments": [{"id": "rm -rf /"}, {"id": "dflash2_default"}]}
    )
    assert [s.id for s in specs] == ["baseline", "dflash2_default"]
    assert decision.refused[0]["id"] == "rm -rf /"


def test_invented_candidate_is_refused():
    specs, decision = validate_plan(
        {"experiments": [{"id": "turbo_mode_9000"}, {"id": "dflash2_block6"}]}
    )
    assert "turbo_mode_9000" not in [s.id for s in specs]
    assert decision.refused[0]["why"] == "not in candidate whitelist"


def test_baseline_is_always_present_and_first():
    specs, decision = validate_plan({"experiments": [{"id": "dflash2_default"}]})
    assert specs[0].id == "baseline"
    assert any("baseline injected" in n for n in decision.notes)


def test_plan_without_real_candidate_is_rejected():
    with pytest.raises(PolicyError):
        validate_plan({"experiments": [{"id": "baseline"}]})


@pytest.mark.parametrize(
    "bad", [{}, {"experiments": []}, "not a dict", {"experiments": "nope"}, None]
)
def test_malformed_plans_are_rejected(bad):
    with pytest.raises(PolicyError):
        validate_plan(bad)


def test_temperature_is_forced_deterministic():
    specs, _ = validate_plan(default_plan())
    assert all(s.temperature == 0.0 for s in specs)


def test_absurd_thresholds_are_clamped():
    out = sanitise_acceptance(
        {
            "acceptance_policy": {
                "max_memory_gb": 9999,
                "minimum_speedup_percent": -50,
            }
        }
    )
    assert out["max_memory_gb"] == 44.0
    assert out["minimum_speedup_percent"] == 1.0


def test_non_numeric_thresholds_are_ignored():
    assert sanitise_acceptance({"acceptance_policy": {"max_memory_gb": "abc"}}) == {}


def test_effective_acceptance_policy_cannot_be_weakened_by_planner():
    resolved = CandidateRegistry.builtin().resolve_all()
    effective, sources = resolve_acceptance_policy(
        {
            "acceptance_policy": {
                "max_memory_gb": 44,
                "minimum_speedup_percent": 1,
            }
        },
        AutonomyPolicy(
            max_peak_memory_gb=40,
            minimum_speedup_percent=10,
            require_all_quality_gates=True,
            require_zero_errors=True,
        ),
        resolved,
        ["baseline", "dflash2_default"],
    )

    assert effective == {
        "quality_must_pass": True,
        "max_memory_gb": 28.0,
        "minimum_speedup_percent": 10.0,
        "max_error_rate": 0.0,
    }
    assert sources["planner"]["minimum_speedup_percent"] == 1.0
    assert sources["effective"] == effective


def test_effective_acceptance_policy_preserves_stricter_planner_gate():
    resolved = CandidateRegistry.builtin().resolve_all()
    effective, _ = resolve_acceptance_policy(
        {
            "acceptance_policy": {
                "max_memory_gb": 20,
                "minimum_speedup_percent": 25,
            }
        },
        AutonomyPolicy(),
        resolved,
        ["baseline", "dflash2_default"],
    )

    assert effective["max_memory_gb"] == 20.0
    assert effective["minimum_speedup_percent"] == 25.0


# --------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------


def test_faster_and_valid_candidate_is_accepted():
    verdict = select(
        make_result("baseline", 18.6), [make_result("dflash2_default", 25.2)]
    )
    assert verdict.accepted is True
    assert verdict.selected_id == "dflash2_default"
    assert verdict.speedup_percent == pytest.approx(35.48, abs=0.1)


def test_fast_but_low_quality_candidate_is_rejected():
    """The central guarantee: speed never overrides correctness."""
    verdict = select(
        make_result("baseline", 18.6),
        [make_result("cheater", 99.0, quality_pass=False)],
    )
    assert verdict.accepted is False
    assert verdict.selected_id == "baseline"


def test_marginal_gain_is_rejected_as_noise():
    verdict = select(make_result("baseline", 18.6), [make_result("meh", 19.1)])
    assert verdict.accepted is False


def test_memory_budget_is_enforced():
    verdict = select(
        make_result("baseline", 18.6), [make_result("fat", 30.0, memory=44.0)]
    )
    assert verdict.accepted is False


def test_missing_memory_measurement_is_not_treated_as_zero():
    candidate = make_result("unknown-memory", 30.0)
    candidate.performance["peak_memory_gb"] = None
    verdict = select(make_result("baseline", 18.6), [candidate])
    assert verdict.accepted is False
    evaluation = next(
        item for item in verdict.evaluations if item["id"] == "unknown-memory"
    )
    assert "peak memory was not measured" in evaluation["disqualifications"]


def test_any_error_disqualifies():
    verdict = select(
        make_result("baseline", 18.6), [make_result("flaky", 30.0, error_rate=0.05)]
    )
    assert verdict.accepted is False


def test_crashed_candidate_leaves_baseline_intact():
    crashed = CandidateResult(id="boom", ok=False, spec={}, error="out of memory")
    verdict = select(make_result("baseline", 18.6), [crashed])
    assert verdict.accepted is False
    assert verdict.selected_id == "baseline"


def test_fastest_qualifying_candidate_wins():
    verdict = select(
        make_result("baseline", 18.6),
        [make_result("a", 25.2), make_result("b", 28.9), make_result("c", 23.8)],
    )
    assert verdict.selected_id == "b"


def test_untrustworthy_baseline_blocks_all_comparison():
    verdict = select(
        make_result("baseline", 18.6, quality_pass=False), [make_result("x", 30.0)]
    )
    assert verdict.accepted is False
    assert "baseline itself did not qualify" in verdict.reason


def test_every_candidate_is_explained():
    verdict = select(
        make_result("baseline", 18.6),
        [make_result("slow", 10.0), make_result("good", 30.0)],
    )
    ids = {e["id"] for e in verdict.evaluations}
    assert ids == {"baseline", "slow", "good"}
    slow = next(e for e in verdict.evaluations if e["id"] == "slow")
    assert slow["disqualifications"]


def test_stricter_policy_is_honoured():
    policy = AcceptancePolicy(minimum_speedup_percent=50.0)
    verdict = select(
        make_result("baseline", 18.6), [make_result("ok_but_not_enough", 25.2)], policy
    )
    assert verdict.accepted is False
