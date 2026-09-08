"""Fixed task suite: quality gate + benchmark prompts.

Two roles:

1. Quality gate  - deterministic pass/fail checks that decide whether a
   candidate configuration is *allowed* to be considered at all.
2. Benchmark set - fixed prompts used for speed measurement.

The gate is intentionally checked by plain Python, never by the model
itself. A candidate that generates faster but fails a check is rejected.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------
# Quality gate checks
# --------------------------------------------------------------------------


def _norm(text: str) -> str:
    return text.strip().lower()


def _strip_think(text: str) -> str:
    """Remove a thinking block if the template emitted one."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _has_replacement_chars(text: str) -> bool:
    """Detect mojibake / decoder breakage."""
    return "\ufffd" in text


def _has_pathological_repetition(text: str, run: int = 12) -> bool:
    """Detect a degenerate loop: same token-ish chunk repeated many times.

    Speculative decoding bugs often surface as runaway repetition, so this
    is a cheap but meaningful guard.
    """
    words = _strip_think(text).split()
    if len(words) < run:
        return False
    streak = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            streak += 1
            if streak >= run:
                return True
        else:
            streak = 1
    # also catch a short phrase repeated verbatim
    for size in (2, 3, 4):
        if len(words) >= size * run:
            for start in range(0, min(len(words) - size * run, 40)):
                chunk = words[start : start + size]
                reps = 1
                idx = start + size
                while (
                    idx + size <= len(words) and words[idx : idx + size] == chunk
                ):
                    reps += 1
                    idx += size
                if reps >= run:
                    return True
    return False


def check_arithmetic(text: str) -> tuple[bool, str]:
    """Deterministic arithmetic: 17 * 23 = 391."""
    body = _strip_think(text)
    if "391" in body.replace(",", ""):
        return True, "found 391"
    return False, "expected 391 in output"


def check_json_schema(text: str) -> tuple[bool, str]:
    """Model must emit a JSON object with the required keys."""
    body = _strip_think(text)
    match = re.search(r"\{.*\}", body, flags=re.DOTALL)
    if not match:
        return False, "no JSON object found"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc.msg}"
    if not isinstance(obj, dict):
        return False, "JSON is not an object"
    missing = [k for k in ("city", "country") if k not in obj]
    if missing:
        return False, f"missing keys: {missing}"
    if "paris" not in _norm(str(obj.get("city", ""))):
        return False, "city should be Paris"
    return True, "valid JSON with required keys"


def check_instruction_following(text: str) -> tuple[bool, str]:
    """Must answer with exactly the single requested word."""
    body = _norm(_strip_think(text)).strip(" .!\"'\n")
    if body == "blue":
        return True, "exact single-word answer"
    if "blue" in body and len(body.split()) <= 6:
        return True, "contains answer, near-exact"
    return False, f"expected 'blue', got {body[:60]!r}"


def check_code_generation(text: str) -> tuple[bool, str]:
    """Validate the requested add function statically without executing model code."""
    body = _strip_think(text)
    block_matches = list(
        re.finditer(r"```(?:python)?\s*(.*?)```", body, flags=re.DOTALL)
    )
    if block_matches:
        if len(block_matches) != 1:
            return False, "output must contain exactly one code block"
        outside = body[: block_matches[0].start()] + body[block_matches[0].end() :]
        if outside.strip():
            return False, "output contains prose outside the code block"
        code = block_matches[0].group(1)
    else:
        code = body
    try:
        tree = ast.parse(code, filename="<candidate>", mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "output must contain exactly one synchronous function"
    function = tree.body[0]
    if function.name != "add" or function.decorator_list:
        return False, "no plain add function definition found"
    positional = [*function.args.posonlyargs, *function.args.args]
    if (
        len(positional) != 2
        or len({argument.arg for argument in positional}) != 2
        or function.args.defaults
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
    ):
        return False, "add must accept exactly two positional arguments"

    statements = list(function.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements.pop(0)
    if len(statements) != 1 or not isinstance(statements[0], ast.Return):
        return False, "add must directly return one expression"
    value = statements[0].value
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return False, "add must return the sum of its two arguments"
    operands = (value.left, value.right)
    argument_names = {argument.arg for argument in positional}
    if not all(isinstance(operand, ast.Name) for operand in operands):
        return False, "add must return the sum of its two arguments"
    if {operand.id for operand in operands} != argument_names:
        return False, "add must return the sum of its two arguments"
    return True, "valid add function with two-argument addition"


@dataclass(frozen=True)
class QualityTask:
    """A single deterministic gate check."""

    id: str
    prompt: str
    checker: Callable[[str], tuple[bool, str]]
    max_tokens: int = 192


QUALITY_TASKS: tuple[QualityTask, ...] = (
    QualityTask(
        id="arithmetic",
        prompt="What is 17 multiplied by 23? Reply with just the number.",
        checker=check_arithmetic,
        max_tokens=96,
    ),
    QualityTask(
        id="json_schema",
        prompt=(
            "Return only a JSON object, no prose and no code fence, with "
            'exactly the keys "city" and "country" for the capital of France.'
        ),
        checker=check_json_schema,
        max_tokens=96,
    ),
    QualityTask(
        id="instruction_following",
        prompt=(
            "Answer with exactly one lowercase word and no punctuation: "
            "what colour is a clear midday sky?"
        ),
        checker=check_instruction_following,
        max_tokens=64,
    ),
    QualityTask(
        id="code_generation",
        prompt=(
            "Write a Python function named add that returns the sum of two "
            "arguments. Reply with only a python code block."
        ),
        checker=check_code_generation,
        max_tokens=192,
    ),
)

# --------------------------------------------------------------------------
# Benchmark prompts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    prompt: str
    max_tokens: int


BENCHMARK_PROMPTS: tuple[BenchmarkPrompt, ...] = (
    BenchmarkPrompt(
        id="code_quicksort",
        prompt="Write a quicksort implementation in Python and briefly explain how it partitions.",
        max_tokens=256,
    ),
    BenchmarkPrompt(
        id="reasoning_math",
        prompt=(
            "A train leaves at 09:15 and arrives at 13:40, stopping twice for "
            "8 minutes each. How long was it moving? Show your reasoning."
        ),
        max_tokens=256,
    ),
    BenchmarkPrompt(
        id="prose_explain",
        prompt="Explain speculative decoding to a backend engineer in one paragraph.",
        max_tokens=256,
    ),
)

WARMUP_PROMPT = BenchmarkPrompt(
    id="warmup", prompt="Say hello in one short sentence.", max_tokens=32
)


def evaluate_quality(outputs: dict[str, str]) -> dict:
    """Run every gate check over candidate outputs.

    `outputs` maps task id -> generated text. Missing or empty output is a
    failure, not a skip.
    """
    results = []
    for task in QUALITY_TASKS:
        text = outputs.get(task.id, "")
        if not text or not text.strip():
            results.append(
                {"id": task.id, "passed": False, "detail": "empty output"}
            )
            continue
        if _has_replacement_chars(text):
            results.append(
                {"id": task.id, "passed": False, "detail": "replacement chars"}
            )
            continue
        if _has_pathological_repetition(text):
            results.append(
                {
                    "id": task.id,
                    "passed": False,
                    "detail": "pathological repetition",
                }
            )
            continue
        passed, detail = task.checker(text)
        results.append({"id": task.id, "passed": passed, "detail": detail})

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "checks": results,
        "passed": passed_count,
        "total": len(QUALITY_TASKS),
        "quality_pass": passed_count == len(QUALITY_TASKS),
    }
