"""Candidate runner: loads a configuration, runs the gate and the benchmark.

Runs as a *subprocess* (see runner_worker.py) so that a 16GB model load is
fully released between candidates. This keeps peak-memory measurements
honest and prevents one candidate from inheriting another's warm cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

WORKER = os.path.join(os.path.dirname(__file__), "runner_worker.py")


@dataclass
class CandidateSpec:
    """A validated, executable candidate configuration."""

    id: str
    target_model: str
    draft_model: str | None = None
    draft_kind: str | None = None
    draft_block_size: int | None = None
    temperature: float = 0.0
    enable_thinking: bool = False
    repeats: int = 3
    target_revision: str | None = None
    candidate_manifest_id: str | None = None
    candidate_manifest_hash: str | None = None
    runtime_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateResult:
    """Everything measured for one candidate."""

    id: str
    ok: bool
    spec: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    load_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- convenience accessors used by the selector -----------------------
    @property
    def quality_pass(self) -> bool:
        return bool(self.quality.get("quality_pass"))

    @property
    def generation_tps(self) -> float:
        return float(self.performance.get("generation_tps_median") or 0.0)

    @property
    def peak_memory_gb(self) -> float:
        return float(self.performance.get("peak_memory_gb") or 0.0)

    @property
    def error_rate(self) -> float:
        return float(self.performance.get("error_rate") or 0.0)


def run_candidate(
    spec: CandidateSpec,
    python_executable: str | None = None,
    timeout: int = 3600,
    log_path: str | None = None,
) -> CandidateResult:
    """Execute one candidate in a clean subprocess and parse its report.

    Isolation is deliberate: the model is loaded and torn down inside the
    child, so peak memory belongs to this candidate alone.
    """
    python_executable = python_executable or sys.executable
    payload = json.dumps(spec.to_dict())

    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        proc = subprocess.run(
            [python_executable, WORKER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return CandidateResult(
            id=spec.id,
            ok=False,
            spec=spec.to_dict(),
            error=f"candidate timed out after {timeout}s",
        )

    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== candidate {spec.id} =====\n")
            fh.write(proc.stderr or "")

    # The worker prints exactly one JSON report line prefixed with REPORT:
    report = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("REPORT:"):
            try:
                report = json.loads(line[len("REPORT:") :])
            except json.JSONDecodeError:
                report = None

    if report is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        return CandidateResult(
            id=spec.id,
            ok=False,
            spec=spec.to_dict(),
            error="worker produced no report: " + " | ".join(tail),
        )

    if not report.get("ok"):
        return CandidateResult(
            id=spec.id,
            ok=False,
            spec=spec.to_dict(),
            error=report.get("error", "unknown worker error"),
            quality=report.get("quality", {}),
            performance=report.get("performance", {}),
        )

    return CandidateResult(
        id=spec.id,
        ok=True,
        spec=spec.to_dict(),
        quality=report.get("quality", {}),
        performance=report.get("performance", {}),
        load_seconds=report.get("load_seconds"),
    )
