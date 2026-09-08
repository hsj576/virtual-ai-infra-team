#!/usr/bin/env python3
"""Recompute the published headline values from the sanitized evidence bundle."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    samples = json.loads((ROOT / "benchmark_samples.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in samples["rows"]}

    for row in rows.values():
        flattened = [
            sample["generation_tps"]
            for prompt_samples in row["per_prompt"].values()
            for sample in prompt_samples
        ]
        assert len(flattened) == row["samples"] == 9
        assert round(statistics.median(flattened), 2) == row["generation_tps_median"]

    baseline = rows["baseline"]["generation_tps_median"]
    selected = rows[summary["selected_id"]]["generation_tps_median"]
    speedup = round((selected / baseline - 1) * 100, 2)
    assert baseline == summary["baseline_tps"]
    assert selected == summary["selected_tps"]
    assert speedup == summary["speedup_percent"]
    print(
        f"verified: {baseline:.2f} -> {selected:.2f} tok/s "
        f"(+{speedup:.2f}%), quality {summary['quality']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
