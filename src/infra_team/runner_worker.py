"""Worker process: loads one configuration, measures it, reports JSON.

Runs in its own process so the model is fully released on exit. Reads a
CandidateSpec as JSON on stdin, writes a single line to stdout:

    REPORT:{...}

All human-readable logging goes to stderr so stdout stays parseable.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra_team.tasks import (  # noqa: E402
    BENCHMARK_PROMPTS,
    QUALITY_TASKS,
    WARMUP_PROMPT,
    evaluate_quality,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _clear_cache(mx) -> None:
    for fn in ("clear_cache", "reset_peak_memory"):
        f = getattr(mx, fn, None) or getattr(getattr(mx, "metal", object), fn, None)
        if callable(f):
            try:
                f()
            except Exception:
                pass


def _peak_memory_gb(mx) -> float | None:
    for holder in (mx, getattr(mx, "metal", None)):
        if holder is None:
            continue
        fn = getattr(holder, "get_peak_memory", None)
        if callable(fn):
            try:
                return round(fn() / (1024**3), 3)
            except Exception:
                continue
    return None


def main() -> None:
    raw = sys.stdin.read()
    spec = json.loads(raw)

    report: dict = {"ok": False, "id": spec.get("id")}

    try:
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        target = spec["target_model"]
        draft_model_path = spec.get("draft_model")
        draft_kind = spec.get("draft_kind")
        draft_block_size = spec.get("draft_block_size")
        temperature = float(spec.get("temperature", 0.0))
        enable_thinking = bool(spec.get("enable_thinking", False))
        repeats = int(spec.get("repeats", 3))

        _clear_cache(mx)

        t0 = time.time()
        log(f"[{spec['id']}] loading target {target}")
        model, processor = load(target)
        config = model.config

        draft_model = None
        resolved_kind = None
        if draft_model_path:
            from mlx_vlm.speculative.drafters import load_drafter

            log(f"[{spec['id']}] loading drafter {draft_model_path}")
            draft_model, resolved_kind = load_drafter(
                draft_model_path, kind=draft_kind
            )
            if draft_block_size:
                try:
                    draft_model.config.block_size = int(draft_block_size)
                except Exception as exc:  # pragma: no cover
                    log(f"[{spec['id']}] block_size override failed: {exc}")
        load_seconds = round(time.time() - t0, 2)
        report["load_seconds"] = load_seconds
        log(f"[{spec['id']}] loaded in {load_seconds}s kind={resolved_kind}")

        def run_once(prompt_text: str, max_tokens: int) -> dict:
            """One generation. Returns text + metrics."""
            formatted = apply_chat_template(
                processor,
                config,
                prompt_text,
                num_images=0,
                enable_thinking=enable_thinking,
            )
            kwargs: dict = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "enable_thinking": enable_thinking,
            }
            if draft_model is not None:
                kwargs["draft_model"] = draft_model
                kwargs["draft_kind"] = resolved_kind

            started = time.time()
            result = generate(model, processor, formatted, verbose=False, **kwargs)
            wall = time.time() - started

            text = getattr(result, "text", "") or ""
            return {
                "text": text,
                "generation_tps": getattr(result, "generation_tps", None),
                "prompt_tps": getattr(result, "prompt_tps", None),
                "generation_tokens": getattr(result, "generation_tokens", None),
                "prompt_tokens": getattr(result, "prompt_tokens", None),
                "wall_seconds": round(wall, 3),
                "finish_reason": getattr(result, "finish_reason", None),
            }

        # ---------------- warmup (never measured) ----------------
        log(f"[{spec['id']}] warmup")
        run_once(WARMUP_PROMPT.prompt, WARMUP_PROMPT.max_tokens)

        # ---------------- quality gate ----------------
        outputs: dict[str, str] = {}
        quality_errors = 0
        for task in QUALITY_TASKS:
            try:
                res = run_once(task.prompt, task.max_tokens)
                outputs[task.id] = res["text"]
                log(f"[{spec['id']}] gate {task.id}: {res['text'][:70]!r}")
            except Exception as exc:
                quality_errors += 1
                outputs[task.id] = ""
                log(f"[{spec['id']}] gate {task.id} FAILED: {exc}")

        quality = evaluate_quality(outputs)
        quality["raw_outputs"] = outputs
        report["quality"] = quality

        # ---------------- benchmark ----------------
        _clear_cache(mx)
        per_prompt: dict[str, list[dict]] = {}
        errors = quality_errors
        attempts = len(QUALITY_TASKS)

        for bp in BENCHMARK_PROMPTS:
            runs: list[dict] = []
            for i in range(repeats):
                attempts += 1
                try:
                    res = run_once(bp.prompt, bp.max_tokens)
                    runs.append(res)
                    log(
                        f"[{spec['id']}] bench {bp.id} #{i + 1}: "
                        f"{res['generation_tps']} tok/s "
                        f"({res['generation_tokens']} tok)"
                    )
                except Exception as exc:
                    errors += 1
                    log(f"[{spec['id']}] bench {bp.id} #{i + 1} FAILED: {exc}")
            per_prompt[bp.id] = runs

        all_tps = [
            r["generation_tps"]
            for runs in per_prompt.values()
            for r in runs
            if r.get("generation_tps")
        ]
        all_prompt_tps = [
            r["prompt_tps"]
            for runs in per_prompt.values()
            for r in runs
            if r.get("prompt_tps")
        ]
        all_wall = [
            r["wall_seconds"] for runs in per_prompt.values() for r in runs
        ]
        total_gen_tokens = sum(
            r.get("generation_tokens") or 0
            for runs in per_prompt.values()
            for r in runs
        )

        if not all_tps:
            report["ok"] = False
            report["error"] = "no successful benchmark runs"
            report["performance"] = {"error_rate": 1.0}
            print("REPORT:" + json.dumps(report), flush=True)
            return

        performance = {
            "generation_tps_median": round(statistics.median(all_tps), 2),
            "generation_tps_mean": round(statistics.fmean(all_tps), 2),
            "generation_tps_min": round(min(all_tps), 2),
            "generation_tps_max": round(max(all_tps), 2),
            "generation_tps_stdev": (
                round(statistics.stdev(all_tps), 3) if len(all_tps) > 1 else 0.0
            ),
            "prompt_tps_median": (
                round(statistics.median(all_prompt_tps), 2)
                if all_prompt_tps
                else None
            ),
            "wall_seconds_median": round(statistics.median(all_wall), 3),
            "total_generation_tokens": total_gen_tokens,
            "samples": len(all_tps),
            "attempts": attempts,
            "errors": errors,
            "error_rate": round(errors / attempts, 4) if attempts else 0.0,
            "peak_memory_gb": _peak_memory_gb(mx),
            "per_prompt": {
                pid: [
                    {
                        "generation_tps": r.get("generation_tps"),
                        "generation_tokens": r.get("generation_tokens"),
                        "wall_seconds": r.get("wall_seconds"),
                        "finish_reason": r.get("finish_reason"),
                    }
                    for r in runs
                ]
                for pid, runs in per_prompt.items()
            },
            "benchmark_texts": {
                pid: (runs[0]["text"] if runs else "")
                for pid, runs in per_prompt.items()
            },
        }
        report["performance"] = performance
        report["resolved_draft_kind"] = resolved_kind
        report["ok"] = True

    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        log(traceback.format_exc())

    print("REPORT:" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
