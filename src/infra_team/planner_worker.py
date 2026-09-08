"""On-demand local fallback planner worker.

The primary planner is the already-running target service. This worker is
started only if failover to a configured small local model is required. It
reads {"target_model", "system_prompt", "context"} as JSON on stdin and
writes a single PLAN_RAW JSON envelope on stdout.

Parsing and validation happen in the parent. This process only produces text,
so a confused fallback model cannot affect anything but its own output.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    spec = json.loads(sys.stdin.read())
    target = spec["target_model"]
    system_prompt = spec["system_prompt"]
    context = spec["context"]

    text = ""
    try:
        from mlx_vlm import load
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        log(f"[planner] loading {target}")
        model, processor = load(target)

        user_msg = (
            "Here is the machine, runtime and measured baseline:\n\n"
            + json.dumps(context, indent=2)
            + "\n\nReturn the ExperimentPlan JSON now."
        )
        prompt = f"{system_prompt}\n\n{user_msg}"

        formatted = apply_chat_template(
            processor, model.config, prompt, num_images=0, enable_thinking=False
        )

        log("[planner] requesting plan")
        result = generate(
            model,
            processor,
            formatted,
            max_tokens=700,
            temperature=0.0,
            enable_thinking=False,
            verbose=False,
        )
        text = getattr(result, "text", "") or ""
        log(f"[planner] response ({len(text)} chars): {text[:400]}")

    except Exception as exc:
        log(f"[planner] FAILED: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())

    print("PLAN_RAW:" + json.dumps({"text": text}), flush=True)


if __name__ == "__main__":
    main()
