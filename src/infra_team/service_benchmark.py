"""Benchmark an already-running OpenAI-compatible MLX-VLM service.

This is the baseline path for the product-shaped demo: the user's model is
already serving a stable local API, so we measure that service rather than
loading a duplicate 27B model in another process. MLX-VLM's /v1/metrics
endpoint supplies decode throughput, TTFT and peak-memory measurements.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

from .runner import CandidateResult, CandidateSpec
from .tasks import BENCHMARK_PROMPTS, QUALITY_TASKS, WARMUP_PROMPT, evaluate_quality


class ServiceBenchmarkError(RuntimeError):
    pass


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ServiceBenchmarkError(f"GET {url} failed: {exc}") from exc


def _chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "enable_thinking": False,
    }
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ServiceBenchmarkError(f"POST {url} failed: {exc}") from exc
    elapsed = time.time() - started
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServiceBenchmarkError("chat response missing message content") from exc
    if not isinstance(content, str):
        raise ServiceBenchmarkError("chat response content is not text")
    return content, elapsed


def _latest_metrics(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        url = base + "/metrics"
    else:
        url = base + "/v1/metrics"
    payload = _get_json(url)
    latest = payload.get("latest")
    if not isinstance(latest, dict):
        raise ServiceBenchmarkError("metrics endpoint has no latest request")
    return latest


def quality_gate_service(
    base_url: str,
    model: str,
    *,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Run the strict 4/4 gate against a live service."""
    outputs: dict[str, str] = {}
    request_errors = 0
    for task in QUALITY_TASKS:
        try:
            text, _ = _chat(
                base_url, model, task.prompt, task.max_tokens, 0.0, timeout
            )
            outputs[task.id] = text
        except ServiceBenchmarkError:
            request_errors += 1
            outputs[task.id] = ""
    quality = evaluate_quality(outputs)
    quality["raw_outputs"] = outputs
    quality["request_errors"] = request_errors
    return quality


def benchmark_service(
    base_url: str,
    model: str,
    *,
    repeats: int = 3,
    timeout: float = 900.0,
) -> CandidateResult:
    """Measure quality and performance through the deployed service API."""
    spec = CandidateSpec(id="baseline", target_model=model, repeats=repeats)
    attempts = 0
    errors = 0

    try:
        _chat(
            base_url,
            model,
            WARMUP_PROMPT.prompt,
            WARMUP_PROMPT.max_tokens,
            0.0,
            timeout,
        )
    except ServiceBenchmarkError as exc:
        return CandidateResult(
            id="baseline",
            ok=False,
            spec=spec.to_dict(),
            error=f"service warmup failed: {exc}",
        )

    quality = quality_gate_service(base_url, model, timeout=timeout)
    attempts += len(QUALITY_TASKS)
    errors += int(quality.get("request_errors") or 0)

    samples: list[dict[str, Any]] = []
    per_prompt: dict[str, list[dict[str, Any]]] = {}
    for bp in BENCHMARK_PROMPTS:
        runs: list[dict[str, Any]] = []
        for _ in range(repeats):
            attempts += 1
            request_started = time.time()
            try:
                text, client_elapsed = _chat(
                    base_url, model, bp.prompt, bp.max_tokens, 0.0, timeout
                )
                metrics = _latest_metrics(base_url)
                timestamp = float(metrics.get("timestamp_unix") or 0)
                if timestamp + 1 < request_started:
                    raise ServiceBenchmarkError("metrics latest entry is stale")
                sample = {
                    "generation_tps": metrics.get("decode_tok_s"),
                    "prompt_tps": metrics.get("prefill_tok_s"),
                    "generation_tokens": metrics.get("generated_tokens"),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "wall_seconds": round(client_elapsed, 3),
                    "request_elapsed_s": metrics.get("request_elapsed_s"),
                    "ttft_s": metrics.get("ttft_s"),
                    "peak_memory_gb": metrics.get("peak_memory_gb"),
                    "finish_reason": metrics.get("finish_reason"),
                    "text": text,
                }
                if not sample["generation_tps"]:
                    raise ServiceBenchmarkError("metrics missing decode_tok_s")
                samples.append(sample)
                runs.append(sample)
            except ServiceBenchmarkError:
                errors += 1
        per_prompt[bp.id] = runs

    if not samples:
        return CandidateResult(
            id="baseline",
            ok=False,
            spec=spec.to_dict(),
            quality=quality,
            performance={"error_rate": 1.0},
            error="no successful API benchmark samples",
        )

    tps = [float(s["generation_tps"]) for s in samples]
    prompt_tps = [float(s["prompt_tps"]) for s in samples if s.get("prompt_tps")]
    wall = [float(s["wall_seconds"]) for s in samples]
    ttft = [float(s["ttft_s"]) for s in samples if s.get("ttft_s") is not None]
    memory = [
        float(s["peak_memory_gb"])
        for s in samples
        if s.get("peak_memory_gb") is not None
    ]
    performance = {
        "generation_tps_median": round(statistics.median(tps), 2),
        "generation_tps_mean": round(statistics.fmean(tps), 2),
        "generation_tps_min": round(min(tps), 2),
        "generation_tps_max": round(max(tps), 2),
        "generation_tps_stdev": (
            round(statistics.stdev(tps), 3) if len(tps) > 1 else 0.0
        ),
        "prompt_tps_median": (
            round(statistics.median(prompt_tps), 2) if prompt_tps else None
        ),
        "wall_seconds_median": round(statistics.median(wall), 3),
        "ttft_seconds_median": round(statistics.median(ttft), 3) if ttft else None,
        "peak_memory_gb": round(max(memory), 3) if memory else None,
        "samples": len(samples),
        "attempts": attempts,
        "errors": errors,
        "error_rate": round(errors / attempts, 4) if attempts else 0.0,
        "source": "deployed_openai_compatible_service",
        "per_prompt": {
            pid: [
                {k: v for k, v in run.items() if k != "text"}
                for run in runs
            ]
            for pid, runs in per_prompt.items()
        },
        "benchmark_texts": {
            pid: runs[0]["text"] if runs else "" for pid, runs in per_prompt.items()
        },
    }
    return CandidateResult(
        id="baseline",
        ok=True,
        spec=spec.to_dict(),
        quality=quality,
        performance=performance,
    )
