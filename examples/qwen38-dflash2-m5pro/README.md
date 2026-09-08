# Qwen3.8-27B + DFlash2 on Apple M5 Pro

This directory is a sanitized, read-only evidence bundle from a real local evolution run.
It is historical evidence, not a live benchmark and not a universal performance claim.

## Result

| Configuration | Generation TPS median | TTFT median | Peak memory | Quality | vs. baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 18.14 | 0.230 s | 16.449 GB | 4/4 | reference |
| DFlash2 block 6 | 35.31 | 0.194 s | 21.876 GB | 4/4 | +94.65% |
| DFlash2 native block 8 | 37.52 | 0.189 s | 21.725 GB | 4/4 | +106.84% |

The selected recipe passed online quality verification after restart while keeping the same OpenAI-compatible endpoint.

## Files

- `summary.json`: compact outcome and scope.
- `environment.redacted.json`: hardware and runtime fingerprint without user paths.
- `benchmark_samples.json`: raw numeric samples sufficient to recompute medians and speedup.
- `quality_results.json`: gate outcomes without raw generated text.
- `promotion_event.json`: selected recipe, fixed revisions and online verification.
- `evolution_events.ndjson`: sanitized outer-loop timeline.
- `verify_evidence.py`: recomputes the published headline values.

## Scope and limitations

- Hardware: Apple M5 Pro, 48 GB unified memory.
- Target: `mlx-community/Qwen3.8-27B-4bit`, fixed commit `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2`, fixed commit `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- Runtime: MLX 0.32.2, MLX-VLM 0.6.16, MLX-LM 0.31.3.
- Workload: three prompts, three repeats per prompt, nine performance samples per configuration.
- These results do not imply that every Mac, model or workload will obtain a similar speedup.
- Peak memory increased by 5.276 GB for the selected candidate.
- A public real fault-injection capture is still pending; rollback behavior is covered by the automated test suite but is not represented here as real-world evidence.

Verify the bundle:

```bash
python examples/qwen38-dflash2-m5pro/verify_evidence.py
```
