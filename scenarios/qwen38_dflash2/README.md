# Scenario: Qwen3.8-27B + DFlash2 on Apple Silicon

A reproducible case for the first job on the virtual team: **local inference
performance optimization**.

## Question

Does DFlash2 speculative decoding actually make `Qwen3.8-27B-4bit` faster on an
Apple Silicon Mac — and if so, at which draft block size?

This is genuinely open. The published DFlash2 numbers come from an NVIDIA H200
with FlashAttention 3 at up to 3.4x. None of that transfers automatically to a
unified-memory M-series GPU, where the drafter's weights compete with the target
model for the same memory pool and the verification step is shaped by GPU
threadgroup occupancy rather than tensor-core throughput.

So the honest position before running: **unknown**.

## Setup

| | |
|---|---|
| Target | `mlx-community/Qwen3.8-27B-4bit` (16.1 GB) |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB, native block size 8) |
| Engine | MLX-VLM |
| Decoding | greedy, `temperature=0` |

Greedy decoding matters: DFlash-family drafting is documented as lossless under
greedy sampling, so a correct implementation should not change what the model
says — only how fast it says it. That makes any quality-gate failure a real
signal rather than expected sampling drift.

## Candidates

| id | configuration |
|---|---|
| `baseline` | plain autoregressive, no drafter |
| `dflash2_default` | DFlash2 at its trained block size (8) |
| `dflash2_block4` | shorter block — less wasted work per rejection |
| `dflash2_block6` | intermediate block |

## Method

1. Probe the machine and record runtime versions.
2. Measure the baseline: 3 benchmark prompts x 3 repeats, after a warmup.
3. Ask the target model which candidates to test.
4. Validate the plan against the whitelist.
5. Run each candidate in an isolated subprocess.
6. Judge quality with the deterministic gate.
7. Select the fastest candidate that passes; otherwise keep the baseline.

Every candidate is measured with the same prompts, the same token limits, the
same temperature, and a discarded warmup pass. The reported speed is the median
across runs, with min/max/stdev retained in the artifacts so noise is visible.

## Acceptance

A candidate is accepted only if it passes every quality check, produces no
errors, stays under 40 GB peak memory, and is at least 5% faster than baseline.
The 5% floor exists so that measurement noise cannot be reported as a win.

## Results

Run `infra-team optimize` to generate them. Numbers are written to
`.infra-team/runs/<run-id>/` and are never committed as literals in this
document — the point of the project is that the figures come from the machine
in front of you, not from a README.

## Interpreting a rejection

If no candidate clears the threshold, the run reports `OPTIMIZATION REJECTED`
and retains the baseline. That is a real finding about this hardware, not a
failure of the tool. A system that cannot say "no" cannot be trusted when it
says "yes".
