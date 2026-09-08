# Changelog

All notable changes to this project are documented here.

## [Unreleased]

- Clean-machine installation testing and external-user validation.
- Public real fault-injection and recovery evidence.
- p95 latency, service-switch interruption, acceptance-rate, and cold-restart measurements.

## [0.1.0] - Unreleased

### Added

- Stable OpenAI-compatible local model service with signed lifecycle control.
- Target-first experiment planning with deterministic policy enforcement.
- Strict 4/4 quality gates, performance/resource gates, promotion, online verification, and rollback.
- Trusted local candidate Registry with fixed revisions and code-owned launch templates.
- Candidate Store with download budgets, SHA-256 inventory, atomic `READY`, and tamper detection.
- `evolve --once` discovery-to-memory outer loop.
- SQLite Recipe Memory and environment fingerprints.
- `evolve --watch` with maintenance windows, persistent backoff, deduplication, and notifications.
- Loopback-only Dashboard with full evolution control, real-time chat streaming, Watch/Memory summaries, and read-only artifact replay.
- Sanitized M5 Pro + Qwen3.8-27B + DFlash2 evidence bundle.
- macOS bootstrap/start/stop scripts and GitHub Actions release checks.

### Verified

- Baseline 18.14 generation tok/s.
- DFlash2 native 37.52 generation tok/s (+106.84%).
- Quality 4/4 before and after promotion.
- Stable API retained.
- 167 automated tests passed in the local release-candidate source tree.

[Unreleased]: https://github.com/hsj576/virtual-ai-infra-team/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hsj576/virtual-ai-infra-team/releases/tag/v0.1.0
