# Contributing

Thank you for improving Virtual AI Infra Team.

## Before opening a pull request

1. Discuss substantial new runtimes, candidate kinds, policy changes, or service-control behavior in an issue first.
2. Keep planner output untrusted. New behavior must map to code-owned schemas and fixed capabilities.
3. Do not commit model weights, caches, access tokens, local service state, raw private prompts, or absolute user paths.
4. Add tests before changing safety, quality, promotion, rollback, Memory, Watch, or Dashboard semantics.
5. Do not weaken the 4/4 quality gate or claim performance from a smoke run.

## Development setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -e '.[dev]'
./.venv/bin/python -m pytest tests -q
```

The real model benchmark is intentionally not part of ordinary CI. Unit and integration tests use fixtures and fake services. Hardware results must be accompanied by a sanitized evidence bundle and precise environment scope.

## Pull request checklist

- [ ] Tests added or updated.
- [ ] Full test suite passes.
- [ ] `python -m compileall -q src` passes.
- [ ] Wheel and sdist build successfully.
- [ ] Wheel installs and runs outside the source checkout.
- [ ] No secret, cache, model weight, generated launchd file, or absolute personal path is included.
- [ ] User-visible behavior and limitations are documented.
- [ ] Performance claims state hardware, revisions, runtime, workload, sample count and variance.

## Candidate contributions

A candidate manifest must use a fixed repository commit, approved license, `allow_remote_code: false`, a supported target/runtime/platform, bounded resources, and an existing code-owned launch template. A manifest cannot contain executable commands.

## Code style

Prefer small deterministic functions, standard-library components, explicit schemas, and auditable artifacts. Avoid adding dependencies unless they materially improve correctness or installation.
