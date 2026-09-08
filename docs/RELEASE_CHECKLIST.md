# v0.1.0 release checklist

This repository is prepared locally. The maintainer performs the final Git commit, push, tag, and GitHub Release.

## Before committing

- [ ] Review every file in `git status` and `git diff`.
- [ ] Run `python scripts/check-release.py`.
- [ ] Run the full test suite once and update all public test-count claims from that single run.
- [ ] Build wheel and sdist, then run `twine check dist/*`.
- [ ] Install the wheel into a fresh virtual environment.
- [ ] Change to a directory outside the repository and run:
  - `infra-team --help`
  - `infra-team registry list --json`
  - `infra-team memory show --json`
  - start the Dashboard on an ephemeral local workspace and verify bundled replay.
- [ ] Verify `python examples/qwen38-dflash2-m5pro/verify_evidence.py`.
- [ ] Confirm no model service or Dashboard process is left running.
- [ ] Confirm no model weights, caches, local `.infra-team/` state, credentials, or generated launchd plist are tracked.

## Apple Silicon installation evidence

- [ ] Install on a clean Apple Silicon Mac without an existing project virtual environment.
- [ ] Record Python, macOS, chip, unified memory, free disk, install duration, and failure points.
- [ ] Verify target download, stable API, Dashboard, one smoke evolution, stop, and restart.
- [ ] Repeat with at least three non-author users before calling the installation path validated.

## GitHub actions performed by the maintainer

```bash
git status
git add --all
git diff --cached --check
git commit -m "Prepare v0.1.0 developer preview"
git push origin main
git tag -a v0.1.0 -m "Virtual AI Infra Team v0.1.0"
git push origin v0.1.0
```

Then create a GitHub Release from `v0.1.0`, attach wheel and sdist, and paste the relevant section from `CHANGELOG.md`.

## GitHub repository metadata

- Repository: `hsj576/virtual-ai-infra-team`
- Description: `An open-source virtual AI Infra team that deploys, tests, optimizes, upgrades, and safely rolls back local AI services.`
- Topics: `local-ai`, `llm-inference`, `ai-infrastructure`, `apple-silicon`, `mlx`, `speculative-decoding`, `model-serving`, `self-hosted`, `ai-agent`, `model-optimization`

## Honest release boundaries

Do not claim universal 2× speedup, arbitrary model support, production SLA, zero downtime, real-world fault injection before it is captured, or validated clean-machine installation before non-author tests finish.
