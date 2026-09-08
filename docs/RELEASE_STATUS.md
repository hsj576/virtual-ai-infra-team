# v0.1.0 local release-candidate status

Updated: 2026-09-08

## Passed locally

- 167 automated tests.
- Python source compilation.
- Dashboard JavaScript syntax check.
- Default Qwen target resolves its fixed Hugging Face commit to a concrete local snapshot before service launch.
- Evidence-bundle recomputation: 18.14 → 37.52 tok/s (+106.84%), quality 4/4.
- Release-tree scan for model weights, large files, common credentials, private keys, and absolute macOS user paths.
- Wheel and sdist build.
- `twine check` for both artifacts.
- Wheel installation with dependencies into a new virtual environment.
- From a directory outside the repository:
  - `infra-team --version`;
  - `infra-team registry list --json` using the packaged default policy;
  - `infra-team memory show --json`;
  - Dashboard API loading the packaged historical evidence replay.
- Desktop and mobile Dashboard browser checks were completed in the source project before extraction.

## Intentionally not performed

- No Git commit, push, tag, or GitHub Release was created by the preparation workflow.
- No launchd task was installed.
- No model or Dashboard process was left running.

## Still requires maintainer or external-user validation

- Review the complete Git diff and commit history.
- Push to `hsj576/virtual-ai-infra-team` and create tag `v0.1.0`.
- Run GitHub Actions on the remote repository.
- Complete a clean Apple Silicon installation by non-author users.
- Capture public real fault-injection and rollback evidence.
- Produce the 90-second demo and uncut benchmark recording.
