# Security Policy

## Supported version

Virtual AI Infra Team is experimental developer-preview software. Security fixes are applied to the latest `main` branch and the latest tagged release.

## Reporting a vulnerability

Please do not open a public issue for vulnerabilities involving process control, credential exposure, path traversal, candidate integrity, model download trust, or rollback safety.

Report privately through GitHub Security Advisories for this repository. Include:

- affected version or commit;
- operating system and Python version;
- reproduction steps;
- expected and actual behavior;
- whether a model service or local data was exposed or modified.

Do not include real API keys, access tokens, private model outputs, or personal paths in the report.

## Security boundaries

The v0.1 design enforces these boundaries:

- Dashboard binds to loopback only.
- Mutating requests require same-origin checks, an unguessable session token, and explicit confirmation.
- Model or planner output is never executed as shell.
- Candidate manifests select only code-owned launch templates.
- Candidate repositories and the target model use fixed revisions.
- Candidate assets are verified before becoming `READY`.
- A process can be stopped only through its signed private control channel.
- Unknown processes occupying a port are left untouched.
- Optimization and evolution share a workspace file lock.
- Promotion requires all quality gates, resource limits, zero request errors, service health, and online verification.
- Failed promotion attempts restore the previous known-good recipe when possible.
- Evidence replay is read-only and redacts control capabilities and absolute user paths.

## Out of scope for v0.1

This release does not provide a production SLA, multi-user authentication, public-network deployment, arbitrary remote Registry execution, enterprise tenancy, or server-grade high availability. Do not expose the Dashboard or model control endpoints to untrusted networks.
