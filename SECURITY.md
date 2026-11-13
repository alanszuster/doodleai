# Security policy

## Reporting a vulnerability

Please report security problems **privately** through GitHub: open the repository's *Security* tab and
choose *Report a vulnerability*. Do not open a public issue for something that could be exploited.

You can expect an answer within a few days. This is a personal project, so there is no formal SLA,
but reports are taken seriously and fixed first.

## What is in scope

The API in `src/` and its container image, for example:

- bypassing authentication or the per-caller rate limits,
- requests that crash the service or exhaust its memory or CPU (crafted images, large bodies),
- leaking internal details, credentials or other callers' data,
- a vulnerable dependency or base image that the project does not yet track.

Only the latest version on `main` is supported.

## How the project protects itself

Authentication uses short-lived OIDC tokens (no stored secret), requests are size- and
format-checked before decoding, the container runs as a non-root user, and CI scans the code
(Bandit), the locked dependencies (`pip-audit`) and the image (Trivy) on every change.
