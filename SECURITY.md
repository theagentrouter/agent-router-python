# Security

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting: the **Security** tab of this repository, then **Report a vulnerability**.

Include the version, what an attacker could do, and steps to reproduce. We will reply to confirm we received it.

## Supported versions

Only the latest release gets security fixes while the SDK is below 1.0.

## How releases are protected

- The package has one runtime dependency, `httpx`.
- Releases are built and published by GitHub Actions with PyPI trusted publishing. No PyPI token is stored in the repository or on a laptop.
- Release tags are created by maintainers only.

## What the SDK does with your keys

Provider keys are read from the environment, from `api_key` on a call, or from `AGENTROUTER_HOME/config.json`. They are sent only to the address the call goes to: the provider, `api_base`, or the gateway set in `AGENTROUTER_BASE_URL`. The SDK does not log them and sends no telemetry.
