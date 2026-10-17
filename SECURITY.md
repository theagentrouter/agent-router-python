# Security

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting: the **Security** tab of this repository, then **Report a vulnerability**. Or email [TODO: add the security contact address].

Include the version, what an attacker could do, and steps to reproduce. We will reply to confirm we received it.

## Supported versions

Only the latest release gets security fixes while the SDK is below 1.0.

## How releases are protected

- The package has one runtime dependency, `httpx`.
- Releases are built and published by GitHub Actions with PyPI trusted publishing. No PyPI token is stored in the repository or on a laptop.
- Release tags are created by maintainers only.

## What the SDK does with your keys

The SDK never calls a provider itself. Provider keys are read from the environment and given to the local gateway in private files (mode 0600, removed when it stops); the gateway sends each key only to its own provider, and its config holds no key. A gateway key, from `api_key` on a call, `AGENTROUTER_API_KEY` or `AGENTROUTER_HOME/config.json`, is sent only to the gateway the call goes to: `api_base`, the gateway set in `AGENTROUTER_BASE_URL`, or the gateway stored with that key. The SDK does not log keys and sends no telemetry.
