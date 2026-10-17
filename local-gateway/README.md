# theagentrouter-local-gateway

Agent Router's `aigw` program, packaged for [theagentrouter](https://github.com/theagentrouter/agent-router-python)'s local gateway. `pip install theagentrouter` installs it with the SDK on macOS 13+ (Apple chips) and Linux:

```sh
pip install theagentrouter
```

The SDK starts the gateway on the first call. Nothing here is run by hand.

Wheels are built for macOS 13+ arm64 and Linux glibc 2.36+ amd64/arm64, plus a `py3-none-any` fallback wheel with no program (`AGENTROUTER_NO_BINARY_WHEEL=1 python -m build --wheel local-gateway`), which pip installs everywhere else so `pip install theagentrouter` never fails. There are no Windows or Intel macOS builds. The unreleased server-authentication fix needs the [reviewed source patch](patches/README.md). Packaging CI builds the pinned patched source and runs the gateway integration tests. The published a4 wheels still contain the unmodified upstream binary identified by `aigw-release.json`.

To reproduce a release wheel on a supported platform:

```sh
python scripts/build_review_gateway.py
python -m build --wheel local-gateway
```
