# theagentrouter

LiteLLM's API with Agent Router underneath. This package has no code of its own: it installs

- `theagentrouter-core`, the SDK (`import theagentrouter`), and
- `theagentrouter-local-gateway`, Agent Router's `aigw` program, on macOS (Apple chips) and Linux (x86-64, Arm).

```sh
pip install theagentrouter          # a developer computer: the SDK and the local gateway
pip install theagentrouter-core     # a server, container, serverless function or CI using a remote gateway
```

Both import as `import theagentrouter`. Documentation: [agent-router-python](https://github.com/theagentrouter/agent-router-python).
