# theagentrouter-core

The SDK of [theagentrouter](https://pypi.org/project/theagentrouter/) without the local gateway: LiteLLM's API, with every call going through an Agent Router gateway.

Install it on a server, container, serverless function or CI that uses a remote gateway:

```sh
pip install theagentrouter-core
export AGENTROUTER_BASE_URL=https://gateway.acme.example/v1
export AGENTROUTER_API_KEY=...
```

On a developer computer, install `theagentrouter` instead; it adds the local gateway. Both import as `import theagentrouter`.

The documentation and examples are on [GitHub](https://github.com/theagentrouter/agent-router-python).
