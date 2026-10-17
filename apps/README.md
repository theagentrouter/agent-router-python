# Apps

Two small apps written against LiteLLM, kept here to test this SDK the way a user meets it: the same code, run on real LiteLLM and on this SDK.

| App | What it is |
|---|---|
| [`adk-support-agent/`](adk-support-agent/) | A Google ADK agent: ADK's `LiteLlm` adapter, MCP ticket tools over stdio, a local knowledge base, saved reports and ticket notes. The migration is one call in its `__init__.py` |
| [`litellm-assistant/`](litellm-assistant/) | Plain `import litellm` code with no framework: questions, streaming, async streaming, a tool round and embeddings |

Each app works on either SDK, installed beside it, never both in one environment. Tests name the SDK with `APP_SDK` and stop when the environment holds the other one.

## Tests

```sh
make apps                                   # both apps, offline against tests/fake_provider.py, on LiteLLM and on this SDK
make apps SDK=theagentrouter APP=litellm-assistant
make apps-live                              # against real OpenAI and Anthropic; keys from the environment or .env
```

`make apps` keeps one virtual environment per SDK (`.venv-apps-litellm`, `.venv-apps-theagentrouter`). LiteLLM is pinned in each app's `litellm` extra.

`litellm-assistant/tests/test_litellm_behaviour.py` states LiteLLM's behaviour, one test per behaviour: stream chunks, `include_usage`, Claude's parameters and `drop_params`, error classes, callbacks, `mock_response`, costs, LiteLLM's modules. On LiteLLM a run proves the expectation; on this SDK it proves the drop-in. Where this SDK still differs, the test is marked `known_gap`: a strict expected failure, so a fix makes it fail until the mark is removed.

`litellm-assistant/tests/test_agent_router_gateway.py` runs only on this SDK: the app through the real local gateway, who can call that gateway, which gateway a stored key goes to, and what one program's routes or model names do to the others.

CI runs both apps on both SDKs on every push (`ci.yml`, job `apps`), and the gateway tests against the built local gateway on every platform (`local-gateway-build.yml`), where a skipped test fails the job.

## Adding a test

Write what LiteLLM does, and run it on LiteLLM first (`make apps SDK=litellm`). If it fails there, the expectation is wrong. Then run it on this SDK; if it fails and the fix is not part of the change, mark it `known_gap("...")` with the reason.
