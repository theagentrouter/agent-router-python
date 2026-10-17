# AGENTS.md

Instructions for coding agents (and people) working in this repository. `CLAUDE.md` imports this file.

## What this is

`theagentrouter` (imported as `theagentrouter`) is an open-source Python library. Agent frameworks use it to call AI models. It keeps LiteLLM's way of calling models (`completion`, `provider/model` names, response fields, error names) in a small install whose only dependency is `httpx`.

It never calls a provider itself: every call goes to an Agent Router gateway, the local one or one set with `AGENTROUTER_BASE_URL`, and both get the same requests. Claude goes in Anthropic Messages, everything else in OpenAI Chat Completions, with the model name as the caller wrote it.

## Layout

| Path | What it holds |
|---|---|
| `src/theagentrouter/__init__.py` | Public API: `completion`, `acompletion`, model lookups |
| `src/theagentrouter/routing.py` | Where a call goes: `api_base`, a gateway you set, or the local gateway. Never a provider directly; settings from the environment and `AGENTROUTER_HOME/config.json` |
| `src/theagentrouter/routes.py` | Routes: names of your own with fallbacks and retries (`route`, `Retry`), stored in `AGENTROUTER_HOME/routes.json` and served by the local gateway |
| `src/theagentrouter/local_gateway.py` | The local gateway: which providers the environment has, the `aigw` config, starting, finding and stopping `aigw` |
| `local-gateway/` | The `theagentrouter-local-gateway` package: the `aigw` program, one wheel per platform |
| `packages/theagentrouter/` | The `theagentrouter` package: no code; installs `theagentrouter-core` (the root package) and the local gateway |
| `src/theagentrouter/openai_format.py` | Build OpenAI requests; read responses and SSE streams |
| `src/theagentrouter/anthropic_format.py` | Convert OpenAI-style messages and tools to Anthropic Messages; read responses and streams back |
| `src/theagentrouter/transport.py` | HTTP with `httpx`, sync and async, streaming, error mapping |
| `src/theagentrouter/exceptions.py` | LiteLLM's error names; `from_http` maps HTTP errors to them |
| `src/theagentrouter/types.py` | LiteLLM-shaped response types; readable by attribute and as dicts |
| `src/theagentrouter/models.py`, `model_catalog.json` | Model prices and limits from tetrateio/agent-router-models (bundled; never fetched at run time) |
| `src/theagentrouter/compat/` | The LiteLLM-compatible module and `install_as_litellm()` |
| `surface.txt` | The recorded public surface |
| `tests/` | Tests against a local fake provider; `test_adk.py` and `test_openai_agents.py` run real Google ADK and OpenAI Agents SDK agents |
| `scripts/` | `surface.py`, `sync_models.py`, `smoke_real.py`, `smoke_adk_real.py`, `smoke_local_gateway.py`, `smoke_routes.py` |

## Rules

1. **Keep LiteLLM's API.** Same function names, arguments, response fields and error names. A framework that works with LiteLLM must work after `install_as_litellm()`.
2. **No provider code beyond the two formats.** The SDK never calls a provider; a new provider is one row in `local_gateway.PROVIDERS` (its `aigw` schema, address and key variables), which the local gateway's config is written from. Anything that needs translation belongs in the gateway, not here.
3. **One dependency.** Do not add runtime dependencies without a strong reason; `httpx` is the only one.
4. **The public surface is a contract.** `tests/test_surface.py` fails when the surface changes. After a deliberate change, run `python scripts/surface.py --write` and say why in the change description.
5. **Settings come from the environment, then `AGENTROUTER_HOME/config.json`.** Do not add a second place to read them.
6. **No secrets in tests or examples.** Tests use the fake provider in `tests/fake_provider.py`. Real-provider checks live in `scripts/smoke_real.py` and read keys from the environment.

## The local gateway

- The local gateway (`src/theagentrouter/local_gateway.py`, package in `local-gateway/`) has end-to-end tests that are skipped unless the package is installed. Before finishing a change to routing or the gateway, run `make local-gateway AIGW=/path/to/aigw`, then `pytest`: a skipped run proves nothing about the gateway. With keys in `.env`, also run `scripts/smoke_local_gateway.py`.
- Never commit the `aigw` program (`local-gateway/src/theagentrouter_local_gateway/bin/`, git-ignored), and never write a key value into the gateway's config: it holds `${VAR}` references that `aigw` fills in from its environment.
- The local gateway and a gateway set with `AGENTROUTER_BASE_URL` get the same requests: the model name as written, Claude in Messages under `/anthropic/v1`, the rest in OpenAI format under `/v1`. Never add an SDK-only header, and never send a gateway key as `Authorization` on a Messages call (Agent Router passes it on to Anthropic). The gateway's config has one rule per model that passes the provider its own name, spread over routes of at most 15 rules (Gateway API's limit is 16 and `aigw` adds one). Anthropic goes in its own Messages format to `/anthropic/v1/messages`, because `aigw` does not translate OpenAI to Anthropic. Details in `docs/local-gateway.md`.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                         # all tests, no keys needed
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python scripts/surface.py --write    # after a deliberate API change
.venv/bin/python scripts/sync_models.py        # refresh the model table
```

`DEVELOPMENT.md` has the details, including the framework tests and the release steps. User docs are in `docs/`: keep `install.md`, `usage.md`, `coming-from-litellm.md` and `frameworks.md` in step with the code, especially the LiteLLM lists when `compat/litellm.py` or `transport.IGNORED` changes.

## Before you finish a change

- `pytest`, `ruff check` and `ruff format --check` pass.
- New behavior has a test against the fake provider.
- `CHANGELOG.md` has a line under "Unreleased" for anything a user would notice.
