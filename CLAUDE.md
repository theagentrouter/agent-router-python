# CLAUDE.md

Read [AGENTS.md](AGENTS.md): it holds the instructions for this repository, for every coding agent.

Reminders that matter most:

- `tests/test_surface.py` guards the public API. A failing surface test after an intended change means: run `python scripts/surface.py --write`, and explain the change.
- Never add a reference to the Enterprise package (`tetrate-agentrouter` / `tetrate.agentrouter`) here; the open-source SDK must not know the Enterprise SDK exists.
- The local gateway (`src/theagentrouter/local_gateway.py`, package in `local-gateway/`) has end-to-end tests that are skipped unless the package is installed. Before finishing a change to routing or the gateway, run `make local-gateway AIGW=/path/to/aigw`, then `pytest` — a skipped run proves nothing about the gateway. With keys in `.env`, also run `scripts/smoke_local_gateway.py`.
- Never commit the `aigw` program (`local-gateway/src/theagentrouter_local_gateway/bin/`, git-ignored), and never write a key value into the gateway's config: it holds `${VAR}` references that `aigw` fills in from its environment.
- The local gateway and a gateway set with `AGENTROUTER_BASE_URL` get the same requests: the model name as written, Claude in Messages under `/anthropic/v1`, the rest in OpenAI format under `/v1`. Never add an SDK-only header, and never send a gateway key as `Authorization` on a Messages call (Agent Router passes it on to Anthropic). The gateway's config has one rule per model that passes the provider its own name, spread over routes of at most 15 rules (Gateway API's limit is 16 and `aigw` adds one). Anthropic goes in its own Messages format to `/anthropic/v1/messages`, because `aigw` does not translate OpenAI to Anthropic. Details in `docs/local-gateway.md`.
