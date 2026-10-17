# ADK support agent

A Google ADK agent first built on LiteLLM and moved to Agent Router's SDK with one call. The model looks up a support ticket through MCP, searches a Markdown knowledge base, saves a cited response draft and adds an internal ticket note. The example checks the saved report and the stored note before it reports success.

It runs on either SDK, installed beside it: [`src/adk_agent/__init__.py`](src/adk_agent/__init__.py) calls `install_as_litellm()` when Agent Router's SDK is installed, and does nothing on LiteLLM. [MIGRATION.md](MIGRATION.md) has the steps for a project of your own.

## Run

```sh
pip install -e ".[litellm,test]"            # on LiteLLM
pip install -e ../.. -e ".[test]"            # on Agent Router's SDK, from this repository
cp .env.example .env                         # add your keys
python examples/support_agent.py
```

Set `ADK_MODEL` to choose a model (`openai/gpt-5.4`, `anthropic/claude-sonnet-5-5`) and `ADK_STREAMING=true` for ADK's streaming mode. Reports and the check results go to `outputs/<provider>/`; ticket notes are kept in `data/tickets.db`.

## KB and MCP

`knowledge/` contains fictional Acme policies in Markdown. Replace them with your own UTF-8 `.md` and `.txt` documents. The local index uses SQLite FTS5 and BM25 keyword ranking, with overlapping passages and explicit line numbers. Retrieval does not require embeddings or another service.

The ADK agent has three Python function tools: `search_knowledge`, `read_document`, and `save_report`. Reports must contain valid KB source names and line ranges. This verifies references, not whether every claim follows from the cited text; review generated drafts.

ADK's native `McpToolset` starts `mcp_servers/tickets.py` as a stdio subprocess and discovers its tools. The agent sees `tickets_list_open_tickets`, `tickets_get_ticket`, and `tickets_add_ticket_note`. These read and write a local SQLite database seeded with fictional Acme tickets. They do not connect to an external ticket service, contact anyone, or approve refunds.

The ADK runner manages the model/tool loop and session. Runs allow at most 12 model calls. Closing the application closes the MCP toolset and KB connection. Model session history is in memory; reports and ticket notes are persistent.

## On Agent Router's SDK

On Agent Router's SDK, the example sets `AGENTROUTER_HOME` to `data/agentrouter` unless you already set it. The first model call starts the local gateway with your provider keys, and later calls reuse it. It keeps running after the example ends.

To use a gateway your team runs, set `AGENTROUTER_BASE_URL` and `AGENTROUTER_API_KEY`. Do not add `num_retries` to the ADK model options: the SDK ignores that per-call argument. Define retries and fallbacks as a route in the gateway instead.

## Tests

```sh
APP_SDK=litellm pytest                       # or APP_SDK=theagentrouter
```

The tests run the real ADK runner and MCP subprocess against HTTP fixtures of OpenAI's and Anthropic's replies, with and without streaming, and check the saved report and the ticket note. They make no paid calls. `make apps` at the repository root runs them on both SDKs; [`../README.md`](../README.md) has the rest.

References: [Google ADK's LiteLlm adapter](https://adk.dev/agents/models/litellm/), [ADK MCP tools](https://adk.dev/tools-custom/mcp-tools/), [Agent Router framework integration](https://github.com/theagentrouter/agent-router-python/blob/main/docs/frameworks.md).
