# Examples

Each file runs on its own. Set the provider keys it names in the environment, then run it from the repository root, for example `OPENAI_API_KEY=... python examples/00_two_liner.py`.

Every example goes through the local gateway, Agent Router on this computer, which comes with `pip install theagentrouter` on macOS 13 or newer on Apple chips and on Linux with glibc 2.36 or newer. Elsewhere, set `AGENTROUTER_BASE_URL` to an Agent Router gateway. The local gateway keeps running after an example ends; the next example uses it.

Routes are saved on the computer, so a route one example defines stays until it is redefined or removed with `theagentrouter.remove_route(name)`.

| File | What it shows | Needs |
|---|---|---|
| [`00_two_liner.py`](00_two_liner.py) | The smallest program: one call, one answer | OpenAI key |
| [`01_hello_world.py`](01_hello_world.py) | A text reply and what it cost | Anthropic key |
| [`02_multi_provider.py`](02_multi_provider.py) | The same call to OpenAI, Anthropic and (with AWS credentials) Bedrock; only the model name changes | OpenAI and Anthropic keys; Bedrock optional |
| [`03_streaming.py`](03_streaming.py) | A reply streamed as it is written, with token usage at the end | OpenAI key |
| [`04_tool_use.py`](04_tool_use.py) | A tool call and the answer that uses its result | Anthropic key |
| [`05_async.py`](05_async.py) | Several calls at once with `acompletion` | OpenAI key |
| [`06_structured_output.py`](06_structured_output.py) | A reply that follows a JSON schema | OpenAI key |
| [`07_fallback.py`](07_fallback.py) | A route whose primary fails, answered by its fallback | OpenAI and Anthropic keys |
| [`08_retry.py`](08_retry.py) | A route the local gateway retries on rate limits and server errors | OpenAI key |
| [`09_model_alias.py`](09_model_alias.py) | Names of your own for models ("fast", "smart") | OpenAI and Anthropic keys |
| [`10_share_traffic.py`](10_share_traffic.py) | Two models sharing the traffic of one route | OpenAI and Anthropic keys |
| [`11_from_litellm.py`](11_from_litellm.py) | A LiteLLM script moved over by changing one import | OpenAI key |
| [`12_litellm_router.py`](12_litellm_router.py) | LiteLLM's `Router` with a fallback, unchanged apart from the import | OpenAI and Anthropic keys |
| [`13_handling_errors.py`](13_handling_errors.py) | Errors with LiteLLM's names | OpenAI key |
| [`14_cost.py`](14_cost.py) | Model details and the cost of a call, from the bundled model list | OpenAI key |
| [`15_through_a_gateway.py`](15_through_a_gateway.py) | Every call through a gateway you set with `AGENTROUTER_BASE_URL` | A gateway's address and key; no provider key |
| [`16_adk_agent.py`](16_adk_agent.py) | A Google ADK agent, with no LiteLLM installed | `google-adk`; Anthropic key |
| [`17_openai_agents.py`](17_openai_agents.py) | An OpenAI Agents SDK agent, with no LiteLLM installed | `openai-agents`; Anthropic key |
| [`18_local_gateway.py`](18_local_gateway.py) | The local gateway: how it starts, that it keeps running, and how to stop it | OpenAI key |
| [`19_embeddings.py`](19_embeddings.py) | Embeddings with LiteLLM's arguments, and comparing two texts | OpenAI key |
| [`20_request_limit.py`](20_request_limit.py) | A route that lets 3 requests a minute through and answers the rest with 429 | OpenAI key |
