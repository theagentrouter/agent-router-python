# Examples

Each file runs on its own. Set the provider keys it names in the environment, then run it from the repository root, for example `OPENAI_API_KEY=... python examples/first_calls/one_call.py`.

Every example goes through the local gateway, Agent Router on this computer, which comes with `pip install theagentrouter` on macOS 13 or newer on Apple chips and on Linux with glibc 2.36 or newer. Elsewhere, set `AGENTROUTER_BASE_URL` to an Agent Router gateway. The local gateway keeps running after an example ends; the next example uses it.

Routes are saved on the computer, so a route one example defines stays until it is redefined or removed with `theagentrouter.remove_route(name)`.

## Coming from LiteLLM

Code written for LiteLLM, run on this SDK.

| File | What it shows | Needs |
|---|---|---|
| [`change_the_import.py`](from_litellm/change_the_import.py) | A LiteLLM script moved over by changing one import | OpenAI key |
| [`litellm_router.py`](from_litellm/litellm_router.py) | LiteLLM's `Router` with a fallback, unchanged apart from the import | OpenAI and Anthropic keys |
| [`litellm_errors.py`](from_litellm/litellm_errors.py) | Errors with LiteLLM's names | OpenAI key |
| [`cost_and_model_info.py`](from_litellm/cost_and_model_info.py) | Model details and the cost of a call, from the bundled model list | OpenAI key |

## First calls

| File | What it shows | Needs |
|---|---|---|
| [`one_call.py`](first_calls/one_call.py) | The smallest program: one call, one answer | OpenAI key |
| [`reply_and_cost.py`](first_calls/reply_and_cost.py) | A text reply and what it cost | Anthropic key |
| [`any_provider.py`](first_calls/any_provider.py) | The same call to OpenAI, Anthropic and (with AWS credentials) Bedrock; only the model name changes | OpenAI and Anthropic keys; Bedrock optional |
| [`stream.py`](first_calls/stream.py) | A reply streamed as it is written, with token usage at the end | OpenAI key |
| [`tools.py`](first_calls/tools.py) | A tool call and the answer that uses its result | Anthropic key |
| [`many_at_once.py`](first_calls/many_at_once.py) | Several calls at once with `acompletion` | OpenAI key |
| [`json_schema.py`](first_calls/json_schema.py) | A reply that follows a JSON schema | OpenAI key |
| [`embeddings.py`](first_calls/embeddings.py) | Embeddings with LiteLLM's arguments, and comparing two texts | OpenAI key |

## Routes

A route is a name of your own for one or more models. The gateway does its fallbacks, retries and limits.

| File | What it shows | Needs |
|---|---|---|
| [`fallback.py`](routes/fallback.py) | A route whose primary fails, answered by its fallback | OpenAI and Anthropic keys |
| [`retry.py`](routes/retry.py) | A route the local gateway retries on rate limits and server errors | OpenAI key |
| [`your_own_names.py`](routes/your_own_names.py) | Names of your own for models ("fast", "smart") | OpenAI and Anthropic keys |
| [`split_traffic.py`](routes/split_traffic.py) | Two models sharing the traffic of one route | OpenAI and Anthropic keys |
| [`request_limit.py`](routes/request_limit.py) | A route that lets 3 requests a minute through and answers the rest with 429 | OpenAI key |

## Gateways

| File | What it shows | Needs |
|---|---|---|
| [`local_gateway.py`](gateways/local_gateway.py) | The local gateway: how it starts, that it keeps running, and how to stop it | OpenAI key |
| [`your_gateway.py`](gateways/your_gateway.py) | Every call through a gateway you set with `AGENTROUTER_BASE_URL` | A gateway's address and key; no provider key |

## Agent frameworks

| File | What it shows | Needs |
|---|---|---|
| [`google_adk.py`](agent_frameworks/google_adk.py) | A Google ADK agent, with no LiteLLM installed | `google-adk`; Anthropic key |
| [`openai_agents.py`](agent_frameworks/openai_agents.py) | An OpenAI Agents SDK agent, with no LiteLLM installed | `openai-agents`; Anthropic key |

## Provider APIs

| File | What it shows | Needs |
|---|---|---|
| [`openai_responses.py`](provider_apis/openai_responses.py) | OpenAI's Responses API: a reply, a follow-up that continues it, and what it cost | OpenAI key |
| [`anthropic_messages.py`](provider_apis/anthropic_messages.py) | Anthropic's Messages API: prompt caching and thinking, in Anthropic's own format | Anthropic key |
