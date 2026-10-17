# Usage

## Contents

- **Calls:** [A first call](#a-first-call) · [Models and keys](#models-and-keys) · [Parameters](#parameters) · [Streaming](#streaming) · [Async](#async) · [Tools](#tools) · [The Responses API](#the-responses-api) · [The Messages API](#the-messages-api) · [Embeddings](#embeddings)
- **What comes back:** [What a call returns](#what-a-call-returns) · [Errors](#errors) · [Cost and model details](#cost-and-model-details) ([adding models](#adding-models)) · [Streams put back together](#streams-put-back-together) · [Callbacks](#callbacks) · [Counting tokens](#counting-tokens) · [Tests](#tests)
- **Gateways:** [Gateways](#gateways) · [A gateway you set](#a-gateway-you-set) · [The local gateway](#the-local-gateway) · [Routes](#routes-fallbacks-retries-and-names-of-your-own) · [Running the same gateway for a team](#running-the-same-gateway-for-a-team) · [Settings](#settings)

## A first call

```python
import theagentrouter

resp = theagentrouter.completion(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Summarise yesterday's refunds"}],
)
print(resp.choices[0].message.content)
```

`completion` takes the same arguments as `litellm.completion` and returns the same response shape.

The SDK has four kinds of call, each as in LiteLLM:

| Call | For | Section |
|---|---|---|
| `completion`, `acompletion` | Chat with any model: OpenAI Chat Completions, and Anthropic Messages for Claude | [Streaming](#streaming), [Tools](#tools) |
| `responses`, `aresponses` | OpenAI's Responses API, for OpenAI models | [The Responses API](#the-responses-api) |
| `anthropic.messages.create`, `acreate` | Anthropic's Messages API, for Claude | [The Messages API](#the-messages-api) |
| `embedding`, `aembedding` | Vectors | [Embeddings](#embeddings) |

## Models and keys

A model name is `provider/model`, such as `openai/gpt-4o-mini`. A name with no provider works where it works in LiteLLM: OpenAI and Anthropic names (`gpt-4o-mini`, `claude-haiku-4-5`). Other names need the provider, as in LiteLLM (`mistral/mistral-large-latest`, `groq/llama-3.1-8b-instant`). A bare `gemini-…` name is refused with `BadRequestError`: LiteLLM sends it to Google Cloud's Vertex AI, which the local gateway does not serve yet, so write `gemini/gemini-2.5-flash` for the Gemini API. Every gateway gets the name as you wrote it, with or without the provider, and a gateway you set can have names of its own, such as `claude-sonnet-5`.

Every call goes through a gateway: the local gateway (Agent Router on this computer), or one you set. The local gateway reads each provider's key from the environment when it starts:

| Model name starts with | Key |
|---|---|
| `openai/` | `OPENAI_API_KEY` |
| `anthropic/` | `ANTHROPIC_API_KEY` |
| `gemini/` | `GEMINI_API_KEY` |
| `groq/` | `GROQ_API_KEY` |
| `mistral/` | `MISTRAL_API_KEY` |
| `deepseek/` | `DEEPSEEK_API_KEY` |
| `together/` | `TOGETHER_API_KEY` |
| `bedrock/` | `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; and `AWS_REGION` |

`api_key=` on a call is used in two cases: with `api_base=`, as the key for that endpoint, and with a gateway you set, as that gateway's key. With the local gateway it has no effect and warns once, because the local gateway reads provider keys from the environment when it starts.

To send one call to an endpoint you name, in the provider's own format, use `api_base=` (LiteLLM's argument), for example a local test server.

## Parameters

For OpenAI Chat Completions, these go to the provider as they are:

`temperature`, `top_p`, `n`, `stop`, `max_tokens`, `max_completion_tokens`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `seed`, `response_format`, `tool_choice`, `parallel_tool_calls`, `reasoning_effort`, `user`, `modalities`, `prediction`, `audio`, `service_tier`.

For Claude, which goes to the gateway in Anthropic's Messages format (so prompt caching and thinking keep working), the SDK translates `temperature`, `top_p`, `top_k`, `stop`, `max_tokens`, `max_completion_tokens`, `tool_choice`, `parallel_tool_calls` and `user`, and passes `thinking` (`thinking={"type": "enabled", "budget_tokens": 1024}`). `parallel_tool_calls=False` allows at most one tool call. `n=1` is fine. `reasoning_effort` maps to a thinking budget, and a JSON schema in `response_format` maps to the Messages output format. A plain `json_object` without a schema is accepted without inventing one, as in LiteLLM. OpenAI-only options such as `seed` raise `UnsupportedParamsError`; with `drop_params=True` (or `theagentrouter.drop_params = True`) they are dropped instead. Use `extra_body` for other native Anthropic options. Prompt caching takes LiteLLM's `cache_control` markers: on a content part, on a whole message, or on a tool; cache reads show in `usage.prompt_tokens_details.cached_tokens`. With thinking on, the reply carries `reasoning_content` and `thinking_blocks` (streamed: `delta.reasoning_content`); send the assistant message back as it is with the tool result, since Anthropic needs its signed thinking blocks. Model capability lookups describe the bundled model catalog; they do not promise that this SDK translates every native option.

These control the call itself:

| Parameter | What it does |
|---|---|
| `api_key` | The key for `api_base`, or for a gateway you set. Ignored, with a warning, for the local gateway |
| `api_base` | Send this call to this address instead, in the provider's own format |
| `timeout` | Seconds before giving up; the default is 600, as in LiteLLM |
| `extra_headers` | More HTTP headers on the request |
| `extra_body` | More fields in the request body, for provider-specific options |
| `mock_response` | Answer on this computer without calling anything, for tests ([Tests](#tests)) |

`num_retries` (or `max_retries`) sets bounded request retries for connection failures, timeouts, 408, 429 and 5xx. It defaults to zero; authentication and budget errors are not retried. Completion streams retry only before yielding their first chunk. A route's Retry is separate and runs in the gateway, so enabling both can multiply attempts. Native Messages byte-stream setup does not yet use the common SDK retry loop. Other unknown completion arguments warn and are not sent; see [Coming from LiteLLM](coming-from-litellm.md#ignored-with-a-warning).

## Streaming

```python
for chunk in theagentrouter.completion(model="openai/gpt-4o-mini", messages=messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

Every chunk has a choice, as in LiteLLM. The last chunk, the one with the finish reason, carries `usage` with the token counts; with `stream_options={"include_usage": True}` the usage comes instead on one more chunk with an empty choice, as LiteLLM sends it. Provider error events raise an SDK exception, including errors that arrive after partial output.

## Async

```python
resp = await theagentrouter.acompletion(model="openai/gpt-4o-mini", messages=messages)

stream = await theagentrouter.acompletion(model="openai/gpt-4o-mini", messages=messages, stream=True)
async for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Tools

Tools use the OpenAI format for every provider, Anthropic included:

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "search_refunds",
        "description": "Find refunds for a day",
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
    },
}]

messages = [{"role": "user", "content": "How much was refunded yesterday?"}]
resp = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=messages, tools=tools)
call = resp.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)

messages.append(resp.choices[0].message.to_dict())
messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"total_usd": 42})})
final = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=messages, tools=tools)
print(final.choices[0].message.content)
```

## The Responses API

```python
resp = theagentrouter.responses(input="Summarise yesterday's refunds", model="openai/gpt-4o-mini", max_output_tokens=200)
resp.output_text                       # the text of the output messages
resp.output[0].content[0].text         # the reply as OpenAI sends it, by attribute or key
resp.usage.input_tokens
resp._hidden_params["response_cost"]
```

LiteLLM's `responses` and `aresponses`, through the gateway's `/v1/responses`. Agent Router serves the Responses API for OpenAI models, so other providers are refused with `BadRequestError`; use `completion` for them. `instructions`, `tools`, `tool_choice`, `max_output_tokens`, `temperature`, `reasoning`, `previous_response_id` and the other Responses arguments are passed on. Streaming is not supported yet.

## The Messages API

```python
resp = theagentrouter.anthropic.messages.create(
    model="anthropic/claude-sonnet-4-5",
    max_tokens=1024,
    system=[{"type": "text", "text": policy, "cache_control": {"type": "ephemeral"}}],
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=[{"role": "user", "content": "Can I get a refund after 40 days?"}],
)
resp["content"]                        # thinking and text blocks, as Anthropic sends them
resp["usage"]["cache_read_input_tokens"]
resp._hidden_params["response_cost"]

for event in theagentrouter.anthropic.messages.create(..., stream=True):
    print(event)                       # b'event: content_block_delta\ndata: {...}\n\n'
```

LiteLLM's `anthropic.messages.create` and `acreate` (also `anthropic_messages`, the async one), through the gateway's `/anthropic/v1/messages`. The request and the reply are Anthropic's own format, not translated, so every Messages feature works: prompt caching, thinking, tools, citations and `anthropic-beta` headers (in `extra_headers`). The reply is a dict, as in LiteLLM, that also reads by attribute (`resp.content[0].text`). With `stream=True` you get the server-sent events as bytes, as in LiteLLM.

Agent Router serves the Messages API for Claude, so other providers are refused with `BadRequestError`; use `completion` for them. Routes are refused too, because the local gateway answers a route in the OpenAI format; use `completion` with a route. `completion` also sends Claude in the Messages format, and translates it to and from OpenAI's shape; use `anthropic.messages.create` when your code already speaks Anthropic's format.

## Embeddings

```python
resp = theagentrouter.embedding(model="openai/text-embedding-3-small", input=["refunds", "returns"])
resp.data[0]["embedding"]               # the first vector; resp.data[0].embedding works too
resp.usage.prompt_tokens
resp._hidden_params["response_cost"]    # US dollars, from the model list
```

`aembedding` is the async version. `dimensions`, `encoding_format` and `user` are passed to the provider. Embeddings go where a completion for the same model goes: through the local gateway or one you set. OpenAI, Gemini, Mistral, Together and Bedrock's Titan models have embeddings; Anthropic has no embeddings API.

## What a call returns

What `completion` returns reads like LiteLLM's, both as attributes and as a dict:

```python
resp.choices[0].message.content
resp["choices"][0]["message"]["content"]
resp.usage.total_tokens
resp.to_dict()
```

`model_dump(mode="json", exclude_none=True)` works too: it leaves out optional fields that are `None`, and keeps null values inside payload dictionaries and lists. [Serialization differences when migrating](coming-from-litellm.md#response-serialization).

`resp._hidden_params` says where the call went and what it cost, as LiteLLM's does:

| Key | Example |
|---|---|
| `route` | `local gateway`, `gateway` (one you set) or `api_base` (an endpoint named on the call) |
| `sent_to` | `http://127.0.0.1:52413/anthropic/v1/messages`: the address the SDK sent the request to, here the local gateway |
| `llm_provider` | `anthropic` |
| `model_sent` | `anthropic/claude-sonnet-4-5`: the model name in the request, as you wrote it. The gateway passes the provider its own name |
| `response_cost` | `0.000123`, in US dollars, from the model list. `completion_cost(resp)` gives the same |
| `served_by`, `route_name` | For a route: the model that answered, and the route's name |

## Errors

Errors keep LiteLLM's names and carry `status_code`, `model` and `llm_provider`:

```python
from theagentrouter import exceptions

try:
    theagentrouter.completion(model="openai/gpt-4o-mini", messages=messages)
except exceptions.ContextWindowExceededError:
    ...
except exceptions.RateLimitError:
    ...
```

| Error | When |
|---|---|
| `UnsupportedParamsError` | An option cannot be translated to the selected request format |
| `BadRequestError` | The request is wrong, for example a model name the SDK cannot place with a provider (`acme-model`), or a bare `gemini-…` name |
| `ContextWindowExceededError` | The messages are too long for the model |
| `ContentPolicyViolationError` | The provider refused the content |
| `AuthenticationError` | The key is missing or wrong |
| `PermissionDeniedError` | The key cannot use this model |
| `NotFoundError` | The model does not exist |
| `RateLimitError` | Too many requests |
| `BudgetExceededError` | A gateway's budget is used up. It is a kind of `RateLimitError` |
| `Timeout` | No answer within `timeout` |
| `APIConnectionError` | The provider or gateway could not be reached, or the local gateway could not start (the message says why) |
| `InternalServerError`, `BadGatewayError`, `ServiceUnavailableError` | The provider or gateway failed |

## Cost and model details

```python
theagentrouter.completion_cost(resp)                      # US dollars for this response
theagentrouter.get_model_info("openai/gpt-4o-mini")       # context window, prices, features
theagentrouter.supports_function_calling("anthropic/claude-sonnet-4-5")
theagentrouter.model_cost["openai/gpt-4o-mini"]           # the raw entry, as in litellm.model_cost
```

These read the model list that ships with each release ([tetrateio/agent-router-models](https://github.com/tetrateio/agent-router-models)): OpenAI, Anthropic, Gemini and Groq models, and DeepInfra and xAI models for prices only. It has no Bedrock, Mistral, DeepSeek or Together models, and no output limits: `max_tokens` and `max_output_tokens` are `None`. For OpenAI models, `max_input_tokens` is the whole context window, which includes the output. The SDK never downloads the list while it runs. `get_model_info` raises `NotFoundError` for a model that is not on the list.

### Adding models

`register_model` adds models or changes their details, in LiteLLM's format, as `litellm.register_model` does:

```python
theagentrouter.register_model({
    "ft:gpt-4o-mini:acme": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 3e-07, "output_cost_per_token": 1.2e-06},
})

# Or LiteLLM's whole list, about 4,000 models from every provider:
theagentrouter.register_model("https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json")

# Or a newer copy of the list this SDK ships, one file per provider:
theagentrouter.register_model("https://raw.githubusercontent.com/tetrateio/agent-router-models/main/openai.json")
```

A registered model gets prices and details, a rule in the local gateway when it next starts, and a rule in `local_gateway.export_config()`. Registrations last for the program that makes them.

## Streams put back together

`theagentrouter.stream_chunk_builder(chunks)` turns the chunks of a streamed call back into one response, as LiteLLM's does: each choice on its own, with text and reasoning joined, tool calls joined by their index, thinking blocks kept, the last finish reason, and the last usage.

## Callbacks

Code that sets LiteLLM's callback lists keeps working through `theagentrouter.compat.litellm`:

```python
from theagentrouter.compat import litellm

def log(kwargs, response, start_time, end_time):
    print(kwargs["model"], kwargs["response_cost"], response.usage.total_tokens)

litellm.success_callback = [log]     # also failure_callback, and callbacks for logger objects
```

A function is called as `fn(kwargs, response, start_time, end_time)`; a logger object gets `log_success_event` and `log_failure_event`, or their `async_` versions under `acompletion`. They run in your program after each `completion`, after a stream ends (with the response put back together), and on errors. A callback that fails warns and does not stop the call. LiteLLM's built-in integrations, named by string (`"langfuse"`), are not here: the gateway's OpenTelemetry traces and metrics record calls.

## Counting tokens

```python
theagentrouter.token_counter(model="anthropic/claude-sonnet-4-5", messages=messages, tools=tools)
theagentrouter.token_counter(model="openai/gpt-4o-mini", text="Summarise yesterday's refunds")
```

LiteLLM's arguments. For OpenAI and Anthropic models, with or without the provider in the name, the gateway counts: Agent Router asks the provider (`/responses/input_tokens`, `/messages/count_tokens`), so the number is what the provider will charge for. Every other provider, a name the SDK cannot place with a provider, and a count the gateway cannot give are estimated on this computer the way LiteLLM does: its rules for messages and tools, with tiktoken. Install it with `pip install theagentrouter[tokens]`; without it the estimate is about four characters a token, and the SDK warns once.

The gateway's count is a request each time. Code that counts before every turn can set `AGENTROUTER_TOKEN_COUNT=estimate` to always estimate, with no request, as LiteLLM does. `litellm.disable_token_counter = True` makes it return 0, as in LiteLLM.

## Tests

`mock_response` works as in LiteLLM: the call is answered on your computer, so no gateway starts and no provider is called.

```python
resp = theagentrouter.completion(model="openai/gpt-4o-mini", messages=messages, mock_response="Offline reply")
resp.choices[0].message.content          # "Offline reply"; usage is 10 prompt and 20 completion tokens

theagentrouter.completion(model="openai/gpt-4o-mini", messages=messages, mock_response="litellm.RateLimitError")  # raises RateLimitError
```

It also takes an exception to raise, a dict or `ModelResponse` as the whole response, `stream=True` (the text comes three characters at a time, with usage on the last chunk) and `acompletion`. `mock_tool_calls` adds tool calls, `mock_timeout=True` with a `timeout` raises `Timeout`, and `mock_delay` waits first. Callbacks run as for any call. To test the request the SDK sends, use a test server with `api_base` instead.

## Gateways

The SDK never calls a provider itself. A call goes to the first of these that applies:

1. **`api_base` on the call.** That endpoint, for this call only, in the provider's own format (LiteLLM's argument).
2. **A gateway you set.** With `AGENTROUTER_BASE_URL`, every call goes there.
3. **The local gateway.** Agent Router on this computer, started by the first call (`AGENTROUTER_AUTOSTART=0` turns that off). If the local gateway is not installed, as on Windows, the call fails with a message that says to set `AGENTROUTER_BASE_URL`.

A gateway you set and the local gateway get the same requests, so code that works with one works with the other:

| Call | Path on the gateway | Format |
|---|---|---|
| `completion` to Claude (`anthropic/…` or `claude-…`) | `/anthropic/v1/messages` | Anthropic Messages, so prompt caching and thinking work |
| `completion` to any other model | `/v1/chat/completions` | OpenAI Chat Completions |
| `embedding` | `/v1/embeddings` | OpenAI |
| `responses` | `/v1/responses` | OpenAI Responses |
| `anthropic.messages.create` | `/anthropic/v1/messages` | Anthropic Messages, as you wrote it |
| `token_counter` | `/anthropic/v1/messages/count_tokens` for Claude, `/v1/responses/input_tokens` for OpenAI | |

The model name is sent as you wrote it: `anthropic/claude-sonnet-4-5` or `claude-sonnet-4-5`. The gateway passes the provider its own name.

### A gateway you set

A remote gateway: an Agent Router your team deploys, or your company's Agent Router Enterprise.

```sh
export AGENTROUTER_BASE_URL=https://gateway.acme.example/v1
export AGENTROUTER_API_KEY=...      # the gateway's key, not a provider's
```

Set the gateway's address with or without the `/v1` that OpenAI clients add; the SDK adds the paths above. Every call goes there and the local gateway never starts. Provider keys stay on the gateway, so this computer needs none. The key goes as `Authorization: Bearer` on OpenAI-format calls and as `x-api-key` on Claude calls, as Anthropic's own clients send it. Routes belong to the local gateway: see [Routes and a gateway you set](#routes-and-a-gateway-you-set).

### The local gateway

It comes with `pip install theagentrouter` on macOS 13 or newer on Apple chips and on Linux with glibc 2.36 or newer (not with `theagentrouter-core`, the SDK alone). There is nothing else to run. The first call starts Agent Router's `aigw` on a free port chosen at start, answering only programs on this computer, and says so once:

```
theagentrouter: started the local gateway on http://127.0.0.1:52413 (providers from your environment: anthropic, bedrock, openai).
```

The port changes from start to start, so another Agent Router gateway on 1975 (your own `aigw run`, for example) never blocks it. Every program using the SDK finds the running gateway by itself; to reach it from another tool, read the port from that line or from `port` in `~/.agentrouter/local-gateway/state.json`. Two Agent Router gateways still cannot run on one computer: `aigw` keeps two internal ports fixed (1061 and 18000), so with another Agent Router gateway running the SDK says so and asks you to stop it or set `AGENTROUTER_BASE_URL`.

- The SDK sets the gateway up from the provider keys in your environment. The gateway holds the keys; its config file holds only references such as `${OPENAI_API_KEY}`.
- It starts with a rule for every model in the SDK's bundled model list for those providers. Register a custom model with `register_model` before adding it to a running gateway. An unregistered model name raises `NotFoundError` and leaves the existing process running.
- A new registered model reloads a same-version gateway without interrupting other programs; a new provider restarts it. The caller must have credentials for every provider already running; otherwise the SDK raises and leaves the process running. It cannot merge credentials held only by another process.
- `AGENTROUTER_AUTOSTART=0` stops the SDK from starting it.
- A gateway set with `AGENTROUTER_BASE_URL` always wins, and then the SDK starts nothing.
- Its files, including its log `aigw.log`, are in `~/.agentrouter/local-gateway/`.
- The very first start downloads Envoy (about 100 MB, into `~/.local/share/aigw`), so it needs the internet and can take a minute.
- It keeps running after your program ends. To stop it: `python -c "from theagentrouter import local_gateway; local_gateway.stop()"`. The next call starts it again.
- After an upgrade, existing gateway rules keep working. Stop the gateway explicitly when other programs finish to load the new version. Rotated credentials are reported before a call; stop and retry to load them. Automatic key reload is not implemented.

| Provider | Set in the environment |
|---|---|
| AWS Bedrock | `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; and `AWS_REGION` |
| OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, Together | The key from [Models and keys](#models-and-keys) |

### Routes: fallbacks, retries and names of your own

A route is a model name of your own with the models behind it. The local gateway serves routes.

```python
import theagentrouter as ar

ar.route(
    "chat",
    primary="openai/gpt-4o",
    fallbacks=["anthropic/claude-sonnet-4-5"],
    retry=ar.Retry(attempts=2, on=(429, 500, 502, 503, 504)),
)
resp = ar.completion(model="chat", messages=messages)
resp._hidden_params["served_by"]   # the model that answered
resp._hidden_params["route_name"]  # "chat"
```

The local gateway does the retries and the fallbacks, as Agent Router Enterprise does: the SDK sends one request, named after the route.

- **Retry:** the gateway tries each tier `attempts` more times on the listed status codes and on connection failures, with `backoff` seconds before the first retry, doubling up to 10 seconds, and an optional per-try `timeout`.
- **Fallback:** when a tier still fails with one of those codes, or cannot be reached, the gateway moves to the next tier. A stream falls back only before its reply starts.
- **Sharing:** `primary=["openai/gpt-4o", "openai/gpt-4o-mini"]` puts two models in one tier, which share the traffic. A fallback can be a list too.
- `served_by` is read from the model name in the reply, so it is missing when a provider answers with a name that does not start with the model's own.
- Routes are kept in `~/.agentrouter/routes.json` and shared by every program on the computer, like the local gateway. Defining a route again replaces it; `ar.list_routes()` and `ar.remove_route(name)` list and remove them. A changed route reloads the gateway on the next call that needs it, without interrupting other programs.
- Define routes without local provider keys or a companion package when exporting configuration. The machine that starts the gateway needs credentials for every provider in the saved routes.
- A route of Anthropic and other providers reaches Anthropic through its OpenAI-compatible endpoint.
- **Limit:** `limit=ar.Limit(60, per="minute")` lets at most 60 requests a minute through the route; the gateway answers the rest with 429 (`RateLimitError`). `per` is `"second"`, `"minute"`, `"hour"` or `"day"`. The count covers every program on the computer that uses the route. It counts requests, not tokens: Agent Router counts tokens only with a Redis-backed rate-limit service, which the local gateway does not have.

LiteLLM's `Router` works on top of routes: see [Coming from LiteLLM](coming-from-litellm.md).

### Running the same gateway for a team

`local_gateway.export_config()` gives the config the SDK writes for the local gateway, to run on a server. Developers then set `AGENTROUTER_BASE_URL` to that server, and the same code works there.

```python
from pathlib import Path
from theagentrouter import local_gateway

Path("config.yaml").write_text(local_gateway.export_config(port=1975))
```

```sh
export AGENTROUTER_API_KEY=...             # the server client key; set it on clients too
ENVOY_VERSION=1.39.4 aigw run config.yaml   # patched build; provider keys come from the server environment
```

What the file holds:

- **Providers:** those whose keys are set on your computer, or the ones you name: `export_config(providers=["openai", "anthropic"])`.
- **Models:** a rule for every model in the SDK's bundled model list for those providers, with and without the provider in the name, plus any you added with `register_model`. **The server serves only these.** Unlike the local gateway, it cannot add a model when it is first asked for. The bundled list has no Bedrock, Mistral, DeepSeek or Together models, so for those providers name the models (`export_config(models=["mistral/mistral-large-latest"])`) or register LiteLLM's list first (see [Adding models](#adding-models)). Export again when the models change.
- **Routes:** the routes defined on your computer, with their fallbacks, retries and limits.
- **No key values:** only references such as `${OPENAI_API_KEY}`, which `aigw` fills in from the server's environment.

Exported configurations require a client key by default. Set `AGENTROUTER_API_KEY` on the server and on each client. The gateway accepts `Authorization: Bearer <key>` or `x-api-key: <key>` and strips the client header before forwarding. `export_config(authentication=False)` explicitly exports an open endpoint. Terminate TLS in front of the server when callers use an untrusted network. The server-authentication change requires the [patched aigw build](../local-gateway/patches/README.md); unmodified upstream v1.2.0 drops the client-key Secret. On the server, also set `AIGW_MCP_SESSION_ENCRYPTION_SEED` to a random secret: `aigw` always starts an MCP proxy, which otherwise uses a publicly known seed. The patched build reads it; unmodified v1.2.0 has no setting for it.

#### Routes and a gateway you set

Routes program the local gateway only. With `AGENTROUTER_BASE_URL` set:

- `route()` refuses to define a route, and says to set up fallbacks on that gateway instead.
- A route defined earlier is not used: its name goes to that gateway as a plain model name, and the SDK warns once. That gateway's own routes and fallbacks apply.

## Settings

| Setting | What it does |
|---|---|
| `AGENTROUTER_BASE_URL` | Send every call through this gateway instead of the local one |
| `AGENTROUTER_API_KEY` | The gateway's key |
| `AGENTROUTER_HOME` | Folder for the SDK's files (default `~/.agentrouter`) |
| `AGENTROUTER_AUTOSTART` | `0` stops the SDK from starting the local gateway |
| `AGENTROUTER_TOKEN_COUNT` | `estimate` makes `token_counter` estimate on this computer instead of asking the gateway |

`AGENTROUTER_BASE_URL`, `AGENTROUTER_API_KEY` and `AGENTROUTER_TOKEN_COUNT` can also be kept in `AGENTROUTER_HOME/config.json`, as `base_url`, `api_key` and `token_count`. Environment variables win over the file. When `AGENTROUTER_BASE_URL` names a different gateway from the file, the file's key is not used, so it never goes to a gateway it was not made for.

To check that the keys a model needs are set, use LiteLLM's `validate_environment(model)`. For the local gateway it checks what the gateway needs for that provider (for Bedrock: a profile, or a key and secret, and a region); with a gateway set, it checks `AGENTROUTER_API_KEY`.
