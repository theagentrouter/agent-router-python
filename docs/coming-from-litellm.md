# Coming from LiteLLM

theagentrouter keeps LiteLLM's way of calling models: the same `completion` arguments, the same response shape and the same error names. Most code moves with one changed import. This page says exactly what moves, what is missing, and what is ignored.

## Change the import

In your own code:

```python
from theagentrouter.compat import litellm   # was: import litellm

resp = litellm.completion(model="openai/gpt-5.4-mini", messages=messages)
```

When a library you use imports `litellm` itself, as Google ADK and the OpenAI Agents SDK do, make that import answer with theagentrouter instead. Call this once, before importing the library:

```python
from theagentrouter.compat import install_as_litellm
install_as_litellm()
```

LiteLLM does not need to be installed. See [Agent frameworks](frameworks.md).

A full example: [`examples/from_litellm/change_the_import.py`](../examples/from_litellm/change_the_import.py). With `Router`: [`examples/from_litellm/litellm_router.py`](../examples/from_litellm/litellm_router.py).

## What works the same

| LiteLLM | theagentrouter |
|---|---|
| `completion`, `acompletion` | Same arguments and response shape |
| `embedding`, `aembedding` | Same arguments (`input`, `dimensions`, `encoding_format`, `user`) and response shape. Anthropic has no embeddings API, so `anthropic/` models raise `BadRequestError` |
| `stream=True` | Same chunks, with `usage` on the last one |
| `tools`, `tool_choice`, `parallel_tool_calls` | OpenAI-shaped inputs, translated for Anthropic too; `parallel_tool_calls=False` disables parallel Anthropic tool calls |
| `response_format`, `temperature`, `max_tokens` and the other OpenAI parameters | Passed through for OpenAI Chat Completions. Unsupported Anthropic options raise `UnsupportedParamsError` ([the full list](usage.md#parameters)) |
| `api_base` | Same: the call goes to that address in place of the gateway, so routes, fallbacks and limits apply only if it is an Agent Router gateway that has them. Warns once per address. Remove an `api_base` that points at a provider to call through the gateway |
| `timeout`, `extra_headers`, `extra_body` | Same |
| `num_retries`, `max_retries` | Bounded request retries before response/stream output; a partially consumed completion stream is never replayed. Default is zero. Route retries remain gateway settings |
| Module `api_base`, `api_key`, `request_timeout`, `num_retries` | Defaults for completion, embeddings and Responses calls; per-call values win. `modify_params=True` warns because message rewriting is unsupported |
| `api_key` | Used with `api_base`, or as the key of a gateway you set. With the local gateway it is ignored with a warning: the gateway reads provider keys from the environment ([Models and keys](usage.md#models-and-keys)) |
| Errors: `RateLimitError`, `ContextWindowExceededError`, `AuthenticationError` and the rest | Same names, under `litellm.` and `theagentrouter.exceptions` |
| `responses`, `aresponses` | Same arguments, for OpenAI models, through the gateway's `/v1/responses` ([The Responses API](usage.md#the-responses-api)) |
| `anthropic.messages.create`, `acreate`, `anthropic_messages` | Same arguments, for Claude, through the gateway's `/anthropic/v1/messages`, in Anthropic's format ([The Messages API](usage.md#the-messages-api)) |
| `success_callback`, `failure_callback`, `callbacks` | Your functions and logger objects are called after each call ([Callbacks](usage.md#callbacks)) |
| `mock_response`, `mock_tool_calls`, `mock_timeout`, `mock_delay` | Same: the answer is made on your computer, with no gateway and no provider call, so tests that use them stay offline ([Tests](usage.md#tests)) |
| `stream_chunk_builder`, `validate_environment`, `model_list`, `cost_calculator.completion_cost` | Same |
| `token_counter` | Same arguments. OpenAI and Anthropic are counted by the gateway, exactly; the rest are estimated as LiteLLM does, with tiktoken from `theagentrouter[tokens]` ([Counting tokens](usage.md#counting-tokens)) |
| `completion_cost`, `cost_per_token`, `get_model_info`, `model_cost`, `supports_*`, `register_model` | Same, from the model list that ships with each release plus the models you register |
| `get_llm_provider`, `get_supported_openai_params`, `provider_list` | Same names; parameter discovery excludes options the SDK cannot translate for Anthropic calls |
| `drop_params`, `additional_drop_params` | Same, on the call or on the module, off by default: with it, Claude drops the parameters it does not take (`seed`, `presence_penalty` and others, and on newer models `temperature` and `top_p`) instead of raising, and a forced tool call becomes `auto` on the models that refuse one ([the full list](usage.md#parameters)). `n=1` is always fine |
| `set_verbose`, `suppress_debug_info` | Accepted |
| Response types under `litellm.types.utils` (`ModelResponse`, `Choices`, `Message`, `Usage` and others) | Same names |
| `Router` | For `model_list`, `fallbacks` and `num_retries`: the groups become routes in the local gateway, which does the retries and fallbacks. Each route is named after its group and its definition (`router-1a2b3c4d:chat`), so programs with a group of the same name keep their own routes, and a group named after a model does not take over that model in other programs. Call the Router's methods with the group name. A new definition of a group removes the route an earlier one left, and a program still using that route adds it back on its next call. Any change restarts the shared gateway on the next call. Deployments with one `model_name` share its traffic as LiteLLM's default `simple-shuffle` shares it: by `weight`, `rpm` or `tpm` in `litellm_params`, the first that any deployment sets, and equally when none does. Keys come from the environment. Other `Router` options warn and are ignored, and so do other `litellm_params`, such as `api_base` (see below). With `AGENTROUTER_BASE_URL` set, nothing is set up on this computer: calls send the group name, `chat`, and that gateway's route of that name applies. `export_config()` names each group's route `chat` too. See [Routes](usage.md#routes-fallbacks-retries-and-names-of-your-own) |

## What is missing

These are not in theagentrouter. Code that uses them fails with `AttributeError`:

| LiteLLM | Instead |
|---|---|
| `image_generation`, `transcription`, `speech` | Not available |
| `text_completion` | Use `completion` with messages |
| LiteLLM's built-in logging integrations (`success_callback = ["langfuse"]`) | The gateway's OpenTelemetry traces and metrics. Your own callback functions and loggers work: [Callbacks](usage.md#callbacks) |
| The LiteLLM proxy server | Agent Router, the gateway: [Gateways](usage.md#gateways) |

## Ignored, with a warning

These `completion` arguments work in LiteLLM but do nothing in theagentrouter. Code that relied on one, say `fallbacks`, would keep running without it and nobody would notice, so each of these warns the first time it is used:

| Argument | Instead |
|---|---|
| `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks` | Define a route with fallbacks, or use `Router(fallbacks=...)` |
| `caching`, `cache` | A gateway caches |
| `metadata`, `tags`, `callbacks`, `logger_fn` (as arguments on one call) | The gateway's request logs record calls; for callbacks, set the module's lists ([Callbacks](usage.md#callbacks)) |
| `custom_llm_provider` | Name the provider in the model, for example `openai/gpt-5.4-mini` |
| `api_version` | Set `AZURE_API_VERSION` where the gateway starts |

`Router` warns in the same way, once for each `Router`, for what it does not use:

| `Router` setting | Instead |
|---|---|
| `routing_strategy` other than `simple-shuffle` | The gateway shares a group's traffic as `simple-shuffle` does. LiteLLM's usage-based strategies also hold each deployment to its `rpm` and `tpm`; with the default strategy LiteLLM does not, and neither does the SDK. For a request limit, define a route with `limit=Limit(...)`; it counts requests per route, not tokens |
| `api_base`, `api_version` in `litellm_params` | Set the provider's address where the gateway starts, such as `AZURE_API_BASE`. `os.environ/AZURE_API_BASE` and `os.environ/AZURE_API_VERSION` do not warn |
| Other `litellm_params`, such as `timeout` | Not used |

A value that asks for nothing does not warn. Other unknown completion arguments warn instead of disappearing silently.

## Other differences

| | LiteLLM | theagentrouter |
|---|---|---|
| Model names | OpenAI and Anthropic names work without a provider; a bare `gemini-…` goes to Vertex AI; the rest need `provider/model` | The same for OpenAI and Anthropic, and the rest. A bare `gemini-…` is refused with the reason, because Vertex AI is not in the local gateway yet: write `gemini/…` for the Gemini API. Every gateway gets the name as you wrote it |
| Where calls go | The library calls more than 100 providers itself | Always through a gateway: Agent Router on this computer, or one you set with `AGENTROUTER_BASE_URL`. The SDK holds no provider addresses or keys; the gateway does |
| Model list | About 4,000 entries, downloaded from GitHub every time `litellm` is imported | About 230 entries, shipped with each release and never downloaded while running. `register_model` adds more, including LiteLLM's whole list. `get_model_info` raises `NotFoundError` for a model that is not on the list |
| Install | Many dependencies | `theagentrouter-core`: one dependency, `httpx`. `theagentrouter`: that plus the local gateway (about 80 MB) |

## Response serialization

Chat Completions responses and their nested messages, choices, tool calls and usage objects are dataclasses with attribute and dictionary-style access. They are not Pydantic models. They support `model_dump(mode="python" | "json", exclude_none=True)` and `model_dump_json(exclude_none=True, indent=2)`. Other Pydantic serialization options, such as `include`, `exclude`, `by_alias` and `exclude_unset`, are not supported. `exclude_none` removes optional model fields recursively; null values inside payload dictionaries and lists are preserved.

`to_dict()` gives the same fields and keeps the ones whose values are `None`.

Anthropic Messages responses are dictionaries. OpenAI Responses replies are dictionary subclasses. For code that handles all three APIs, convert only the Chat Completions object:

```python
data = dict(response) if isinstance(response, dict) else response.to_dict()
```

## Check your code

1. Change the import, or call `install_as_litellm()` before the library that imports `litellm`.
2. Check model names with no provider: OpenAI and Anthropic names work as they are; add the provider to the rest (`mistral/…`, `groq/…`), and write `gemini/…` for Gemini.
3. Remove `api_key=` from calls that use the local gateway, and set the provider's key in the environment instead.
4. Check code that passes Pydantic options other than `mode` and `exclude_none` to `model_dump` ([Response serialization](#response-serialization)).
5. Run your tests with warnings shown (`python -W default`, or pytest's warnings summary). Every ignored argument is named there once.
6. Search your code for the names under [What is missing](#what-is-missing).
