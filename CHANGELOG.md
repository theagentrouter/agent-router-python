# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- The SDK is part of the Agent Router project. It follows the LF Projects Code of Conduct, security reports can also go to Agent Router's security team, and the packages' author is Agent Router Authors.

### Fixed

- A route from Claude to an OpenAI or Azure model no longer fails on `max_tokens` at the fallback. A route of those models now sends `max_completion_tokens`, which OpenAI's newer models require and Anthropic's OpenAI-compatible endpoint also takes.
- `max_tokens` works when an OpenAI model refuses it behind a name the SDK does not know, such as a route on a gateway you set. The SDK makes the call once more with `max_completion_tokens`, and the limit holds.
- A route whose fallback is a newer Claude model no longer fails on `temperature` or `top_p`: the gateway removes them before the request reaches that model. A gateway already running keeps its old configuration until `local_gateway.stop()`.
- Claude's thinking follows LiteLLM's rules for each model. A token budget stays as written on Opus 4.6 and Sonnet 4.6, and becomes an effort level from Opus 4.7 on. Adaptive thinking becomes a token budget before Opus 4.6 and Sonnet 4.6. `thinking={"type": "disabled"}` is left out on the models that always think, which refuse it.
- `reasoning_effort` on Claude 4.6 and later asks for summarized thinking, so `reasoning_content` is no longer empty. It also takes `{"effort": "low"}`, and a dated Claude name not on the model list follows the rules for its name.
- Without `max_tokens`, Claude gets the model's own limit, as in LiteLLM, instead of 4096, so long answers are no longer cut off. With a token budget, it gets the budget plus 4096. `max_completion_tokens` wins over `max_tokens`.
- What a Claude name the SDK does not know refuses is left for Anthropic to answer, as in LiteLLM, instead of being refused before the call.
- A model name without a rule works with a running local gateway, which restarts with a rule for it, as LiteLLM sends any name to the provider. Before, it raised `NotFoundError`, so a second Bedrock or Azure name failed. A mistyped name now restarts the gateway too, which interrupts other programs' requests.
- A gateway route named like `claude-first` goes in the OpenAI format: only a name with a version, as in LiteLLM, counts as Claude.
- Claude on Bedrock follows the same rules. `thinking` no longer goes missing, signed thinking goes back with a tool result, `reasoning_effort` no longer reaches Claude as an unknown field, and what a newer model refuses raises `UnsupportedParamsError` unless `drop_params=True`. On Bedrock, `tool_choice="none"` no longer forces a tool named "none", and what a model does not take, such as `seed`, raises in the same way. `bedrock/converse/...` and `bedrock/invoke/...` send Bedrock the model's own name. The SDK warns once for what the gateway cannot send to Bedrock: an effort level, `top_k`, `parallel_tool_calls=False` and `response_format`.
- `anthropic.messages` follows LiteLLM's rules for what a model refuses, instead of sending every setting unchanged. It takes `reasoning_effort`, no longer loses `cache_control`, `output_config` and the other Messages options LiteLLM passes, and adds the `anthropic-beta` header each of them needs.
- `response_format` takes a pydantic class, as in LiteLLM, on every provider. For Claude, a JSON schema is changed as in LiteLLM, so a schema without `additionalProperties` or with `maximum` no longer fails.
- More of LiteLLM's Claude behaviour on `completion`: `max_tokens=0` stays 0, an email address or phone number in `user` is left out, Anthropic's own tool shapes and hosted tools go as written, `tool_choice={"type": "none"}` is kept, and `thinking=True` is the medium setting.
- `additional_drop_params` drops the options it names, and `drop_params` on the module holds whatever a call says, as in LiteLLM.
- A route warns when `thinking` or `top_k` cannot be sent, instead of losing them silently, and a framework's private flag such as `_skip_mcp_handler` no longer warns.
- `drop_params` works for what newer Claude models refuse. From Opus 4.7 on, `temperature` other than 1, `top_p` and `top_k` are dropped. On Sonnet 5.5, Opus 5.5 and Fable 5.1, a forced tool call becomes `auto`, with a warning. Without `drop_params`, these now raise `UnsupportedParamsError` before the call, as in LiteLLM, instead of `BadRequestError` from the provider. A gateway you set still gets them, because its model names are its own.
- `reasoning_effort` and `thinking` with a budget work on Opus 4.5, which refuses adaptive thinking.
- OpenAI's "Input tokens exceed the configured limit" raises `ContextWindowExceededError`, as in LiteLLM, instead of `BadRequestError`.
- A web page can no longer reach the local gateway through DNS rebinding: its `SecurityPolicy` also checks the `Host` header, so a browser sending the page's own name gets 403.
- A provider address that does not answer from your network no longer fails the call with 503. Bedrock's hostname has many addresses; the gateway now tries the next one, as an HTTP client does.
- The local gateway no longer starts, rarely, with its admin page on the port the SDK calls, which answered every call with "invalid path".
- No `ResourceWarning: subprocess N is still running` when a program starts the local gateway.
- Docs: the Messages API example and the thinking advice for frameworks used a form that Claude 5 models refuse; the Responses API example read the wrong output item on models that reason; running an exported config says where `aigw` comes from.

## [0.1.0a8] - 2026-11-02

### Changed

- LiteLLM's `Router` names each route after its group and its definition, such as `router-1a2b3c4d:chat`. Two programs with a group of the same name no longer replace each other's route, and a group named after a model no longer takes over that model's calls in other programs. A new definition of a group removes the route an earlier one left. Call the Router's methods with the group name, as before.
- A 429 is a `BudgetExceededError`, and is not retried, only when the gateway says it is a budget block (code `budget_exceeded`, as Agent Router sends, or "budget exceeded" in the message). A provider rate limit that mentions a token budget is a `RateLimitError` and is retried.
- A plain `response_format={"type": "json_object"}` for Claude warns once that it is not applied. Claude has no JSON mode without a schema; pass a JSON schema.

### Fixed

- An image or document in a tool result reaches Claude as an image or document block. It was sent as the text of Python's list printout.
- `Limit` checks its arguments as `Retry` does: `Limit(2.5)`, `Limit(True)` and `Limit("10")` raise `ValueError`.
- `local_gateway.stop()` signals the stored process only while it is still `aigw`, so a process that later got the same number is left alone.

## [0.1.0a7] - 2026-10-20

### Added

- Azure OpenAI through the local gateway: set `AZURE_API_KEY` and `AZURE_API_BASE` (and `AZURE_API_VERSION` for a version other than `2024-10-21`), then call `azure/<deployment>`. `max_tokens` reaches Azure as `max_completion_tokens`, as for OpenAI. Exported configuration includes Azure.

### Changed

- The local gateway package carries the published Agent Router `aigw` v1.2.0 again instead of a build with local patches.
- A new route, registered model or provider restarts the shared local gateway, which interrupts other programs' requests.
- The local gateway's ports listen on every address again. Its `SecurityPolicy` still refuses any caller whose connection does not come from this computer.
- The local gateway no longer sets an MCP session seed. The published `aigw` has no setting for it, and the local gateway configures no MCP servers.

### Removed

- `local_gateway.reload_keys()`. This breaks code that calls it, which now gets `AttributeError`. After rotating a provider key, call `local_gateway.stop()` when other programs have finished; the next call starts the gateway with the new key.

### Security

- Exported server configuration checks the client key with an authorization rule that works on the published `aigw`. A missing or wrong key gets 403 (was 401), a server started without `AGENTROUTER_API_KEY` refuses every request, and the client key is removed before a request reaches a provider.

## [0.1.0a6] - 2026-10-17

### Fixed

- `max_tokens` reaches OpenAI as `max_completion_tokens`, which OpenAI's newer models require; they refuse `max_tokens`. Other providers in OpenAI's format still get `max_tokens`.
- `response_format` with a JSON schema works for Claude: it is sent as `output_config.format`, the field Anthropic takes now. `output_format` was refused.
- `reasoning_effort` and LiteLLM's `thinking={"type": "enabled", ...}` work on Claude 5 models, which think adaptively and refuse a fixed budget: they get `thinking: adaptive` with `output_config.effort`. The model list says which models these are. For older Claude models, a thinking budget that would not fit the caller's `max_tokens` is reduced to fit.
- Adding a route or model to a running local gateway with an HTTPS provider (every real provider) no longer stops it. The gateway's config named `BackendTLSPolicy` by an older API version, so the reload could not update it. Two examples no longer assume Claude's reply starts with text, and the tool example uses a model that accepts a forced tool call.
- A relative or `~` `AGENTROUTER_HOME` works: the local gateway resolved it a second time inside that folder and did not start.

### Changed

- The docs, examples, skills and apps use current models: `openai/gpt-5.4-mini` and `openai/gpt-5.4` for Chat Completions, `openai/gpt-6-luna` for the Responses API, and `anthropic/claude-haiku-5-5` and `anthropic/claude-sonnet-5-5`. OpenAI deprecated GPT-4o, GPT-4o mini, GPT-4.1 and GPT-4.1 mini on 2026-10-14.
- `scripts/smoke_examples.py` runs every example against real providers; the release steps include it.
- The examples are grouped by what they show: `from_litellm/`, `first_calls/`, `routes/`, `gateways/`, `agent_frameworks/` and `provider_apis/`, with task names instead of numbers. `examples/README.md` lists them.

## [0.1.0a5] - 2026-10-17

### Fixed

- Retry settings are checked before saving a route. Fractional timeouts and long backoffs produce valid gateway durations, so one malformed retry policy cannot prevent every program's local gateway from starting.
- Stream establishment errors raise at the call or await, malformed responses raise SDK errors, loopback prompts bypass proxy environment variables, and HTTP connections are reused. HTTP exceptions retain response headers; bounded request retries are available without replaying partial completion streams.
- Cache-aware cost estimates, JSON serialization, response metadata, Claude file/image handling and LiteLLM module defaults/imports are supported. Unsupported arguments and audio content raise or warn instead of silently disappearing.
- Interrupted gateway starts are stopped and reaped. Startup locking has a timeout, runtime/state permissions are restricted, stopped runs' generated key files are removed, and supplied rotated credentials produce an actionable error.
- Unknown models cannot restart an existing shared gateway. Existing model rules are reused across SDK versions, and reconfiguration cannot silently drop providers whose credentials the caller lacks. Custom models must be registered before adding them to a running gateway.
- With the bundled build, every port the local gateway opens is on loopback only: Envoy's listener and readiness listener, Envoy Gateway's xDS, wasm cache and health servers, the MCP proxy and the external processor's admin port. Envoy Gateway's metrics server is not started. This needs the fourth bundled `aigw` patch and a patched Envoy Gateway; the builder applies both.
- A new route or model reloads the shared local gateway instead of restarting it, so other programs' requests and streams continue. This needs the third bundled upstream patch; a companion without it restarts as before. A new provider is added the same way, except Bedrock.
- `get_model_info` and `model_cost` give each model's output limit (`max_tokens`, `max_output_tokens`) and, where the model list has one, its input limit. The SDK read a field the model list does not have, so both were `None` for every model. The bundled list is refreshed.
- A model the list still offers but marks deprecated has LiteLLM's `deprecation_date`, and the first call to it warns with the list's reason. A model the list no longer offers raises `NotFoundError` saying why, instead of calling it unknown.
- A call through the local gateway reads its settings, routes and the gateway's state only when they change, and checks that the gateway is listening at most every 30 seconds; it opened four files and a probe connection each time.
- A streamed `anthropic.messages.create` or `acreate` retries as `num_retries` asks, before its first event, like completion streams. It sent the request once.
- After a provider key is rotated, `local_gateway.reload_keys()` gives the running local gateway the new key without interrupting other programs; the error for a changed key says so. The local config names a private key file for each provider instead of an environment reference, so a reload reads the new key.
- Routes can be defined for export without local keys or the companion. Provider base-URL variables are used, AWS regions are validated, zero timeouts are preserved, and request representations hide credentials and explicit authentication headers.

### Security

- Exported server configuration requires client-key authentication by default; an open endpoint requires `authentication=False`. This needs the bundled upstream Secret-retention patch and source build described in `local-gateway/patches/README.md`; no patched upstream release has been published.
- The local gateway gives `aigw` a random MCP session seed at each start, so the MCP proxy that `aigw run` always starts no longer uses its public default. This needs the second bundled upstream patch. An exported server reads the seed from `AIGW_MCP_SESSION_ENCRYPTION_SEED`.
- Release workflows use exact companion artifact names, publish the verified bundle by artifact ID, and pin third-party actions to full commit SHAs.

### Added

- Python 3.14 classifier and CI coverage.

## [0.1.0a4] - 2026-10-16

### Security

- The local gateway answers only callers on this computer. Envoy listens on every network address, so anyone who could reach the port could make calls with the developer's provider keys; calls from another address now get 403. The docs said it listened on `127.0.0.1` only. `aigw`'s own internal ports still listen on every address; that needs a change in Agent Router.
- A key stored in `AGENTROUTER_HOME/config.json` is no longer sent to a different gateway set with `AGENTROUTER_BASE_URL`. It goes only to the gateway stored with it.

### Fixed

- The local gateway starts on macOS 13 and 14 again. `aigw` downloaded the newest Envoy at each start, and Envoy 1.40.0 needs macOS 15.5. The SDK now asks for Envoy 1.39.4 (`local_gateway.ENVOY_VERSION`); setting `ENVOY_VERSION` overrides it. A server running an exported config should set it too.
- Streams have a choice in every chunk, as in LiteLLM, so `chunk.choices[0].delta.content` works to the end of the stream. The usage comes on the chunk with the finish reason; with `stream_options={"include_usage": True}` it comes on one more chunk with an empty choice, as LiteLLM sends it.
- Claude takes `n=1`, and `drop_params=True` (on the call, or `litellm.drop_params = True`) drops the parameters Claude does not take, such as `seed` and `presence_penalty`, instead of raising. `drop_params` is off by default, as in LiteLLM; it was on and did nothing.
- `thinking=` reaches Claude. It was left out of the request. With no `max_tokens`, the request leaves room for the answer after the thinking budget.

## [0.1.0a3] - 2026-10-13

### Added

- LiteLLM's `mock_response`, with `mock_tool_calls`, `mock_timeout` and `mock_delay`: `completion` and `acompletion` answer on this computer, streamed or not, with no gateway and no provider call. It was ignored with a warning, so a test that used it called the provider.

### Fixed

- Chat Completions responses and stream chunks accept `model_dump(mode="json", exclude_none=True)`. Optional model fields are omitted recursively while null values inside payload dictionaries and lists are kept.

### Changed

- The migration guide explains how Chat Completions responses serialize.

## [0.1.0a2] - 2026-10-13

### Changed

- A gateway set with `AGENTROUTER_BASE_URL` gets the same requests as the local gateway. Claude goes in Anthropic Messages to `/anthropic/v1/messages`, so prompt caching and thinking work through it too; before, every call went there in the OpenAI format. `AGENTROUTER_BASE_URL` works with or without `/v1`. Claude options that Messages lacks, such as `response_format`, now fail the same way with both gateways.
- The model name is sent as written. `claude-sonnet-4-5` and `gpt-4o-mini` reach the local gateway as they are, and its config matches them as well as `anthropic/claude-sonnet-4-5` and `openai/gpt-4o-mini`, as Agent Router Enterprise names them.

### Added

- LiteLLM's Anthropic Messages API: `anthropic.messages.create` and `acreate`, and `anthropic_messages`. Requests and replies in Anthropic's own format, for Claude, through the gateway's `/anthropic/v1/messages`; streaming gives the events as bytes, as in LiteLLM. Also as `litellm.anthropic.messages` after `install_as_litellm()`.
- Examples `21_responses.py` (OpenAI's Responses API) and `22_messages.py` (Anthropic's Messages API with prompt caching and thinking).
- `register_model`, as in LiteLLM: add models or change their details from a dict or a URL in LiteLLM's format, including LiteLLM's whole model list, or a file of tetrateio/agent-router-models. Registered models get prices, details and a gateway rule.
- `local_gateway.export_config()`: the local gateway's config, to run the same gateway on a server with `aigw run`. It holds the bundled model list for the chosen providers, any extra models named with `models=`, and this computer's routes, with key references and no key values.

### Fixed

- For Claude, `usage.prompt_tokens_details.cached_tokens` is 0 when nothing was read from the cache, as in LiteLLM; it was `None`, so reading `cached_tokens` failed.
- `max_tokens` in the model list is LiteLLM's output limit; it held each model's whole context window. The list has no output limits yet, so it is now `None` from `get_model_info`, as LiteLLM gives a limit it does not know.
- httpx is capped below 1.0. `pip install --pre theagentrouter` installed httpx 1.0.dev6, which removes `httpx.Timeout` and `httpx.HTTPError`, and every call failed. A plain `pip install theagentrouter` was not affected.

## [0.1.0a1] - 2026-10-12

The first release.

### Added

- LiteLLM's API, for what agent frameworks call: `completion` and `acompletion` (streaming, tools, structured output), `embedding` and `aembedding`, `responses` and `aresponses` for OpenAI models, `token_counter`, `stream_chunk_builder`, `validate_environment`, `completion_cost`, `model_list`, `cost_calculator`, LiteLLM's errors, and `success_callback`, `failure_callback` and `callbacks`.
- `theagentrouter.compat.litellm`: change one import to move a LiteLLM program over. LiteLLM's `Router`, with `model_list`, `fallbacks` and `num_retries`, is served by routes.
- `install_as_litellm()` in `theagentrouter.compat`, so Google ADK and the OpenAI Agents SDK run on the SDK without LiteLLM installed.
- Every call goes through an Agent Router gateway: the local gateway (`aigw` v1.2.0), started by the first call and shared by every program on the computer, or a gateway set with `AGENTROUTER_BASE_URL`. The local gateway chooses its port at each start and restarts after the SDK is upgraded.
- Claude goes to the gateway in Anthropic's Messages format, so prompt caching (`cache_control`) and thinking (`thinking_blocks`, `reasoning_content`) work as in LiteLLM.
- Routes: `route(name, primary, fallbacks, retry, limit)`, `list_routes` and `remove_route`. The local gateway does the retries, the fallbacks and the request limits. Routes are saved in `~/.agentrouter/routes.json`.
- Three packages: `theagentrouter` (the SDK and the local gateway), `theagentrouter-core` (the SDK alone, for a remote gateway) and `theagentrouter-local-gateway` (`aigw`, for macOS 13+ on Apple chips and Linux with glibc 2.36+ on x86-64 and Arm, plus a wheel with no program for every other computer).
- A bundled model list of 228 models, for prices and model details.
- Twenty-one numbered examples, the docs, and two agent skills.
