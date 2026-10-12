# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Added (2026-10-12)

- After an SDK upgrade, the next call restarts a local gateway that the old version started. `state.json` records the SDK's and the local gateway's versions.

### Fixed (2026-10-12)

- `pip install theagentrouter` no longer fails where the local gateway has no wheel (Intel macOS, macOS 12 and older, Alpine, Linux before glibc 2.36, Windows). A `py3-none-any` local gateway wheel with no program is published next to the platform wheels; pip picks it only where none fits, and the SDK then runs without a local gateway. The macOS wheel needs macOS 13 or newer; the docs said 12.
- Two programs defining routes at the same time no longer lose one: the route file is locked while it is changed and written in one step.
- Removing a route named like a model (`openai/gpt-4o-mini`) restarts the local gateway on the next call; the route's fallbacks, retries and limit stayed active before.
- With `AGENTROUTER_AUTOSTART=0`, a route works when the running gateway already has it.
- Claude: `cache_control` on content parts, on a whole message (LiteLLM's form) and on tools reaches Anthropic, so prompt caching works; the system prompt goes as text blocks when one is marked. Cache reads show in `usage.prompt_tokens_details.cached_tokens`.
- Claude: streamed thinking arrives as `delta.reasoning_content`, and responses and rebuilt streams carry `thinking_blocks` with their signatures, as LiteLLM's do. Streamed, each fragment of thinking text comes in its own chunk and the signature alone in a last chunk, LiteLLM's shape, so Google ADK puts the block back together unchanged. Sent back with a tool result, the blocks go to Anthropic first in the turn, unchanged, which Anthropic requires when thinking is on.
- `stream_chunk_builder` rebuilds each choice on its own; with `n=2` the two answers were joined into one.
- `acompletion`, `aembedding` and `aresponses` no longer block the event loop while the local gateway starts.
- `completion_cost` works on a `responses()` reply (it read only Chat Completions usage).
- `validate_environment` for Bedrock asks for `AWS_SECRET_ACCESS_KEY` and `AWS_REGION` too, as the local gateway does.
- The release workflow runs the route and embedding tests against the real `aigw`, not only the local gateway tests.

### Changed (2026-10-12)

- The bundled model list is read from `tetrateio/agent-router-models` (the SDK pointed at a `theagentrouter/` copy that does not exist) and refreshed: 228 models. DeepInfra models are keyed `deepinfra/<model>`; they were `deepinfra/deepinfra/<model>`, so lookups by their usual name found nothing.
- **The local gateway chooses its port at each start** instead of always using 1975, so Tetrate Desktop or another `aigw run` on 1975 no longer blocks it. SDK programs find it through `state.json`. When `aigw`'s fixed internal ports (1061, 18000) are taken, the error names the program holding them.
- **The SDK talks only to a gateway.** It no longer calls OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek or Together directly: every call goes to the local gateway (Agent Router on this computer), to a gateway set with `AGENTROUTER_BASE_URL`, or to an endpoint named with `api_base=` (LiteLLM's argument). Claude goes to the gateway in Anthropic's Messages format, so prompt caching and thinking keep working.
- **Three packages.** `theagentrouter` installs the SDK and, on macOS 13+ (Apple chips) and Linux (x86-64, Arm), the local gateway; it has no code of its own. `theagentrouter-core` is the SDK alone, for a remote gateway (servers, containers, serverless, CI). `theagentrouter-local-gateway` holds `aigw`. All import as `import theagentrouter`. `theagentrouter[local-gateway]` still works.
- `api_key=` without `api_base=` warns and has no effect: the local gateway reads provider keys from the environment.
- `_hidden_params["route"]` is `local gateway`, `gateway` or `api_base` (was `direct` for calls to providers).

### Added

- The rest of what agent frameworks call from LiteLLM (counted across 20 frameworks that import it): OpenAI and Anthropic model names work without a provider, as in LiteLLM (`gpt-4o-mini`, `claude-haiku-4-5`); a bare `gemini-…` is refused with the reason, since LiteLLM sends it to Vertex AI; `responses` and `aresponses` for OpenAI models, through the gateway's `/v1/responses`; `stream_chunk_builder`; `validate_environment`; `success_callback`, `failure_callback` and `callbacks` in `theagentrouter.compat.litellm`, calling your functions and loggers after each call and stream; `model_list`, `cost_calculator`, and `ChatCompletionDeltaToolCall`, `CostPerToken` and `ChatCompletionMessageToolCall` under `types.utils`.
- A route defined in Python warns when `AGENTROUTER_BASE_URL` is set: routes program the local gateway only.
- `token_counter`, with LiteLLM's arguments, also under `theagentrouter.compat.litellm`. OpenAI and Anthropic are counted by the gateway (`/responses/input_tokens`, `/messages/count_tokens`); everything else is estimated with LiteLLM's rules, using tiktoken from the new `tokens` extra or about four characters a token without it. `AGENTROUTER_TOKEN_COUNT=estimate` always estimates.
- Embeddings: `theagentrouter.embedding` and `aembedding`, also under `theagentrouter.compat.litellm`, with LiteLLM's arguments and `EmbeddingResponse`, priced from the model list. Direct for OpenAI-format providers, through a gateway for the rest; Anthropic has none.
- Request limits on routes: `route(..., limit=Limit(requests, per="minute"))`, enforced by the local gateway with Envoy's local rate limit. Requests, not tokens: counting tokens needs Redis, which `aigw run` does not have.
- Routes: `theagentrouter.route(name, primary, fallbacks, retry)`, with `Retry`, `list_routes` and `remove_route`. A route is a model name of your own; the local gateway does its retries and fallbacks, with the models at priority 0, 1, …, as Agent Router Enterprise does. Models in one tier share the traffic.
- LiteLLM's `Router` in `theagentrouter.compat.litellm`, for `model_list`, `fallbacks` and `num_retries`, served by routes.
- Responses from a route carry `served_by` (read from the reply's model name) and `route_name` in `_hidden_params`, and are priced as the model that answered.
- The local gateway writes a backend at `127.0.0.1` as `localhost`, and waits after starting until Envoy has looked up every route's backends.
- Numbered examples, `examples/00_two_liner.py` to `examples/20_request_limit.py`, with an index.

### Changed

- The local gateway uses `aigw` v1.2.0 (was v1.1.0). Request limits need it: v1.1.0 is built on Envoy Gateway v1.8.1, whose rate-limit schema makes `aigw run` reject every `rateLimit` (`requests` declared int32 with a maximum of 4294967295); Envoy Gateway v1.9 fixed it.

### Changed

- The warning for `fallbacks` and `num_retries` now points to routes.

## [0.1.0a1] - 2026-10-07

### Added

- `completion` and `acompletion` with LiteLLM's arguments, streaming and tool calls.
- Direct calls with no gateway to OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek and Together, using the key in the environment.
- Anthropic Messages support: LiteLLM-style messages and tools are converted, and replies and streams are converted back.
- Gateway mode: with `AGENTROUTER_BASE_URL` set, every call goes to that gateway in the OpenAI format.
- Settings from `AGENTROUTER_HOME/config.json` when the environment does not set them.
- LiteLLM's error names, mapped from provider HTTP errors.
- `get_model_info`, `model_cost`, `completion_cost` and capability lookups, from a snapshot of tetrateio/agent-router-models.
- `theagentrouter.compat.litellm` and `install_as_litellm()`, so Google ADK's `LiteLlm` class and the OpenAI Agents SDK's `LitellmModel` work without LiteLLM installed.
- A warning, once per argument, when a LiteLLM-only argument such as `fallbacks` or `num_retries` is passed and ignored.
- `usage.prompt_tokens_details`, `usage.completion_tokens_details`, `choice.logprobs`, `delta.refusal` and `model_copy()` on responses, as in LiteLLM.
- Documentation: install, usage, coming from LiteLLM, and agent frameworks, under `docs/`.
- The local gateway: `pip install "theagentrouter[local-gateway]"` installs Agent Router's `aigw` (package `theagentrouter-local-gateway`). The first call that needs it, such as a `bedrock/` model, starts it in the background with the providers whose keys are in the environment. While it runs, calls to its configured providers go through it, except direct-provider calls with an explicit `api_key`. `AGENTROUTER_AUTOSTART=0` turns starting off.

### Fixed

- An explicit provider `api_key` is honored even while the local gateway runs. Direct providers bypass it for that call.
- Anthropic maps `parallel_tool_calls=False` to `disable_parallel_tool_use`. OpenAI-only parameters that cannot be translated raise `UnsupportedParamsError`; parameter discovery reports the supported set.
- OpenAI responses retain refusal, reasoning, log probabilities and cached/reasoning token details, including streams.
- Provider error events in streams raise the matching SDK exception with the requested model and provider.
- Preparing a request leaves the caller's tool-call history unchanged.
- Releases build, verify and publish the SDK and three companion wheels together. The companion uses checksum-pinned upstream binaries; required integration tests cannot silently skip.

### Changed

- A provider that needs a gateway now says how to install the local gateway, instead of only asking for `AGENTROUTER_BASE_URL`.
- With a gateway set, a model name with no provider, such as `claude-sonnet-5`, goes to the gateway unchanged instead of being refused.
- The local gateway receives the full `provider/model` name, as any gateway does, and routes it with one rule per model. No SDK-only header is sent.
