# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

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
- Twenty-one numbered examples, the docs, `llms.txt` and `llms-full.txt`, and two agent skills.
