# Local gateway

The SDK talks only to a gateway: it never calls a provider itself. `pip install theagentrouter` installs the SDK (`theagentrouter-core`) and `theagentrouter-local-gateway`, which holds the `aigw` program; `pip install theagentrouter-core` is the SDK alone, for a remote gateway. The SDK starts it on the first call, writes its config from the provider keys in the environment, and sends every call through it. The developer runs no command. The SDK sends any format the gateway accepts (Claude in Anthropic Messages, the rest in OpenAI's), and holds no provider address, key or authentication of its own.

## Packages

| Package | Contents | Wheels |
|---|---|---|
| `theagentrouter` | No code (`packages/theagentrouter/`, a hatchling wheel with `bypass-selection`). Depends on `theagentrouter-core==<same version>` and `theagentrouter-local-gateway==<same version>` on every platform, with no environment marker. Its empty `local-gateway` extra keeps older instructions working | One wheel, metadata only |
| `theagentrouter-core` | The SDK, the repository's root package. Depends on `httpx` only; its `local-gateway` extra names the companion | One pure-Python wheel |
| `theagentrouter-local-gateway` | `theagentrouter_local_gateway/bin/aigw`, `binary_path()` and `has_binary()` | One per platform, built on that platform by `local-gateway/hatch_build.py`: macOS arm64 (`macosx_13_0`, the minimum read from the executable) and Linux amd64/arm64 (`manylinux_2_36`). Plus one `py3-none-any` fallback with no program (`AGENTROUTER_NO_BINARY_WHEEL=1`), which pip installs only where no platform wheel fits, so `pip install theagentrouter` never fails |

The second package lives in this repo under `local-gateway/` and is released with the same version as `theagentrouter`. `make local-gateway AIGW=/path/to/aigw` puts a program in it and installs it for development.

The SDK never imports it at module load. `local_gateway.installed()` is checked only when a call needs the gateway: the package must be importable and `has_binary()` true, so the fallback wheel counts as not installed.

### Size

The macOS arm64 wheel is about 80 MB compressed (the program itself is 281 MB). That is under PyPI's default limit of 100 MB per file. The workflow checks the wheel metadata and builds Linux wheels too. Each wheel must fit PyPI's per-file limit before publication.

## Where a call goes

`routing.resolve`, first match wins:

1. `api_base` on the call: that endpoint, in the provider's own format (LiteLLM's argument).
2. A gateway set with `AGENTROUTER_BASE_URL` (or `config.json`).
3. A route name: the local gateway's route for it. The running gateway is used only if it has this computer's current routes (`routes_fingerprint` in `state.json`); otherwise it restarts. With `AGENTROUTER_AUTOSTART=0`, a running gateway that already has the route is still used.
4. Anything else: the running local gateway, if it has a rule for this name and the name was not a route when it started (`local_gateway.serves`); otherwise the gateway is started or restarted with this model, if installed and `AGENTROUTER_AUTOSTART` is not `0`. Otherwise an error says to set `AGENTROUTER_BASE_URL`. An `api_key=` here cannot reach the provider (the gateway holds the keys), so it warns once.

## Starting and stopping

State lives in `AGENTROUTER_HOME/local-gateway/`: `config.yaml`, `state.json` (pid, port, providers), `aigw.log`, `lock`, and `aigw`'s own config and state folders. `aigw`'s runtime folder, which holds Unix sockets, is a short path under `/tmp`, one per `AGENTROUTER_HOME`.

1. A program takes a file lock on that folder before starting or restarting, so two programs never start two gateways.
2. If `state.json` names a live process and its port accepts connections, the SDK uses that gateway.
3. Otherwise it writes `config.yaml`, starts `aigw run` in its own session (so it outlives the program), and waits for `aigw`'s "listening on" line and an open port. It prints the one-line notice once per start.
4. A call to a provider the running gateway lacks, with that provider's key set, stops the gateway and starts it again with every provider in this program's environment. `aigw` reads its config once at start.
5. The gateway's own ports are chosen free at each start, so another gateway on 1975 does not block it; if a chosen port is taken before `aigw` binds it, the SDK chooses again (three tries). `aigw`'s internal ports (1061, 18000) cannot move: when one is taken, the SDK names the program holding it (via `lsof`) and asks to stop it or set `AGENTROUTER_BASE_URL`.

`local_gateway.stop()` stops it; the docs give it as a `python -c` line, since there is no command (see Open). `state.json` records the SDK and companion versions. Existing rules are reused across versions; configuration changes require an explicit stop if another version started the process. Unknown models cannot restart an existing process; register custom models first.

The start lock times out after 10 seconds. The SDK checks that its runtime directory is owned by the current user, rejects symlinks, and sets mode 0700. It writes state atomically with mode 0600 immediately after spawning, so an interrupted start can be stopped and reaped. Stop removes generated key/state files once the process has stopped. Credential fingerprints detect supplied key changes; reloading requires an explicit stop after other programs finish. A process with no provider key can continue using an already configured provider, but cannot reconfigure the gateway while dropping its credentials.

## Config

At each start the SDK also gives `aigw` a random `AIGW_MCP_SESSION_ENCRYPTION_SEED`, unless one is set: `aigw run` always starts an MCP proxy, which otherwise derives its session keys from a public default.

The SDK writes the resources `aigw run` reads: a `GatewayClass` and `Gateway` at a free port chosen at each start (kept in `state.json`; `--admin-port` is chosen the same way), a buffer-limit `ClientTrafficPolicy`, a `SecurityPolicy` that answers only callers on this computer (`127.0.0.1` and `::1`; others get 403), one `AIGatewayRoute`, and per provider a `Backend`, a `BackendTLSPolicy` for HTTPS, an `AIServiceBackend`, a `BackendSecurityPolicy` and, for API keys, a `Secret`.

**The SDK's `config.yaml` holds key references, not key values.** Each `Secret` holds `${OPENAI_API_KEY}`-style references. `aigw` fills them in from its environment and writes its translated resources and runtime configuration under `aigw-state/`. Those generated files can contain credentials. The end-to-end test checks the SDK input file and the key received by the provider; it does not assert that the upstream process keeps every generated file free of keys.

| Provider | Turned on when these are set | Schema |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | OpenAI |
| Anthropic | `ANTHROPIC_API_KEY` | Anthropic |
| AWS Bedrock | `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`; and `AWS_REGION` or `AWS_DEFAULT_REGION` | AWSBedrock, with the AWS default credential chain (no credentials file) |
| Gemini, Groq, Mistral, DeepSeek, Together | That provider's key (`GEMINI_API_KEY` and so on) | OpenAI, with the provider's base path |

Azure OpenAI and Google Vertex AI are not set up yet. A call to them says so.

## Routing

The SDK sends the model name as the caller wrote it (`bedrock/anthropic.claude-sonnet-4-5`, `claude-sonnet-4-5`), on the same paths and in the same formats as to a gateway you set. There is no SDK-only header: the local gateway is stock Agent Router, and the same config serves a gateway that a team runs elsewhere.

`aigw` routes on the model name as it is sent and cannot strip a `bedrock/` prefix. So the config has **one route rule per model**. Each rule matches the full name in the `x-ai-eg-model` header that `aigw` sets from the request, and for OpenAI and Anthropic the name without a provider too (`gpt-4o-mini`, `claude-sonnet-4-5`, as LiteLLM and Agent Router Enterprise write them). It passes the provider its own name with `modelNameOverride`.

- The rules cover the bundled model list for the providers whose keys are set. A new model added to a running gateway must be explicitly registered. Known configuration changes can restart a same-version gateway; a typo cannot.
- `export_config()` writes the same config for a server, on a fixed port. A server cannot add rules on demand, so it gets the bundled list plus the models named in `models=`.
- Gateway API allows at most 16 rules in one route, and `aigw` adds one rule of its own to each `AIGatewayRoute`. So the rules are spread over several `AIGatewayRoute`s of at most 15 (`RULES_PER_ROUTE`). A route with more never starts: `aigw` logs `HTTPRoute ... is invalid: spec.rules: Too many` only with `--debug`, and waits.

**Anthropic goes in its own format.** This `aigw` build has no translation from OpenAI Chat Completions to the Anthropic schema ("unsupported API schema"). It takes Anthropic Messages as they are, under `/anthropic/v1/messages` ([`theagentrouter/agent-router`](https://github.com/theagentrouter/agent-router) `internal/internalapi/internalapi.go`). So for Anthropic the SDK sends Messages to that path. Every other provider gets Chat Completions on `/v1/chat/completions`.

### Routes

A route (`theagentrouter.route()`, or LiteLLM's `Router`) becomes **one `AIGatewayRoute`** matching the route's name, with its models as backends at priority 0, 1, … (one per tier, models in a tier at the same priority), and a `BackendTrafficPolicy` that tries each tier `attempts + 1` times before moving to the next. This is Agent Router's provider fallback, the way Agent Router Enterprise renders it. The SDK sends one request named after the route; the gateway does the retries and the fallback.

**Every backend must be a hostname.** Under `aigw run`, a backend written as an IP address gets its endpoints over EDS, and the AI Gateway extension server then labels the whole rule with its **first** backend ([`theagentrouter/agent-router`](https://github.com/theagentrouter/agent-router) `internal/extensionserver/post_translate_modify.go`, the `LoadAssignment == nil` case). A fallback is then sent the first backend's model name and key, and the priorities are lost. The same providers written as `localhost` fall back correctly. So the config writes `127.0.0.1` and `::1` as `localhost`. A provider at another IP address cannot fall back correctly until this is fixed in Agent Router.

**Envoy looks up hostnames after it starts listening.** Until it has, a fallback has nowhere to go, and requests sent right after the start get 503. So after a start the SDK waits, up to 15 seconds, until Envoy's `/clusters` lists a host for every backend of every route.

A route of Anthropic and other providers reaches Anthropic through its OpenAI-compatible endpoint (`https://api.anthropic.com/v1`, backend `anthropic-openai`), because routes go through the OpenAI format and `aigw` does not translate it to Anthropic Messages.

**Request limits** (`Limit`) go in the same `BackendTrafficPolicy` as the retry, because Envoy Gateway takes one per route: `rateLimit.type: Local`, one rule with `requests` and `unit`. Envoy's local rate limit filter answers 429 before the router, so a limited request is never retried. Two limits on what the local gateway can do:

- It counts **requests, not tokens.** Agent Router's token limits (`QuotaPolicy`, usage-based rate limiting) need Redis and a rate-limit service configured in Envoy Gateway at install; `aigw run` has neither, and Envoy Gateway's local rate limit cannot take a cost from the response (`envoyproxy/gateway` v1.9 `api/v1alpha1/ratelimit_types.go`, "response cost is not supported for Local Rate Limits").
- It needs an **`aigw` built on Envoy Gateway newer than v1.8.1.** With Envoy Gateway v1.8.1, `aigw run` rejects any `rateLimit`: the schema declares `requests` as int32 with a maximum of 4294967295, so validation fails for every value.

**Embeddings** need no config of their own: `aigw` sets the model header from the body on `/v1/embeddings` too, so the same per-model rules apply.

A prefix-strip option in Agent Router itself would replace the per-model rules with one rule per provider, and would let a stock gateway in Docker take `provider/model` names too. That needs a change in Agent Router first.

## Tests

| What | Where |
|---|---|
| Config per provider, no key in the file, one rule per model, rules spread over routes of 15; where calls go (not installed, the fallback wheel, `AGENTROUTER_AUTOSTART=0`, unknown provider, missing credentials, running gateway, set gateway wins, stale state, restart on a new provider, model or removed route) | `tests/test_local_gateway.py`, `tests/test_release_regressions.py`, no `aigw` needed |
| Real `aigw` against the fake provider: start on first call, key filled in by `aigw`, streaming tool call, Anthropic through the gateway, a second program reuses it, restart on a new key and model, the provider gets its own model name, error mapping, ports in use | `tests/test_local_gateway.py`, skipped unless the package is installed |
| Routes: config per model, retry policy, limits, the order models are tried, errors that do not move on, LiteLLM's `Router`, two programs defining routes at once | `tests/test_routes.py`, `tests/test_release_regressions.py` |
| Real `aigw` against two fake providers: fallback with each provider's own model name and key, retries then giving up, no retry on 401, `Router` with a fallback, a request limit answering 429, embeddings | `tests/test_routes.py`, `tests/test_embeddings.py`, skipped unless the package is installed. CI runs them on every platform with a program, and none may skip |
| Real OpenAI and Anthropic through the real gateway, and a second program | `scripts/smoke_local_gateway.py` |
| Real OpenAI and Anthropic routes: primary answers, a failing primary falls back to Anthropic, `Router`, a shared tier | `scripts/smoke_routes.py` |
| All 21 examples on real providers through the local gateway | `examples/` |

Run the real-provider scripts one at a time and stop the gateway between them: see `DEVELOPMENT.md`.

## Open

1. **The listener and `aigw`'s own ports listen on every network address.** Envoy Gateway binds every listener to `0.0.0.0` and has no setting to change it (`envoyproxy/gateway` v1.9 `internal/gatewayapi/listener.go`), so the SDK's `SecurityPolicy` refuses calls from other computers rather than not listening for them. `aigw`'s internal ports (xDS 18000, health 8081, 18002, 19001) also listen on every address, and `aigw run` has no flag for it; that needs a change in Agent Router.
2. **Fixed internal ports.** `aigw run` binds fixed ports inside itself (extension server 1061, Envoy Gateway xDS 18000, health 8081, admin 19000 and others), so only one Agent Router gateway runs per computer. The SDK chooses its own listener and admin ports at each start and names the program holding 1061 or 18000. Moving the internal ones needs changes in Agent Router and Envoy Gateway.
3. **Windows and Intel macOS have no program.** Upstream provides macOS arm64 and Linux amd64/arm64 binaries only; those computers get the fallback wheel. The start lock and the route lock are POSIX only.
4. **Envoy is downloaded on first start** into `~/.local/share/aigw` (about 100 MB, and older versions stay). The SDK sets `ENVOY_VERSION` (`local_gateway.ENVOY_VERSION`, the 1.39 line that Envoy Gateway v1.9 targets) unless it is already set; without it `aigw` takes the newest Envoy, and Envoy 1.40.0 does not start on macOS 14. To ship it in the wheel, test `aigw run` with a pre-filled `AIGW_DATA_HOME` and no network.
5. **Cross-process credential merging is not implemented.** Reconfiguration is refused unless the caller has credentials for every provider already running. A provider is no longer silently dropped.
6. **Every route change restarts the gateway**, because `aigw` reads its config once at start.
7. **Backends at an IP address cannot fall back** correctly until Agent Router labels each backend on its own (see "Routes").
8. **Bedrock with real AWS credentials**, including `AWS_PROFILE`, is untested.
9. **Azure OpenAI and Google Vertex AI** are not set up.
10. **No command to stop the gateway**, only `local_gateway.stop()`. Not planned for now.
