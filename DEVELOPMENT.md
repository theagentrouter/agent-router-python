# Development

## What is where

| Folder | What it holds |
|---|---|
| [`src/theagentrouter/`](src/theagentrouter/) | The SDK. [`local_gateway.py`](src/theagentrouter/local_gateway.py) starts and finds the local gateway |
| [`local-gateway/`](local-gateway/) | The `theagentrouter-local-gateway` package. Its `aigw` program goes in `src/theagentrouter_local_gateway/bin/`, which git ignores |
| [`packages/theagentrouter/`](packages/theagentrouter/) | The `theagentrouter` package: no code, it installs `theagentrouter-core` (this repository's root package, the SDK) and the local gateway |
| [`tests/`](tests/) | Tests against a local fake provider; [`test_local_gateway.py`](tests/test_local_gateway.py) also runs the real `aigw` when it is installed |
| [`scripts/`](scripts/) | Real-provider checks, including [`smoke_local_gateway.py`](scripts/smoke_local_gateway.py), and the surface and model-table tools |
| [`examples/`](examples/) | Runnable examples |
| [`docs/`](docs/) | The documentation linked from the README |
| [`skills/`](skills/) | Agent skills; `llms.txt` and `llms-full.txt` at the root are built from the docs, examples and skills by `make llms` |

## Set up

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Python 3.10 or newer. The only runtime dependency is `httpx`.

## Test

```sh
.venv/bin/pytest
```

The tests run over real HTTP against `tests/fake_provider.py`, a local server that answers like OpenAI Chat Completions and Anthropic Messages, including streaming and errors. They need no keys and no network.

### The Google ADK test

`tests/test_adk.py` runs a real Google ADK agent on this SDK. It is skipped unless `google-adk` is installed, and it must run where LiteLLM is **not** installed, so that only this SDK can answer `import litellm`:

```sh
python3 -m venv .venv-adk
.venv-adk/bin/pip install -e ".[test]" google-adk
.venv-adk/bin/pytest tests/test_adk.py
```

### The OpenAI Agents SDK test

`tests/test_openai_agents.py` runs a real OpenAI Agents SDK agent, through the Agents SDK's `LitellmModel`, on this SDK. Same rule as the ADK test: skipped unless `openai-agents` is installed, and run where LiteLLM is **not** installed:

```sh
python3 -m venv .venv-agents
.venv-agents/bin/pip install -e ".[test]" openai-agents
.venv-agents/bin/pytest tests/test_openai_agents.py
```

### Real providers

`scripts/smoke_real.py` calls real providers for each key it finds in the environment. It checks a text reply, a tool call, a streamed tool call with usage, and the error for an unknown model. It costs a few cents at most.

```sh
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... .venv/bin/python scripts/smoke_real.py
```

`scripts/smoke_adk_real.py` runs a real Google ADK agent through the SDK on the same providers, with LiteLLM not installed. It needs a virtual environment with `google-adk` and this package, as in "The Google ADK test" above.

```sh
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... .venv-adk/bin/python scripts/smoke_adk_real.py
```

`scripts/smoke_agents_real.py` does the same with a real OpenAI Agents SDK agent, in the environment from "The OpenAI Agents SDK test" above.

```sh
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... .venv-agents/bin/python scripts/smoke_agents_real.py
```

Instead of setting keys on the command line, put `export OPENAI_API_KEY=...` lines in `.env` at the repo root (git ignores it; keep it mode 0600). `make smoke` reads it.

Run all three before every release. Last run with `gpt-4o-mini` and `claude-haiku-4-5`: `smoke_real.py` PASS=8 FAIL=0 and `smoke_adk_real.py` (ADK 2.11.0) PASS=4 FAIL=0 on 2026-10-07; `smoke_agents_real.py` (openai-agents 0.23.1) PASS=4 FAIL=0 on 2026-10-07.

### The local gateway

`tests/test_local_gateway.py` checks the config and where calls go with no `aigw` needed. Its end-to-end tests start the real `aigw` against the fake provider on free ports, and are skipped unless the local-gateway package is installed. Install it with an `aigw` program, from an Agent Router release or your own build:

```sh
make local-gateway AIGW=/path/to/aigw
.venv/bin/pytest tests/test_local_gateway.py
```

The first start downloads Envoy into `~/.local/share/aigw` unless it is already there.

`scripts/smoke_local_gateway.py` sends real OpenAI and Anthropic calls through the real local gateway, then checks that another program uses the same one. It uses a temporary `AGENTROUTER_HOME` and stops the gateway at the end.

```sh
set -a; . ./.env; set +a; .venv/bin/python scripts/smoke_local_gateway.py
```

`scripts/smoke_routes.py` checks routes on real providers: a primary answering, a failing primary falling back to Anthropic, LiteLLM's `Router`, and a shared tier.

```sh
set -a; . ./.env; set +a; .venv/bin/python scripts/smoke_routes.py
```

**When `aigw`'s fixed ports are taken on your computer** (a Docker container on 18000, say), the end-to-end tests skip. Run them in a Linux container instead, which has its own ports, with an `aigw` Linux binary from an Agent Router release in `<dir>`:

```sh
docker run --rm --platform linux/arm64 -v "$PWD":/src:ro -v <dir>:/aigw:ro python:3.12-slim bash -c '
  mkdir /w && cd /src && tar --exclude=.venv --exclude=.venv-adk --exclude=.venv-agents --exclude=dist \
    --exclude=local-gateway/src/theagentrouter_local_gateway/bin -cf - . | tar -xf - -C /w && cd /w &&
  mkdir -p local-gateway/src/theagentrouter_local_gateway/bin &&
  cp /aigw/aigw-linux-arm64 local-gateway/src/theagentrouter_local_gateway/bin/aigw &&
  pip install -q -e ".[test]" -e local-gateway && python -m pytest -q'
```

Last run 2026-10-12 (evening) with `aigw` v1.2.0, after the fixes from the pre-release review (CHANGELOG, Fixed 2026-10-12): macOS arm64 151 passed, 8 skipped (the framework tests, run separately); `smoke_real.py` PASS=8, `smoke_local_gateway.py` PASS=10, `smoke_routes.py` PASS=9, `smoke_adk_real.py` PASS=4, `smoke_agents_real.py` PASS=4; a live check through the local gateway of Claude prompt caching (12,616 of 12,628 prompt tokens read from the cache), streamed thinking, a thinking and tool-result round trip, and `acompletion` leaving the event loop running while the gateway starts PASS=5. After the second review round (streamed thinking in LiteLLM's fragment shape): ADK tests 5/5, including one where the fake provider refuses a changed thinking block, and OpenAI Agents SDK tests 4/4; a real ADK 2.11.0 agent with Claude thinking on called a tool and answered through the local gateway, streaming and not, PASS=2. The fallback wheel was built and installed: pip picks `macosx_13_0_arm64` when both are offered, and with only the fallback `pip install theagentrouter` succeeds and the SDK reports no local gateway. Earlier run 2026-10-12 (later the same day), after the minimum LiteLLM surface was completed (token_counter, responses, callbacks, names without a provider): macOS arm64 135 passed, 8 skipped; `smoke_real.py` PASS=8, `smoke_local_gateway.py` PASS=10, `smoke_routes.py` PASS=9, `smoke_adk_real.py` PASS=4, `smoke_agents_real.py` PASS=4, ADK and OpenAI Agents SDK tests 4/4 each; a live check of names without a provider, `responses`/`aresponses`, a stream callback and `validate_environment` through the local gateway PASS=6. Earlier run 2026-10-12 with `aigw` v1.2.0, after every call moved to the gateway (no direct provider calls): macOS arm64 113 passed, 8 skipped (the framework tests); `smoke_real.py` PASS=8, `smoke_local_gateway.py` PASS=10, `smoke_routes.py` PASS=9, ADK and OpenAI Agents SDK 4/4 against the fake provider and PASS=4 each against real OpenAI and Anthropic, and examples 00–07, 11–14, 18 and 19 answered through the local gateway. Linux arm64 container: 113 passed, 8 skipped; `smoke_real.py` PASS=8; `smoke_routes.py` PASS=9 FAIL=1 once and PASS=9 FAIL=0 on two reruns (the one failure was not captured; cause unknown). Linux amd64 not run with v1.2.0 yet. Stop the local gateway between real-provider scripts (`python -c "from theagentrouter import local_gateway; local_gateway.stop()"`): `smoke_real.py` and the framework scripts leave the default home's gateway running, as the SDK does, and a second `aigw` from another home cannot start while it holds `aigw`'s fixed internal port 1061. Tetrate Desktop holds 1061 too.

Last run 2026-10-07 with checksum-pinned `aigw` v1.1.0 and the installed macOS arm64 wheel: PASS=10 FAIL=0. All 19 fake-provider local gateway tests also pass against the installed wheels on macOS arm64 and Linux amd64.

To build the wheel for this computer's platform: `.venv/bin/python -m build --wheel local-gateway`. It is about 80 MB compressed on macOS arm64.

## Lint and format

```sh
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## The public surface

`surface.txt` lists every public function with its arguments, every public class, and every setting. `tests/test_surface.py` fails when the package and the file disagree.

`llms.txt` and `llms-full.txt` are built from the README, `docs/`, `examples/` and `skills/` by `make llms` (`scripts/build_llms.py`). `tests/test_llms.py` fails when they are out of date, so run `make llms` after changing any of those. Their links use `Homepage` in `pyproject.toml`; change it when the repository moves.

After a deliberate change:

```sh
.venv/bin/python scripts/surface.py --write
```

Explain the change in the pull request. Removing or changing an item breaks users and the Enterprise SDK, which must contain the whole surface unchanged.

## The model table

`src/theagentrouter/model_catalog.json` is a snapshot of the project's model list, [tetrateio/agent-router-models](https://github.com/tetrateio/agent-router-models). Each release ships the snapshot; the SDK never downloads the list while it runs. To add or fix a model, send a pull request to that repository, then refresh the snapshot here:

```sh
.venv/bin/python scripts/sync_models.py
```

It reads `tetrateio/agent-router-models` by default; `MODELS_REPO=owner/repo` reads another copy.

## How a call works

1. `routing.resolve` picks the target, first match wins: `api_base` on the call, then a gateway from settings (`AGENTROUTER_BASE_URL` or `config.json`), then a route defined in Python, then the running local gateway when it already serves the model, then starting or restarting the local gateway (`local_gateway.ensure`). The SDK never calls a provider itself; there is no table of provider addresses.
2. `openai_format` or `anthropic_format` builds the request body.
3. `transport` sends it with `httpx`, maps HTTP errors with `exceptions.from_http`, and parses the reply or the stream.
4. Responses are the types in `types.py`, readable by attribute and as dicts, as LiteLLM's are.

## Release

The first release is `0.1.0a1`. Three packages always use the same version: `theagentrouter-core` (the SDK, the root `pyproject.toml`), `theagentrouter` (`packages/theagentrouter/`, no code) and `theagentrouter-local-gateway` (`local-gateway/`). The local gateway uses the upstream release and SHA-256 checksums recorded in `local-gateway/aigw-release.json`.

1. Run `pytest`, lint, both framework suites, and all four `scripts/smoke_*.py` checks. Required local gateway tests must run, not skip. Use a fresh machine or container if another program holds the upstream binary's internal ports.
2. Set the version in the three `pyproject.toml` files and both package `__init__.py` files, including `theagentrouter`'s exact pins on the other two and the companion pin in `theagentrouter-core`'s extra. Move Unreleased notes under a dated version heading in `CHANGELOG.md`.
3. Run `python scripts/check_release.py --tag vX.Y.Z`. This checks all five version declarations, the pins, the tag, and the dated changelog entry.
4. Run `python scripts/fetch_aigw.py` on each supported platform. It downloads only the pinned upstream release and refuses a checksum mismatch. Run `python -m build` for the SDK, `python -m build packages/theagentrouter --outdir dist` for `theagentrouter`, and `python -m build --wheel local-gateway` for the companion. Build the fallback companion once, from a checkout with no program in `local-gateway/src/theagentrouter_local_gateway/bin/`: `AGENTROUTER_NO_BINARY_WHEEL=1 python -m build --wheel local-gateway`. The companion is distributed as wheels only; building it from a source checkout requires a binary first.
5. Check the distributions with `python -m twine check --strict dist/* local-gateway/dist/*`. In one fresh environment install `theagentrouter` (it must bring in the other two); in another install `theagentrouter-core` (it must not bring in the companion); in a third, offered only the fallback wheel, install `theagentrouter` (it must succeed, and `theagentrouter.local_gateway.installed()` must be False). Run `pip check` in both, and the fake-provider and local gateway tests against the installed wheels.
6. Push the reviewed source to [`theagentrouter/agent-router-python`](https://github.com/theagentrouter/agent-router-python) when authorized. Before the first publication, configure a pending PyPI trusted publisher for **each** project, with owner `theagentrouter`, repository `agent-router-python` and workflow `release.yml`. Each project has its own environment, because PyPI accepts only one pending publisher per owner, repository, workflow and environment:

   | PyPI project | Environment |
   |---|---|
   | `theagentrouter` | `pypi` |
   | `theagentrouter-core` | `pypi-core` |
   | `theagentrouter-local-gateway` | `pypi-local-gateway` |

   GitHub creates the three environments on the first release run; add reviewers to them under Settings → Environments if releases need approval. PyPI setup requires the owner's authenticated account; it cannot be inferred from the workflow file.
7. Tag the reviewed commit and push the tag when publication is authorized. `release.yml` runs the full CI workflow, checks the tag, and downloads the SDK distributions and all four companion wheels: three with `aigw` and the `py3-none-any` fallback without it. It publishes in dependency order through trusted publishing, one job per package: the companion, then `theagentrouter-core`, then `theagentrouter`, so each package's dependencies exist when it goes up.

`.github/workflows/local-gateway-build.yml` builds and tests companion wheels on macOS 13+ arm64 and Linux amd64/arm64, running every test that starts the real `aigw` (`test_local_gateway.py`, `test_routes.py`, `test_embeddings.py`). It fails if any of them is skipped. Its `fallback` job builds the `py3-none-any` companion with no program and checks that the SDK then reports no local gateway. There are no Windows or Intel macOS companion wheels with a program: pip installs the fallback there, so `pip install theagentrouter` gives the SDK alone. `theagentrouter-core` installs on every supported Python platform.

Changing the upstream binary requires updating `local-gateway/aigw-release.json`, checking the new binary's CLI and resource schema, and passing the local gateway tests on every supported platform. Linux wheels require glibc 2.36 or newer (`manylinux_2_36_*`), and the build refuses dynamically linked launcher binaries. Alpine Linux and older glibc get the fallback wheel. The macOS wheel's deployment target comes from the executable: `aigw` v1.2.0 declares macOS 13.0, so the wheel is `macosx_13_0_arm64`.
