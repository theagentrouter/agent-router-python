# Development

## What is where

| Folder | What it holds |
|---|---|
| [`src/theagentrouter/`](src/theagentrouter/) | The SDK. [`local_gateway.py`](src/theagentrouter/local_gateway.py) starts and finds the local gateway |
| [`local-gateway/`](local-gateway/) | The `theagentrouter-local-gateway` package. Its `aigw` program goes in `src/theagentrouter_local_gateway/bin/`, which git ignores |
| [`packages/theagentrouter/`](packages/theagentrouter/) | The `theagentrouter` package: no code, it installs `theagentrouter-core` (this repository's root package, the SDK) and the local gateway. Its PyPI page is the repository README |
| [`packages/core-README.md`](packages/core-README.md) | The PyPI page of `theagentrouter-core` |
| [`tests/`](tests/) | Tests against a local fake provider; [`test_local_gateway.py`](tests/test_local_gateway.py) also runs the real `aigw` when it is installed |
| [`scripts/`](scripts/) | Real-provider checks, including [`smoke_local_gateway.py`](scripts/smoke_local_gateway.py), and the surface and model-table tools |
| [`examples/`](examples/) | Runnable examples |
| [`apps/`](apps/) | Two apps written against LiteLLM (a Google ADK agent and plain LiteLLM code), tested on real LiteLLM and on this SDK |
| [`docs/`](docs/) | The documentation linked from the README |
| [`skills/`](skills/) | Agent skills |

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

### The apps

`make apps` runs both apps under [`apps/`](apps/) offline, on real LiteLLM and on this SDK; `make apps-live` runs them against real providers. What they cover and how to add a test: [`apps/README.md`](apps/README.md).

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

Run all three before every release.

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

Stop the local gateway between real-provider scripts (`python -c "from theagentrouter import local_gateway; local_gateway.stop()"`): `smoke_real.py` and the framework scripts leave the default home's gateway running, as the SDK does, and a second `aigw` from another home cannot start while it holds `aigw`'s fixed internal port 1061. Another Agent Router gateway on the computer holds 1061 too.

To build the wheel for this computer's platform: `.venv/bin/python -m build --wheel local-gateway`. It is about 80 MB compressed on macOS arm64.

## Lint and format

```sh
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## The public surface

`surface.txt` lists every public function with its arguments, every public class, and every setting. `tests/test_surface.py` fails when the package and the file disagree.

After a deliberate change:

```sh
.venv/bin/python scripts/surface.py --write
```

Explain the change in the pull request: removing or changing an item breaks code that uses it.

## The model table

`src/theagentrouter/model_catalog.json` is a snapshot of the project's model list, [tetrateio/agent-router-models](https://github.com/tetrateio/agent-router-models). Each release ships the snapshot; the SDK never downloads the list while it runs. To add or fix a model, send a pull request to that repository, then refresh the snapshot here:

```sh
.venv/bin/python scripts/sync_models.py
```

It reads `tetrateio/agent-router-models` by default; `MODELS_REPO=owner/repo` reads another copy. It also writes `src/theagentrouter/model_lifecycle.json`: the models the list marks deprecated, outside `model_cost`, with the list's own reason text.

`.github/workflows/update-models.yml` runs the same script every Monday (or by hand) and opens a pull request when the snapshot changes, listing the models added, removed and changed and the source commit. The list has no tags or releases, so it is checked on a schedule.

## How a call works

1. `routing.resolve` picks the target, first match wins: `api_base` on the call, then a gateway from settings (`AGENTROUTER_BASE_URL` or `config.json`), then a route defined in Python, then the running local gateway when it already serves the model, then starting or restarting the local gateway (`local_gateway.ensure`). The SDK never calls a provider itself; there is no table of provider addresses.
2. `openai_format` or `anthropic_format` builds the request body.
3. `transport` sends it with `httpx`, maps HTTP errors with `exceptions.from_http`, and parses the reply or the stream.
4. Responses are the types in `types.py`, readable by attribute and as dicts, as LiteLLM's are.

## Release

Three packages always use the same version: `theagentrouter-core` (the SDK, the root `pyproject.toml`), `theagentrouter` (`packages/theagentrouter/`, no code) and `theagentrouter-local-gateway` (`local-gateway/`). The local gateway uses the upstream release and SHA-256 checksums recorded in `local-gateway/aigw-release.json`.

1. Run `pytest`, lint, both framework suites, `make apps`, `make apps-live` and every `scripts/smoke_*.py` check, including `smoke_examples.py`, which runs each example against real providers. Required local gateway tests must run, not skip. Use a fresh machine or container if another program holds the upstream binary's internal ports. Run the real-provider checks with the companion the release ships (the binary `scripts/fetch_aigw.py` downloads, installed with `make local-gateway AIGW=...`), and do not skip them: the offline tests use a plain-HTTP fake provider, so they cannot show what HTTPS providers, current models and provider-side parameter rules do.
2. Write the release notes under `## Unreleased` in `CHANGELOG.md`.
3. Run `make release V=X.Y.Z`, from a clean working tree. It changes the version in the three `pyproject.toml` files, the exact pins between the packages and both `__init__.py` files, and turns `## Unreleased` into `## [X.Y.Z] - <today>`. Then it runs `scripts/check_release.py --tag vX.Y.Z`, ruff and pytest, and puts the files back if any of them fails. If they pass, it commits `Release X.Y.Z`, creates the annotated tag `vX.Y.Z` and prints the push commands. It never pushes. `AMEND=1` folds the release into the last commit instead of making a new one.
4. Run `python scripts/fetch_aigw.py` on each supported platform. It downloads only the pinned upstream release and refuses a checksum mismatch. Run `python -m build` for the SDK, `python -m build packages/theagentrouter --outdir dist` for `theagentrouter`, and `python -m build --wheel local-gateway` for the companion. Build the fallback companion once, from a checkout with no program in `local-gateway/src/theagentrouter_local_gateway/bin/`: `AGENTROUTER_NO_BINARY_WHEEL=1 python -m build --wheel local-gateway`. The companion is distributed as wheels only; building it from a source checkout requires a binary first.
5. Check the distributions with `python -m twine check --strict dist/* local-gateway/dist/*`. In one fresh environment install `theagentrouter` (it must bring in the other two); in another install `theagentrouter-core` (it must not bring in the companion); in a third, offered only the fallback wheel, install `theagentrouter` (it must succeed, and `theagentrouter.local_gateway.installed()` must be False). Run `pip check` in both, and the fake-provider and local gateway tests against the installed wheels.
6. Push the reviewed source to [`theagentrouter/agent-router-python`](https://github.com/theagentrouter/agent-router-python). Each PyPI project has a trusted publisher with owner `theagentrouter`, repository `agent-router-python`, workflow `release.yml` and its own GitHub environment:

   | PyPI project | Environment |
   |---|---|
   | `theagentrouter` | `pypi` |
   | `theagentrouter-core` | `pypi-core` |
   | `theagentrouter-local-gateway` | `pypi-local-gateway` |

   Add reviewers to these environments under Settings → Environments if releases need approval. Changing a trusted publisher needs the PyPI project owner's account.
7. Push the commit, then the tag; `make release` printed both commands. `release.yml` runs the full CI workflow, checks the tag, and downloads the SDK distributions and all four companion wheels: three with `aigw` and the `py3-none-any` fallback without it. It publishes in dependency order through trusted publishing, one job per package: the companion, then `theagentrouter-core`, then `theagentrouter`, so each package's dependencies exist when it goes up.

`.github/workflows/local-gateway-build.yml` builds and tests companion wheels on macOS 13+ arm64 and Linux amd64/arm64, running every test that starts the real `aigw` (`test_local_gateway.py`, `test_routes.py`, `test_embeddings.py`, `test_go_live.py`). This includes refusing calls from a network address while allowing loopback calls. It fails if any of them is skipped. Its `fallback` job builds the `py3-none-any` companion with no program and checks that the SDK then reports no local gateway. There are no Windows or Intel macOS companion wheels with a program: pip installs the fallback there, so `pip install theagentrouter` gives the SDK alone. `theagentrouter-core` installs on every supported Python platform.

`.github/workflows/update-aigw.yml` checks every Monday for a new `aigw` release (not pre-releases) and opens a pull request that updates `local-gateway/aigw-release.json` with the new version and the SHA-256 GitHub records for each platform's file. It starts CI and Security on that branch itself, because a pull request opened by a workflow starts no workflows. The pull request lists the files that still name the old version.

`.github/workflows/security.yml` checks for known vulnerabilities: `pip-audit` on the SDK's Python dependencies, which fails the job, and `govulncheck` on `aigw`, which only warns, because fixing a finding needs a new `aigw` release. It runs when a dependency file, `aigw-release.json` or the workflow changes, and every Monday. No merge or release waits for it.

Changing the upstream binary requires updating `local-gateway/aigw-release.json`, checking the new binary's CLI and resource schema, and passing the local gateway tests on every supported platform. Linux wheels require glibc 2.36 or newer (`manylinux_2_36_*`), and the build refuses dynamically linked launcher binaries. Alpine Linux and older glibc get the fallback wheel. The macOS wheel's deployment target comes from the executable's minimum macOS version.
