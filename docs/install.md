# Install

## What you need

- Python 3.10 or newer.
- A key for at least one provider, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

There are three packages, all with the same version:

| Package | What it is | Install it for |
|---|---|---|
| `theagentrouter` | No code of its own: installs the two below | A developer computer: the SDK and the local gateway |
| `theagentrouter-core` | The SDK (`import theagentrouter`); depends on `httpx` only | A remote gateway: servers, containers, serverless functions, CI |
| `theagentrouter-local-gateway` | Agent Router's `aigw` program (about 80 MB) | Installed by `theagentrouter`. On computers with no build of `aigw`, pip installs a small version with no program |

## The SDK

```sh
pip install theagentrouter          # the SDK and the local gateway
pip install theagentrouter-core     # the SDK alone, for a remote gateway
```

With `uv`: `uv add theagentrouter`, or `uv add theagentrouter-core`.

**From source.** To install from a checkout of this repository:

```sh
git clone https://github.com/theagentrouter/agent-router-python
pip install ./agent-router-python      # theagentrouter-core, the SDK alone
```

To get the local gateway too, build its package (below), then `pip install ./agent-router-python/packages/theagentrouter`.

## The local gateway

`pip install theagentrouter` also installs `theagentrouter-local-gateway`, which holds Agent Router's `aigw` program. It runs on:

- macOS 13 or newer on Apple chips
- Linux with glibc 2.36 or newer, on x86-64 or Arm

There is nothing else to run: the first call starts it. The very first start also downloads Envoy, the proxy `aigw` runs, into `~/.local/share/aigw` (about 100 MB), so it needs the internet and can take a minute; later starts take a few seconds. An SDK upgrade that moves to a newer `aigw` downloads its Envoy the same way and leaves the old one in that folder. See [Usage: the local gateway](usage.md#the-local-gateway).

The package is about 80 MB to download and roughly 280 MB installed, plus Envoy's first-start download. Windows, Intel Macs, macOS 12 or older, Alpine Linux and Linux with glibc older than 2.36 have no local gateway. This includes RHEL 9 and Amazon Linux 2023 (glibc 2.34) and Ubuntu 22.04 (2.35). On those, `pip install theagentrouter` still succeeds: pip picks a small version of the package with no program in it, and you get the SDK alone. Set `AGENTROUTER_BASE_URL` to an Agent Router gateway; until you do, calls fail with a message that says so.

Use the local gateway for development. For production, serverless functions and CI jobs, install `theagentrouter-core` and use a remote gateway (below). The local process has shared configuration and fixed internal ports; a new route, model or provider restarts it and interrupts other programs. Local configuration does not support Azure OpenAI or Vertex AI. See [How the local gateway works](local-gateway.md#open) for the exact limits.

**From source.** Build the local gateway package from this repository with an `aigw` program you have, then install the SDK. `make local-gateway` puts the program in the package and installs it into the repository's own `.venv`; for any other environment, install the folder first:

```sh
make local-gateway AIGW=/path/to/aigw
pip install ./local-gateway      # into another environment, before the SDK
```

`pip install "theagentrouter[local-gateway]"` still works and does the same as `pip install theagentrouter`. With `theagentrouter-core`, `pip install theagentrouter` later adds the local gateway; nothing conflicts.

## A remote gateway

Instead of the local gateway, the SDK can send every call to a gateway that runs elsewhere: an Agent Router your team deploys or your company's Agent Router Enterprise. It gets the same requests the local gateway gets, so code that works with one works with the other. This works on every computer, including those with no local gateway:

```sh
export AGENTROUTER_BASE_URL=https://gateway.acme.example/v1
export AGENTROUTER_API_KEY=...      # the gateway's key, not a provider's
```

Install `theagentrouter-core` for this: it leaves out the local gateway. With `AGENTROUTER_BASE_URL` set, the local gateway never starts, even where it is installed, and no provider keys are needed on this computer: the gateway holds them. See [Usage: a gateway you set](usage.md#a-gateway-you-set).

## Check that it works

```sh
python -c "import theagentrouter; print(theagentrouter.__version__)"
```

Then make a first call with a key in your environment:

```sh
export OPENAI_API_KEY=...
python -c "
import theagentrouter
r = theagentrouter.completion(model='openai/gpt-5.4-mini', messages=[{'role': 'user', 'content': 'Say hi.'}])
print(r.choices[0].message.content)"
```

The first call prints one line when it starts the local gateway, such as `theagentrouter: started the local gateway on http://127.0.0.1:52413 (providers from your environment: openai).`, then the answer.

## Upgrade

```sh
pip install -U theagentrouter
```

Each release carries its own copy of the model list (prices and context windows), so upgrading also updates the list. A running gateway continues serving its existing models across SDK versions. Stop it explicitly after other programs finish to load the new binary/configuration: `python -c "from theagentrouter import local_gateway; local_gateway.stop()"`. The next call starts the installed version. A different version cannot implicitly reconfigure the shared process.

## Remove

Stop the local gateway first; it keeps running after your programs end, and uninstalling does not stop it:

```sh
python -c "from theagentrouter import local_gateway; local_gateway.stop()"
pip uninstall theagentrouter theagentrouter-core theagentrouter-local-gateway
```

The SDK keeps its settings, routes and the local gateway's files in `~/.agentrouter/` (or `AGENTROUTER_HOME`), and `aigw` keeps the Envoy it downloaded in `~/.local/share/aigw`. Delete those folders to remove them too.

## Next

- [Usage](usage.md): calls, streaming, tools, errors, gateways and settings.
- [Coming from LiteLLM](coming-from-litellm.md): what changes, and what does not.
- [Agent frameworks](frameworks.md): Google ADK and the OpenAI Agents SDK.
