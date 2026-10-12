"""The local gateway: config, where calls go, and (with aigw installed) the real program.

The end-to-end tests run only when theagentrouter-local-gateway is installed, which needs an aigw
program: `make local-gateway AIGW=/path/to/aigw`. They start aigw on free ports against the fake
provider, so they need no keys and no network once Envoy is cached.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest

import theagentrouter
from theagentrouter import exceptions, local_gateway

from .fake_provider import FakeProvider

HELLO = [{"role": "user", "content": "refunds yesterday"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_refunds",
            "description": "Find refunds",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]
CLEARED = (
    "AGENTROUTER_BASE_URL",
    "AGENTROUTER_API_KEY",
    "AGENTROUTER_PROJECT",
    "AGENTROUTER_AUTOSTART",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    for var in CLEARED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return tmp_path / "home"


@pytest.fixture
def fake():
    server = FakeProvider()
    yield server
    server.close()


def _port(url: str) -> int:
    return int(url.split(":")[2].split("/")[0])


def _write_state(home, port: int, providers: list[str], extra: set[str] = frozenset()) -> None:
    folder = home / "local-gateway"
    folder.mkdir(parents=True)
    # A live pid and a listening port: what any program sees once the gateway runs.
    models = local_gateway.routed_models(providers, set(extra))
    state = {
        "pid": os.getpid(),
        "port": port,
        "providers": providers,
        "models": models,
        "versions": local_gateway.versions(),
    }
    (folder / "state.json").write_text(json.dumps(state))


# --- config ---------------------------------------------------------------------------------------


def test_providers_from_the_environment(env, monkeypatch):
    assert local_gateway.available(dict(os.environ)) == ["anthropic", "openai"]
    monkeypatch.setenv("AWS_PROFILE", "acme-dev")
    assert local_gateway.missing("bedrock", dict(os.environ)) == "AWS_REGION"
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert local_gateway.available(dict(os.environ)) == ["anthropic", "bedrock", "openai"]


def test_config_holds_references_not_keys(env, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "acme-dev")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    models = ["openai/gpt-4o-mini", "bedrock/anthropic.claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"]
    config = local_gateway.config_yaml(["anthropic", "bedrock", "openai"], dict(os.environ), port=1975, models=models)
    assert "sk-test" not in config and "sk-ant-test" not in config
    assert "apiKey: ${OPENAI_API_KEY}" in config and "apiKey: ${ANTHROPIC_API_KEY}" in config
    assert "hostname: bedrock-runtime.us-west-2.amazonaws.com" in config
    assert "type: AWSCredentials" in config and "region: us-west-2" in config
    assert "type: AnthropicAPIKey" in config
    # One route rule per model: the full name in (and for OpenAI and Anthropic the name without a
    # provider, as LiteLLM and Agent Router Enterprise write it), the provider's own name out.
    for model, bare in (
        ("openai/gpt-4o-mini", True),
        ("anthropic/claude-sonnet-4-5", True),
        ("bedrock/anthropic.claude-sonnet-4-5", False),
    ):
        provider, own = model.split("/", 1)
        names = [model, own] if bare else [model]
        matches = "".join(
            f'\n        - headers:\n            - type: Exact\n              name: x-ai-eg-model\n              value: "{n}"'
            for n in names
        )
        assert f"    - matches:{matches}\n      backendRefs:\n        - name: {provider}" in config
        assert f'modelNameOverride: "{own}"' in config
    assert "x-agentrouter-provider" not in config  # stock Agent Router routing only


def test_export_config_for_a_team_gateway(env, monkeypatch):
    config = local_gateway.export_config()
    assert "port: 1975" in config  # a fixed port, unlike the local gateway's
    assert "apiKey: ${OPENAI_API_KEY}" in config and "apiKey: ${ANTHROPIC_API_KEY}" in config
    assert "sk-test" not in config and "sk-ant-test" not in config
    assert 'value: "claude-sonnet-4-5"' in config and 'value: "anthropic/claude-sonnet-4-5"' in config
    assert "bedrock" not in config  # only the providers whose keys are set here
    config = local_gateway.export_config(port=8080, providers=["openai"])
    assert "port: 8080" in config and "anthropic" not in config
    with pytest.raises(exceptions.BadRequestError, match="unknown providers: acme"):
        local_gateway.export_config(providers=["acme"])
    config = local_gateway.export_config(providers=["openai"], models=["openai/ft:gpt-4o-mini:acme"])
    assert 'value: "openai/ft:gpt-4o-mini:acme"' in config
    with pytest.raises(exceptions.BadRequestError, match="need their provider"):
        local_gateway.export_config(models=["gpt-4o-tuned"])
    with pytest.raises(exceptions.BadRequestError, match="not in the config: mistral"):
        local_gateway.export_config(providers=["openai"], models=["mistral/acme-tuned"])
    with pytest.raises(exceptions.BadRequestError, match="bedrock needs AWS_REGION"):
        local_gateway.export_config(providers=["bedrock"])
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(exceptions.BadRequestError, match="no provider keys are set here"):
        local_gateway.export_config()


def test_rules_are_spread_over_routes_of_at_most_15(env):
    models = [f"openai/m{i}" for i in range(40)]
    config = local_gateway.config_yaml(["openai"], dict(os.environ), port=1975, models=models)
    routes = [d for d in config.split("\n---\n") if "kind: AIGatewayRoute" in d]
    # 15 + 15 + 10: Gateway API rejects a route with more than 16 rules, and aigw adds one per route
    assert [r.count("x-ai-eg-model") for r in routes] == [15, 15, 10]
    assert "port: 1975" in config


def test_config_for_a_plain_http_address(env):
    local_gateway.PROVIDERS["acme"] = {"schema": "OpenAI", "base": "http://127.0.0.1:8080/v1", "keys": ["ACME_KEY"]}
    try:
        config = local_gateway.config_yaml(["acme"], {}, port=1975)
    finally:
        del local_gateway.PROVIDERS["acme"]
    # A hostname, not the IP: under aigw run IP backends of one rule cannot be told apart.
    assert "hostname: localhost" in config and "port: 8080" in config
    assert "BackendTLSPolicy" not in config


# --- where calls go -------------------------------------------------------------------------------


def test_needs_the_local_gateway_when_not_installed(env, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: False)
    with pytest.raises(
        exceptions.BadRequestError,
        match=r"pip install theagentrouter adds it.*theagentrouter-core.*AGENTROUTER_BASE_URL",
    ):
        theagentrouter.completion(model="bedrock/anthropic.claude-sonnet-4-5", messages=HELLO)


def test_autostart_off(env, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")
    with pytest.raises(exceptions.BadRequestError, match="AGENTROUTER_AUTOSTART=0"):
        theagentrouter.completion(model="bedrock/anthropic.claude-sonnet-4-5", messages=HELLO)


def test_provider_the_local_gateway_does_not_set_up(env, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    with pytest.raises(exceptions.BadRequestError, match="does not set up 'vertex'"):
        theagentrouter.completion(model="vertex/gemini-2.5-pro", messages=HELLO)


def test_missing_credentials_for_the_local_gateway(env, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    with pytest.raises(exceptions.AuthenticationError, match="AWS_PROFILE or AWS_ACCESS_KEY_ID"):
        theagentrouter.completion(model="bedrock/anthropic.claude-sonnet-4-5", messages=HELLO)


def test_api_base_does_not_start_the_gateway(env, monkeypatch, fake):
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    monkeypatch.setattr(local_gateway, "ensure", lambda provider: pytest.fail("started the gateway"))
    r = theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    assert r._hidden_params["route"] == "api_base"


def test_openai_goes_through_the_local_gateway_too(env, monkeypatch, fake):
    # No direct calls: with nothing set, OpenAI starts the local gateway like any other provider.
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    started = []

    def ensure(model):
        started.append(model)
        raise RuntimeError("stop before starting a real gateway")

    monkeypatch.setattr(local_gateway, "ensure", ensure)
    with pytest.raises(RuntimeError, match="stop before"):
        theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO)
    assert started == ["openai/gpt-4o-mini"]


def test_routed_models_cover_the_model_list(env):
    models = local_gateway.routed_models(["openai"], {"openai/acme-tuned", "bedrock/x"})
    assert "openai/gpt-4o-mini" in models and "openai/acme-tuned" in models
    assert not [m for m in models if not m.startswith("openai/")]  # only providers that are set up


def test_running_gateway_takes_every_call_it_can(env, fake):
    _write_state(env, _port(fake.url), ["bedrock", "openai"], {"bedrock/anthropic.claude-sonnet-4-5"})
    r = theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO)
    assert r._hidden_params["route"] == "local gateway"
    assert r._hidden_params["response_cost"] > 0  # priced as openai/gpt-4o-mini
    path, headers, body = fake.requests[-1]
    assert path == "/v1/chat/completions"
    assert body["model"] == "openai/gpt-4o-mini"  # the full name, as to any gateway
    assert "x-agentrouter-provider" not in headers
    assert "authorization" not in headers  # the gateway holds the provider keys

    theagentrouter.completion(model="bedrock/anthropic.claude-sonnet-4-5", messages=HELLO)
    _, headers, body = fake.requests[-1]
    assert body["model"] == "bedrock/anthropic.claude-sonnet-4-5" and "x-agentrouter-provider" not in headers


def test_anthropic_goes_to_the_gateways_anthropic_path(env, fake):
    # aigw takes Anthropic Messages under /anthropic, without translating from the OpenAI format.
    _write_state(env, _port(fake.url), ["anthropic"])
    r = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO)
    assert r._hidden_params["sent_to"].endswith("/anthropic/v1/messages")
    path, headers, body = fake.requests[-1]
    assert path == "/anthropic/v1/messages" and body["model"] == "anthropic/claude-sonnet-4-5"
    assert "x-api-key" not in headers and "x-agentrouter-provider" not in headers


def test_a_set_gateway_wins_over_the_local_one(env, fake, monkeypatch):
    _write_state(env, 9, ["openai"])  # a stale entry would fail; the set gateway must win before it is read
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    r = theagentrouter.completion(model="bedrock/anthropic.claude-sonnet-4-5", messages=HELLO)
    assert r._hidden_params["route"] == "gateway"
    assert fake.requests[-1][2]["model"] == "bedrock/anthropic.claude-sonnet-4-5"


def test_stale_state_is_ignored(env, monkeypatch):
    _write_state(env, 9, ["bedrock"])  # nothing listens on port 9
    monkeypatch.setattr(local_gateway, "installed", lambda: False)
    with pytest.raises(exceptions.BadRequestError, match="needs the local gateway"):
        theagentrouter.completion(model="bedrock/anthropic.claude-sonnet-4-5", messages=HELLO)


def _record(monkeypatch, state):
    calls = []
    monkeypatch.setattr(local_gateway, "running", lambda: state)
    monkeypatch.setattr(local_gateway, "stop", lambda state=None: calls.append(("stop", state["providers"])))
    monkeypatch.setattr(
        local_gateway, "_start", lambda providers, models, routes=None: calls.append(("start", providers, models)) or {}
    )
    return calls


def test_new_provider_restarts_the_gateway(env, monkeypatch):
    state = {
        "pid": 1,
        "port": 1975,
        "providers": ["openai"],
        "models": ["openai/gpt-4o-mini"],
        "versions": local_gateway.versions(),
    }
    calls = _record(monkeypatch, state)
    monkeypatch.setenv("AWS_PROFILE", "acme-dev")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    local_gateway.ensure("bedrock/anthropic.claude-sonnet-4-5")
    assert calls[0] == ("stop", ["openai"])
    _, providers, models = calls[1]
    assert providers == ["anthropic", "bedrock", "openai"]
    assert "bedrock/anthropic.claude-sonnet-4-5" in models and "openai/gpt-4o-mini" in models


def test_new_model_restarts_the_gateway_with_one_more_rule(env, monkeypatch):
    state = {
        "pid": 1,
        "port": 1975,
        "providers": ["openai"],
        "models": ["openai/gpt-4o-mini"],
        "versions": local_gateway.versions(),
    }
    calls = _record(monkeypatch, state)
    assert local_gateway.ensure("openai/gpt-4o-mini") is state  # already routed: no restart
    assert calls == []
    local_gateway.ensure("openai/acme-tuned")
    assert calls[0] == ("stop", ["openai"]) and "openai/acme-tuned" in calls[1][2]


# --- the real aigw --------------------------------------------------------------------------------

needs_aigw = pytest.mark.skipif(
    not local_gateway.installed(), reason="theagentrouter-local-gateway is not installed (make local-gateway AIGW=...)"
)
needs_xds_port = pytest.mark.skipif(
    local_gateway._listening(local_gateway.XDS_PORT),
    reason=f"aigw's fixed xDS port {local_gateway.XDS_PORT} is already in use",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def real_gateway(env, fake, monkeypatch):
    """aigw on ports it chooses at start (the default), with providers "acme" and "acme2" from the fake provider."""
    monkeypatch.setattr(local_gateway, "PORT", 0)
    monkeypatch.setattr(local_gateway, "ADMIN_PORT", 0)
    for name in ("acme", "acme2"):
        monkeypatch.setitem(
            local_gateway.PROVIDERS, name, {"schema": "OpenAI", "base": fake.url, "keys": [f"{name.upper()}_KEY"]}
        )
    monkeypatch.setitem(
        local_gateway.PROVIDERS, "anthropic", {**local_gateway.PROVIDERS["anthropic"], "base": fake.url[:-3]}
    )
    monkeypatch.delenv("OPENAI_API_KEY")  # keep real providers out of the config
    monkeypatch.setenv("ACME_KEY", "acme-secret")
    yield fake
    local_gateway.stop()


@needs_aigw
@needs_xds_port
def test_real_gateway_end_to_end(real_gateway, env, capsys, monkeypatch):
    fake = real_gateway

    # 1. The first call that needs the gateway starts it.
    r = theagentrouter.completion(model="acme/acme-large", messages=HELLO)
    assert r.choices[0].message.content == "echo: refunds yesterday"
    assert r._hidden_params["route"] == "local gateway"
    assert "started the local gateway" in capsys.readouterr().err
    path, headers, body = fake.requests[-1]
    assert body["model"] == "acme-large"  # the SDK sent acme/acme-large; aigw's rule passed on the provider's own name
    assert "x-agentrouter-provider" not in headers
    assert headers["authorization"] == "Bearer acme-secret"  # aigw filled in the key from its environment
    config = (env / "local-gateway" / "config.yaml").read_text()
    assert "acme-secret" not in config and "sk-ant-test" not in config
    first = local_gateway.running()
    assert first and first["providers"] == ["acme", "anthropic"] and "acme/acme-large" in first["models"]

    # 2. Streaming with a tool call, through the gateway.
    args, usage = "", None
    for chunk in theagentrouter.completion(model="acme/acme-large", messages=HELLO, tools=TOOLS, stream=True):
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            args += chunk.choices[0].delta.tool_calls[0].function.arguments or ""
        usage = chunk.usage or usage
    assert json.loads(args) == {"query": "refunds yesterday"}

    # 3. A provider the SDK could call directly also goes through the running gateway; aigw translates.
    r = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO)
    assert r._hidden_params["route"] == "local gateway"
    assert r.choices[0].message.content == "echo: refunds yesterday"
    path, headers, body = fake.requests[-1]
    assert path.endswith("/messages") and headers["x-api-key"] == "sk-ant-test"
    assert body["model"] == "claude-sonnet-4-5"

    # 4. Another program on the computer uses the same gateway.
    code = (
        "import theagentrouter; "
        "r = theagentrouter.completion(model='acme/acme-large', messages=[{'role': 'user', 'content': 'hi'}]); "
        "print(r._hidden_params['route'], r.choices[0].message.content)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ), timeout=60)
    assert out.stdout.strip() == "local gateway echo: hi", out.stderr
    assert local_gateway.running()["pid"] == first["pid"]

    # 5. A new provider's key restarts the gateway with it.
    monkeypatch.setenv("ACME2_KEY", "acme2-secret")
    r = theagentrouter.completion(model="acme2/acme-small", messages=HELLO)
    assert r.choices[0].message.content == "echo: refunds yesterday"
    assert fake.requests[-1][1]["authorization"] == "Bearer acme2-secret"
    second = local_gateway.running()
    assert second["pid"] != first["pid"] and second["providers"] == ["acme", "acme2", "anthropic"]

    # 6. A model the gateway has no rule for yet gets one: the gateway restarts with it.
    r = theagentrouter.completion(model="acme/acme-other", messages=HELLO)
    assert r.choices[0].message.content == "echo: refunds yesterday"
    assert fake.requests[-1][2]["model"] == "acme-other"
    assert local_gateway.running()["pid"] != second["pid"]

    # 7. Errors from the provider keep LiteLLM's names through the gateway.
    with pytest.raises(exceptions.RateLimitError):
        theagentrouter.completion(model="acme/err-429", messages=HELLO)


@needs_aigw
@needs_xds_port
def test_port_in_use_is_reported(real_gateway, monkeypatch):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        monkeypatch.setattr(local_gateway, "PORT", busy.getsockname()[1])
        with pytest.raises(exceptions.APIConnectionError, match="already in use"):
            theagentrouter.completion(model="acme/acme-large", messages=HELLO)


@needs_aigw
@needs_xds_port
def test_same_calls_work_through_a_gateway_you_set(real_gateway, monkeypatch):
    """One aigw, reached as the local gateway and then as a gateway set with AGENTROUTER_BASE_URL:
    the same calls give the same answers, and the provider gets the same requests."""
    fake = real_gateway
    calls = [
        ("acme/acme-large", {}),
        ("acme/acme-large", {"tools": TOOLS}),
        ("anthropic/claude-sonnet-4-5", {}),
        ("claude-sonnet-4-5", {}),  # the name without a provider, as Agent Router Enterprise writes it
        ("anthropic/claude-sonnet-4-5", {"tools": TOOLS}),
    ]

    def run() -> list:
        out = []
        for model, extra in calls:
            r = theagentrouter.completion(model=model, messages=HELLO, **extra)
            path, headers, body = fake.requests[-1]
            message = r.choices[0].message
            calls_made = [(c.function.name, c.function.arguments) for c in message.tool_calls or []]
            out.append((message.content, calls_made, r._hidden_params["format"], path, body["model"]))
            stream = theagentrouter.completion(model=model, messages=HELLO, stream=True, **extra)
            text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
            out.append((text, fake.requests[-1][0]))
        # Anthropic's Messages format, as litellm.anthropic.messages.create sends it.
        r = theagentrouter.anthropic.messages.create(model="claude-sonnet-4-5", max_tokens=100, messages=HELLO)
        out.append((r["content"][0]["text"], fake.requests[-1][0], fake.requests[-1][2]["model"]))
        return out

    local = run()
    state = local_gateway.running()
    assert state and all(
        r[2] == ("anthropic" if "claude" in m else "openai")
        for r, (m, _) in zip(local[: 2 * len(calls) : 2], calls, strict=True)
    )

    monkeypatch.setenv("AGENTROUTER_BASE_URL", f"http://127.0.0.1:{state['port']}/v1")
    monkeypatch.setenv("AGENTROUTER_API_KEY", "team-key")
    monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")  # nothing may start: every call goes to the set gateway
    remote = run()
    assert remote == local
    headers = fake.requests[-1][1]
    # The provider gets the gateway's key for it; the caller's gateway key goes no further.
    assert headers["x-api-key"] == "sk-ant-test" and "team-key" not in json.dumps(headers)
    assert local_gateway.running()["pid"] == state["pid"]


@needs_aigw
@needs_xds_port
def test_another_gateway_on_1975_does_not_block(real_gateway, monkeypatch):
    # Another Agent Router gateway (a plain `aigw run`) listens on 1975; the SDK's gateway takes a free port instead.
    with socket.socket() as other:
        try:
            other.bind(("127.0.0.1", 1975))
            other.listen()
        except OSError:
            pass  # something already holds 1975, which is the case being tested
        r = theagentrouter.completion(model="acme/acme-large", messages=HELLO)
        state = local_gateway.running()
        assert r.choices[0].message.content and state and state["port"] != 1975
        assert r._hidden_params["sent_to"].startswith(f"http://127.0.0.1:{state['port']}/")


def test_internal_port_owner_is_named(monkeypatch):
    monkeypatch.setattr(local_gateway, "_listening", lambda port, timeout=0.5: port == local_gateway.EXTPROC_PORT)
    monkeypatch.setattr(local_gateway, "_port_owner", lambda port: "aigw")
    with pytest.raises(exceptions.APIConnectionError, match="in use by aigw.*another Agent Router gateway"):
        local_gateway._busy_internal()
    monkeypatch.setattr(local_gateway, "_port_owner", lambda port: None)
    with pytest.raises(
        exceptions.APIConnectionError, match="1061 on 127.0.0.1 is already in use.*another Agent Router gateway"
    ):
        local_gateway._busy_internal()


@needs_aigw
def test_xds_port_in_use_is_reported(real_gateway, monkeypatch):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        monkeypatch.setattr(local_gateway, "XDS_PORT", busy.getsockname()[1])
        with pytest.raises(exceptions.APIConnectionError, match="xDS port"):
            theagentrouter.completion(model="acme/acme-large", messages=HELLO)


# --- upgrades -------------------------------------------------------


def test_a_gateway_started_by_another_version_is_restarted(env, monkeypatch):
    # After pip install -U, the old aigw keeps running until something stops it; the next call does.
    old = {
        "pid": 1,
        "port": 1975,
        "providers": ["openai"],
        "models": ["openai/gpt-4o-mini"],
        "versions": {"sdk": "0.0.1", "local_gateway": "0.0.1"},
    }
    calls = []
    monkeypatch.setattr(local_gateway, "running", lambda: old)
    monkeypatch.setattr(local_gateway, "stop", lambda state=None: calls.append("stop"))
    monkeypatch.setattr(local_gateway, "_start", lambda providers, models, routes=None: calls.append("start") or {})
    assert not local_gateway.serves(old, "openai/gpt-4o-mini", {})
    local_gateway.ensure("openai/gpt-4o-mini")
    assert calls == ["stop", "start"]
    # A gateway from before versions were recorded is restarted once, too.
    assert not local_gateway.serves({k: v for k, v in old.items() if k != "versions"}, "openai/gpt-4o-mini", {})
