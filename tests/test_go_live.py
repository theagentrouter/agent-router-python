# SPDX-License-Identifier: Apache-2.0
"""Who can call the local gateway, which key goes where, the Envoy version, LiteLLM's stream shape
and LiteLLM's drop_params for Claude."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import signal
import socket
import stat
import subprocess
import time
import warnings

import httpx
import pytest

import theagentrouter
from theagentrouter import anthropic_format, exceptions, local_gateway, openai_format, routing, transport
from theagentrouter.compat import litellm

from .test_local_gateway import _free_port, needs_aigw, needs_xds_port
from .test_sdk import HELLO

MODELS = ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]


# ---- who can call the local gateway -------------------------------------------------------------


def test_local_gateway_config_answers_only_this_computer(env):
    config = local_gateway.config_yaml(["openai"], {"OPENAI_API_KEY": "k"}, port=1975, models=[])
    policy = next(d for d in config.split("\n---\n") if "kind: SecurityPolicy" in d)
    assert "defaultAction: Deny" in policy
    assert "- 127.0.0.1/32" in policy and "- ::1/128" in policy
    assert '- "127.0.0.1:1975"' in policy and '- "localhost:1975"' in policy and '- "[::1]:1975"' in policy
    assert "clientCIDRs" not in local_gateway.export_config(providers=["openai"])  # a server serves the network


@pytest.mark.parametrize("exported", [False, True])
def test_the_gateway_tries_a_failed_connection_again(env, exported):
    config = (
        local_gateway.export_config(providers=["openai"], authentication=False)
        if exported
        else local_gateway.config_yaml(["openai"], {"OPENAI_API_KEY": "k"}, port=1975, models=[])
    )
    policy = next(d for d in config.split("\n---\n") if "name: retry-failed-connections" in d)
    assert "kind: Gateway" in policy and "connect-failure" in policy
    assert "retriable-status-codes" not in policy  # only what never reached the provider


def test_exported_config_authenticates_by_default(env):
    config = local_gateway.export_config(providers=["openai"])
    policy = next(d for d in config.split("\n---\n") if "name: theagentrouter-client-auth" in d)
    assert "defaultAction: Deny" in policy
    assert 'values: ["${AGENTROUTER_API_KEY:-\\n}"]' in policy
    assert 'values: ["Bearer ${AGENTROUTER_API_KEY:-\\n}"]' in policy
    assert "remove: [authorization, x-api-key]" in config  # the client key goes no further
    open_config = local_gateway.export_config(providers=["openai"], authentication=False)
    assert "theagentrouter-client-auth" not in open_config and "headerMutation" not in open_config


def _network_address() -> str | None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))  # no packet is sent; this only picks the outgoing interface
        except OSError:
            return None
        address = s.getsockname()[0]
    return None if address.startswith("127.") else address


@needs_aigw
@needs_xds_port
def test_real_gateway_refuses_callers_from_the_network(real_gateway):
    address = _network_address()
    if address is None:
        pytest.skip("this computer has no network address besides loopback")
    theagentrouter.completion(model="acme/acme-large", messages=HELLO)
    port = local_gateway.running()["port"]
    body = {"model": "acme/acme-large", "messages": HELLO}
    sent = len(real_gateway.requests)
    # Envoy listens on every address; the policy refuses a caller that is not on this computer.
    assert httpx.post(f"http://{address}:{port}/v1/chat/completions", json=body, timeout=20).status_code == 403
    assert len(real_gateway.requests) == sent  # the provider never saw it, so no key was used
    assert httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=20).status_code == 200


@needs_aigw
@needs_xds_port
def test_real_gateway_refuses_a_web_page_through_dns_rebinding(real_gateway):
    # The browser connects from this computer, but sends the name of the page's site as the Host.
    theagentrouter.completion(model="acme/acme-large", messages=HELLO)
    port = local_gateway.running()["port"]
    body = {"model": "acme/acme-large", "messages": HELLO}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    sent = len(real_gateway.requests)
    for host in ("acme.example", f"acme.example:{port}"):
        assert httpx.post(url, json=body, headers={"Host": host}, timeout=20).status_code == 403
    assert len(real_gateway.requests) == sent  # the provider never saw them, so no key was used
    assert httpx.post(url, json=body, headers={"Host": f"localhost:{port}"}, timeout=20).status_code == 200


@contextlib.contextmanager
def _exported_gateway(config: str, folder, environ: dict[str, str]):
    """``aigw run`` on an exported config, as a team would run it on a server."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(config)
    log = folder / "aigw.log"
    environ = {
        **os.environ,
        "ENVOY_VERSION": local_gateway.ENVOY_VERSION,
        "AIGW_CONFIG_HOME": str(folder / "config-home"),
        "AIGW_STATE_HOME": str(folder / "state"),
        **environ,
    }
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [local_gateway._binary(), "run", str(folder / "config.yaml"), f"--admin-port={_free_port()}"],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=environ,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + local_gateway.START_TIMEOUT
        while "listening on" not in log.read_text(errors="replace"):
            assert proc.poll() is None and time.monotonic() < deadline, log.read_text(errors="replace")
            time.sleep(0.2)
        yield
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)


@needs_aigw
@needs_xds_port
def test_real_exported_gateway_checks_the_client_key(real_gateway, tmp_path):
    fake = real_gateway
    port = _free_port()
    config = local_gateway.export_config(port=port, providers=["acme", "anthropic"], models=["acme/acme-large"])
    chat = {"model": "acme/acme-large", "messages": HELLO}
    claude = {"model": "claude-sonnet-4-5", "max_tokens": 50, "messages": HELLO}
    base = f"http://127.0.0.1:{port}"
    with _exported_gateway(config, tmp_path / "server", {"AGENTROUTER_API_KEY": "team-key"}):
        sent = len(fake.requests)
        for headers in ({}, {"x-api-key": "wrong"}, {"authorization": "Bearer wrong"}, {"authorization": "team-key"}):
            r = httpx.post(f"{base}/v1/chat/completions", json=chat, headers=headers, timeout=20)
            assert r.status_code == 403, headers
        assert len(fake.requests) == sent  # refused before any provider was called
        for headers in ({"authorization": "Bearer team-key"}, {"x-api-key": "team-key"}):
            r = httpx.post(f"{base}/v1/chat/completions", json=chat, headers=headers, timeout=20)
            assert r.status_code == 200, r.text
            seen = fake.requests[-1][1]
            assert seen["authorization"] == "Bearer acme-secret" and "team-key" not in json.dumps(seen)
        # A Messages call: the client key is removed whichever header carries it, and Anthropic gets its own.
        for headers in ({"authorization": "Bearer team-key"}, {"x-api-key": "team-key"}):
            r = httpx.post(f"{base}/anthropic/v1/messages", json=claude, headers=headers, timeout=20)
            assert r.status_code == 200, r.text
            seen = fake.requests[-1][1]
            assert seen["x-api-key"] == "sk-ant-test" and "team-key" not in json.dumps(seen)


@needs_aigw
@needs_xds_port
def test_real_exported_gateway_without_its_key_refuses_everyone(real_gateway, tmp_path):
    port = _free_port()
    config = local_gateway.export_config(port=port, providers=["acme"], models=["acme/acme-large"])
    chat = {"model": "acme/acme-large", "messages": HELLO}
    body = json.dumps(chat).encode()
    with _exported_gateway(config, tmp_path / "server", {"AGENTROUTER_API_KEY": ""}):
        # Raw requests, since HTTP clients refuse to send an empty key or one with a trailing space.
        for header in (b"", b"x-api-key: \r\n", b"authorization: Bearer \r\n", b"authorization: Bearer\r\n"):
            with socket.create_connection(("127.0.0.1", port), timeout=20) as s:
                s.sendall(
                    b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                    + header
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                assert s.recv(64).startswith(b"HTTP/1.1 403"), header
    assert not real_gateway.requests


def test_the_gateway_runs_the_pinned_envoy(env, monkeypatch):
    seen = {}

    class Started(Exception):
        pass

    def launch(home, environ, *args):
        seen.update(environ)
        raise Started

    monkeypatch.setattr(local_gateway, "_launch", launch)
    monkeypatch.setattr(local_gateway, "_busy_internal", lambda: None)
    monkeypatch.delenv("ENVOY_VERSION", raising=False)
    with pytest.raises(Started):
        local_gateway._start(["openai"], [])
    assert seen["ENVOY_VERSION"] == local_gateway.ENVOY_VERSION
    monkeypatch.setenv("ENVOY_VERSION", "1.39.3")
    with pytest.raises(Started):
        local_gateway._start(["openai"], [])
    assert seen["ENVOY_VERSION"] == "1.39.3"  # the user's choice wins


# ---- a route change and the other programs on the gateway ----------------------------------------


def test_a_route_change_restarts_the_gateway(env, monkeypatch):
    calls = []
    state = {
        "pid": 1,
        "port": 9,
        "providers": ["openai"],
        "models": ["openai/gpt-4o-mini"],
        "routes": [],
        "versions": local_gateway.versions(),
    }
    monkeypatch.setattr(local_gateway, "running", lambda: state)
    monkeypatch.setattr(local_gateway, "stop", lambda *a, **k: calls.append(("stop",)))
    monkeypatch.setattr(local_gateway, "_start", lambda *a, **k: calls.append(("start",)) or state)
    theagentrouter.route("acme-chat", "openai/gpt-4o-mini", fallbacks=["openai/gpt-4o"])
    local_gateway.ensure("acme-chat")
    assert calls == [("stop",), ("start",)]  # aigw reads its config once, at start


# ---- a rotated provider key -----------------------------------------------------------------------


def test_local_config_reads_api_keys_from_private_files(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "acme-openai-key")
    config = local_gateway.config_yaml(["openai"], dict(os.environ), port=1975, models=[])
    assert "${OPENAI_API_KEY}" not in config and "acme-openai-key" not in config
    assert f"substitution.aigw.run/file/apiKey: {json.dumps(str(local_gateway._key_file('OPENAI_API_KEY')))}" in config
    assert "${OPENAI_API_KEY}" in local_gateway.export_config(providers=["openai"])  # a server keeps its environment
    local_gateway._write_keys(["openai"], dict(os.environ))
    key = local_gateway._key_file("OPENAI_API_KEY")
    assert key.read_text() == "acme-openai-key" and stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700


def test_a_changed_key_says_how_to_load_it(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "acme-old")
    state = {
        "providers": ["openai"],
        "credential_fingerprints": local_gateway._credential_fingerprints(["openai"], dict(os.environ)),
    }
    monkeypatch.setenv("OPENAI_API_KEY", "acme-new")
    with pytest.raises(exceptions.AuthenticationError, match="local_gateway.stop"):
        local_gateway.check_credentials(state)


@needs_aigw
@needs_xds_port
def test_real_gateway_adds_a_route_with_an_https_backend(env, monkeypatch):
    # An HTTPS provider gets a BackendTLSPolicy; a route change must not stop Envoy Gateway on it.
    monkeypatch.setattr(local_gateway, "PORT", 0)
    monkeypatch.setattr(local_gateway, "ADMIN_PORT", 0)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://localhost:1/v1")  # no DNS and no call needed
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    try:
        local_gateway.ensure("openai/gpt-4o-mini")
        theagentrouter.route("acme-chat", "openai/gpt-4o-mini", fallbacks=["openai/gpt-4o"])
        state = local_gateway.ensure("acme-chat")
        assert local_gateway.running()["pid"] == state["pid"]
        assert "exiting on provider runner error" not in (local_gateway.folder() / "aigw.log").read_text()
    finally:
        local_gateway.stop()


# ---- which key goes where -----------------------------------------------------------------------


def _store(env, **values) -> None:
    env.mkdir(parents=True, exist_ok=True)
    (env / "config.json").write_text(json.dumps(values))


def test_stored_key_is_not_sent_to_another_gateway(env, monkeypatch):
    _store(env, base_url="https://gw.acme.example/v1", api_key="acme-stored-key", project="payments")
    monkeypatch.setenv("AGENTROUTER_BASE_URL", "https://other.example/v1")
    assert routing.settings() == {"base_url": "https://other.example/v1"}


def test_stored_key_goes_to_its_own_gateway(env, monkeypatch):
    _store(env, base_url="https://gw.acme.example/v1", api_key="acme-stored-key")
    assert routing.settings()["api_key"] == "acme-stored-key"
    monkeypatch.setenv("AGENTROUTER_BASE_URL", "https://gw.acme.example/v1/")  # the same gateway
    assert routing.settings()["api_key"] == "acme-stored-key"
    monkeypatch.setenv("AGENTROUTER_BASE_URL", "https://other.example/v1")
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-env-key")
    assert routing.settings()["api_key"] == "acme-env-key"


# ---- streams shaped as LiteLLM's ----------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_litellm_stream_loop_runs_to_the_end(env, fake, model):
    # The loop from LiteLLM's README and docs/usage.md, streamed from the fake provider, not a mock.
    text, chunks = "", []
    for chunk in theagentrouter.completion(model=model, messages=HELLO, stream=True, api_base=fake.url):
        text += chunk.choices[0].delta.content or ""
        chunks.append(chunk)
    assert text.strip() == "echo: refunds yesterday"
    assert all(chunk.choices for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.total_tokens > 0  # usage still arrives, on the finish chunk
    assert theagentrouter.stream_chunk_builder(chunks).usage.total_tokens == chunks[-1].usage.total_tokens


@pytest.mark.parametrize("model", MODELS)
def test_include_usage_adds_a_last_chunk_with_an_empty_choice(env, fake, model):
    chunks = list(
        theagentrouter.completion(
            model=model, messages=HELLO, stream=True, api_base=fake.url, stream_options={"include_usage": True}
        )
    )
    *rest, last = chunks
    assert len(last.choices) == 1 and last.choices[0].delta.content is None and last.choices[0].finish_reason is None
    assert last.usage.total_tokens > 0
    assert rest[-1].choices[0].finish_reason == "stop" and rest[-1].usage is None


async def test_async_stream_has_a_choice_in_every_chunk(env, fake):
    stream = await theagentrouter.acompletion(model=MODELS[1], messages=HELLO, stream=True, api_base=fake.url)
    chunks = [chunk async for chunk in stream]
    assert all(chunk.choices for chunk in chunks) and chunks[-1].usage.total_tokens > 0


@pytest.mark.parametrize("include_usage", [False, True])
def test_mock_stream_has_the_same_shape(env, include_usage):
    options = {"stream_options": {"include_usage": True}} if include_usage else {}
    chunks = list(
        theagentrouter.completion(model=MODELS[0], messages=HELLO, mock_response="hi", stream=True, **options)
    )
    assert all(chunk.choices for chunk in chunks)
    assert chunks[-1].usage.completion_tokens == 20
    assert (chunks[-1].choices[0].finish_reason is None) == include_usage


# ---- drop_params for Claude ---------------------------------------------------------------------


def test_claude_takes_n_1(env, fake):
    theagentrouter.completion(model=MODELS[1], messages=HELLO, n=1, api_base=fake.url)
    assert "n" not in fake.requests[-1][2]


@pytest.mark.parametrize("param", [{"seed": 1}, {"presence_penalty": 0.0}, {"n": 2}, {"logprobs": True}])
def test_claude_refuses_other_parameters_unless_dropped(env, fake, param):
    with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
        theagentrouter.completion(model=MODELS[1], messages=HELLO, api_base=fake.url, **param)
    theagentrouter.completion(model=MODELS[1], messages=HELLO, api_base=fake.url, drop_params=True, **param)
    assert not set(param) & set(fake.requests[-1][2])


@pytest.mark.parametrize("module", [theagentrouter, litellm])
def test_module_drop_params(env, fake, monkeypatch, module):
    monkeypatch.setattr(module, "drop_params", True)
    theagentrouter.completion(model=MODELS[1], messages=HELLO, seed=1, api_base=fake.url)
    assert "seed" not in fake.requests[-1][2]
    # As in LiteLLM, the module's setting holds whatever the call says.
    theagentrouter.completion(model=MODELS[1], messages=HELLO, seed=1, api_base=fake.url, drop_params=False)
    assert "seed" not in fake.requests[-1][2]


def test_drop_params_is_off_by_default_as_in_litellm():
    assert theagentrouter.drop_params is False and litellm.drop_params is False


def test_claude_thinking_is_sent(env, fake):
    thinking = {"type": "enabled", "budget_tokens": 1024}
    theagentrouter.completion(model=MODELS[1], messages=HELLO, thinking=thinking, api_base=fake.url)
    assert fake.requests[-1][2]["thinking"] == thinking
    big = {"type": "enabled", "budget_tokens": 8000}
    theagentrouter.completion(model=MODELS[1], messages=HELLO, thinking=big, api_base=fake.url)
    assert fake.requests[-1][2]["max_tokens"] > 8000  # room for the answer after thinking
    theagentrouter.completion(model=MODELS[1], messages=HELLO, thinking=big, max_tokens=9000, api_base=fake.url)
    assert fake.requests[-1][2]["max_tokens"] == 9000


def test_claude_5_thinks_adaptively_and_older_claude_within_max_tokens(env, fake):
    base = dict(messages=HELLO, api_base=fake.url)
    theagentrouter.completion(model="anthropic/claude-sonnet-5-5", reasoning_effort="high", **base)
    body = fake.requests[-1][2]
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    theagentrouter.completion(
        model="anthropic/claude-sonnet-5-5", thinking={"type": "enabled", "budget_tokens": 9}, **base
    )
    body = fake.requests[-1][2]
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "low"}
    with pytest.raises(exceptions.BadRequestError, match="Invalid reasoning_effort"):
        theagentrouter.completion(model="anthropic/claude-sonnet-5-5", reasoning_effort="extreme", **base)
    theagentrouter.completion(model="anthropic/claude-haiku-4-5", reasoning_effort="high", max_tokens=2048, **base)
    body = fake.requests[-1][2]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2047} and body["max_tokens"] == 2048
    with pytest.raises(exceptions.BadRequestError, match="max_tokens above 1024"):
        theagentrouter.completion(model="anthropic/claude-haiku-4-5", reasoning_effort="low", max_tokens=1000, **base)


def test_openai_gets_max_completion_tokens(env, fake):
    # OpenAI's newer models refuse max_tokens; other OpenAI-format providers keep it.
    theagentrouter.completion(model="openai/gpt-5.4-mini", messages=HELLO, max_tokens=50, api_base=fake.url)
    body = fake.requests[-1][2]
    assert body["max_completion_tokens"] == 50 and "max_tokens" not in body
    theagentrouter.completion(model="openai/gpt-5.4-mini", messages=HELLO, max_completion_tokens=40, api_base=fake.url)
    assert fake.requests[-1][2]["max_completion_tokens"] == 40


# ---- what each Claude model refuses, as Anthropic's API answered --------------------------------


@pytest.mark.parametrize(
    ("model", "refuses"),
    [
        ("claude-fable-5", True),
        ("claude-fable-5-1", True),
        ("claude-haiku-5-5", True),
        ("claude-opus-4-7", True),
        ("claude-opus-4-8", True),
        ("claude-opus-5", True),
        ("claude-opus-5-5", True),
        ("claude-sonnet-5", True),
        ("claude-sonnet-5-5", True),
        ("anthropic/claude-sonnet-5-5", True),
        ("claude-haiku-4-5", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-opus-4-5", False),
        ("claude-opus-4-5-20251101", False),
        ("claude-opus-4-6", False),
        ("claude-sonnet-4-5", False),
        ("claude-sonnet-4-6", False),
    ],
)
def test_which_claude_models_refuse_sampling(model, refuses):
    assert anthropic_format._refuses_sampling(model) is refuses


@pytest.mark.parametrize("param", [{"temperature": 0.2}, {"temperature": 0}, {"top_p": 0.9}, {"top_k": 5}])
def test_newer_claude_refuses_sampling_unless_dropped(env, fake, param):
    with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
        theagentrouter.completion(model="anthropic/claude-sonnet-5-5", messages=HELLO, api_base=fake.url, **param)
    assert not fake.requests
    theagentrouter.completion(
        model="anthropic/claude-sonnet-5-5", messages=HELLO, api_base=fake.url, drop_params=True, **param
    )
    assert not set(param) & set(fake.requests[-1][2])


def test_newer_claude_takes_temperature_1_and_older_claude_takes_the_rest(env, fake):
    base = dict(messages=HELLO, api_base=fake.url)
    theagentrouter.completion(model="anthropic/claude-sonnet-5-5", temperature=1, **base)
    assert fake.requests[-1][2]["temperature"] == 1
    theagentrouter.completion(model="anthropic/claude-sonnet-5-5", temperature=1, drop_params=True, **base)
    assert fake.requests[-1][2]["temperature"] == 1
    theagentrouter.completion(model="anthropic/claude-sonnet-4-6", temperature=0.2, top_p=0.9, top_k=5, **base)
    body = fake.requests[-1][2]
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.2, 0.9, 5)


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "lookup"}}])
def test_a_forced_tool_call_is_refused_or_becomes_auto(env, fake, choice, monkeypatch):
    monkeypatch.setattr(anthropic_format, "_warned", set())
    base = dict(messages=HELLO, api_base=fake.url, tool_choice=choice, tools=TOOLS)
    with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
        theagentrouter.completion(model="anthropic/claude-sonnet-5-5", **base)
    assert not fake.requests
    with pytest.warns(UserWarning, match="forced tool call"):
        theagentrouter.completion(model="anthropic/claude-sonnet-5-5", drop_params=True, **base)
    assert fake.requests[-1][2]["tool_choice"] == {"type": "auto"}
    # Haiku 5.5 takes a forced call, with or without drop_params.
    theagentrouter.completion(model="anthropic/claude-haiku-5-5", **base)
    assert fake.requests[-1][2]["tool_choice"]["type"] in ("any", "tool")


@pytest.mark.parametrize(
    "thinking", [{"reasoning_effort": "low"}, {"thinking": {"type": "enabled", "budget_tokens": 1024}}]
)
def test_opus_4_5_thinks_with_a_token_budget(env, fake, thinking):
    # It has effort levels on the model list, and Anthropic refuses adaptive thinking on it.
    theagentrouter.completion(
        model="anthropic/claude-opus-4-5", messages=HELLO, api_base=fake.url, max_tokens=2048, **thinking
    )
    assert fake.requests[-1][2]["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_a_gateway_you_set_decides_what_its_model_names_refuse(env, fake, monkeypatch):
    # Its names are its own: claude-sonnet-5 there may be any model, and it may remove the option itself.
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-gateway-key")
    theagentrouter.completion(model="claude-sonnet-5", messages=HELLO, temperature=0.2)
    assert fake.requests[-1][2]["temperature"] == 0.2
    theagentrouter.completion(model="claude-sonnet-5-5", messages=HELLO, tools=TOOLS, tool_choice="required")
    assert fake.requests[-1][2]["tool_choice"]["type"] == "any"
    theagentrouter.completion(model="claude-sonnet-5", messages=HELLO, temperature=0.2, drop_params=True)
    assert "temperature" not in fake.requests[-1][2]


# ---- a model behind a name the SDK does not know refuses max_tokens -------------------------------


NO_MAX_TOKENS = "refuses-max-tokens"


def _sent_limits(fake) -> list[tuple]:
    return [(b.get("max_tokens"), b.get("max_completion_tokens")) for _, _, b in fake.requests]


@pytest.fixture
def team_gateway(env, fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-gateway-key")
    return fake


def test_max_tokens_works_on_a_name_the_sdk_does_not_know(team_gateway):
    resp = theagentrouter.completion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=50)
    assert resp.choices[0].message.content == "echo: refunds yesterday"
    assert _sent_limits(team_gateway) == [(50, None), (None, 50)]


def test_max_tokens_works_on_a_name_the_sdk_does_not_know_streaming(team_gateway):
    chunks = list(theagentrouter.completion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=50, stream=True))
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices).strip() == "echo: refunds yesterday"
    assert _sent_limits(team_gateway) == [(50, None), (None, 50)]


async def test_max_tokens_works_on_a_name_the_sdk_does_not_know_async(team_gateway):
    resp = await theagentrouter.acompletion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=50)
    assert resp.choices[0].message.content == "echo: refunds yesterday"
    stream = await theagentrouter.acompletion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=50, stream=True)
    assert [c async for c in stream]
    assert _sent_limits(team_gateway) == [(50, None), (None, 50)] * 2


def test_each_call_starts_from_what_the_caller_wrote(team_gateway):
    # Behind a name, the model that answers can change from one call to the next.
    theagentrouter.completion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=50)
    theagentrouter.completion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=50)
    assert _sent_limits(team_gateway) == [(50, None), (None, 50), (50, None), (None, 50)]


def test_any_other_bad_request_is_not_sent_again(team_gateway):
    with pytest.raises(exceptions.ContextWindowExceededError):
        theagentrouter.completion(model="err-context", messages=HELLO, max_tokens=50)
    assert len(team_gateway.requests) == 1


# ---- Claude on Bedrock: LiteLLM's rules, in the format the gateway turns into Converse --------------


def _bedrock(fake, name, **params):
    """The request for a Bedrock model, as it reaches the gateway."""
    theagentrouter.completion(model=f"bedrock/{name}", messages=HELLO, api_base=fake.url, **params)
    return fake.requests[-1][2]


OLD_BEDROCK = "anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_4_6 = "us.anthropic.claude-sonnet-4-6"
NEW_BEDROCK = "us.anthropic.claude-sonnet-5-5"


@pytest.mark.parametrize("param", [{"temperature": 0.2}, {"top_p": 0.9}, {"top_k": 5}])
def test_newer_claude_on_bedrock_refuses_sampling_unless_dropped(env, fake, param, monkeypatch):
    monkeypatch.setattr(anthropic_format, "_warned", {"bedrock_top_k"})
    with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
        _bedrock(fake, NEW_BEDROCK, **param)
    assert not fake.requests
    assert not set(param) & set(_bedrock(fake, NEW_BEDROCK, drop_params=True, **param))
    assert "temperature" in _bedrock(fake, OLD_BEDROCK, temperature=0.2)


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "lookup"}}])
def test_a_forced_tool_call_on_bedrock_is_refused_or_becomes_auto(env, fake, choice, monkeypatch):
    monkeypatch.setattr(anthropic_format, "_warned", {"forced_tool"})
    with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
        _bedrock(fake, NEW_BEDROCK, tools=TOOLS, tool_choice=choice)
    assert _bedrock(fake, NEW_BEDROCK, tools=TOOLS, tool_choice=choice, drop_params=True)["tool_choice"] == "auto"
    assert _bedrock(fake, BEDROCK_4_6, tools=TOOLS, tool_choice=choice)["tool_choice"] == choice


def test_tool_choice_none_on_bedrock_is_refused_or_dropped(env, fake):
    # The gateway would read "none" as the name of a tool to force.
    for name in (NEW_BEDROCK, "amazon.nova-pro-v1:0"):
        with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
            _bedrock(fake, name, tools=TOOLS, tool_choice="none")
        assert "tool_choice" not in _bedrock(fake, name, tools=TOOLS, tool_choice="none", drop_params=True)


@pytest.mark.parametrize("param", [{"seed": 1}, {"presence_penalty": 0.1}, {"n": 2}, {"logprobs": True}])
def test_bedrock_refuses_what_it_does_not_take(env, fake, param):
    for name in (OLD_BEDROCK, "amazon.nova-pro-v1:0"):
        with pytest.raises(exceptions.UnsupportedParamsError, match="drop_params"):
            _bedrock(fake, name, **param)
    _bedrock(fake, OLD_BEDROCK, n=1)  # as LiteLLM: n=1 is always fine
    assert not set(param) & set(_bedrock(fake, OLD_BEDROCK, drop_params=True, **param))


@pytest.mark.parametrize(
    ("name", "params", "thinking", "max_tokens"),
    [
        (OLD_BEDROCK, {"reasoning_effort": "low"}, {"type": "enabled", "budget_tokens": 1024}, 5120),
        (OLD_BEDROCK, {"reasoning_effort": "high"}, {"type": "enabled", "budget_tokens": 4096}, 8192),
        (OLD_BEDROCK, {"thinking": {"type": "adaptive"}}, {"type": "enabled", "budget_tokens": 2048}, 6144),
        (
            OLD_BEDROCK,
            {"thinking": {"type": "enabled", "budget_tokens": 500}},
            {"type": "enabled", "budget_tokens": 1024},
            5120,
        ),
        (
            BEDROCK_4_6,
            {"thinking": {"type": "enabled", "budget_tokens": 2000}},
            {"type": "enabled", "budget_tokens": 2000},
            6096,
        ),
        (BEDROCK_4_6, {"reasoning_effort": "low"}, {"type": "adaptive", "display": "summarized"}, None),
        (NEW_BEDROCK, {"thinking": {"type": "enabled", "budget_tokens": 2000}}, {"type": "adaptive"}, None),
        (NEW_BEDROCK, {"thinking": {"type": "adaptive"}}, {"type": "adaptive"}, None),
        (NEW_BEDROCK, {"thinking": {"type": "disabled"}}, None, None),
        (BEDROCK_4_6, {"thinking": {"type": "disabled"}}, {"type": "disabled"}, None),
    ],
)
def test_claude_thinking_on_bedrock(env, fake, name, params, thinking, max_tokens, monkeypatch):
    monkeypatch.setattr(anthropic_format, "_warned", {"bedrock_effort"})
    body = _bedrock(fake, name, **params)
    assert body.get("thinking") == thinking
    assert body.get("max_tokens") == max_tokens
    assert "reasoning_effort" not in body


def test_other_bedrock_models_keep_reasoning_effort(env, fake):
    body = _bedrock(fake, "amazon.nova-pro-v1:0", reasoning_effort="low")
    assert body["reasoning_effort"] == "low" and "thinking" not in body


def test_xhigh_on_bedrock_follows_litellm(env, fake, monkeypatch):
    monkeypatch.setattr(anthropic_format, "_warned", {"bedrock_effort"})
    with pytest.raises(exceptions.BadRequestError, match="xhigh"):
        _bedrock(fake, BEDROCK_4_6, reasoning_effort="xhigh")
    assert _bedrock(fake, "us.anthropic.claude-opus-4-6-v1", reasoning_effort="xhigh")["thinking"]["type"] == "adaptive"


@pytest.mark.parametrize(
    ("params", "warning"),
    [
        ({"reasoning_effort": "low"}, "effort level"),
        ({"top_k": 5}, "top_k"),
        ({"tools": TOOLS, "parallel_tool_calls": False}, "parallel_tool_calls"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
    ],
)
def test_what_the_gateway_cannot_send_to_bedrock_warns(env, fake, params, warning, monkeypatch):
    monkeypatch.setattr(anthropic_format, "_warned", set())
    with pytest.warns(UserWarning, match=warning):
        _bedrock(fake, BEDROCK_4_6, **params)


def test_a_gateway_you_set_decides_what_its_bedrock_names_refuse(env, fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-gateway-key")
    monkeypatch.setattr(anthropic_format, "_warned", {"bedrock_effort"})
    model = f"bedrock/{NEW_BEDROCK}"
    theagentrouter.completion(model=model, messages=HELLO, temperature=0.2, seed=1)
    assert fake.requests[-1][2]["temperature"] == 0.2
    theagentrouter.completion(model=model, messages=HELLO, tools=TOOLS, tool_choice="required")
    assert fake.requests[-1][2]["tool_choice"] == "required"
    theagentrouter.completion(model=model, messages=HELLO, reasoning_effort="low")
    assert fake.requests[-1][2]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "reasoning_effort" not in fake.requests[-1][2]


def test_a_started_gateway_gives_no_resource_warning(env, monkeypatch):
    # aigw outlives the program that started it, on purpose.
    children = [subprocess.Popen(["sleep", "30"], start_new_session=True)]
    pid = children[0].pid
    monkeypatch.setattr(local_gateway, "_busy_internal", lambda: None)
    monkeypatch.setattr(local_gateway, "_launch", lambda *args: children.pop())
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            local_gateway._start([], [])
            gc.collect()
        assert not [w for w in seen if issubclass(w.category, ResourceWarning)]
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)


# ---- thinking through a tool loop on Bedrock, and the gateway's stream ------------------------------


def test_signed_thinking_goes_back_to_bedrock_as_content_parts(env, fake):
    blocks = [
        {"type": "thinking", "thinking": "Check the refunds.", "signature": "c2ln"},
        {"type": "redacted_thinking", "data": "cmVkYWN0ZWQ="},
    ]
    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    history = [
        {"role": "user", "content": "refunds yesterday"},
        {"role": "assistant", "content": "Let me look.", "tool_calls": [call], "thinking_blocks": blocks},
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]
    theagentrouter.completion(model=f"bedrock/{BEDROCK_4_6}", messages=history, api_base=fake.url)
    assistant = fake.requests[-1][2]["messages"][1]
    assert assistant["content"] == [
        {"type": "thinking", "text": "Check the refunds.", "signature": "c2ln"},
        {"type": "redacted_thinking", "redactedContent": "cmVkYWN0ZWQ="},
        {"type": "text", "text": "Let me look."},
    ]
    assert "thinking_blocks" not in assistant and assistant["tool_calls"][0]["id"] == "call_1"
    # Any other provider's request has no place for them.
    theagentrouter.completion(model="openai/gpt-5.4-mini", messages=history, api_base=fake.url)
    assistant = fake.requests[-1][2]["messages"][1]
    assert assistant["content"] == "Let me look." and "thinking_blocks" not in assistant


def test_the_gateways_bedrock_replies_keep_their_thinking():
    message = (
        openai_format.parse_response(
            {"choices": [{"message": {"content": "hi", "thinking_blocks": [{"type": "thinking", "signature": "s"}]}}]}
        )
        .choices[0]
        .message
    )
    assert message.thinking_blocks == [{"type": "thinking", "signature": "s"}]
    parser = openai_format.StreamParser("bedrock")
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "Check "}}]},
        {"choices": [{"delta": {"reasoning_content": "the refunds."}}]},
        {"choices": [{"delta": {"thinking_blocks": [{"type": "thinking", "signature": "c2ln"}]}}]},
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]},
    ]
    parsed = [chunk for data in chunks for chunk in parser.feed(json.dumps(data))]
    built = theagentrouter.stream_chunk_builder(parsed).choices[0].message
    assert built.reasoning_content == "Check the refunds."
    assert built.thinking_blocks == [{"type": "thinking", "thinking": "Check the refunds.", "signature": "c2ln"}]
    # Other providers' reasoning text stays reasoning text, as in LiteLLM.
    other = openai_format.StreamParser("deepseek").feed(json.dumps(chunks[0]))[0]
    assert other.choices[0].delta.thinking_blocks is None


# ---- more of LiteLLM's rules ------------------------------------------------------------------------


def test_max_tokens_0_stays_0_for_claude(env, fake):
    theagentrouter.completion(model="anthropic/claude-opus-5", messages=HELLO, max_tokens=0, api_base=fake.url)
    assert fake.requests[-1][2]["max_tokens"] == 0


def test_a_gateway_name_that_is_not_a_claude_version_goes_in_the_openai_format(env, fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-gateway-key")
    theagentrouter.completion(model="claude-first", messages=HELLO, max_tokens=50)
    assert fake.requests[-1][0].endswith("/chat/completions")
    theagentrouter.completion(model="claude-sonnet-6", messages=HELLO, max_tokens=50)
    assert fake.requests[-1][0].endswith("/messages")


def test_both_limits_on_a_name_the_sdk_does_not_know(env, fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-gateway-key")
    theagentrouter.completion(model=NO_MAX_TOKENS, messages=HELLO, max_tokens=100, max_completion_tokens=200)
    assert _sent_limits(fake) == [(100, 200), (None, 200)]


def test_both_limits_on_openai_send_max_completion_tokens(env, fake):
    theagentrouter.completion(
        model="openai/gpt-5.4-mini", messages=HELLO, max_tokens=100, max_completion_tokens=200, api_base=fake.url
    )
    assert _sent_limits(fake) == [(None, 200)]


def test_a_route_warns_about_what_the_openai_format_cannot_carry(env, fake, monkeypatch):
    monkeypatch.setattr(transport, "_warned", set())
    target = routing.Target("local gateway", routing.OPENAI, fake.url, None, "chat", "openai", {}, ("openai/gpt-5.4",))
    monkeypatch.setattr(routing, "resolve", lambda *a, **k: target)
    with pytest.warns(UserWarning, match="no thinking"):
        transport._prepare("chat", HELLO, None, False, {"thinking": {"type": "adaptive"}})
    with pytest.warns(UserWarning, match="no top_k"):
        transport._prepare("chat", HELLO, None, False, {"top_k": 5})


def test_a_frameworks_private_flag_does_not_warn(env, fake, monkeypatch, recwarn):
    # The OpenAI Agents SDK passes _skip_mcp_handler with tools.
    monkeypatch.setattr(transport, "_warned", set())
    theagentrouter.completion(model="openai/gpt-5.4-mini", messages=HELLO, _skip_mcp_handler=True, api_base=fake.url)
    assert not [w for w in recwarn.list if "_skip_mcp_handler" in str(w.message)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("us.anthropic.claude-sonnet-4-6", ("sonnet", 4, 6)),
        ("anthropic/claude-opus-4-5-20251101", ("opus", 4, 5)),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", ("sonnet", 4, 5)),
        ("claude-sonnet-4-20250514", ("sonnet", 4, 0)),
        ("claude-mythos-preview", ("mythos", 5, 0)),
        ("claude-3-7-sonnet-20250219", None),
        ("claude-opus-41", None),
    ],
)
def test_claude_family_and_version_from_a_name(name, expected):
    assert anthropic_format._claude(name) == expected


def test_a_cohere_model_on_bedrock_keeps_what_it_takes(env, fake):
    name = "cohere.command-r-plus-v1:0"
    body = _bedrock(fake, name, seed=1, presence_penalty=0.1, n=2)
    assert (body["seed"], body["presence_penalty"], body["n"]) == (1, 0.1, 2)
    assert "tool_choice" not in _bedrock(fake, name, tools=TOOLS, tool_choice="none")
    with pytest.raises(exceptions.UnsupportedParamsError):
        _bedrock(fake, name, logprobs=True)


def test_an_auto_tool_choice_on_bedrock_is_not_a_forced_call(env, fake):
    assert _bedrock(fake, NEW_BEDROCK, tools=TOOLS, tool_choice={"type": "auto"})["tool_choice"] == {"type": "auto"}


def test_an_inference_profile_arn_gets_claudes_thinking_rules(env, fake, monkeypatch):
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/acme"
    body = _bedrock(fake, arn, reasoning_effort="low")
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024} and "reasoning_effort" not in body
    assert _bedrock(fake, arn, thinking={"type": "disabled"})["thinking"] == {"type": "disabled"}


def test_the_bedrock_api_in_a_name_is_not_sent_to_bedrock():
    assert local_gateway._provider_name("bedrock", "converse/us.anthropic.claude-sonnet-4-6") == (
        "us.anthropic.claude-sonnet-4-6"
    )
    assert (
        local_gateway._provider_name("bedrock", "invoke/anthropic.claude-sonnet-4-6") == "anthropic.claude-sonnet-4-6"
    )
    assert local_gateway._provider_name("openai", "converse/x") == "converse/x"


def test_anthropic_messages_beta_headers_stay_beside_the_callers(env, fake):
    headers = {"anthropic-version": "2023-06-01", "Anthropic-Beta": "acme-1"}
    out = anthropic_format.with_betas(headers, {"speed": "fast", "output_format": {"type": "json_schema"}})
    assert out["anthropic-beta"] == "acme-1,structured-outputs-2025-11-13,fast-mode-2026-02-01"
    assert "Anthropic-Beta" not in out
    assert anthropic_format.with_betas(headers, {"max_tokens": 5}) is headers


class _FakePydantic:
    """What the SDK reads from a pydantic class; the SDK has no pydantic dependency."""

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "title": "_FakePydantic",
            "properties": {
                "id": {"type": "string"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            },
            "required": ["id"],
        }


def test_a_pydantic_class_as_response_format(env, fake):
    theagentrouter.completion(
        model="openai/gpt-5.4-mini", messages=HELLO, response_format=_FakePydantic, api_base=fake.url
    )
    sent = fake.requests[-1][2]["response_format"]["json_schema"]
    assert sent["name"] == "_FakePydantic" and sent["strict"] is True
    assert sent["schema"]["required"] == ["id", "note"] and sent["schema"]["additionalProperties"] is False
    assert "default" not in sent["schema"]["properties"]["note"]
    theagentrouter.completion(
        model="anthropic/claude-opus-5", messages=HELLO, response_format=_FakePydantic, api_base=fake.url
    )
    assert fake.requests[-1][2]["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_a_schema_for_claude_moves_what_anthropic_refuses_into_the_description():
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 3}, "minItems": 1, "uniqueItems": False},
            "kind": {"oneOf": [{"type": "string"}, {"type": "integer", "minimum": 1}]},
        },
        "$defs": {"inner": {"type": "object", "properties": {}}},
    }
    out = anthropic_format._claude_schema(schema)
    assert out["additionalProperties"] is False and out["$defs"]["inner"]["additionalProperties"] is False
    tags = out["properties"]["tags"]
    assert "minItems" not in tags and "uniqueItems" not in tags and "minItems: 1" in tags["description"]
    assert "uniqueItems" not in tags["description"]  # false asks for nothing
    assert "maxLength" not in tags["items"] and "maxLength: 3" in tags["items"]["description"]
    assert out["properties"]["kind"]["anyOf"][1] == {"type": "integer", "description": "Note: minimum: 1."}
    assert "oneOf" not in out["properties"]["kind"]
