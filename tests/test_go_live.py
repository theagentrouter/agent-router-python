"""Fixes from the go-live readiness review: who can call the local gateway, which key goes where,
the Envoy version, LiteLLM's stream shape and LiteLLM's drop_params for Claude."""

from __future__ import annotations

import json
import socket

import httpx
import pytest

import theagentrouter
from theagentrouter import exceptions, local_gateway, routing
from theagentrouter.compat import litellm

from .test_local_gateway import needs_aigw, needs_xds_port
from .test_sdk import HELLO

MODELS = ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"]


# ---- who can call the local gateway -------------------------------------------------------------


def test_local_gateway_config_answers_only_this_computer():
    config = local_gateway.config_yaml(["openai"], {"OPENAI_API_KEY": "k"}, port=1975, models=[])
    policy = next(d for d in config.split("\n---\n") if "kind: SecurityPolicy" in d)
    assert "defaultAction: Deny" in policy
    assert "- 127.0.0.1/32" in policy and "- ::1/128" in policy


def test_exported_config_has_no_loopback_policy(env):
    # A server gateway is called from other computers; its authentication is the team's to add.
    assert "kind: SecurityPolicy" not in local_gateway.export_config(providers=["openai"])


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
    assert httpx.post(f"http://{address}:{port}/v1/chat/completions", json=body, timeout=20).status_code == 403
    assert len(real_gateway.requests) == sent  # the provider never saw it, so no key was used
    assert httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=20).status_code == 200


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
    with pytest.raises(exceptions.UnsupportedParamsError):  # the call's own setting wins
        theagentrouter.completion(model=MODELS[1], messages=HELLO, seed=1, api_base=fake.url, drop_params=False)


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
