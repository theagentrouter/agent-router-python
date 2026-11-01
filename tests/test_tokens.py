"""token_counter: the gateway's count for OpenAI and Anthropic, LiteLLM's estimate otherwise."""

from __future__ import annotations

import sys

import pytest

import theagentrouter as ar
from theagentrouter import routing, tokens
from theagentrouter.compat import litellm

from .fake_provider import FakeProvider

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]


@pytest.fixture
def server(monkeypatch):
    s = FakeProvider()
    for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_API_KEY", "AGENTROUTER_PROJECT", "AGENTROUTER_TOKEN_COUNT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", "/nonexistent-agentrouter-home")
    monkeypatch.setattr(tokens, "_warned", set())
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # the expected numbers are the estimate without tiktoken
    yield s
    s.close()


@pytest.fixture
def gateway(server, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", server.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "gw-key")
    return server


def test_gateway_counts_openai_format(gateway):
    assert ar.token_counter(model="openai/gpt-4o-mini", messages=CHAT, tools=TOOLS) == 42
    path, headers, body = gateway.requests[-1]
    assert path == "/v1/responses/input_tokens" and headers["authorization"] == "Bearer gw-key"
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["input"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city":"Paris"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
    ]
    assert body["tools"] == [{"type": "function", **TOOLS[0]["function"]}]


def test_gateway_counts_text(gateway):
    assert ar.token_counter(model="openai/gpt-4o-mini", text=["a", "b"]) == 42
    assert gateway.requests[-1][2]["input"] == "ab"


def test_no_count_endpoint_falls_back_to_the_estimate(gateway):
    with pytest.warns(UserWarning, match="did not count tokens"):
        n = ar.token_counter(model="openai/no-count", text="abcdefgh")
    assert n == 2  # about four characters a token, without tiktoken
    assert gateway.requests[-1][0] == "/v1/responses/input_tokens"


def test_estimate_setting_sends_nothing(gateway, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_TOKEN_COUNT", "estimate")
    tokens._warned.add("tiktoken")
    assert (
        ar.token_counter(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "abcd"}]) == 3 + 1 + 1 + 3
    )  # message, role, content, reply
    assert gateway.requests == []


def test_providers_without_a_count_are_estimated_with_no_request(server):
    tokens._warned.add("tiktoken")
    # No gateway set and not OpenAI or Anthropic: the local gateway is not even started.
    assert ar.token_counter(model="groq/llama-3.1-8b-instant", text="abcdefgh") == 2
    assert ar.token_counter(model="gpt-4o-mini", text="abcdefgh") == 2  # no provider: estimated
    assert server.requests == []


def test_anthropic_counts_in_messages_format(server, monkeypatch):
    target = routing.Target(
        "local gateway",
        routing.ANTHROPIC,
        server.url.removesuffix("/v1") + "/anthropic/v1",
        None,
        "anthropic/claude-haiku-4-5",
        "anthropic",
    )
    monkeypatch.setattr(routing, "resolve", lambda model, **_: target)
    assert ar.token_counter(model="anthropic/claude-haiku-4-5", messages=CHAT, tools=TOOLS) == 42
    path, _, body = server.requests[-1]
    assert path == "/anthropic/v1/messages/count_tokens"
    assert body["system"] == "Be brief." and body["messages"][0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Weather in Paris?"}],
    }
    assert body["tools"][0]["name"] == "get_weather" and "max_tokens" not in body


def test_estimate_follows_litellm_rules(server, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_TOKEN_COUNT", "estimate")
    tokens._warned.add("tiktoken")
    one = [{"role": "user", "content": "abcd", "name": "abcd"}]
    assert (
        ar.token_counter(model="x/y", messages=one) == 3 + 1 + 1 + 1 + 1 + 3
    )  # message, role, content, name, name bonus, reply
    assert ar.token_counter(model="x/y", messages=one, count_response_tokens=True) == 3 + 1 + 1 + 1 + 1
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://acme.example/a.png"}}]}]
    assert ar.token_counter(model="x/y", messages=image) == 3 + 1 + 250 + 3


def test_arguments_checked_like_litellm(server):
    with pytest.raises(ValueError):
        ar.token_counter(model="x/y", text="a", messages=[])
    with pytest.raises(ValueError):
        ar.token_counter(model="x/y")
    with pytest.raises(ValueError):
        ar.token_counter(model="x/y", text="a", tools=TOOLS)


def test_compat_disable_token_counter(server, monkeypatch):
    monkeypatch.setattr(litellm, "disable_token_counter", True)
    assert litellm.token_counter(model="openai/gpt-4o-mini", text="hello") == 0
    assert server.requests == []
