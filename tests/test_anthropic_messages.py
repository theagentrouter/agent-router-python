"""litellm.anthropic.messages.create and acreate: Anthropic's own format, through the gateway."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

import theagentrouter
from theagentrouter import exceptions

from .fake_provider import FakeProvider

HELLO = [{"role": "user", "content": "refunds yesterday"}]


@pytest.fixture
def fake(monkeypatch):
    server = FakeProvider()
    for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_API_KEY", "AGENTROUTER_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", "/nonexistent-agentrouter-home")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    yield server
    server.close()


def test_as_in_litellm(fake):
    from theagentrouter.compat import litellm

    assert litellm.anthropic.messages.create is theagentrouter.anthropic.messages.create
    assert litellm.anthropic_messages is theagentrouter.anthropic_messages
    r = litellm.anthropic.messages.create(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=100,
        messages=HELLO,
        system=[{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "enabled", "budget_tokens": 1024},
        api_base=fake.url,
    )
    path, headers, body = fake.requests[-1]
    # Sent as written, in Anthropic's format: nothing translated or dropped.
    assert path == "/v1/messages" and body["model"] == "claude-sonnet-4-5" and body["max_tokens"] == 100
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"} and body["thinking"]["budget_tokens"] == 1024
    assert headers["x-api-key"] == "sk-ant-test" and "anthropic-version" in headers
    # A dict, as LiteLLM returns, that also reads by attribute.
    assert isinstance(r, dict) and r["content"][0]["text"] == r.content[0].text == "echo: refunds yesterday"
    assert r._hidden_params["route"] == "api_base" and r._hidden_params["response_cost"] > 0


def test_gateway_you_set_gets_messages(fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "gw-key")
    for model in (
        "anthropic/claude-sonnet-4-5",
        "claude-sonnet-4-5",
        "acme-claude-9",
    ):  # the last: the gateway's own name
        r = theagentrouter.anthropic.messages.create(model=model, max_tokens=50, messages=HELLO)
        path, headers, body = fake.requests[-1]
        assert (path, body["model"]) == ("/anthropic/v1/messages", model)
        assert headers["x-api-key"] == "gw-key" and "authorization" not in headers
        assert r._hidden_params["route"] == "gateway"


def test_stream_gives_the_events_as_bytes(fake):
    chunks = list(
        theagentrouter.anthropic.messages.create(
            model="anthropic/claude-sonnet-4-5", max_tokens=50, messages=HELLO, stream=True, api_base=fake.url
        )
    )
    assert fake.requests[-1][2]["stream"] is True
    assert all(isinstance(c, bytes) and c.endswith(b"\n\n") for c in chunks)
    events = [json.loads(c.split(b"data: ", 1)[1]) for c in chunks if b"data: " in c]
    assert events[0]["type"] == "message_start" and events[-1]["type"] == "message_stop"
    text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
    assert text.strip() == "echo: refunds yesterday"


def test_async(fake):
    async def run():
        r = await theagentrouter.anthropic_messages(
            model="anthropic/claude-sonnet-4-5", max_tokens=50, messages=HELLO, api_base=fake.url
        )
        stream = await theagentrouter.anthropic.messages.acreate(
            model="anthropic/claude-sonnet-4-5", max_tokens=50, messages=HELLO, stream=True, api_base=fake.url
        )
        return r, [c async for c in stream]

    r, chunks = asyncio.run(run())
    assert r["content"][0]["text"] == "echo: refunds yesterday" and chunks[-1].startswith(b"event: message_stop")


def test_only_claude(fake):
    for model in ("openai/gpt-4o-mini", "gpt-4o-mini", "gemini/gemini-2.5-flash"):
        with pytest.raises(exceptions.BadRequestError, match="Messages API for Claude.*use completion"):
            theagentrouter.anthropic.messages.create(model=model, max_tokens=5, messages=HELLO)
    with pytest.raises(exceptions.BadRequestError, match="cannot tell which provider"):
        theagentrouter.anthropic.messages.create(model="acme-model-9", max_tokens=5, messages=HELLO)
    assert fake.requests == []


def test_errors_keep_litellm_names(fake):
    with pytest.raises(exceptions.ContextWindowExceededError):
        theagentrouter.anthropic.messages.create(
            model="anthropic/err-context", max_tokens=5, messages=HELLO, api_base=fake.url
        )
    with pytest.raises(exceptions.ContextWindowExceededError):
        list(
            theagentrouter.anthropic.messages.create(
                model="anthropic/err-context", max_tokens=5, messages=HELLO, api_base=fake.url, stream=True
            )
        )


def test_import_litellm_anthropic(monkeypatch):
    from theagentrouter.compat import install_as_litellm

    for name in [n for n in sys.modules if n == "litellm" or n.startswith("litellm.")]:
        monkeypatch.delitem(sys.modules, name)
    install_as_litellm()
    import litellm
    from litellm.anthropic import messages

    assert messages.create is theagentrouter.anthropic.messages.create and litellm.anthropic.messages is messages
