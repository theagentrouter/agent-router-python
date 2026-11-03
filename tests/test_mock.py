# SPDX-License-Identifier: Apache-2.0
"""LiteLLM's mock_response: answered on this computer, with no gateway and no provider call."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import theagentrouter
from theagentrouter import exceptions, local_gateway
from theagentrouter.compat import litellm

HELLO = [{"role": "user", "content": "refunds yesterday"}]
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "search_refunds", "arguments": "{}"}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a mocked call used the network")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
    monkeypatch.setattr(local_gateway, "ensure", refuse)
    monkeypatch.setattr(local_gateway, "running", refuse)


def test_reply_as_in_litellm():
    r = litellm.completion(model="gpt-4o-mini", messages=HELLO, mock_response="Offline reply")
    assert r.choices[0].message.content == "Offline reply" and r.choices[0].finish_reason == "stop"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens) == (10, 20, 30)
    assert r.model == "gpt-4o-mini" and r._hidden_params["route"] == "mock"
    assert r._hidden_params["response_cost"] == theagentrouter.completion_cost(r) > 0
    two = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO, mock_response="x", n=2)
    assert [c.message.content for c in two.choices] == ["x", "x"]


def test_tool_calls_and_whole_responses():
    r = theagentrouter.completion(model="openai/gpt-4o", messages=HELLO, mock_tool_calls=[TOOL_CALL])
    assert r.choices[0].message.content == "This is a mock request"  # LiteLLM's default text
    assert r.choices[0].message.tool_calls[0].function.name == "search_refunds"
    given = theagentrouter.completion(model="openai/gpt-4o", messages=HELLO, mock_response="a")
    assert theagentrouter.completion(model="openai/gpt-4o", messages=HELLO, mock_response=given) is given
    r = theagentrouter.completion(
        model="openai/gpt-4o",
        messages=HELLO,
        mock_response={"choices": [{"message": {"role": "assistant", "content": "from a dict"}}]},
    )
    assert r.choices[0].message.content == "from a dict"


@pytest.mark.parametrize(
    "mock,cls",
    [
        ("litellm.RateLimitError", exceptions.RateLimitError),
        ("litellm.ContextWindowExceededError", exceptions.ContextWindowExceededError),
        ("litellm.InternalServerError", exceptions.InternalServerError),
        (exceptions.AuthenticationError("bad key"), exceptions.AuthenticationError),
        (ValueError("boom"), exceptions.InternalServerError),
    ],
)
def test_errors(mock, cls):
    with pytest.raises(cls):
        litellm.completion(model="gpt-4o-mini", messages=HELLO, mock_response=mock)
    with pytest.raises(cls):  # before the first chunk, as in LiteLLM
        litellm.completion(model="gpt-4o-mini", messages=HELLO, mock_response=mock, stream=True)


def test_timeout():
    with pytest.raises(exceptions.Timeout):
        litellm.completion(model="gpt-4o-mini", messages=HELLO, mock_timeout=True, timeout=1)


def test_stream():
    chunks = list(theagentrouter.completion(model="gpt-4o-mini", messages=HELLO, mock_response="streamed", stream=True))
    assert [c.choices[0].delta.content for c in chunks[:-1]] == ["str", "eam", "ed"]
    assert chunks[-1].choices[0].finish_reason == "stop" and chunks[-1].usage.total_tokens == 30
    assert theagentrouter.stream_chunk_builder(chunks).choices[0].message.content == "streamed"


def test_async():
    async def run():
        r = await litellm.acompletion(model="gpt-4o-mini", messages=HELLO, mock_response="async reply")
        stream = await litellm.acompletion(model="gpt-4o-mini", messages=HELLO, mock_response="abcd", stream=True)
        return r, [c async for c in stream]

    r, chunks = asyncio.run(run())
    assert r.choices[0].message.content == "async reply"
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "abcd"


def test_callbacks_run_and_no_warning(recwarn):
    seen = []
    litellm.success_callback.append(lambda kwargs, response, *_: seen.append(response.choices[0].message.content))
    try:
        litellm.completion(model="gpt-4o-mini", messages=HELLO, mock_response="logged")
    finally:
        litellm.success_callback.clear()
    assert seen == ["logged"]
    assert not [w for w in recwarn if "mock_response" in str(w.message)]


def test_router_route_names(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))  # the Router saves its routes there
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        local_gateway, "installed", lambda: True
    )  # a Router needs it to exist; CI installs the SDK alone
    router = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
    assert (
        router.completion(model="chat", messages=HELLO, mock_response="routed").choices[0].message.content == "routed"
    )
