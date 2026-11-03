# SPDX-License-Identifier: Apache-2.0
"""The rest of what agent frameworks call from LiteLLM: names without a provider, validate_environment,
responses, stream_chunk_builder, callbacks, model_list and the types they import."""

from __future__ import annotations

import asyncio
import warnings

import pytest

import theagentrouter as ar
from theagentrouter import exceptions, local_gateway, routing
from theagentrouter.compat import _callbacks, litellm
from theagentrouter.types import Delta, ModelResponseStream, StreamingChoices

from .fake_provider import FakeProvider

HELLO = [{"role": "user", "content": "hi"}]


@pytest.fixture
def fake(monkeypatch, tmp_path):
    server = FakeProvider()
    for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_API_KEY", "AGENTROUTER_PROJECT", "AGENTROUTER_TOKEN_COUNT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(routing, "_route_warned", set())
    monkeypatch.setattr(_callbacks, "_warned", set())
    for name in ("success_callback", "failure_callback", "callbacks", "_async_success_callback"):
        monkeypatch.setattr(litellm, name, [])
    yield server
    server.close()


# --- model names without a provider ---------------------------------------------------------------


def test_name_without_provider_found_in_the_model_list(fake):
    r = ar.completion(model="gpt-4o-mini", messages=HELLO, api_base=fake.url)
    path, _, body = fake.requests[-1]
    assert path == "/v1/chat/completions" and body["model"] == "gpt-4o-mini"
    assert r._hidden_params["llm_provider"] == "openai"
    ar.completion(model="claude-haiku-4-5", messages=HELLO, api_base=fake.url, max_tokens=5)
    assert fake.requests[-1][0].endswith("/messages")  # Claude found, sent in Messages format
    assert routing.with_provider("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert routing.with_provider("acme-chat-1") == "acme-chat-1"
    # As in LiteLLM: only OpenAI and Anthropic names; the rest need provider/model.
    assert routing.with_provider("gpt-5.5") == "openai/gpt-5.5"  # not on the list yet
    for name in ("mistral-large-latest", "deepseek-chat", "grok-4", "llama-3.1-8b-instant"):
        assert routing.with_provider(name) == name
    # A bare Gemini name means Vertex AI in LiteLLM: refused rather than sent to the Gemini API.
    with pytest.raises(exceptions.BadRequestError, match="Vertex AI.*gemini/gemini-2.5-flash"):
        routing.with_provider("gemini-2.5-flash")
    assert litellm.get_llm_provider("gpt-4o-mini")[:2] == ("gpt-4o-mini", "openai")
    assert litellm.get_llm_provider("gemini-2.5-flash")[:2] == ("gemini-2.5-flash", "vertex_ai")  # as LiteLLM says


def test_route_in_python_warns_with_a_gateway_set(fake, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: True)  # CI installs theagentrouter-core alone
    ar.route("chat", primary="openai/gpt-4o-mini")
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    with pytest.warns(UserWarning, match="routes apply to the local gateway only"):
        ar.completion(model="chat", messages=HELLO)
    assert fake.requests[-1][2]["model"] == "chat"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ar.completion(model="chat", messages=HELLO)  # once


# --- validate_environment -------------------------------------------------------------------------


def test_validate_environment(fake, monkeypatch):
    assert litellm.validate_environment("openai/gpt-4o-mini") == {"keys_in_environment": True, "missing_keys": []}
    monkeypatch.setattr(local_gateway, "installed", lambda: True)  # CI installs theagentrouter-core alone
    ar.route("chat", primary="openai/gpt-4o-mini", fallbacks=["anthropic/claude-haiku-4-5"])
    monkeypatch.delenv("OPENAI_API_KEY")
    assert litellm.validate_environment("gpt-4o-mini") == {
        "keys_in_environment": False,
        "missing_keys": ["OPENAI_API_KEY"],
    }
    assert litellm.validate_environment("gpt-4o-mini", api_key="sk")["keys_in_environment"] is True
    assert litellm.validate_environment() == {"keys_in_environment": False, "missing_keys": []}
    assert litellm.validate_environment("acme/unknown") == {"keys_in_environment": False, "missing_keys": []}
    assert litellm.validate_environment("chat")["missing_keys"] == ["OPENAI_API_KEY"]
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)  # the gateway holds provider keys
    assert litellm.validate_environment("gpt-4o-mini")["missing_keys"] == ["AGENTROUTER_API_KEY"]
    monkeypatch.setenv("AGENTROUTER_API_KEY", "gw")
    assert litellm.validate_environment("gpt-4o-mini")["keys_in_environment"] is True


# --- responses ------------------------------------------------------------------------------------


def test_responses_through_a_gateway(fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    r = litellm.responses(
        input="Say hi", model="openai/gpt-4o-mini", instructions="Be brief", max_output_tokens=20, store=None
    )
    path, _, body = fake.requests[-1]
    assert path == "/v1/responses"
    assert body == {
        "model": "openai/gpt-4o-mini",
        "input": "Say hi",
        "instructions": "Be brief",
        "max_output_tokens": 20,
    }
    assert r.output[0].content[0].text == r["output"][0]["content"][0]["text"] == "ok:Say hi"
    assert r.output_text == "ok:Say hi" and r.usage.input_tokens == 10
    assert r.model_dump()["id"] == "resp_1" and "_hidden_params" not in r.model_dump()
    assert r._hidden_params["route"] == "gateway" and r._hidden_params["response_cost"] > 0
    assert asyncio.run(litellm.aresponses(input="x", model="openai/gpt-4o-mini")).output_text == "ok:x"


def test_responses_only_for_openai(fake):
    with pytest.raises(exceptions.BadRequestError, match="Responses API for OpenAI models, not anthropic"):
        ar.responses(input="hi", model="claude-haiku-4-5")
    assert fake.requests == []  # refused before anything starts
    with pytest.raises(exceptions.UnsupportedParamsError):
        ar.responses(input="hi", model="openai/gpt-4o-mini", stream=True)


# --- stream_chunk_builder -------------------------------------------------------------------------


def _chunk(content=None, calls=None, finish=None, usage=None):
    return ModelResponseStream(
        id="c1",
        model="gpt-4o-mini",
        choices=[StreamingChoices(delta=Delta(content=content, tool_calls=calls), finish_reason=finish)],
        usage=usage,
    )


def test_stream_chunk_builder_joins_text_and_tool_calls():
    calls = [
        _chunk(calls=[{"index": 0, "id": "t1", "function": {"name": "get_weather", "arguments": '{"ci'}}]),
        _chunk(calls=[{"index": 0, "function": {"arguments": 'ty":"Paris"}'}}]),
        _chunk(
            calls=[{"index": 1, "id": "t2", "function": {"name": "get_time", "arguments": "{}"}}], finish="tool_calls"
        ),
    ]
    r = litellm.stream_chunk_builder(calls)
    tc = r.choices[0].message.tool_calls
    assert [(c.id, c.function.name, c.function.arguments) for c in tc] == [
        ("t1", "get_weather", '{"city":"Paris"}'),
        ("t2", "get_time", "{}"),
    ]
    assert r.choices[0].finish_reason == "tool_calls"
    text = [
        _chunk("Hel"),
        _chunk("lo"),
        {"id": "c1", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    ]
    r = ar.stream_chunk_builder(text)
    assert r.choices[0].message.content == "Hello" and r.usage.total_tokens == 5 and r.id == "c1"
    assert ar.stream_chunk_builder([]) is None


# --- callbacks ------------------------------------------------------------------------------------


class Logger:
    def __init__(self):
        self.events = []

    def log_success_event(self, kwargs, response, start, end):
        self.events.append(("ok", kwargs["model"], response.usage.total_tokens, end >= start))

    def log_failure_event(self, kwargs, response, start, end):
        self.events.append(("fail", type(kwargs["exception"]).__name__))


def test_callbacks_after_a_call(fake, monkeypatch):
    seen, logger = [], Logger()
    litellm.success_callback = [
        lambda kw, resp, s, e: seen.append((kw["model"], kw["response_cost"] >= 0, resp.choices[0].message.content))
    ]
    litellm.callbacks = [logger]
    litellm.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    assert seen == [("openai/gpt-4o-mini", True, seen[0][2])] and seen[0][2]
    assert logger.events[0][:2] == ("ok", "openai/gpt-4o-mini") and logger.events[0][3] is True
    litellm.failure_callback = [logger]  # also in callbacks: called once
    with pytest.raises(exceptions.AuthenticationError):
        litellm.completion(model="openai/err-401", messages=HELLO, api_base=fake.url)
    assert logger.events[-1] == ("fail", "AuthenticationError") and logger.events[-2][0] == "ok"


def test_callbacks_after_a_stream(fake):
    seen = []
    litellm.success_callback = [
        lambda kw, resp, s, e: seen.append(kw["complete_streaming_response"].choices[0].message.content)
    ]
    chunks = list(litellm.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, stream=True))
    assert seen == ["".join(c.choices[0].delta.content or "" for c in chunks if c.choices)]


def test_async_callbacks_and_strings(fake):
    events = []

    class AsyncLogger:
        async def async_log_success_event(self, kwargs, response, start, end):
            events.append("async")

    litellm.callbacks = [AsyncLogger()]
    litellm.success_callback = ["langfuse"]
    with pytest.warns(UserWarning, match="'langfuse' integration"):
        asyncio.run(litellm.acompletion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url))
    assert events == ["async"]


def test_a_failing_callback_does_not_break_the_call(fake):
    def broken(*_):
        raise RuntimeError("logger down")

    litellm.success_callback = [broken]
    with pytest.warns(UserWarning, match="logger down"):
        r = litellm.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    assert r.choices[0].message.content


# --- names frameworks import ----------------------------------------------------------------------


def test_names_frameworks_import():
    assert "gpt-4o-mini" in litellm.model_list and "claude-haiku-4-5" in litellm.model_list
    assert litellm.cost_calculator.completion_cost is litellm.completion_cost
    for name in (
        "ChatCompletionMessageToolCall",
        "ChatCompletionDeltaToolCall",
        "CostPerToken",
        "ResponsesAPIResponse",
    ):
        assert hasattr(litellm.types.utils, name)
    assert litellm.cache is None and litellm._async_success_callback == []
