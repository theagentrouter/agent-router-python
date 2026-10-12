from __future__ import annotations

import json
import warnings

import pytest

import theagentrouter
from theagentrouter import exceptions, local_gateway
from theagentrouter.compat import litellm

from .fake_provider import FakeProvider

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
HELLO = [{"role": "user", "content": "refunds yesterday"}]


@pytest.fixture
def fake(monkeypatch):
    server = FakeProvider()
    for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_API_KEY", "AGENTROUTER_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", "/nonexistent-agentrouter-home")  # never read the real one
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    yield server
    server.close()


def test_openai_text(fake):
    r = theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    assert r.choices[0].message.content == "echo: refunds yesterday"
    assert r.usage.total_tokens == 18
    path, headers, body = fake.requests[-1]
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-4o-mini"  # the provider gets its own model name
    assert r._hidden_params["route"] == "api_base"  # the endpoint named on the call
    assert r._hidden_params["response_cost"] > 0  # gpt-4o-mini is in the bundled table


def test_openai_tool_call_and_dict_access(fake):
    r = theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO, tools=TOOLS, api_base=fake.url)
    call = r.choices[0].message.tool_calls[0]
    assert call.function.name == "search_refunds"
    assert json.loads(call.function.arguments) == {"query": "refunds yesterday"}
    assert r.choices[0].finish_reason == "tool_calls"
    assert r.get("choices")[0]["message"]["tool_calls"][0]["id"] == "call_1"  # frameworks read both ways


def test_openai_stream_tool_call_and_usage(fake):
    args, usage, finish = "", None, None
    for chunk in theagentrouter.completion(
        model="openai/gpt-4o-mini", messages=HELLO, tools=TOOLS, api_base=fake.url, stream=True
    ):
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            args += chunk.choices[0].delta.tool_calls[0].function.arguments
        if chunk.choices and chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
        if chunk.usage:
            usage = chunk.usage
    assert json.loads(args) == {"query": "refunds yesterday"}
    assert finish == "tool_calls"
    assert usage.total_tokens == 18
    assert fake.requests[-1][2]["stream_options"] == {"include_usage": True}


def test_anthropic_request_conversion(fake):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "refunds yesterday"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_9",
                    "type": "function",
                    "function": {"name": "search_refunds", "arguments": '{"query": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_9", "content": "3 refunds"},
    ]
    r = theagentrouter.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        api_base=fake.url,
        max_tokens=256,
    )
    path, headers, body = fake.requests[-1]
    assert path == "/v1/messages"
    assert headers["x-api-key"] == "sk-ant-test" and headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "Be brief."
    assert body["max_tokens"] == 256
    assert body["tools"][0]["input_schema"]["properties"]["query"]["type"] == "string"
    assert body["tool_choice"] == {"type": "auto"}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "toolu_9",
        "name": "search_refunds",
        "input": {"query": "x"},
    }
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert r.choices[0].message.content == "echo: 3 refunds"  # a tool result came back, so the model answers


def test_anthropic_tool_call_response(fake):
    r = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO, tools=TOOLS, api_base=fake.url)
    call = r.choices[0].message.tool_calls[0]
    assert (call.id, call.function.name) == ("toolu_1", "search_refunds")
    assert json.loads(call.function.arguments) == {"query": "refunds yesterday"}
    assert r.choices[0].finish_reason == "tool_calls"
    assert r.usage.prompt_tokens == 11 and r.usage.completion_tokens == 7


def test_anthropic_stream(fake):
    text, args, usage, finish = "", "", None, None
    for chunk in theagentrouter.completion(
        model="anthropic/claude-sonnet-4-5", messages=HELLO, tools=TOOLS, api_base=fake.url, stream=True
    ):
        for c in chunk.choices:
            if c.delta.tool_calls:
                args += c.delta.tool_calls[0].function.arguments
            text += c.delta.content or ""
            finish = c.finish_reason or finish
        usage = chunk.usage or usage
    assert json.loads(args) == {"query": "refunds yesterday"}
    assert finish == "tool_calls" and usage.completion_tokens == 7 and usage.prompt_tokens == 11
    text = "".join(
        c.choices[0].delta.content or ""
        for c in theagentrouter.completion(
            model="anthropic/claude-sonnet-4-5", messages=HELLO, api_base=fake.url, stream=True
        )
        if c.choices
    )
    assert text.strip() == "echo: refunds yesterday"


@pytest.mark.parametrize(
    "model,cls",
    [
        ("openai/err-429", exceptions.RateLimitError),
        ("openai/err-401", exceptions.AuthenticationError),
        ("openai/err-500", exceptions.InternalServerError),
        ("openai/err-context", exceptions.ContextWindowExceededError),
        ("anthropic/err-context", exceptions.ContextWindowExceededError),
        ("openai/err-budget", exceptions.BudgetExceededError),
    ],
)
def test_errors_keep_litellm_names(fake, model, cls):
    with pytest.raises(cls) as info:
        theagentrouter.completion(model=model, messages=HELLO, api_base=fake.url)
    assert isinstance(info.value, getattr(litellm, cls.__name__))


def test_stream_error_raises(fake):
    with pytest.raises(exceptions.RateLimitError):
        list(theagentrouter.completion(model="openai/err-429", messages=HELLO, api_base=fake.url, stream=True))


def test_gateway_gets_every_call_in_openai_format(fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "gw-key")
    r = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO)
    path, headers, body = fake.requests[-1]
    assert path == "/v1/chat/completions"
    assert body["model"] == "anthropic/claude-sonnet-4-5"  # unchanged; the gateway routes on it
    assert headers["authorization"] == "Bearer gw-key"
    assert r._hidden_params["route"] == "gateway"


def test_every_call_needs_a_gateway(fake, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: False)  # tests/test_local_gateway.py covers installed
    # No direct calls: without the local gateway, OpenAI needs a gateway as much as Bedrock does.
    for model in ("openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5", "bedrock/claude-sonnet"):
        with pytest.raises(exceptions.BadRequestError, match="needs the local gateway.*AGENTROUTER_BASE_URL"):
            theagentrouter.completion(model=model, messages=HELLO)
    assert fake.requests == []
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(exceptions.AuthenticationError, match="OPENAI_API_KEY"):
        theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    # A name without a provider takes it from the model list, as in LiteLLM; an unknown one is refused.
    with pytest.raises(exceptions.BadRequestError, match="openai/gpt-4o-mini needs the local gateway"):
        theagentrouter.completion(model="gpt-4o-mini", messages=HELLO)
    with pytest.raises(exceptions.BadRequestError, match="provider prefix"):
        theagentrouter.completion(model="acme-model-9", messages=HELLO)


def test_connection_error():
    with pytest.raises(exceptions.APIConnectionError):
        theagentrouter.completion(
            model="openai/gpt-4o-mini", messages=HELLO, api_key="k", api_base="http://127.0.0.1:9/v1", timeout=2
        )


async def test_async_completion_and_stream(fake):
    r = await theagentrouter.acompletion(model="anthropic/claude-sonnet-4-5", messages=HELLO, api_base=fake.url)
    assert r.choices[0].message.content == "echo: refunds yesterday"
    args = ""
    stream = await theagentrouter.acompletion(
        model="openai/gpt-4o-mini", messages=HELLO, tools=TOOLS, api_base=fake.url, stream=True
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            args += chunk.choices[0].delta.tool_calls[0].function.arguments
    assert json.loads(args) == {"query": "refunds yesterday"}


def test_model_info():
    info = theagentrouter.get_model_info("anthropic/claude-sonnet-4-5")
    assert info["max_input_tokens"] > 0 and info["supports_function_calling"]
    assert litellm.supports_response_schema("anthropic/claude-sonnet-4-5")
    with pytest.raises(exceptions.NotFoundError):
        theagentrouter.get_model_info("openai/no-such-model")


def test_compat_module(fake):
    r = litellm.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, drop_params=True)
    assert r.choices[0].message.content.startswith("echo")
    assert litellm.get_llm_provider("anthropic/claude-sonnet-4-5")[:2] == ("claude-sonnet-4-5", "anthropic")


def test_settings_file_and_env_precedence(fake, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"base_url": fake.url, "api_key": "file-key"}))
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    r = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO)
    assert r._hidden_params["route"] == "gateway"
    assert fake.requests[-1][1]["authorization"] == "Bearer file-key"
    monkeypatch.setenv("AGENTROUTER_API_KEY", "env-key")  # the environment wins over the file
    theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO)
    assert fake.requests[-1][1]["authorization"] == "Bearer env-key"


def test_litellm_only_arguments_warn_once_and_are_not_sent(fake, monkeypatch):
    from theagentrouter import transport

    monkeypatch.setattr(transport, "_warned", set())
    with pytest.warns(UserWarning, match=r"ignores fallbacks=; define a route"):
        theagentrouter.completion(
            model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, fallbacks=["openai/gpt-4o"], temperature=0
        )
    body = fake.requests[-1][2]
    assert "fallbacks" not in body and body["temperature"] == 0  # supported arguments still go through
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second call with the same argument does not warn again
        theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, fallbacks=[])


def test_a_set_gateway_takes_its_own_model_names(fake, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    r = theagentrouter.completion(model="claude-sonnet-5", messages=HELLO)
    assert r._hidden_params["route"] == "gateway"
    assert fake.requests[-1][2]["model"] == "claude-sonnet-5"  # sent as the gateway names it
    monkeypatch.delenv("AGENTROUTER_BASE_URL")
    with pytest.raises(exceptions.BadRequestError, match="no provider prefix"):
        theagentrouter.completion(
            model="acme-chat-1", messages=HELLO
        )  # without a gateway: from the model list, or refused


def test_model_list_keys_name_the_provider_once():
    # DeepInfra's catalog already prefixes its model names; scripts/sync_models.py must not add it twice.
    doubled = [k for k in theagentrouter.model_cost if k.split("/")[0] == k.split("/")[1]]
    assert doubled == []
