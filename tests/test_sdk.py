from __future__ import annotations

import copy
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


def test_gateway_gets_the_requests_the_local_gateway_gets(fake, monkeypatch):
    # Same paths and formats as the local gateway, so one Agent Router config serves both.
    monkeypatch.setenv("AGENTROUTER_API_KEY", "gw-key")
    root = fake.url[: -len("/v1")]
    for base in (fake.url, root, root + "/"):  # with or without the /v1 that OpenAI clients add
        monkeypatch.setenv("AGENTROUTER_BASE_URL", base)
        for model, path, fmt in (
            ("anthropic/claude-sonnet-4-5", "/anthropic/v1/messages", "anthropic"),
            ("claude-sonnet-4-5", "/anthropic/v1/messages", "anthropic"),
            ("openai/gpt-4o-mini", "/v1/chat/completions", "openai"),
            ("gpt-4o-mini", "/v1/chat/completions", "openai"),
            ("acme-model-9", "/v1/chat/completions", "openai"),  # the gateway's own name
        ):
            r = theagentrouter.completion(model=model, messages=HELLO)
            sent, headers, body = fake.requests[-1]
            assert (sent, body["model"]) == (path, model)  # the name as written; the gateway routes on it
            if fmt == "anthropic":  # never as Authorization, which Agent Router passes on to Anthropic
                assert headers["x-api-key"] == "gw-key" and "authorization" not in headers
            else:
                assert headers["authorization"] == "Bearer gw-key"
            assert (r._hidden_params["route"], r._hidden_params["format"]) == ("gateway", fmt)
            assert r.choices[0].message.content == "echo: refunds yesterday"


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


def test_catalog_records_give_their_limits():
    from theagentrouter.models import from_catalog

    record = {"provider": "openai", "model": "acme-1", "contextWindow": 1000, "limits": {"max_output_tokens": 100}}
    assert from_catalog({"models": [record]})["openai/acme-1"]["max_input_tokens"] == 900
    record["limits"]["max_input_tokens"] = 850  # an explicit input limit wins
    entry = from_catalog({"models": [record]})["openai/acme-1"]
    assert entry["max_input_tokens"] == 850 and entry["max_output_tokens"] == entry["max_tokens"] == 100
    del record["limits"]
    assert "max_tokens" not in from_catalog({"models": [record]})["openai/acme-1"]


def test_model_info():
    info = theagentrouter.get_model_info("anthropic/claude-sonnet-4-5")
    assert info["max_input_tokens"] > 0 and info["supports_function_calling"]
    # The output limit is the catalog's limits.max_output_tokens; LiteLLM's max_tokens is that, never the window.
    assert info["max_output_tokens"] == info["max_tokens"] == 64000
    mini = theagentrouter.get_model_info("openai/gpt-4o-mini")
    assert (
        mini["max_tokens"] == 16384 and mini["max_input_tokens"] == 128000 - 16384
    )  # OpenAI's window holds the output
    assert litellm.supports_response_schema("anthropic/claude-sonnet-4-5")
    with pytest.raises(exceptions.NotFoundError):
        theagentrouter.get_model_info("openai/no-such-model")


def test_register_model_as_in_litellm(monkeypatch, tmp_path):
    import httpx

    from theagentrouter import models

    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    saved = copy.deepcopy(models.model_cost)
    try:
        table = litellm.register_model(
            {
                "acme-tuned": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 1e-6},
                "together_ai/acme/llama": {"litellm_provider": "together_ai", "input_cost_per_token": 2e-7},
                "openai/gpt-4o-mini": {"input_cost_per_token": 9e-6},  # changes an entry, keeps the rest
                "sample_spec": {"mode": "chat"},  # LiteLLM's file documents its fields under this name
                "fallback_generalizations": {"x": ["y"]},  # not a model
            }
        )
        assert table is theagentrouter.model_cost
        assert theagentrouter.get_model_info("openai/acme-tuned")["input_cost_per_token"] == 1e-6
        assert theagentrouter.get_model_info("together/acme/llama")["litellm_provider"] == "together_ai"
        mini = theagentrouter.model_cost["openai/gpt-4o-mini"]
        assert mini["input_cost_per_token"] == 9e-6 and mini["max_input_tokens"] > 0
        assert not [k for k in table if "sample_spec" in k or "fallback_generalizations" in k]
        # Registered models get a rule in the local gateway and in an exported config.
        assert "openai/acme-tuned" in local_gateway.routed_models(["openai"])
        assert 'value: "openai/acme-tuned"' in local_gateway.export_config(providers=["openai"])
        with pytest.raises(exceptions.BadRequestError, match="has no provider"):
            theagentrouter.register_model({"acme-9": {"mode": "chat", "input_cost_per_token": 1e-6}})
        # A file of theagentrouter/agent-router-models, converted as the bundled list is.
        theagentrouter.register_model(
            {
                "version": "1.0",
                "models": [
                    {
                        "model": "gpt-acme",
                        "provider": "openai",
                        "mode": "chat",
                        "contextWindow": 400000,
                        "limits": {"max_output_tokens": 128000},
                        "inputTokensPricePerMillion": "1.25",
                        "capabilities": ["tool_choice"],
                    },
                    {"model": "claude-acme-old", "provider": "anthropic", "isEnabled": False},
                ],
            }
        )
        acme = theagentrouter.get_model_info("openai/gpt-acme")
        assert (acme["max_input_tokens"], acme["max_output_tokens"]) == (
            272000,
            128000,
        )  # OpenAI's window holds the output
        assert acme["input_cost_per_token"] == 1.25e-06 and acme["supports_function_calling"]
        assert "anthropic/claude-acme-old" not in theagentrouter.model_cost
        # A URL, such as LiteLLM's model_prices_and_context_window.json.
        url = "https://models.acme.example/models.json"
        served = {"claude-acme": {"litellm_provider": "anthropic", "mode": "chat"}}
        monkeypatch.setattr(
            httpx, "get", lambda u, **_: httpx.Response(200, json=served, request=httpx.Request("GET", u))
        )
        theagentrouter.register_model(url)
        assert "anthropic/claude-acme" in theagentrouter.model_cost
    finally:
        models.model_cost.clear()
        models.model_cost.update(saved)


def test_compat_module(fake):
    r = litellm.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, drop_params=True)
    assert r.choices[0].message.content.startswith("echo")
    assert litellm.get_llm_provider("anthropic/claude-sonnet-4-5")[:2] == ("claude-sonnet-4-5", "anthropic")


def test_settings_file_and_env_precedence(fake, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"base_url": fake.url, "api_key": "file-key"}))
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    r = theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO)
    assert r._hidden_params["route"] == "gateway"
    assert fake.requests[-1][1]["authorization"] == "Bearer file-key"
    monkeypatch.setenv("AGENTROUTER_API_KEY", "env-key")  # the environment wins over the file
    theagentrouter.completion(model="openai/gpt-4o-mini", messages=HELLO)
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


@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"])
def test_model_dump_exclude_none_keeps_tool_calls_for_the_next_turn(fake, model):
    response = theagentrouter.completion(model=model, messages=HELLO, tools=TOOLS, api_base=fake.url)
    assert response.model_dump() == response.to_dict()
    data = response.model_dump(mode="json", exclude_none=True)
    assert "_hidden_params" not in data
    message = data["choices"][0]["message"]
    assert "content" not in message and "reasoning_content" not in message
    if response.choices[0].logprobs is None:
        assert "logprobs" not in data["choices"][0]
    else:
        assert data["choices"][0]["logprobs"] == response.choices[0].logprobs
    call = message["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"query": "refunds yesterday"}
    reply = theagentrouter.completion(
        model=model,
        api_base=fake.url,
        messages=[*HELLO, message, {"role": "tool", "tool_call_id": call["id"], "content": "3 refunds"}],
    )
    assert reply.choices[0].message.content == "echo: 3 refunds"
    assert response.choices[0].message.content is None


def test_model_dump_preserves_payload_nulls_and_does_not_mutate_response():
    from theagentrouter.types import Message, Usage

    usage = Usage(prompt_tokens_details={"cached_tokens": 0, "custom": None, "values": (None, 1)})
    data = usage.model_dump(mode="json", exclude_none=True)
    assert "completion_tokens_details" not in data
    assert data["prompt_tokens_details"] == {"cached_tokens": 0, "custom": None, "values": [None, 1]}
    assert usage.model_dump()["prompt_tokens_details"]["values"] == (None, 1)
    data["prompt_tokens_details"]["cached_tokens"] = 99
    assert usage.prompt_tokens_details["cached_tokens"] == 0
    message = Message(thinking_blocks=[{"type": "thinking", "thinking": "", "signature": None}])
    assert message.model_dump(exclude_none=True)["thinking_blocks"][0]["signature"] is None
    with pytest.raises(ValueError, match="mode"):
        message.model_dump(mode="invalid")


@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"])
def test_stream_chunk_model_dump_exclude_none(fake, model):
    chunks = list(theagentrouter.completion(model=model, messages=HELLO, api_base=fake.url, stream=True))
    dumped = [chunk.model_dump(mode="json", exclude_none=True) for chunk in chunks]
    assert any("usage" not in chunk for chunk in dumped)
    assert all("_hidden_params" not in chunk for chunk in dumped)
    text = "".join(choice["delta"].get("content", "") for chunk in dumped for choice in chunk["choices"])
    original_text = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
    assert text == original_text
    assert text.strip() == "echo: refunds yesterday"
    assert json.loads(json.dumps(dumped)) == dumped


def test_the_model_list_marks_retired_and_deprecated_models():
    from theagentrouter.models import lifecycle_from_catalog

    data = {
        "models": [
            {
                "provider": "openai",
                "model": "acme-old",
                "isEnabled": False,
                "metadata": {"deprecated": True, "retirement_date": "2026-01-01", "deprecated_reason": "Use acme-new."},
            },
            {
                "provider": "deepinfra",
                "model": "deepinfra/acme/soon",
                "metadata": {"deprecated": True, "retirement_date": "2027-01-01"},
            },
            {"provider": "openai", "model": "acme-new", "metadata": {}},
        ]
    }
    assert lifecycle_from_catalog(data) == {
        "retired": {"openai/acme-old": "Use acme-new."},
        "deprecated": {"deepinfra/acme/soon": {"date": "2027-01-01", "reason": "retires on 2027-01-01"}},
    }


def test_a_retired_model_says_so_instead_of_unknown(monkeypatch):
    monkeypatch.setitem(
        theagentrouter.models.lifecycle["retired"], "openai/acme-old", "Retired on 2026-01-01. Replacement: acme-new."
    )
    with pytest.raises(
        exceptions.NotFoundError,
        match=r"openai/acme-old is no longer offered: Retired on 2026-01-01\. Replacement: acme-new\.",
    ):
        theagentrouter.completion(model="openai/acme-old", messages=HELLO)
    with pytest.raises(exceptions.NotFoundError, match="no longer offered"):
        theagentrouter.get_model_info("openai/acme-old")


def test_a_deprecated_model_warns_once_and_has_litellms_deprecation_date(monkeypatch):
    from theagentrouter import models

    info = theagentrouter.get_model_info("anthropic/claude-sonnet-4-5")
    assert info["deprecation_date"] == models.lifecycle["deprecated"]["anthropic/claude-sonnet-4-5"]["date"]
    monkeypatch.setattr(models, "_deprecation_warned", set())
    with pytest.warns(UserWarning, match="anthropic/claude-sonnet-4-5 is deprecated"):
        models.warn_if_deprecated("anthropic/claude-sonnet-4-5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        models.warn_if_deprecated("anthropic/claude-sonnet-4-5")  # once per program
