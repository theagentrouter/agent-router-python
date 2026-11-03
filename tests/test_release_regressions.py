# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy

import pytest

import theagentrouter
from theagentrouter import exceptions, local_gateway
from theagentrouter.compat import litellm

from .test_sdk import HELLO, TOOLS
from .test_sdk import fake as sdk_fake

fake = sdk_fake


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_explicit_key_without_api_base_goes_through_the_gateway_and_warns(fake, monkeypatch, provider):
    # The gateway holds the provider keys; an api_key on the call cannot reach the provider, so it warns.
    from theagentrouter import routing

    model = f"{provider}/acme-model"
    port = int(fake.url.rsplit(":", 1)[1].split("/")[0])
    state = {"pid": 1, "port": port, "providers": [provider], "models": [model], "versions": local_gateway.versions()}
    monkeypatch.setattr(local_gateway, "running", lambda: state)
    monkeypatch.setattr(routing, "_key_warned", False)
    with pytest.warns(UserWarning, match="ignores api_key= without api_base="):
        response = theagentrouter.completion(model=model, messages=HELLO, api_key="acme-explicit-key")
    assert response._hidden_params["route"] == "local gateway"
    headers = fake.requests[-1][1]
    assert "acme-explicit-key" not in (headers.get("x-api-key", "") + headers.get("authorization", ""))


@pytest.mark.parametrize("choice,expected", [(None, "auto"), ("auto", "auto"), ("required", "any")])
def test_anthropic_disables_parallel_tools(fake, choice, expected):
    theagentrouter.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=HELLO,
        tools=TOOLS,
        api_base=fake.url,
        tool_choice=choice,
        parallel_tool_calls=False,
    )
    assert fake.requests[-1][2]["tool_choice"] == {"type": expected, "disable_parallel_tool_use": True}


@pytest.mark.parametrize(
    "param,value",
    [
        ("seed", 0),
        ("n", 2),
    ],
)
def test_anthropic_rejects_untranslated_options(fake, param, value):
    with pytest.raises(exceptions.UnsupportedParamsError, match=param):
        theagentrouter.completion(
            model="anthropic/claude-sonnet-4-5",
            messages=HELLO,
            api_base=fake.url,
            **{param: value},
        )
    assert not fake.requests
    assert param not in litellm.get_supported_openai_params("anthropic/claude-sonnet-4-5")


def test_configured_gateway_takes_claude_options_as_the_local_gateway_does(fake, monkeypatch):
    # Both routes use Anthropic Messages, including LiteLLM's translated options.
    monkeypatch.setenv("AGENTROUTER_BASE_URL", fake.url)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    theagentrouter.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=HELLO,
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        reasoning_effort="high",
    )
    body = fake.requests[-1][2]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert body["max_tokens"] > body["thinking"]["budget_tokens"]
    # Anthropic's current field, with the schema closed as LiteLLM and Anthropic's own SDK close it.
    assert body["output_config"] == {
        "format": {"type": "json_schema", "schema": {**schema, "additionalProperties": False}}
    }
    for name in ("anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"):
        assert {"response_format", "reasoning_effort"} <= set(litellm.get_supported_openai_params(name))


def test_anthropic_json_object_without_a_schema_warns_and_invents_none(fake, monkeypatch):
    from theagentrouter import anthropic_format

    monkeypatch.setattr(anthropic_format, "_warned", set())
    with pytest.warns(UserWarning, match="json_object is not applied"):
        theagentrouter.completion(
            model="anthropic/claude-sonnet-4-5",
            messages=HELLO,
            api_base=fake.url,
            response_format={"type": "json_object"},
        )
    assert "output_config" not in fake.requests[-1][2] and "output_format" not in fake.requests[-1][2]


def test_an_image_in_a_tool_result_reaches_claude_as_an_image():
    from theagentrouter import anthropic_format

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    messages = [
        {"role": "user", "content": "Look at the chart."},
        {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "chart", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": [{"type": "text", "text": "Refunds by day"}, image]},
        {"role": "assistant", "tool_calls": [{"id": "t2", "function": {"name": "chart", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t2", "content": [{**image, "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "tool_calls": [{"id": "t3", "function": {"name": "note", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t3", "content": [{"type": "text", "text": "done"}]},
    ]
    _, out = anthropic_format.convert_messages(messages)
    results = [b for m in out for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    png = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    assert results[0]["content"] == [{"type": "text", "text": "Refunds by day"}, png]
    assert results[1]["content"] == [png] and results[1]["cache_control"] == {"type": "ephemeral"}
    assert results[2]["content"] == "done"


@pytest.mark.parametrize(
    "body,cls",
    [
        ({"error": {"message": "Token budget per minute exceeded", "type": "rate_limit"}}, exceptions.RateLimitError),
        ({"error": {"message": "Request blocked", "code": "budget_exceeded"}}, exceptions.BudgetExceededError),
        ({"error": {"message": "Budget exceeded for project payments"}}, exceptions.BudgetExceededError),
    ],
)
def test_only_a_budget_block_is_a_budget_error(body, cls):
    import json

    error = exceptions.from_http(429, json.dumps(body), "openai/gpt-4o", "openai")
    assert type(error) is cls


def test_openai_response_details_and_serialization(fake):
    response = theagentrouter.completion(model="openai/response-details", messages=HELLO, api_base=fake.url)
    assert response.usage.prompt_tokens_details.cached_tokens == 5
    assert response.usage.completion_tokens_details.reasoning_tokens == 3
    assert response.choices[0].message.refusal == "refused"
    assert response.choices[0].message.reasoning_content == "thinking"
    assert response.choices[0].logprobs == {"content": []}
    assert response.to_dict()["usage"]["prompt_tokens_details"] == {"cached_tokens": 5}


def test_openai_stream_details(fake):
    chunks = list(
        theagentrouter.completion(
            model="openai/response-details",
            messages=HELLO,
            api_base=fake.url,
            stream=True,
        )
    )
    assert chunks[0].choices[0].delta.refusal == "refused"
    assert chunks[0].choices[0].delta.reasoning_content == "thinking"
    assert chunks[0].choices[0].logprobs == {"content": []}
    assert chunks[-1].usage.prompt_tokens_details.cached_tokens == 5
    assert chunks[-1].usage.completion_tokens_details.reasoning_tokens == 3


def test_openai_stream_error_is_raised_after_partial_output(fake):
    stream = theagentrouter.completion(model="openai/stream-error", messages=HELLO, api_base=fake.url, stream=True)
    assert next(stream).choices[0].delta.content == "partial"
    with pytest.raises(exceptions.RateLimitError) as error:
        list(stream)
    assert error.value.model == "openai/stream-error"
    assert error.value.llm_provider == "openai"


async def test_async_openai_stream_error_is_raised(fake):
    stream = await theagentrouter.acompletion(
        model="openai/stream-error", messages=HELLO, api_base=fake.url, stream=True
    )
    with pytest.raises(exceptions.RateLimitError) as error:
        async for _ in stream:
            pass
    assert error.value.model == "openai/stream-error"
    assert error.value.llm_provider == "openai"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_request_does_not_modify_caller_history(fake, provider):
    messages = [
        *HELLO,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "acme-call",
                    "index": 2,
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
    ]
    original = copy.deepcopy(messages)
    theagentrouter.completion(model=f"{provider}/acme-model", messages=messages, api_base=fake.url)
    assert messages == original


# --- one test per problem ----------------------------------------------------------------------------


def test_fallback_wheel_without_a_program_counts_as_not_installed(monkeypatch):
    # pip installs the py3-none-any fallback wheel where no platform wheel fits; it has no aigw.
    import types

    monkeypatch.setattr(local_gateway.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        local_gateway.importlib, "import_module", lambda name: types.SimpleNamespace(has_binary=lambda: False)
    )
    assert local_gateway.installed() is False


def test_routes_defined_at_the_same_time_all_stay(tmp_path, monkeypatch):
    import threading

    from theagentrouter import routes

    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    start = threading.Barrier(2)

    def define(prefix: str) -> None:
        start.wait()
        for n in range(25):
            routes.route(f"{prefix}-{n}", primary="openai/gpt-4o-mini")

    threads = [threading.Thread(target=define, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(routes.list_routes()) == 50
    assert not list(tmp_path.glob(".routes-*"))  # no temporary files left behind


def _former_route_state(model: str) -> dict:
    return {
        "pid": 1,
        "port": 1975,
        "providers": ["openai"],
        "models": [model],
        "routes": [model],
        "routes_fingerprint": "from-when-it-was-a-route",
        "versions": local_gateway.versions(),
    }


def test_a_removed_route_named_like_a_model_restarts_the_gateway(tmp_path, monkeypatch):
    # The running gateway still has the route's fallbacks, retries and limit under that name.
    model = "openai/gpt-4o-mini"
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTROUTER_AUTOSTART", raising=False)
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    old = _former_route_state(model)
    calls = []
    monkeypatch.setattr(local_gateway, "running", lambda: old)
    monkeypatch.setattr(local_gateway, "stop", lambda state=None: calls.append("stop"))
    fresh = {**old, "routes": [], "routes_fingerprint": "none"}
    monkeypatch.setattr(local_gateway, "_start", lambda providers, models, routes=None: calls.append("start") or fresh)
    assert not local_gateway.serves(old, model, {})
    target = theagentrouter.routing.resolve(model)
    assert calls == ["stop", "start"] and target.route == "local gateway"


def test_autostart_off_still_uses_a_running_gateway_that_has_the_route(tmp_path, monkeypatch):
    from theagentrouter import routes, routing

    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    routes.route("chat", primary="openai/gpt-4o-mini")
    state = {
        "pid": 1,
        "port": 1975,
        "providers": ["openai"],
        "models": [],
        "routes": ["chat"],
        "routes_fingerprint": routes.fingerprint(routes.list_routes()),
        "versions": local_gateway.versions(),
    }
    monkeypatch.setattr(local_gateway, "running", lambda: state)
    monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")
    assert routing.resolve("chat").base_url == "http://127.0.0.1:1975/v1"
    # A route the running gateway does not have yet still needs a start, which is off.
    routes.route("other", primary="openai/gpt-4o-mini")
    with pytest.raises(exceptions.BadRequestError, match="AGENTROUTER_AUTOSTART=0"):
        routing.resolve("other")


def test_anthropic_keeps_cache_control():
    from theagentrouter import anthropic_format

    marker = {"type": "ephemeral"}
    body = anthropic_format.build_body(
        "claude-sonnet-4-5",
        [
            {"role": "system", "content": [{"type": "text", "text": "Long rules", "cache_control": marker}]},
            {"role": "user", "content": [{"type": "text", "text": "Big document", "cache_control": marker}]},
            {"role": "assistant", "content": "Read it."},
            {"role": "user", "content": "Summarise", "cache_control": marker},  # LiteLLM's message-level form
        ],
        [{**TOOLS[0], "cache_control": marker}],
        False,
        {},
    )
    assert body["system"] == [{"type": "text", "text": "Long rules", "cache_control": marker}]
    assert body["messages"][0]["content"][0]["cache_control"] == marker
    assert body["messages"][2]["content"][-1] == {"type": "text", "text": "Summarise", "cache_control": marker}
    assert body["tools"][0]["cache_control"] == marker


def test_anthropic_system_without_cache_control_stays_a_string():
    from theagentrouter import anthropic_format

    system, _ = anthropic_format.convert_messages(
        [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, *HELLO]
    )
    assert system == "a\n\nb"


def _anthropic_thinking_stream() -> list[str]:
    import json

    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-5", "usage": {}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Check the "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "refunds."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search_refunds", "input": {}},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]
    return [json.dumps(e) for e in events]


def test_anthropic_stream_keeps_thinking_and_its_signature():
    from theagentrouter import anthropic_format, types

    parser = anthropic_format.StreamParser()
    chunks = [c for payload in _anthropic_thinking_stream() for c in parser.feed(payload)]
    reasoning = "".join(c.choices[0].delta.reasoning_content or "" for c in chunks if c.choices)
    assert reasoning == "Check the refunds."
    message = types.stream_chunk_builder(chunks).choices[0].message
    assert message.reasoning_content == "Check the refunds."
    assert message.thinking_blocks == [{"type": "thinking", "thinking": "Check the refunds.", "signature": "sig-1"}]
    # Sent back for the tool result, the thinking block goes first, signature included, as Anthropic requires.
    _, converted = anthropic_format.convert_messages([*HELLO, message])
    assert converted[1]["content"][0] == message.thinking_blocks[0]
    assert converted[1]["content"][1]["type"] == "tool_use"


def test_anthropic_reply_keeps_thinking_blocks():
    from theagentrouter import anthropic_format

    block = {"type": "thinking", "thinking": "Check.", "signature": "sig-2"}
    resp = anthropic_format.parse_response({"id": "m", "content": [block, {"type": "text", "text": "Done"}]})
    assert resp.choices[0].message.thinking_blocks == [block]
    assert resp.choices[0].message.reasoning_content == "Check."


def test_thinking_blocks_are_not_sent_in_the_openai_format():
    from theagentrouter import openai_format

    message = {"role": "assistant", "content": "Done", "thinking_blocks": [{"type": "thinking", "signature": "s"}]}
    body = openai_format.build_body("gpt-4o-mini", [*HELLO, message], None, False, {})
    assert "thinking_blocks" not in body["messages"][1]


def test_stream_chunk_builder_keeps_choices_apart():
    from theagentrouter import types

    def chunk(index: int, **delta):
        return {"id": "c", "model": "m", "choices": [{"index": index, "delta": delta}]}

    call = {"index": 0, "id": "call_b", "function": {"name": "search_refunds", "arguments": "{}"}}
    resp = types.stream_chunk_builder(
        [chunk(0, content="fir"), chunk(1, content="sec"), chunk(0, content="st"), chunk(1, tool_calls=[call])]
    )
    assert [c.index for c in resp.choices] == [0, 1]
    assert resp.choices[0].message.content == "first" and resp.choices[0].message.tool_calls is None
    assert resp.choices[1].message.content == "sec"
    assert resp.choices[1].message.tool_calls[0].id == "call_b"


async def test_async_call_does_not_block_the_event_loop_while_preparing(monkeypatch):
    import asyncio
    import time

    from theagentrouter import transport

    def slow_prepare(*args, **kwargs):
        time.sleep(0.5)  # what starting the local gateway looks like to the event loop
        raise exceptions.APIConnectionError("stopped after preparing")

    monkeypatch.setattr(transport, "_prepare", slow_prepare)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    with pytest.raises(exceptions.APIConnectionError):
        await theagentrouter.acompletion(model="openai/gpt-4o-mini", messages=HELLO)
    task.cancel()
    assert ticks >= 20


def test_completion_cost_reads_responses_api_usage():
    from theagentrouter import models
    from theagentrouter.responses import ResponsesAPIResponse

    resp = ResponsesAPIResponse({"model": "gpt-4o-mini", "usage": {"input_tokens": 1000, "output_tokens": 500}})
    resp._hidden_params = {"model_key": "openai/gpt-4o-mini"}
    expected = models.completion_cost(model="openai/gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    assert expected > 0
    assert models.completion_cost(resp) == expected


def test_validate_environment_uses_the_gateways_bedrock_check(monkeypatch, tmp_path):
    from theagentrouter import routing

    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    for var in ("AWS_PROFILE", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    model = "bedrock/anthropic.claude-sonnet-4-5"
    assert routing.validate_environment(model) == {
        "keys_in_environment": False,
        "missing_keys": ["AWS_SECRET_ACCESS_KEY"],
    }
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "example")
    assert routing.validate_environment(model)["missing_keys"] == ["AWS_REGION"]
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert routing.validate_environment(model) == {"keys_in_environment": True, "missing_keys": []}


def test_anthropic_cache_reads_show_as_cached_prompt_tokens():
    from theagentrouter import anthropic_format

    usage = {"input_tokens": 10, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0, "output_tokens": 5}
    resp = anthropic_format.parse_response({"id": "m", "content": [], "usage": usage})
    assert resp.usage.prompt_tokens == 2010
    assert resp.usage.prompt_tokens_details.cached_tokens == 2000
