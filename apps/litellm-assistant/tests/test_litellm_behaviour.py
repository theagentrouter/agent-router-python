# SPDX-License-Identifier: Apache-2.0
"""LiteLLM's behaviour, checked on LiteLLM and on Agent Router's SDK.

Each test states what LiteLLM does. On LiteLLM the run proves the expectation; on Agent Router's SDK
it proves the drop-in. Known differences are marked known_gap (strict).
"""

import time

import litellm
import pydantic
import pytest
from conftest import CLAUDE, MODELS, OPENAI, SDK

HELLO = [{"role": "user", "content": "refunds yesterday"}]
NEW_CLAUDE = "anthropic/claude-sonnet-5-5"


# ---- streaming ----------------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_every_stream_chunk_has_a_choice(base, model):
    chunks = list(litellm.completion(model=model, messages=HELLO, stream=True, **base))
    assert chunks and all(len(c.choices) == 1 for c in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.parametrize("model", MODELS)
def test_include_usage_comes_on_a_last_chunk_with_an_empty_choice(base, model):
    chunks = list(
        litellm.completion(model=model, messages=HELLO, stream=True, stream_options={"include_usage": True}, **base)
    )
    last = chunks[-1]
    assert len(last.choices) == 1 and not last.choices[0].delta.content and last.choices[0].finish_reason is None
    assert last.usage.total_tokens > 0
    assert chunks[-2].choices[0].finish_reason == "stop"


@pytest.mark.parametrize("model", MODELS)
async def test_async_stream_chunks_have_a_choice(base, model):
    response = await litellm.acompletion(model=model, messages=HELLO, stream=True, **base)
    chunks = [c async for c in response]
    assert all(len(c.choices) == 1 for c in chunks)


@pytest.mark.parametrize("model", MODELS)
def test_stream_chunk_builder_rebuilds_the_reply(base, model):
    chunks = list(litellm.completion(model=model, messages=HELLO, stream=True, **base))
    rebuilt = litellm.stream_chunk_builder(chunks, messages=HELLO)
    assert rebuilt.choices[0].message.content.strip() == "echo: refunds yesterday"


@pytest.mark.parametrize("model", MODELS)
def test_streamed_tool_call_arguments_are_joined(base, model):
    from assistant.app import TOOLS

    chunks = list(litellm.completion(model=model, messages=HELLO, tools=TOOLS, stream=True, **base))
    call = litellm.stream_chunk_builder(chunks, messages=HELLO).choices[0].message.tool_calls[0]
    assert call.function.name == "order_status" and call.function.arguments == '{"query": "refunds yesterday"}'


@pytest.mark.parametrize("model", MODELS)
def test_stream_error_raises_at_the_call(base, model):
    with pytest.raises(litellm.AuthenticationError):
        litellm.completion(model=f"{model.split('/')[0]}/err-401", messages=HELLO, stream=True, **base)


# ---- Claude parameters --------------------------------------------------------------------------


def test_claude_takes_n_1(base, fake):
    litellm.completion(model=CLAUDE, messages=HELLO, n=1, **base)
    assert "n" not in fake.requests[-1][2]


@pytest.mark.parametrize("param", [{"seed": 1}, {"presence_penalty": 0.0}, {"n": 2}, {"logprobs": True}])
def test_claude_refuses_what_it_does_not_take(base, param):
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(model=CLAUDE, messages=HELLO, **param, **base)


@pytest.mark.parametrize("param", [{"seed": 1}, {"presence_penalty": 0.0}, {"n": 2}])
def test_drop_params_on_the_call(base, fake, param):
    litellm.completion(model=CLAUDE, messages=HELLO, drop_params=True, **param, **base)
    assert not set(param) & set(fake.requests[-1][2])


def test_drop_params_on_the_module(base, fake, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    litellm.completion(model=CLAUDE, messages=HELLO, seed=1, **base)
    assert "seed" not in fake.requests[-1][2]


def test_drop_params_is_off_by_default():
    assert litellm.drop_params is False


@pytest.mark.parametrize("param", [{"temperature": 0.2}, {"top_p": 0.9}, {"top_k": 5}])
def test_newer_claude_refuses_sampling(base, param):
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(model=NEW_CLAUDE, messages=HELLO, **param, **base)


@pytest.mark.parametrize("param", [{"temperature": 0.2}, {"top_p": 0.9}, {"top_k": 5}])
def test_newer_claude_drops_sampling_with_drop_params(base, fake, param):
    litellm.completion(model=NEW_CLAUDE, messages=HELLO, drop_params=True, **param, **base)
    assert not set(param) & set(fake.requests[-1][2])


def test_newer_claude_takes_temperature_1(base, fake):
    litellm.completion(model=NEW_CLAUDE, messages=HELLO, temperature=1, **base)
    assert fake.requests[-1][2]["temperature"] == 1


def test_claude_thinking_is_sent(base, fake):
    thinking = {"type": "enabled", "budget_tokens": 1024}
    litellm.completion(model=CLAUDE, messages=HELLO, thinking=thinking, max_tokens=2048, **base)
    assert fake.requests[-1][2]["thinking"] == thinking


def test_claude_reasoning_effort_becomes_thinking(base, fake):
    litellm.completion(model=CLAUDE, messages=HELLO, reasoning_effort="low", **base)
    assert fake.requests[-1][2]["thinking"]["type"] == "enabled"


SHOWN = {"type": "adaptive", "display": "summarized"}
ADAPTIVE = {"type": "adaptive"}
DISABLED = {"type": "disabled"}


def _budget(tokens):
    return {"type": "enabled", "budget_tokens": tokens}


@pytest.mark.parametrize(
    ("model", "params", "thinking", "effort", "max_tokens"),
    [
        # A token budget: as written up to 4.6, and from 4.7 on an effort level, which those models take instead.
        ("anthropic/claude-sonnet-4-5", {"thinking": _budget(2000)}, _budget(2000), None, 6096),
        ("anthropic/claude-sonnet-4-6", {"thinking": _budget(2000)}, _budget(2000), None, 6096),
        ("anthropic/claude-opus-4-6", {"thinking": _budget(9000)}, _budget(9000), None, 13096),
        ("anthropic/claude-opus-4-7", {"thinking": _budget(2000)}, ADAPTIVE, "low", 128000),
        ("anthropic/claude-opus-4-7", {"thinking": _budget(2048)}, ADAPTIVE, "medium", 128000),
        ("anthropic/claude-opus-4-7", {"thinking": _budget(4096)}, ADAPTIVE, "high", 128000),
        ("anthropic/claude-sonnet-5-5", {"thinking": _budget(8192)}, ADAPTIVE, "xhigh", 128000),
        # Adaptive thinking: as written from 4.6 on, and before that the "medium" budget, below max_tokens.
        ("anthropic/claude-sonnet-4-6", {"thinking": ADAPTIVE}, ADAPTIVE, None, 128000),
        ("anthropic/claude-sonnet-4-5", {"thinking": ADAPTIVE}, _budget(2048), None, 6144),
        ("anthropic/claude-opus-4-5", {"thinking": ADAPTIVE}, _budget(2048), None, 6144),
        ("anthropic/claude-haiku-4-5", {"thinking": ADAPTIVE, "max_tokens": 1500}, _budget(1499), None, 1500),
        ("anthropic/claude-opus-4-5", {"thinking": ADAPTIVE, "max_tokens": 500}, None, None, 500),
        # Off: as written, except on the models that always think.
        ("anthropic/claude-sonnet-4-5", {"thinking": DISABLED}, DISABLED, None, 64000),
        ("anthropic/claude-sonnet-5", {"thinking": DISABLED}, DISABLED, None, 128000),
        ("anthropic/claude-sonnet-5-5", {"thinking": DISABLED}, None, None, 128000),
        ("anthropic/claude-fable-5", {"thinking": DISABLED}, None, None, 128000),
        # reasoning_effort: a budget before 4.6, and adaptive thinking with the level from then on.
        ("anthropic/claude-sonnet-4-5", {"reasoning_effort": "minimal"}, _budget(1024), None, 5120),
        ("anthropic/claude-sonnet-4-5", {"reasoning_effort": "max"}, _budget(16384), None, 20480),
        ("anthropic/claude-opus-4-5", {"reasoning_effort": "low"}, _budget(1024), None, 5120),
        ("anthropic/claude-sonnet-4-6", {"reasoning_effort": "low"}, SHOWN, "low", 128000),
        ("anthropic/claude-sonnet-5-5", {"reasoning_effort": "minimal"}, SHOWN, "low", 128000),
        ("anthropic/claude-opus-5", {"reasoning_effort": "max"}, SHOWN, "max", 128000),
        ("anthropic/claude-sonnet-4-5", {"reasoning_effort": "none"}, None, None, 64000),
        ("anthropic/claude-sonnet-4-5", {"reasoning_effort": {"effort": "low"}}, _budget(1024), None, 5120),
        # thinking wins over reasoning_effort, which then only sets a level on a model that has one.
        (
            "anthropic/claude-sonnet-4-5",
            {"thinking": _budget(2000), "reasoning_effort": "low"},
            _budget(2000),
            None,
            6096,
        ),
        (
            "anthropic/claude-sonnet-4-6",
            {"thinking": _budget(2000), "reasoning_effort": "low"},
            _budget(2000),
            "low",
            6096,
        ),
        ("anthropic/claude-opus-5", {"thinking": ADAPTIVE, "reasoning_effort": "low"}, ADAPTIVE, "low", 128000),
        ("anthropic/claude-opus-5", {"thinking": _budget(9000), "reasoning_effort": "low"}, ADAPTIVE, "low", 128000),
        # A date after the name is not on the model list, and the name's rules still apply.
        ("anthropic/claude-opus-5-20270101", {"reasoning_effort": "low"}, SHOWN, "low", 4096),
    ],
)
def test_claude_thinking_by_model(base, fake, model, params, thinking, effort, max_tokens):
    litellm.completion(model=model, messages=HELLO, **params, **base)
    body = fake.requests[-1][2]
    assert body.get("thinking") == thinking
    assert (body.get("output_config") or {}).get("effort") == effort
    assert body["max_tokens"] == max_tokens


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-6"])
def test_xhigh_is_refused_where_a_model_has_no_such_level(base, model):
    with pytest.raises(litellm.BadRequestError):
        litellm.completion(model=model, messages=HELLO, reasoning_effort="xhigh", **base)


def test_max_completion_tokens_wins_over_max_tokens(base, fake):
    litellm.completion(model=CLAUDE, messages=HELLO, max_tokens=300, max_completion_tokens=400, **base)
    assert fake.requests[-1][2]["max_tokens"] == 400


def test_claude_gets_the_models_own_max_tokens_by_default(base, fake):
    litellm.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO, **base)
    assert fake.requests[-1][2]["max_tokens"] == 64000
    litellm.completion(model="anthropic/claude-opus-5", messages=HELLO, **base)
    assert fake.requests[-1][2]["max_tokens"] == 128000
    litellm.completion(model="anthropic/claude-sonnet-6", messages=HELLO, **base)
    assert fake.requests[-1][2]["max_tokens"] == 4096


def test_a_claude_name_on_no_list_is_left_for_the_api_to_answer(base, fake):
    litellm.completion(model="anthropic/claude-sonnet-6", messages=HELLO, temperature=0.2, **base)
    assert fake.requests[-1][2]["temperature"] == 0.2


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        ("anthropic/claude-sonnet-4-5", {"thinking": ADAPTIVE}, {"thinking": _budget(2048)}),
        ("anthropic/claude-sonnet-4-5", {"thinking": ADAPTIVE, "max_tokens": 500}, {}),
        ("anthropic/claude-opus-4-5", {"thinking": ADAPTIVE}, {}),
        ("anthropic/claude-sonnet-4-6", {"thinking": _budget(2000)}, {"thinking": _budget(2000)}),
        (
            "anthropic/claude-opus-5",
            {"thinking": _budget(2000)},
            {"thinking": ADAPTIVE, "output_config": {"effort": "low"}},
        ),
        ("anthropic/claude-sonnet-5-5", {"thinking": DISABLED}, {}),
        ("anthropic/claude-sonnet-4-5", {"reasoning_effort": "low"}, {"thinking": _budget(1024)}),
        (
            "anthropic/claude-opus-5",
            {"reasoning_effort": "low"},
            {"thinking": SHOWN, "output_config": {"effort": "low"}},
        ),
        ("anthropic/claude-sonnet-5", {"output_config": {"effort": "low"}}, {"output_config": {"effort": "low"}}),
        ("anthropic/claude-sonnet-4-5", {"output_config": {"effort": "low"}}, {"thinking": _budget(1024)}),
        ("anthropic/claude-opus-4-5", {"output_config": {"effort": "low"}}, {"output_config": {"effort": "low"}}),
    ],
)
async def test_anthropic_messages_thinking_by_model(base, fake, model, params, expected):
    params = {"max_tokens": 3000, **params}
    await litellm.anthropic.messages.acreate(model=model, messages=HELLO, **params, **base)
    body = fake.requests[-1][2]
    assert {k: body[k] for k in ("thinking", "output_config") if k in body} == expected


def test_thinking_true_is_the_medium_setting(base, fake):
    litellm.completion(model="anthropic/claude-sonnet-4-5", messages=HELLO, thinking=True, **base)
    assert fake.requests[-1][2]["thinking"] == _budget(2048)
    litellm.completion(model="anthropic/claude-opus-5", messages=HELLO, thinking=True, **base)
    body = fake.requests[-1][2]
    assert body["thinking"] == ADAPTIVE and body["output_config"] == {"effort": "medium"}


def test_a_budget_as_text_still_sets_the_level(base, fake):
    litellm.completion(
        model="anthropic/claude-opus-4-7", messages=HELLO, thinking={"type": "enabled", "budget_tokens": "5000"}, **base
    )
    assert fake.requests[-1][2]["output_config"] == {"effort": "high"}


def test_max_tokens_0_stays_0(base, fake):
    litellm.completion(model=CLAUDE, messages=HELLO, max_tokens=0, **base)
    assert fake.requests[-1][2]["max_tokens"] == 0


@pytest.mark.parametrize("user", ["a@acme.example", "+1 555 010 9999", 5])
def test_claude_gets_no_email_phone_or_number_as_a_user_id(base, fake, user):
    litellm.completion(model=CLAUDE, messages=HELLO, user=user, **base)
    assert "metadata" not in fake.requests[-1][2]


def test_claude_gets_a_user_id(base, fake):
    litellm.completion(model=CLAUDE, messages=HELLO, user="u-1", **base)
    assert fake.requests[-1][2]["metadata"] == {"user_id": "u-1"}


def test_additional_drop_params(base, fake):
    litellm.completion(
        model=CLAUDE, messages=HELLO, seed=1, temperature=0.5, additional_drop_params=["seed", "temperature"], **base
    )
    body = fake.requests[-1][2]
    assert "seed" not in body and "temperature" not in body


def test_drop_params_on_the_module_holds_whatever_the_call_says(base, fake, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    litellm.completion(model=CLAUDE, messages=HELLO, seed=1, drop_params=False, **base)
    assert "seed" not in fake.requests[-1][2]


def test_anthropic_shaped_and_hosted_tools_go_as_written(base, fake):
    tools = [
        {"name": "g", "description": "d", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
        {"type": "web_search_20250305", "name": "web_search"},
    ]
    litellm.completion(model=NEW_CLAUDE, messages=HELLO, tools=tools, **base)
    sent = fake.requests[-1][2]["tools"]
    assert sent[0]["input_schema"] == tools[0]["input_schema"] and sent[1] == tools[1]


def test_tool_choice_none_as_a_dict(base, fake):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    litellm.completion(model=NEW_CLAUDE, messages=HELLO, tools=tools, tool_choice={"type": "none"}, **base)
    assert fake.requests[-1][2]["tool_choice"] == {"type": "none"}


class _Item(pydantic.BaseModel):
    name: str
    price: float = pydantic.Field(gt=0, le=100, description="price")


class _Order(pydantic.BaseModel):
    id: str = pydantic.Field(min_length=3)
    note: str | None = None


def test_a_pydantic_class_as_response_format_on_openai(base, fake):
    litellm.completion(model=OPENAI, messages=HELLO, response_format=_Order, **base)
    sent = fake.requests[-1][2]["response_format"]
    assert sent["type"] == "json_schema" and sent["json_schema"]["name"] == "_Order"
    schema = sent["json_schema"]["schema"]
    assert sent["json_schema"]["strict"] is True and schema["additionalProperties"] is False
    assert set(schema["required"]) == {"id", "note"} and "default" not in schema["properties"]["note"]


@pytest.mark.parametrize(
    "response_format",
    [
        _Item,
        {
            "type": "json_schema",
            "json_schema": {
                "name": "r",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number", "exclusiveMinimum": 0, "maximum": 100, "description": "price"},
                    },
                },
            },
        },
    ],
)
def test_a_schema_for_claude_has_only_what_anthropic_takes(base, fake, response_format):
    litellm.completion(model=NEW_CLAUDE, messages=HELLO, response_format=response_format, **base)
    body = fake.requests[-1][2]
    schema = (body.get("output_format") or body["output_config"]["format"])["schema"]
    assert schema["additionalProperties"] is False
    price = schema["properties"]["price"]
    assert "maximum" not in price and "exclusiveMinimum" not in price
    assert price["description"].startswith("price Note:") and "100" in price["description"]


async def test_anthropic_messages_sends_top_level_cache_control(base, fake):
    arguments = dict(model=NEW_CLAUDE, messages=HELLO, max_tokens=300, **base)
    await litellm.anthropic.messages.acreate(**arguments, cache_control={"type": "ephemeral"})
    assert fake.requests[-1][2]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_messages_keeps_only_user_id_in_metadata(base, fake):
    arguments = dict(model=NEW_CLAUDE, messages=HELLO, max_tokens=300, **base)
    await litellm.anthropic.messages.acreate(**arguments, metadata={"user_id": "u", "x": "y"})
    assert fake.requests[-1][2]["metadata"] == {"user_id": "u"}


@pytest.mark.parametrize(
    ("params", "beta"),
    [
        ({"speed": "fast"}, "fast-mode-2026-02-01"),
        ({"compaction": {"type": "auto"}}, "compact-2026-09-04"),
        ({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}, "context-management-2025-06-27"),
        (
            {"output_format": {"type": "json_schema", "schema": {"type": "object", "properties": {}}}},
            "structured-outputs-2025-11-13",
        ),
    ],
)
async def test_anthropic_messages_adds_the_beta_header_an_option_needs(base, fake, params, beta):
    arguments = dict(model="anthropic/claude-opus-4-6", messages=HELLO, max_tokens=300, **base)
    await litellm.anthropic.messages.acreate(**arguments, **params)
    assert beta in fake.requests[-1][1].get("anthropic-beta", "")


async def test_anthropic_messages_speed_on_a_model_without_it(base, fake):
    arguments = dict(model="anthropic/claude-sonnet-5", messages=HELLO, max_tokens=300, **base)
    with pytest.raises(litellm.UnsupportedParamsError):
        await litellm.anthropic.messages.acreate(**arguments, speed="fast")
    await litellm.anthropic.messages.acreate(**arguments, speed="fast", drop_params=True)
    assert "speed" not in fake.requests[-1][2]


@pytest.mark.parametrize("param", [{"temperature": 0.2}, {"top_p": 0.9}, {"top_k": 5}])
async def test_anthropic_messages_on_newer_claude_refuses_sampling(base, fake, param):
    arguments = dict(model=NEW_CLAUDE, messages=HELLO, max_tokens=300, **base)
    with pytest.raises(litellm.UnsupportedParamsError):
        await litellm.anthropic.messages.acreate(**arguments, **param)
    await litellm.anthropic.messages.acreate(**arguments, drop_params=True, **param)
    assert not set(param) & set(fake.requests[-1][2])


async def test_anthropic_messages_drops_a_temperature_thinking_refuses(base, fake):
    arguments = dict(model="anthropic/claude-sonnet-4-5", messages=HELLO, max_tokens=3000, **base)
    await litellm.anthropic.messages.acreate(**arguments, temperature=0.2, thinking=_budget(2000))
    assert "temperature" not in fake.requests[-1][2]
    await litellm.anthropic.messages.acreate(**arguments, temperature=0.2)
    assert fake.requests[-1][2]["temperature"] == 0.2


# ---- errors -------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "name,error",
    [
        ("err-429", litellm.RateLimitError),
        ("err-401", litellm.AuthenticationError),
        ("err-context", litellm.ContextWindowExceededError),
    ],
)
def test_error_classes(base, model, name, error):
    with pytest.raises(error):
        litellm.completion(model=f"{model.split('/')[0]}/{name}", messages=HELLO, **base)


# ---- responses and other calls ------------------------------------------------------------------


def test_response_model_has_no_provider_prefix(base):
    assert litellm.completion(model=OPENAI, messages=HELLO, **base).model == "gpt-4o-mini"


def test_response_model_dump_json(base):
    assert '"choices"' in litellm.completion(model=OPENAI, messages=HELLO, **base).model_dump_json()


def test_success_callback_runs_after_a_stream(base):
    seen = []
    litellm.success_callback = [lambda kwargs, response, start, end: seen.append(response)]
    try:
        for _ in litellm.completion(model=OPENAI, messages=HELLO, stream=True, **base):
            pass
        deadline = time.monotonic() + 5  # LiteLLM calls it from a background thread
        while not [r for r in seen if type(r).__name__ == "ModelResponse"] and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        litellm.success_callback = []
    # Both give the whole response once at the end. LiteLLM 1.105 also calls it with each chunk first.
    assert [type(r).__name__ for r in seen if type(r).__name__ == "ModelResponse"] == ["ModelResponse"]


def test_router_does_not_hold_a_deployment_to_rpm_by_default():
    # LiteLLM's default simple-shuffle uses rpm as a traffic share; only its usage-based strategies limit by it.
    router = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": OPENAI, "rpm": 1}}])
    replies = [router.completion(model="chat", messages=HELLO, mock_response="ok") for _ in range(3)]
    assert [r.choices[0].message.content for r in replies] == ["ok"] * 3


def test_mock_response():
    resp = litellm.completion(model=OPENAI, messages=HELLO, mock_response="Offline reply")
    assert resp.choices[0].message.content == "Offline reply"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (10, 20)


def test_mock_response_stream_chunks_have_a_choice():
    chunks = list(litellm.completion(model=OPENAI, messages=HELLO, mock_response="Offline reply", stream=True))
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Offline reply"


def test_mock_response_error():
    with pytest.raises(litellm.RateLimitError):
        litellm.completion(model=OPENAI, messages=HELLO, mock_response="litellm.RateLimitError")


def test_completion_cost(base):
    assert litellm.completion_cost(completion_response=litellm.completion(model=OPENAI, messages=HELLO, **base)) > 0


def test_completion_cost_from_text():
    assert litellm.completion_cost(model=OPENAI, prompt="hello there", completion="hi") > 0


# ---- LiteLLM's modules: imports after install_as_litellm -----------------------------------------


@pytest.mark.parametrize(
    "module", ["litellm.utils", "litellm.exceptions", "litellm.cost_calculator", "litellm.integrations.custom_logger"]
)
def test_litellm_modules_import(module):
    __import__(module)


def test_sdk_under_test_is_the_one_imported():
    assert litellm.__name__ == ("litellm" if SDK == "litellm" else "theagentrouter.compat.litellm")
