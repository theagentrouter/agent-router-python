"""LiteLLM's behaviour, checked on LiteLLM and on Agent Router's SDK.

Each test states what LiteLLM does. On LiteLLM the run proves the expectation; on Agent Router's SDK
it proves the drop-in. Known differences are marked known_gap (strict), named after the go-live
readiness review's finding where it has one.
"""

import time

import litellm
import pytest
from conftest import CLAUDE, MODELS, OPENAI, SDK

HELLO = [{"role": "user", "content": "refunds yesterday"}]


# ---- streaming (review H1) ----------------------------------------------------------------------


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


# ---- Claude parameters (review H2) --------------------------------------------------------------


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


def test_claude_thinking_is_sent(base, fake):
    thinking = {"type": "enabled", "budget_tokens": 1024}
    litellm.completion(model=CLAUDE, messages=HELLO, thinking=thinking, max_tokens=2048, **base)
    assert fake.requests[-1][2]["thinking"] == thinking


def test_claude_reasoning_effort_becomes_thinking(base, fake):
    litellm.completion(model=CLAUDE, messages=HELLO, reasoning_effort="low", **base)
    assert fake.requests[-1][2]["thinking"]["type"] == "enabled"


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
        while not seen and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        litellm.success_callback = []
    assert len(seen) == 1


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


# ---- LiteLLM's modules (review: imports after install_as_litellm) -------------------------------


@pytest.mark.parametrize(
    "module", ["litellm.utils", "litellm.exceptions", "litellm.cost_calculator", "litellm.integrations.custom_logger"]
)
def test_litellm_modules_import(module):
    __import__(module)


def test_sdk_under_test_is_the_one_imported():
    assert litellm.__name__ == ("litellm" if SDK == "litellm" else "theagentrouter.compat.litellm")
