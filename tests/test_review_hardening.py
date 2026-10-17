"""Regressions for the original go-live review, using the real HTTP fake provider."""

import json

import pytest

import theagentrouter as ar
from theagentrouter import _http, anthropic_format, exceptions, models, routing, transport
from theagentrouter.compat import litellm

HELLO = [{"role": "user", "content": "Say hello"}]


@pytest.fixture(autouse=True)
def environment(env):
    return env


@pytest.mark.parametrize(
    "params",
    [
        {"attempts": 1.5},
        {"attempts": True},
        {"on": (42,)},
        {"on": (600,)},
        {"on": ("429",)},
        {"timeout": -1},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": 0.0001},
        {"backoff": -1},
    ],
)
def test_retry_invalid_values_do_not_change_saved_routes(env, params, monkeypatch):
    monkeypatch.setattr(routing.local_gateway, "installed", lambda: True)
    ar.route("chat", primary="openai/gpt-4o-mini")
    before = (env / "routes.json").read_bytes()
    with pytest.raises(ValueError):
        ar.route("chat", primary="openai/gpt-4o-mini", retry=ar.Retry(**params))
    assert (env / "routes.json").read_bytes() == before


def test_proxy_environment_cannot_receive_loopback_prompts(fake, monkeypatch):
    _http.close()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    assert ar.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url).choices[0].message.content


def test_http_connections_are_reused(fake):
    for _ in range(2):
        ar.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    assert fake.connections[0] == fake.connections[1]


@pytest.mark.parametrize("model", ["openai/err-401", "anthropic/err-401"])
def test_stream_authentication_errors_raise_before_iteration(fake, model):
    with pytest.raises(exceptions.AuthenticationError) as caught:
        ar.completion(model=model, messages=HELLO, api_base=fake.url, stream=True)
    assert caught.value.response.status_code == 401
    assert caught.value.response.headers["x-request-id"] == "acme-request-id"


async def test_async_stream_errors_raise_at_await(fake):
    with pytest.raises(exceptions.RateLimitError) as caught:
        await ar.acompletion(model="openai/err-429", messages=HELLO, api_base=fake.url, stream=True)
    assert caught.value.response.headers["retry-after"] == "3"
    await ar.aclose()


@pytest.mark.parametrize("name", ["stream-html", "stream-json", "stream-redirect", "malformed-stream"])
def test_wrong_stream_responses_raise_instead_of_succeeding_empty(fake, name):
    with pytest.raises(exceptions.APIError):
        ar.completion(model=f"openai/{name}", messages=HELLO, api_base=fake.url, stream=True)


@pytest.mark.parametrize("name", ["malformed-json", "malformed-shape"])
def test_malformed_responses_raise_sdk_errors(fake, name):
    with pytest.raises(exceptions.APIResponseValidationError):
        ar.completion(model=f"openai/{name}", messages=HELLO, api_base=fake.url)


def test_cached_tokens_use_cache_prices(fake, monkeypatch):
    monkeypatch.setitem(
        models.model_cost,
        "openai/cached-cost",
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 8e-6,
            "cache_read_input_token_cost": 0.2e-6,
            "cache_creation_input_token_cost": 2.5e-6,
        },
    )
    response = ar.completion(model="openai/cached-cost", messages=HELLO, api_base=fake.url)
    assert response._hidden_params["response_cost"] == pytest.approx(0.00073)
    assert ar.completion_cost(response) == pytest.approx(0.00073)


def test_module_settings_and_legacy_function_arguments_are_used(fake, monkeypatch):
    monkeypatch.setattr(litellm, "api_base", fake.url)
    monkeypatch.setattr(litellm, "api_key", "acme-call-key")
    litellm.completion(
        model="openai/gpt-4o-mini",
        messages=HELLO,
        functions=[{"name": "search", "parameters": {"type": "object"}}],
        function_call={"name": "search"},
        headers={"x-acme": "custom"},
    )
    _, headers, body = fake.requests[-1]
    assert headers["authorization"] == "Bearer acme-call-key" and headers["x-acme"] == "custom"
    assert body["tools"][0]["function"]["name"] == "search"
    assert body["tool_choice"]["function"]["name"] == "search"


def test_unknown_arguments_warn_instead_of_disappearing(fake):
    with pytest.warns(UserWarning, match="acme_unknown"):
        ar.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, acme_unknown="value")


def test_credentials_are_not_in_representations_or_connection_errors():
    target = routing.Target(
        "api_base", "openai", "http://acme:password@127.0.0.1:9/v1", "acme-secret", "gpt-4o-mini", "openai"
    )
    assert "acme-secret" not in repr(target) and "password" not in repr(target)
    with pytest.raises(exceptions.APIConnectionError) as caught:
        ar.completion(
            model="openai/gpt-4o-mini", messages=HELLO, api_base=target.base_url, api_key=target.api_key, timeout=0.1
        )
    assert "password" not in str(caught.value) and "acme-secret" not in str(caught.value)


def test_claude_pdf_and_string_image_parts_are_kept():
    blocks = anthropic_format._user_blocks(
        [
            {"type": "image_url", "image_url": "https://acme.example/image.png"},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,YWNtZQ=="}},
        ]
    )
    assert [b["type"] for b in blocks] == ["image", "document"]
    assert blocks[1]["source"]["media_type"] == "application/pdf"
    with pytest.raises(exceptions.UnsupportedParamsError, match="input_audio"):
        anthropic_format._user_blocks([{"type": "input_audio", "input_audio": {"data": "YWNtZQ==", "format": "wav"}}])


def test_invalid_tool_arguments_are_not_replaced_with_an_empty_object():
    with pytest.raises(exceptions.BadRequestError, match="valid JSON"):
        anthropic_format.convert_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "acme-call", "function": {"name": "search", "arguments": "not JSON"}}],
                }
            ]
        )


def test_response_json_serialization_round_trips(fake):
    response = ar.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url)
    assert json.loads(response.model_dump_json(exclude_none=True))["choices"][0]["message"]["content"]
    assert response._response_ms >= 0


def test_timeout_zero_is_not_replaced_with_ten_minutes():
    assert transport._timeout({"timeout": 0}).read == 0
