"""LiteLLM's mock_response: an answer made on this computer, for tests. No gateway starts and no provider is called.

As in LiteLLM, a call is mocked when it has ``mock_response``, ``mock_tool_calls`` or ``mock_timeout``:

- a string is the reply text (``"litellm.RateLimitError"``, ``"litellm.ContextWindowExceededError"`` and
  ``"litellm.InternalServerError"`` raise that error instead);
- an exception is raised, as one of LiteLLM's errors;
- a dict or a ModelResponse is the whole response;
- ``mock_tool_calls`` adds tool calls, ``n`` gives that many choices, ``prompt_tokens`` sets the prompt count;
- ``mock_timeout=True`` with a ``timeout`` raises Timeout, and ``mock_delay`` waits that many seconds first.

Usage is 10 prompt and 20 completion tokens, as in LiteLLM. A stream sends the text three characters at a time.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from . import exceptions, models, openai_format, routing
from .types import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage

DEFAULT = "This is a mock request"
PROMPT_TOKENS, COMPLETION_TOKENS = 10, 20
_ERRORS = {
    "litellm.RateLimitError": (exceptions.RateLimitError, "this is a mock rate limit error"),
    "litellm.ContextWindowExceededError": (
        exceptions.ContextWindowExceededError,
        "this is a mock context window exceeded error",
    ),
    "litellm.InternalServerError": (exceptions.InternalServerError, "this is a mock internal server error"),
}


def requested(params: dict[str, Any]) -> bool:
    return bool(params.get("mock_response") or params.get("mock_tool_calls") or params.get("mock_timeout"))


def _provider(model: str) -> str:
    return routing.provider_for(model) or "openai"


def _check(model: str, params: dict[str, Any]) -> Any:
    """The mock answer, after raising any error it asks for."""
    provider = _provider(model)
    if params.get("mock_timeout") and params.get("timeout") is not None:
        raise exceptions.Timeout("This is a mock timeout error", model=model, llm_provider=provider)
    mock = params.get("mock_response") or DEFAULT
    if isinstance(mock, exceptions.APIError):
        raise mock
    if isinstance(mock, Exception):
        status = getattr(mock, "status_code", 500)
        raise exceptions.from_http(status, getattr(mock, "text", str(mock)), model, provider) from mock
    if isinstance(mock, str) and mock in _ERRORS:
        cls, message = _ERRORS[mock]
        raise cls(message, model=model, llm_provider=provider)
    return mock


def _hidden(model: str, usage: Usage) -> dict[str, Any]:
    key = model if "/" in model else f"{_provider(model)}/{model}"
    return {
        "route": "mock",
        "llm_provider": _provider(model),
        "model_sent": model,
        "model_key": key,
        "response_cost": models.completion_cost(
            model=key, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
        ),
    }


def _response(model: str, mock: Any, params: dict[str, Any]) -> ModelResponse:
    if isinstance(mock, ModelResponse):
        return mock
    if isinstance(mock, dict):
        resp = openai_format.parse_response({"id": f"chatcmpl-mock-{uuid.uuid4().hex}", "model": model, **mock})
    else:
        prompt = params.get("prompt_tokens")
        prompt = PROMPT_TOKENS if prompt is None else int(prompt)
        message: dict[str, Any] = {"role": "assistant", "content": str(mock)}
        if params.get("mock_tool_calls"):
            message["tool_calls"] = params["mock_tool_calls"]
        resp = openai_format.parse_response(
            {
                "id": f"chatcmpl-mock-{uuid.uuid4().hex}",
                "model": model,
                "created": int(time.time()),
                "choices": [
                    {"index": i, "finish_reason": "stop", "message": message} for i in range(params.get("n") or 1)
                ],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": prompt + COMPLETION_TOKENS,
                },
            }
        )
    resp._hidden_params = _hidden(model, resp.usage)
    return resp


def _chunks(resp: ModelResponse) -> Iterator[ModelResponseStream]:
    text = resp.choices[0].message.content or "" if resp.choices else ""
    for i in range(0, len(text), 3):
        yield ModelResponseStream(
            id=resp.id,
            model=resp.model,
            created=resp.created,
            choices=[
                StreamingChoices(index=c.index, delta=Delta(role="assistant", content=text[i : i + 3]))
                for c in resp.choices
            ],
        )
    # The last chunk carries the finish reason and the usage, as from a gateway.
    yield ModelResponseStream(
        id=resp.id,
        model=resp.model,
        created=resp.created,
        choices=[
            StreamingChoices(index=c.index, delta=Delta(tool_calls=c.message.tool_calls), finish_reason=c.finish_reason)
            for c in resp.choices
        ],
        usage=resp.usage,
    )


def complete(model: str, params: dict[str, Any]) -> ModelResponse:
    if params.get("mock_delay"):
        time.sleep(float(params["mock_delay"]))
    return _response(model, _check(model, params), params)


def stream(model: str, params: dict[str, Any]) -> Iterator[ModelResponseStream]:
    resp = complete(model, params)  # errors raise here, before the first chunk, as in LiteLLM
    return _chunks(resp)


async def acomplete(model: str, params: dict[str, Any]) -> ModelResponse:
    if params.get("mock_delay"):
        await asyncio.sleep(float(params["mock_delay"]))
    return _response(model, _check(model, params), params)


async def astream(model: str, params: dict[str, Any]) -> AsyncIterator[ModelResponseStream]:
    resp = await acomplete(model, params)

    async def chunks() -> AsyncIterator[ModelResponseStream]:
        for chunk in _chunks(resp):
            yield chunk

    return chunks()
