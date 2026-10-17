"""LiteLLM's responses() and aresponses(): OpenAI's Responses API, through the gateway.

Agent Router serves ``/v1/responses`` for OpenAI-format backends (OpenAI, Azure OpenAI, AWS), so
the call goes where a completion for the same model goes, in that format. Claude is not served
there; use completion() for it. Streaming is not supported yet.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import httpx

from . import exceptions, models, routing
from .transport import _asend, _raise_for, _send

# Responses API arguments passed to the provider as they are.
_PASSED = (
    "instructions",
    "tools",
    "tool_choice",
    "max_output_tokens",
    "temperature",
    "top_p",
    "reasoning",
    "text",
    "previous_response_id",
    "store",
    "metadata",
    "parallel_tool_calls",
    "truncation",
    "include",
    "user",
    "service_tier",
    "background",
    "prompt_cache_key",
)


class ResponsesAPIResponse(dict):
    """The reply as OpenAI sends it, read by key or by attribute (``resp.output[0].content``), like LiteLLM's."""

    def __getattr__(self, key: str) -> Any:
        try:
            return _wrap_value(self[key])
        except KeyError:
            raise AttributeError(key) from None

    _hidden_params: dict[str, Any]  # where the call went and its cost, as on completion responses

    def model_dump(self, mode: str | None = None, **_: Any) -> dict[str, Any]:
        return copy.deepcopy(dict(self))

    @property
    def output_text(self) -> str:
        """The text of every output message, joined, as in OpenAI's SDK."""
        return "".join(
            part.get("text", "")
            for item in self.get("output") or []
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        )


class _Item(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return _wrap_value(self[key])
        except KeyError:
            raise AttributeError(key) from None


def _wrap_value(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, _Item):
        return _Item(value)
    if isinstance(value, list):
        return [_wrap_value(v) for v in value]
    return value


def _prepare(input: Any, model: str, params: dict[str, Any]) -> tuple[routing.Target, dict, dict, str]:
    if params.get("stream"):
        raise exceptions.UnsupportedParamsError(
            "responses(stream=True) is not supported yet; use completion(stream=True)", model=model
        )
    if not (params.get("api_base") or params.get("base_url") or routing.settings().get("base_url")):
        from . import routes

        r = routes.get(model)
        provider = (r.primary if r is not None else routing.with_provider(model)).split("/", 1)[0]
        if provider != "openai":
            raise exceptions.BadRequestError(
                f"responses: Agent Router serves the Responses API for OpenAI models, not {provider}; "
                "use completion() for this model",
                model=model,
                llm_provider=provider,
            )
    target = routing.resolve(
        model,
        api_base=params.get("api_base") or params.get("base_url"),
        api_key=params.get("api_key"),
        extra_headers=params.get("extra_headers"),
    )
    if target.format == routing.ANTHROPIC or (target.route != "gateway" and target.provider not in ("openai", "")):
        raise exceptions.BadRequestError(
            f"responses: Agent Router serves the Responses API for OpenAI models, not {target.provider}; "
            "use completion() for this model",
            model=model,
            llm_provider=target.provider,
        )
    body: dict[str, Any] = {"model": target.model, "input": input}
    body.update({k: params[k] for k in _PASSED if params.get(k) is not None})
    body.update(params.get("extra_body") or {})
    headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
    headers.update({"content-type": "application/json", **target.headers})
    return target, body, headers, target.base_url.rstrip("/") + "/responses"


def _parse(target: routing.Target, data: dict[str, Any], url: str, model: str) -> ResponsesAPIResponse:
    resp = ResponsesAPIResponse(data)
    usage = data.get("usage") or {}
    key = target.model if target.route in ("gateway", "local gateway") else f"{target.provider}/{target.model}"
    resp._hidden_params = {
        "route": target.route,
        "sent_to": url,
        "llm_provider": target.provider,
        "model_sent": target.model,
        "model_key": key,
        "response_cost": models.completion_cost(
            model=key,
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
        ),
    }
    return resp


def _check(r: httpx.Response, target: routing.Target, url: str, model: str) -> ResponsesAPIResponse:
    _raise_for(r, r.text, target, model)
    try:
        return _parse(target, r.json(), url, model)
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        raise exceptions.APIResponseValidationError(
            f"{target.provider}: reply is not JSON", model=model, llm_provider=target.provider
        ) from e


def responses(input: Any, model: str, **params: Any) -> ResponsesAPIResponse:
    """Same arguments as litellm.responses: ``input`` (a string or input items), ``model``, ``tools``, ..."""
    target, body, headers, url = _prepare(input, model, params)
    r = _send(target, body, headers, params, model, url)
    return _check(r, target, url, model)


async def aresponses(input: Any, model: str, **params: Any) -> ResponsesAPIResponse:
    """Same arguments as litellm.aresponses."""
    # In a worker thread: preparing can start the local gateway, which must not block the event loop.
    target, body, headers, url = await asyncio.to_thread(_prepare, input, model, params)
    r = await _asend(target, body, headers, params, model, url)
    return _check(r, target, url, model)
