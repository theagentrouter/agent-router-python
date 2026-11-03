"""LiteLLM's Anthropic Messages API: ``litellm.anthropic.messages.create`` and ``acreate``, through the gateway.

The request and the reply are Anthropic's own format, so every Messages feature works: prompt caching,
thinking, tools, citations and beta headers. As in LiteLLM, the thinking settings and sampling options a
model refuses are changed first. The call goes where a completion for the same Claude model goes, to the
gateway's ``/anthropic/v1/messages``. Agent Router serves Messages for Claude, so other
providers are refused; use completion() for them. Routes are refused too: the local gateway answers a route
in the OpenAI format.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from . import _http, anthropic_format, exceptions, local_gateway, models, routing
from .responses import _wrap_value
from .transport import _asend, _raise_for, _retries, _retry_delay, _retryable, _send, _timeout, _wrap, drop_params

# Messages arguments sent to the gateway as they are; the rest of **kwargs controls the call.
_PASSED = (
    "metadata",
    "stop_sequences",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "container",
    "service_tier",
    "context_management",
    "mcp_servers",
    "cache_control",
    "output_config",
    "output_format",
    "compaction",
    "inference_geo",
    "speed",
    "safeguards",
)


class AnthropicMessagesResponse(dict):
    """The reply as Anthropic sends it: a dict, as LiteLLM returns, also read by attribute (``resp.content[0].text``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return _wrap_value(self[key])
        except KeyError:
            raise AttributeError(key) from None

    _hidden_params: dict[str, Any]  # where the call went and its cost, as on completion responses


def _refuse(model: str, why: str, provider: str = "") -> exceptions.BadRequestError:
    return exceptions.BadRequestError(
        f"anthropic.messages: {why}; use completion() for this model", model=model, llm_provider=provider or None
    )


def _prepare(model: str, params: dict[str, Any]) -> tuple[routing.Target, dict[str, str], str]:
    api_base = params.get("api_base") or params.get("base_url")
    gateway = routing.settings().get("base_url")
    if not api_base:
        from . import routes

        if not gateway and routes.get(model) is not None:
            raise _refuse(model, f"{model!r} is a route, and the local gateway answers routes in the OpenAI format")
        provider = routing.provider_for(model)
        # A gateway you set may have Claude names of its own, such as claude-sonnet-5; anything it can place elsewhere is refused.
        if provider and provider != "anthropic":
            raise _refuse(model, f"Agent Router serves the Messages API for Claude, not {provider}", provider)
        if not provider and not gateway:
            raise _refuse(model, f"cannot tell which provider serves {model!r}; write anthropic/<model>")
    target = routing.resolve(
        model, api_base=api_base, api_key=params.get("api_key"), extra_headers=params.get("extra_headers")
    )
    if target.route == "gateway":
        # Always the Messages path, also for a name of the gateway's own.
        target.base_url = routing.gateway_root(gateway) + local_gateway.PROVIDERS["anthropic"]["path"]
        target.format = routing.ANTHROPIC
    elif target.format != routing.ANTHROPIC:
        raise _refuse(model, f"Agent Router serves the Messages API for Claude, not {target.provider}", target.provider)
    headers = anthropic_format.headers(target.api_key)
    headers.update({"content-type": "application/json", **target.headers})
    return target, headers, target.url


def _body(target: routing.Target, max_tokens: int, messages: list[dict], stream: bool | None, params: dict) -> dict:
    body: dict[str, Any] = {"model": target.model, "max_tokens": max_tokens, "messages": messages}
    params = anthropic_format.messages_params(
        target.model, {**params, "drop_params": drop_params(params)}, max_tokens, target.route == "gateway"
    )
    body.update({k: params[k] for k in _PASSED if params.get(k) is not None})
    if stream:
        body["stream"] = True
    body.update(params.get("extra_body") or {})
    return body


def _hidden(target: routing.Target, url: str, usage: dict[str, Any] | None) -> dict[str, Any]:
    key = target.model if target.route in ("gateway", "local gateway") else f"{target.provider}/{target.model}"
    hidden: dict[str, Any] = {
        "route": target.route,
        "sent_to": url,
        "llm_provider": target.provider or "anthropic",
        "model_sent": target.model,
        "model_key": key,
    }
    if usage is not None:
        u = anthropic_format._usage(usage)  # the same token counts completion() reports for Claude
        hidden["response_cost"] = models.completion_cost(
            model=key, prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens
        )
    return hidden


def _check(r: httpx.Response, target: routing.Target, url: str, model: str) -> AnthropicMessagesResponse:
    _raise_for(r, r.text, target, model)
    try:
        data = r.json()
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise ValueError("missing Messages content")
        resp = AnthropicMessagesResponse(data)
        resp._hidden_params = _hidden(target, url, resp.get("usage") or {})
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        raise exceptions.APIResponseValidationError(
            f"{target.provider}: malformed Messages response", model=model, llm_provider=target.provider, response=r
        ) from e
    return resp


def _events(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Whole server-sent events in ``buffer``, each ending with its blank line, and what is left over."""
    out = []
    buffer = buffer.replace(b"\r\n", b"\n")
    while b"\n\n" in buffer:
        event, buffer = buffer.split(b"\n\n", 1)
        if event.strip():
            out.append(event + b"\n\n")
    return out, buffer


def _open(target: routing.Target, headers: dict, url: str, body: dict, model: str, params: dict) -> httpx.Response:
    """The stream's response, with the retries num_retries asks for. Once events flow, nothing is replayed."""
    client = _http.client(url)
    for attempt in range(_retries(params) + 1):
        try:
            try:
                r = client.send(
                    client.build_request("POST", url, json=body, headers=headers, timeout=_timeout(params)),
                    stream=True,
                )
            except httpx.HTTPError as e:
                raise _wrap(e, target, model) from e
            if r.status_code >= 300:
                text = r.read().decode(errors="replace")
                r.close()
                _raise_for(r, text, target, model)
            return r
        except exceptions.APIError as error:
            if attempt == _retries(params) or not _retryable(error):
                raise
            time.sleep(_retry_delay(error, attempt))
    raise AssertionError("unreachable")


async def _aopen(
    target: routing.Target, headers: dict, url: str, body: dict, model: str, params: dict
) -> httpx.Response:
    client = _http.async_client(url)
    for attempt in range(_retries(params) + 1):
        try:
            try:
                r = await client.send(
                    client.build_request("POST", url, json=body, headers=headers, timeout=_timeout(params)),
                    stream=True,
                )
            except httpx.HTTPError as e:
                raise _wrap(e, target, model) from e
            if r.status_code >= 300:
                text = (await r.aread()).decode(errors="replace")
                await r.aclose()
                _raise_for(r, text, target, model)
            return r
        except exceptions.APIError as error:
            if attempt == _retries(params) or not _retryable(error):
                raise
            await asyncio.sleep(_retry_delay(error, attempt))
    raise AssertionError("unreachable")


def _stream(target: routing.Target, headers: dict, url: str, body: dict, model: str, params: dict) -> Iterator[bytes]:
    r = _open(target, headers, url, body, model, params)
    if "text/event-stream" not in r.headers.get("content-type", "").lower():
        r.close()
        raise exceptions.APIResponseValidationError("Expected a text/event-stream response", model=model, response=r)

    def events() -> Iterator[bytes]:
        buffer = b""
        try:
            for data in r.iter_bytes():
                chunks, buffer = _events(buffer + data)
                yield from chunks
            if buffer.strip():
                yield buffer + b"\n\n"
        except httpx.HTTPError as e:
            raise _wrap(e, target, model) from e
        finally:
            r.close()

    return events()


async def _astream(
    target: routing.Target, headers: dict, url: str, body: dict, model: str, params: dict
) -> AsyncIterator[bytes]:
    r = await _aopen(target, headers, url, body, model, params)
    if "text/event-stream" not in r.headers.get("content-type", "").lower():
        await r.aclose()
        raise exceptions.APIResponseValidationError("Expected a text/event-stream response", model=model, response=r)

    async def events() -> AsyncIterator[bytes]:
        buffer = b""
        try:
            async for data in r.aiter_bytes():
                chunks, buffer = _events(buffer + data)
                for chunk in chunks:
                    yield chunk
            if buffer.strip():
                yield buffer + b"\n\n"
        except httpx.HTTPError as e:
            raise _wrap(e, target, model) from e
        finally:
            await r.aclose()

    return events()


def create(
    max_tokens: int,
    messages: list[dict],
    model: str,
    metadata: dict | None = None,
    stop_sequences: list[str] | None = None,
    stream: bool | None = False,
    system: str | list[dict] | None = None,
    temperature: float | None = None,
    thinking: dict | None = None,
    tool_choice: dict | None = None,
    tools: list[dict] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    container: dict | None = None,
    **kwargs: Any,
) -> AnthropicMessagesResponse | Iterator[bytes]:
    """Same arguments as litellm.anthropic.messages.create. With stream=True, the server-sent events as bytes."""
    params = {**kwargs, **_named(locals())}
    target, headers, url = _prepare(model, params)
    body = _body(target, max_tokens, messages, stream, params)
    headers = anthropic_format.with_betas(headers, body)
    if stream:
        return _stream(target, headers, url, body, model, params)
    r = _send(target, body, headers, params, model, url)
    return _check(r, target, url, model)


async def acreate(
    max_tokens: int,
    messages: list[dict],
    model: str,
    metadata: dict | None = None,
    stop_sequences: list[str] | None = None,
    stream: bool | None = False,
    system: str | list[dict] | None = None,
    temperature: float | None = None,
    thinking: dict | None = None,
    tool_choice: dict | None = None,
    tools: list[dict] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    container: dict | None = None,
    **kwargs: Any,
) -> AnthropicMessagesResponse | AsyncIterator[bytes]:
    """Same arguments as litellm.anthropic.messages.acreate."""
    params = {**kwargs, **_named(locals())}
    # In a worker thread: preparing can start the local gateway, which must not block the event loop.
    target, headers, url = await asyncio.to_thread(_prepare, model, params)
    body = _body(target, max_tokens, messages, stream, params)
    headers = anthropic_format.with_betas(headers, body)
    if stream:
        return await _astream(target, headers, url, body, model, params)
    r = await _asend(target, body, headers, params, model, url)
    return _check(r, target, url, model)


async def anthropic_messages(
    max_tokens: int, messages: list[dict], model: str, **kwargs: Any
) -> AnthropicMessagesResponse | AsyncIterator[bytes]:
    """litellm.anthropic_messages: the async call, as acreate."""
    return await acreate(max_tokens=max_tokens, messages=messages, model=model, **kwargs)


def _named(local: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in local.items() if k in _PASSED}
