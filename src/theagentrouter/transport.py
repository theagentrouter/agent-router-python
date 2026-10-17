"""HTTP: send the request in the target's format and turn the reply into LiteLLM-shaped objects."""

from __future__ import annotations

import asyncio
import sys
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from . import _http, anthropic_format, exceptions, models, openai_format, routing
from .types import ModelResponse, ModelResponseStream, StreamShaper, wants_usage

DEFAULT_TIMEOUT = 600.0  # seconds; LiteLLM's default

# LiteLLM arguments this SDK does not act on. They used to be dropped silently, so code that relied
# on, say, fallbacks kept running without them. Each one now warns once per process.
_GATEWAY = "a gateway does this: set AGENTROUTER_BASE_URL, or use the local gateway"
IGNORED: dict[str, str] = {
    "fallbacks": "define a route: theagentrouter.route() or litellm.Router",
    "context_window_fallbacks": _GATEWAY,
    "content_policy_fallbacks": _GATEWAY,
    "caching": _GATEWAY,
    "cache": _GATEWAY,
    "metadata": "the gateway's request logs record calls",
    "tags": "the gateway's request logs record calls",
    "callbacks": "the gateway's request logs record calls",
    "logger_fn": "the gateway's request logs record calls",
    "custom_llm_provider": "name the provider in the model, for example openai/gpt-4o-mini",
    "api_version": "Azure OpenAI goes through a gateway",
}
_warned: set[str] = set()


def _warn_ignored(params: dict[str, Any]) -> None:
    for name, instead in IGNORED.items():
        # Only values that ask for something: num_retries=0 (the OpenAI Agents SDK sends it) is not a request.
        if params.get(name) and name not in _warned:
            _warned.add(name)
            warnings.warn(f"theagentrouter ignores {name}=; {instead}.", UserWarning, stacklevel=4)


def drop_params(params: dict[str, Any]) -> bool:
    """LiteLLM's drop_params: on the call, else on the module (``litellm.drop_params = True``)."""
    if params.get("drop_params") is not None:
        return bool(params["drop_params"])
    modules = (sys.modules.get("theagentrouter"), sys.modules.get("theagentrouter.compat.litellm"))
    return any(getattr(m, "drop_params", False) for m in modules)


def _prepare(
    model: str, messages: list[Any], tools: list[dict[str, Any]] | None, stream: bool, params: dict[str, Any]
) -> tuple[routing.Target, dict[str, Any], dict[str, str]]:
    _warn_ignored(params)
    params = {**params, "drop_params": drop_params(params)}
    allowed = (
        set(openai_format.PASSTHROUGH)
        | set(IGNORED)
        | {
            "api_base",
            "base_url",
            "api_key",
            "extra_headers",
            "headers",
            "extra_body",
            "timeout",
            "request_timeout",
            "stream_timeout",
            "drop_params",
            "thinking",
            "top_k",
            "stream_options",
            "functions",
            "function_call",
            "num_retries",
            "max_retries",
        }
    )
    for key, value in params.items():
        if key not in allowed and value is not None and key not in _warned:
            _warned.add(key)
            warnings.warn(
                f"theagentrouter does not support {key}=; the argument is not sent.", UserWarning, stacklevel=3
            )
    if tools is None and params.get("functions") is not None:
        tools = [{"type": "function", "function": function} for function in params["functions"]]
    if params.get("function_call") is not None and params.get("tool_choice") is None:
        choice = params["function_call"]
        params["tool_choice"] = {"type": "function", "function": choice} if isinstance(choice, dict) else choice
    target = routing.resolve(
        model,
        api_base=params.get("api_base") or params.get("base_url"),
        api_key=params.get("api_key"),
        extra_headers={**(params.get("headers") or {}), **(params.get("extra_headers") or {})},
    )
    if target.format == routing.ANTHROPIC:
        body = anthropic_format.build_body(target.model, messages, tools, stream, params)
        headers = anthropic_format.headers(target.api_key)
    else:
        body = openai_format.build_body(target.model, messages, tools, stream, params)
        headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
    headers.update({"content-type": "application/json", **target.headers})
    return target, body, headers


def _timeout(params: dict[str, Any]) -> httpx.Timeout:
    value = params.get("timeout")
    if value is None:
        value = params.get("request_timeout")
    if value is None:
        value = DEFAULT_TIMEOUT
    timeout = value if isinstance(value, httpx.Timeout) else httpx.Timeout(float(value))
    return (
        httpx.Timeout(timeout, read=float(params["stream_timeout"]))
        if params.get("stream_timeout") is not None
        else timeout
    )


def _hidden(target: routing.Target) -> dict[str, Any]:
    # Gateways (set or local) get the full provider/model name; an api_base endpoint gets the provider's own name.
    key = target.model if target.route in ("gateway", "local gateway") else f"{target.provider}/{target.model}"
    return {
        "route": target.route,
        "sent_to": routing._safe_url(target.url),
        "llm_provider": target.provider,
        "model_sent": target.model,
        "model_key": key,
        "format": target.format,
    }


def _parse(target: routing.Target, data: dict[str, Any], model: str) -> ModelResponse:
    resp = (
        anthropic_format.parse_response(data)
        if target.format == routing.ANTHROPIC
        else openai_format.parse_response(data)
    )
    resp._hidden_params = _hidden(target)
    if not resp.model:
        resp.model = target.model
    if target.members:  # a route: priced as the model that answered
        resp._hidden_params["route_name"] = target.model
        served = served_by(resp.model, target.members)
        if served:
            resp._hidden_params["served_by"] = served
            resp._hidden_params["model_key"] = served
    # From the bundled table; a gateway that reports cost would override this.
    resp._hidden_params["response_cost"] = models.completion_cost(
        completion_response=resp,
        model=resp._hidden_params["model_key"],
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
    )
    return resp


def _raise_for(response: httpx.Response, text: str, target: routing.Target, model: str) -> None:
    if response.status_code >= 300:
        error = exceptions.from_http(response.status_code, text, model, target.provider)
        error.response = response
        error.headers = response.headers
        error.request = response.request
        raise error


def _wrap(error: Exception, target: routing.Target, model: str) -> exceptions.APIError:
    if isinstance(error, httpx.TimeoutException):
        return exceptions.Timeout(f"{target.provider}: request timed out", model=model, llm_provider=target.provider)
    return exceptions.APIConnectionError(
        f"{target.provider}: cannot reach {routing._safe_url(target.url)} ({type(error).__name__})",
        model=model,
        llm_provider=target.provider,
    )


def _stream_parser(target: routing.Target):
    return anthropic_format.StreamParser() if target.format == routing.ANTHROPIC else openai_format.StreamParser()


def _retries(params: dict[str, Any]) -> int:
    count = params.get("num_retries", params.get("max_retries", 0))
    if count is None:
        count = 0
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("num_retries must be a non-negative integer")
    return count


def _retry_delay(error: exceptions.APIError, attempt: int) -> float:
    headers = getattr(error, "headers", {}) or {}
    try:
        after = float(headers["retry-after"])
        if 0 <= after <= 60:
            return after
    except (KeyError, ValueError, TypeError):
        pass
    return min(0.1 * 2 ** min(attempt, 7), 10)


def _retryable(error: exceptions.APIError) -> bool:
    if isinstance(error, exceptions.BudgetExceededError):
        return False
    return (
        isinstance(error, (exceptions.APIConnectionError, exceptions.Timeout))
        or error.status_code in (408, 429)
        or bool(error.status_code and error.status_code >= 500)
    )


def _send(target, body, headers, params, model, url=None):
    address = url or target.url
    for attempt in range(_retries(params) + 1):
        try:
            try:
                response = _http.client(address).post(address, json=body, headers=headers, timeout=_timeout(params))
            except httpx.HTTPError as error:
                raise _wrap(error, target, model) from error
            _raise_for(response, response.text, target, model)
            return response
        except exceptions.APIError as error:
            if attempt == _retries(params) or not _retryable(error):
                raise
            time.sleep(_retry_delay(error, attempt))


async def _asend(target, body, headers, params, model, url=None):
    address = url or target.url
    for attempt in range(_retries(params) + 1):
        try:
            try:
                response = await _http.async_client(address).post(
                    address, json=body, headers=headers, timeout=_timeout(params)
                )
            except httpx.HTTPError as error:
                raise _wrap(error, target, model) from error
            _raise_for(response, response.text, target, model)
            return response
        except exceptions.APIError as error:
            if attempt == _retries(params) or not _retryable(error):
                raise
            await asyncio.sleep(_retry_delay(error, attempt))


def _named(chunk: ModelResponseStream, target: routing.Target) -> ModelResponseStream:
    if not chunk.model:
        chunk.model = target.model
    return chunk


# ---- routes ------------------------------------------------------------------------------------


def served_by(reply_model: str | None, members: tuple[str, ...]) -> str | None:
    """The route model that answered, from the model name in the reply.

    Providers answer with their own name, often with a date added (``gpt-4o-mini-2024-07-18``), so the
    longest model name the reply starts with wins.
    """
    if not reply_model:
        return None
    found = [m for m in members if reply_model.startswith(m.split("/", 1)[1])]
    return max(found, key=len) if found else None


# ---- sync ---------------------------------------------------------------------------------------


def complete(
    model: str, messages: list[Any], tools: list[dict[str, Any]] | None, params: dict[str, Any]
) -> ModelResponse:
    target, body, headers = _prepare(model, messages, tools, False, params)
    return _complete_one(target, body, headers, params, model)


def _complete_one(
    target: routing.Target, body: dict[str, Any], headers: dict[str, str], params: dict[str, Any], model: str
) -> ModelResponse:
    started = time.perf_counter()
    r = _send(target, body, headers, params, model)
    try:
        response = _parse(target, r.json(), model)
        response._response_ms = (time.perf_counter() - started) * 1000
        return response
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        raise exceptions.APIResponseValidationError(
            f"{target.provider}: malformed completion response", model=model, llm_provider=target.provider, response=r
        ) from e


def stream(
    model: str, messages: list[Any], tools: list[dict[str, Any]] | None, params: dict[str, Any]
) -> Iterator[ModelResponseStream]:
    target, body, headers = _prepare(model, messages, tools, True, params)
    for attempt in range(_retries(params) + 1):
        chunks = _shape_stream(target, body, headers, params, model)
        try:
            first = next(chunks)
            return _prepend(first, chunks)
        except StopIteration as error:
            raise exceptions.APIResponseValidationError(
                "The gateway returned an empty event stream", model=model
            ) from error
        except exceptions.APIError as error:
            chunks.close()
            if attempt == _retries(params) or not _retryable(error):
                raise
            time.sleep(_retry_delay(error, attempt))


def _prepend(first, chunks):
    try:
        yield first
        yield from chunks
    finally:
        chunks.close()


def _shape_stream(target, body, headers, params, model):
    shaper = StreamShaper(wants_usage(params))
    for chunk in _stream_one(target, body, headers, params, model):
        yield from shaper.feed(chunk)
    yield from shaper.end()


def _stream_one(
    target: routing.Target, body: dict[str, Any], headers: dict[str, str], params: dict[str, Any], model: str
) -> Iterator[ModelResponseStream]:
    r = None
    try:
        with _http.client(target.url).stream(
            "POST", target.url, json=body, headers=headers, timeout=_timeout(params)
        ) as r:
            if r.status_code >= 300:
                _raise_for(r, r.read().decode("utf-8", "replace"), target, model)
            if "text/event-stream" not in r.headers.get("content-type", "").lower():
                raise exceptions.APIResponseValidationError(
                    "Expected a text/event-stream response", model=model, response=r
                )
            parser = _stream_parser(target)
            for line in r.iter_lines():
                payload = openai_format.sse_payload(line)
                if payload:
                    for chunk in parser.feed(payload):
                        yield _named(chunk, target)
                if parser.done:
                    break
    except httpx.HTTPError as e:
        raise _wrap(e, target, model) from e
    except exceptions.APIError as e:
        e.model = model
        e.llm_provider = target.provider
        if r is not None and e.response is None:
            e.response, e.headers = r, r.headers
        raise
    except (ValueError, AttributeError, TypeError, KeyError) as error:
        raise exceptions.APIResponseValidationError(
            "Malformed event stream", model=model, llm_provider=target.provider, response=r
        ) from error


# ---- async --------------------------------------------------------------------------------------
# Preparing a call can start the local gateway: a process start, a file lock and up to minutes of
# waiting. It runs in a worker thread so the event loop keeps serving other tasks meanwhile.


async def acomplete(
    model: str, messages: list[Any], tools: list[dict[str, Any]] | None, params: dict[str, Any]
) -> ModelResponse:
    target, body, headers = await asyncio.to_thread(_prepare, model, messages, tools, False, params)
    return await _acomplete_one(target, body, headers, params, model)


async def _acomplete_one(
    target: routing.Target, body: dict[str, Any], headers: dict[str, str], params: dict[str, Any], model: str
) -> ModelResponse:
    started = time.perf_counter()
    r = await _asend(target, body, headers, params, model)
    try:
        response = _parse(target, r.json(), model)
        response._response_ms = (time.perf_counter() - started) * 1000
        return response
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        raise exceptions.APIResponseValidationError(
            f"{target.provider}: malformed completion response", model=model, llm_provider=target.provider, response=r
        ) from e


async def astream(
    model: str, messages: list[Any], tools: list[dict[str, Any]] | None, params: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    target, body, headers = await asyncio.to_thread(_prepare, model, messages, tools, True, params)
    for attempt in range(_retries(params) + 1):
        chunks = _ashape_stream(target, body, headers, params, model)
        try:
            first = await anext(chunks)
            return _aprepend(first, chunks)
        except StopAsyncIteration as error:
            raise exceptions.APIResponseValidationError(
                "The gateway returned an empty event stream", model=model
            ) from error
        except exceptions.APIError as error:
            await chunks.aclose()
            if attempt == _retries(params) or not _retryable(error):
                raise
            await asyncio.sleep(_retry_delay(error, attempt))


async def _aprepend(first, chunks):
    try:
        yield first
        async for chunk in chunks:
            yield chunk
    finally:
        await chunks.aclose()


async def _ashape_stream(target, body, headers, params, model):
    shaper = StreamShaper(wants_usage(params))
    async for chunk in _astream_one(target, body, headers, params, model):
        for out in shaper.feed(chunk):
            yield out
    for out in shaper.end():
        yield out


async def _astream_one(
    target: routing.Target, body: dict[str, Any], headers: dict[str, str], params: dict[str, Any], model: str
) -> AsyncIterator[ModelResponseStream]:
    parser = _stream_parser(target)
    r = None
    try:
        async with _http.async_client(target.url).stream(
            "POST", target.url, json=body, headers=headers, timeout=_timeout(params)
        ) as r:
            if r.status_code >= 300:
                _raise_for(r, (await r.aread()).decode("utf-8", "replace"), target, model)
            if "text/event-stream" not in r.headers.get("content-type", "").lower():
                raise exceptions.APIResponseValidationError(
                    "Expected a text/event-stream response", model=model, response=r
                )
            async for line in r.aiter_lines():
                payload = openai_format.sse_payload(line)
                if payload:
                    for chunk in parser.feed(payload):
                        yield _named(chunk, target)
                if parser.done:
                    break
    except httpx.HTTPError as e:
        raise _wrap(e, target, model) from e
    except exceptions.APIError as e:
        e.model = model
        e.llm_provider = target.provider
        if r is not None and e.response is None:
            e.response, e.headers = r, r.headers
        raise
    except (ValueError, AttributeError, TypeError, KeyError) as error:
        raise exceptions.APIResponseValidationError(
            "Malformed event stream", model=model, llm_provider=target.provider, response=r
        ) from error
