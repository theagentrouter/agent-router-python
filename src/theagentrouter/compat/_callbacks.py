"""LiteLLM's callbacks for completion() and acompletion(), for code that sets litellm.success_callback.

Each entry is a function ``fn(kwargs, response, start_time, end_time)`` or a logger object with
LiteLLM's ``log_success_event`` / ``log_failure_event`` (and ``async_`` versions). They run in the
program, after the call; an error in one warns and does not stop the call. Strings name LiteLLM's
built-in integrations, which are not here.
"""

from __future__ import annotations

import inspect
import uuid
import warnings
from datetime import datetime
from typing import Any

from .. import models
from ..types import stream_chunk_builder

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, UserWarning, stacklevel=5)


def _entries(module: dict[str, Any], failure: bool, is_async: bool) -> list[Any]:
    names = ["failure_callback", "callbacks"] if failure else ["success_callback", "callbacks"]
    if is_async and not failure:
        names.append("_async_success_callback")
    seen: list[Any] = []
    for name in names:
        for entry in module.get(name) or []:
            if isinstance(entry, str):
                _warn_once(
                    f"str:{entry}",
                    f"theagentrouter does not have LiteLLM's {entry!r} integration; "
                    "the gateway's OpenTelemetry traces and metrics record calls instead.",
                )
            elif not any(entry is s for s in seen):
                seen.append(entry)
    return seen


def _any(module: dict[str, Any]) -> bool:
    return any(module.get(n) for n in ("success_callback", "failure_callback", "callbacks", "_async_success_callback"))


def _details(model: str, messages: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    params = {k: v for k, v in kwargs.items() if k not in ("api_key", "messages")}
    return {
        "model": model,
        "messages": messages,
        "stream": bool(kwargs.get("stream")),
        "optional_params": params,
        "litellm_params": {"api_base": kwargs.get("api_base")},
        "litellm_call_id": str(uuid.uuid4()),
    }


def _method(entry: Any, failure: bool, is_async: bool):
    base = "log_failure_event" if failure else "log_success_event"
    if is_async and callable(getattr(entry, "async_" + base, None)):
        return getattr(entry, "async_" + base)
    if callable(getattr(entry, base, None)):
        return getattr(entry, base)
    if callable(entry) and not inspect.isclass(entry):
        return entry
    return None


def _fire(entries: list[Any], failure: bool, details: dict, response: Any, start: datetime, end: datetime) -> None:
    for entry in entries:
        fn = _method(entry, failure, False)
        if fn is None:
            continue
        try:
            result = fn(details, response, start, end)
            if inspect.iscoroutine(result):
                result.close()
                _warn_once(f"coro:{id(entry)}", f"callback {entry!r} is async; it runs only with acompletion().")
        except Exception as e:  # a logger must not break the call
            _warn_once(f"err:{id(entry)}", f"callback {entry!r} failed: {e!r}")


async def _afire(
    entries: list[Any], failure: bool, details: dict, response: Any, start: datetime, end: datetime
) -> None:
    for entry in entries:
        fn = _method(entry, failure, True)
        if fn is None:
            continue
        try:
            result = fn(details, response, start, end)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            _warn_once(f"err:{id(entry)}", f"callback {entry!r} failed: {e!r}")


def _success_details(details: dict, response: Any) -> dict:
    hidden = getattr(response, "_hidden_params", None) or {}
    details["response_cost"] = hidden.get("response_cost", models.completion_cost(response))
    return details


def run(call, model: str, messages: list[Any], kwargs: dict[str, Any], module: dict[str, Any]):
    if not _any(module):
        return call(model, messages, **kwargs)
    details, start = _details(model, messages, kwargs), datetime.now()
    try:
        result = call(model, messages, **kwargs)
    except Exception as e:
        _fire(_entries(module, True, False), True, {**details, "exception": e}, None, start, datetime.now())
        raise
    if not kwargs.get("stream"):
        _fire(_entries(module, False, False), False, _success_details(details, result), result, start, datetime.now())
        return result
    return _stream(result, details, start, module)


def _stream(chunks, details: dict, start: datetime, module: dict[str, Any]):
    seen = []
    try:
        for chunk in chunks:
            seen.append(chunk)
            yield chunk
    except Exception as e:
        _fire(_entries(module, True, False), True, {**details, "exception": e}, None, start, datetime.now())
        raise
    full = stream_chunk_builder(seen)
    details["complete_streaming_response"] = full
    _fire(_entries(module, False, False), False, _success_details(details, full), full, start, datetime.now())


async def arun(call, model: str, messages: list[Any], kwargs: dict[str, Any], module: dict[str, Any]):
    if not _any(module):
        return await call(model, messages, **kwargs)
    details, start = _details(model, messages, kwargs), datetime.now()
    try:
        result = await call(model, messages, **kwargs)
    except Exception as e:
        await _afire(_entries(module, True, True), True, {**details, "exception": e}, None, start, datetime.now())
        raise
    if not kwargs.get("stream"):
        await _afire(
            _entries(module, False, True), False, _success_details(details, result), result, start, datetime.now()
        )
        return result
    return _astream(result, details, start, module)


async def _astream(chunks, details: dict, start: datetime, module: dict[str, Any]):
    seen = []
    try:
        async for chunk in chunks:
            seen.append(chunk)
            yield chunk
    except Exception as e:
        await _afire(_entries(module, True, True), True, {**details, "exception": e}, None, start, datetime.now())
        raise
    full = stream_chunk_builder(seen)
    details["complete_streaming_response"] = full
    await _afire(_entries(module, False, True), False, _success_details(details, full), full, start, datetime.now())
