"""Anthropic Messages: convert OpenAI-style requests to Messages, and responses and streams back.

Callers use LiteLLM's OpenAI-shaped messages and tools. Anthropic's Messages API differs in three
ways this module handles: the system prompt is a separate field, tool calls and tool results are
content blocks, and user and assistant turns must alternate.

Two things pass through as LiteLLM passes them: ``cache_control`` (on a content part, on a whole
message, or on a tool) for prompt caching, and an assistant message's ``thinking_blocks`` with
their signatures, which Anthropic needs back in a tool-use turn when thinking is on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from . import exceptions
from .openai_format import PASSTHROUGH, _plain, iter_sse_data
from .types import (
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    ToolCall,
    Usage,
    _TokenDetails,
)

VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
SUPPORTED_PARAMS = frozenset(
    (
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "tool_choice",
        "parallel_tool_calls",
        "user",
    )
)

_STOP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def headers(api_key: str | None) -> dict[str, str]:
    # A gateway's key goes in x-api-key too, never Authorization: Agent Router replaces x-api-key with
    # the provider's key but passes Authorization on to Anthropic.
    out = {"anthropic-version": VERSION}
    if api_key:
        out["x-api-key"] = api_key
    return out


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _cached(block: dict[str, Any], part: Any) -> dict[str, Any]:
    """Keep a part's cache_control (LiteLLM's prompt-caching marker) on the block made from it."""
    if isinstance(part, dict) and part.get("cache_control"):
        block["cache_control"] = part["cache_control"]
    return block


def _mark_last(blocks: list[dict[str, Any]], message: dict[str, Any]) -> list[dict[str, Any]]:
    """cache_control on a whole message, as LiteLLM takes it, goes on the message's last block."""
    if message.get("cache_control") and blocks and "cache_control" not in blocks[-1]:
        blocks[-1]["cache_control"] = message["cache_control"]
    return blocks


def _text_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [
        _cached({"type": "text", "text": p.get("text", "")}, p)
        for p in content or []
        if isinstance(p, dict) and p.get("type") == "text"
    ]


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            blocks.append(_cached({"type": "text", "text": part.get("text", "")}, part))
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                media, data = url[5:].split(";base64,", 1)
                source = {"type": "base64", "media_type": media, "data": data}
            else:
                source = {"type": "url", "url": url}
            blocks.append(_cached({"type": "image", "source": source}, part))
    return blocks


def _thinking_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The assistant's thinking blocks, as Anthropic sent them (signature included)."""
    out = []
    for block in message.get("thinking_blocks") or []:
        block = dict(block)
        if block.get("type") == "thinking" and block.get("signature"):
            out.append({k: block[k] for k in ("type", "thinking", "signature") if k in block})
        elif block.get("type") == "redacted_thinking" and block.get("data"):
            out.append({"type": "redacted_thinking", "data": block["data"]})
    return out


def _arguments(raw: str) -> Any:
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


def convert_messages(messages: list[Any]) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """System prompt and messages. The system prompt is a string, or text blocks when one carries cache_control."""
    system: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:  # Anthropic needs alternating turns
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for raw in messages:
        m = _plain(raw)
        role = m.get("role")
        if role in ("system", "developer"):
            system.extend(_mark_last(_text_blocks(m.get("content")), m))
        elif role == "user":
            add("user", _mark_last(_user_blocks(m.get("content")), m))
        elif role == "assistant":
            blocks: list[dict[str, Any]] = _thinking_blocks(m) + _text_blocks(m.get("content"))
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": c.get("id"),
                        "name": f.get("name"),
                        "input": _arguments(f.get("arguments") or ""),
                    }
                )
            add("assistant", _mark_last(blocks, m))
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": _text_of(m.get("content")) or str(m.get("content") or ""),
            }
            for part in m.get("content") if isinstance(m.get("content"), list) else []:
                _cached(result, part)
            add("user", _mark_last([result], m))
    system = [b for b in system if b["text"]]
    if any("cache_control" in b for b in system):
        return system, out
    return ("\n\n".join(b["text"] for b in system) or None), out


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        f = t.get("function") or t
        tool = {
            "name": f.get("name"),
            "description": f.get("description", ""),
            "input_schema": f.get("parameters") or {"type": "object", "properties": {}},
        }
        cache = t.get("cache_control") or f.get("cache_control")
        if cache:
            tool["cache_control"] = cache
        out.append(tool)
    return out


def convert_tool_choice(choice: Any) -> dict[str, Any] | None:
    if choice is None:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def build_body(
    model: str, messages: list[Any], tools: list[dict[str, Any]] | None, stream: bool, params: dict[str, Any]
) -> dict[str, Any]:
    # As LiteLLM: n=1 is always fine; the rest raise unless drop_params is set, which drops them.
    unsupported = [
        key
        for key in PASSTHROUGH
        if key not in SUPPORTED_PARAMS and params.get(key) is not None and not (key == "n" and params[key] == 1)
    ]
    if unsupported and not params.get("drop_params"):
        raise exceptions.UnsupportedParamsError(
            "Anthropic Messages does not support these SDK parameters: "
            + ", ".join(unsupported)
            + ". Set drop_params=True to drop them, or use extra_body for native Anthropic options.",
            model=model,
            llm_provider="anthropic",
        )
    system, converted = convert_messages(messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": converted,
        "max_tokens": params.get("max_tokens") or params.get("max_completion_tokens") or DEFAULT_MAX_TOKENS,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = convert_tools(tools)
        choice = convert_tool_choice(params.get("tool_choice"))
        if params.get("parallel_tool_calls") is not None:
            choice = choice or {"type": "auto"}
            if choice["type"] != "none":
                choice["disable_parallel_tool_use"] = not params["parallel_tool_calls"]
        if choice:
            body["tool_choice"] = choice
    for key in ("temperature", "top_p", "top_k"):
        if params.get(key) is not None:
            body[key] = params[key]
    stop = params.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if params.get("user"):
        body["metadata"] = {"user_id": str(params["user"])}
    thinking = params.get("thinking")
    if thinking:
        body["thinking"] = thinking
        budget = (thinking.get("budget_tokens") or 0) if isinstance(thinking, dict) else 0
        if not (params.get("max_tokens") or params.get("max_completion_tokens")) and budget >= body["max_tokens"]:
            body["max_tokens"] = budget + DEFAULT_MAX_TOKENS  # Claude needs room for the answer after thinking
    if params.get("extra_body"):
        body.update(params["extra_body"])
    if stream:
        body["stream"] = True
    return body


def _usage(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    prompt = (
        (raw.get("input_tokens") or 0)
        + (raw.get("cache_read_input_tokens") or 0)
        + (raw.get("cache_creation_input_tokens") or 0)
    )
    completion = raw.get("output_tokens") or 0
    # Cached prompt tokens where LiteLLM puts them; the OpenAI Agents SDK reads them there.
    cached = raw.get("cache_read_input_tokens") or 0
    details = _TokenDetails(cached_tokens=cached)  # 0 when nothing was read, as in LiteLLM
    return Usage(prompt, completion, prompt + completion, prompt_tokens_details=details)


def parse_response(data: dict[str, Any]) -> ModelResponse:
    text, thinking, blocks, calls = [], [], [], []
    for block in data.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            text.append(block.get("text", ""))
        elif kind == "thinking":
            thinking.append(block.get("thinking", ""))
            blocks.append(block)
        elif kind == "redacted_thinking":
            blocks.append(block)
        elif kind == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    index=len(calls),
                    function=Function(block.get("name", ""), json.dumps(block.get("input") or {})),
                )
            )
    message = Message(
        content="".join(text) or None,
        tool_calls=calls or None,
        reasoning_content="".join(thinking) or None,
        thinking_blocks=blocks or None,
    )
    return ModelResponse(
        id=data.get("id", ""),
        model=data.get("model", ""),
        choices=[Choices(message=message, finish_reason=_STOP.get(data.get("stop_reason"), "stop"))],
        usage=_usage(data.get("usage")),
    )


class StreamParser:
    """Turns one Messages stream event at a time into OpenAI-shaped chunks. Keeps state across events."""

    done = False

    def __init__(self) -> None:
        self.rid = ""
        self.model = ""
        self.usage_in: dict[str, Any] = {}
        self.output_tokens = 0
        self.tool_index_by_block: dict[int, int] = {}

    def _chunk(self, delta: Delta, finish: str | None = None) -> ModelResponseStream:
        return ModelResponseStream(self.rid, self.model, [StreamingChoices(delta=delta, finish_reason=finish)])

    def feed(self, payload: str) -> list[ModelResponseStream]:
        event = json.loads(payload)
        kind = event.get("type")
        if kind == "message_start":
            msg = event.get("message") or {}
            self.rid, self.model = msg.get("id", ""), msg.get("model", "")
            self.usage_in = msg.get("usage") or {}
            return [self._chunk(Delta(role="assistant"))]
        if kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "redacted_thinking":
                return [self._chunk(Delta(thinking_blocks=[{"type": "redacted_thinking", "data": block.get("data")}]))]
            if block.get("type") == "tool_use":
                index = len(self.tool_index_by_block)
                self.tool_index_by_block[event.get("index", 0)] = index
                return [
                    self._chunk(
                        Delta(
                            tool_calls=[
                                ToolCall(
                                    id=block.get("id", ""), index=index, function=Function(block.get("name", ""), "")
                                )
                            ]
                        )
                    )
                ]
            return []
        if kind == "content_block_delta":
            d = event.get("delta") or {}
            if d.get("type") == "text_delta":
                return [self._chunk(Delta(content=d.get("text", "")))]
            # LiteLLM's streaming shape, which ADK joins: each fragment of thinking text on its own,
            # then the signature alone in a chunk with no text. Never the whole block again.
            if d.get("type") == "thinking_delta":
                text = d.get("thinking", "")
                return [
                    self._chunk(Delta(reasoning_content=text, thinking_blocks=[{"type": "thinking", "thinking": text}]))
                ]
            if d.get("type") == "signature_delta":
                block = {"type": "thinking", "thinking": "", "signature": d.get("signature", "")}
                return [self._chunk(Delta(thinking_blocks=[block]))]
            if d.get("type") == "input_json_delta":
                index = self.tool_index_by_block.get(event.get("index", 0), 0)
                return [
                    self._chunk(
                        Delta(
                            tool_calls=[ToolCall(id="", index=index, function=Function("", d.get("partial_json", "")))]
                        )
                    )
                ]
            return []
        if kind == "message_delta":
            self.output_tokens = (event.get("usage") or {}).get("output_tokens", self.output_tokens)
            reason = (event.get("delta") or {}).get("stop_reason")
            return [self._chunk(Delta(), _STOP.get(reason, "stop"))]
        if kind == "message_stop":
            self.done = True
            usage = _usage({**self.usage_in, "output_tokens": self.output_tokens})
            return [ModelResponseStream(self.rid, self.model, [], usage=usage)]
        if kind == "error":
            err = event.get("error") or {}
            status = {
                "overloaded_error": 529,
                "rate_limit_error": 429,
                "authentication_error": 401,
                "invalid_request_error": 400,
                "permission_error": 403,
            }.get(err.get("type"), 500)
            raise exceptions.from_http(status, payload, self.model, "anthropic")
        return []


def parse_stream(lines: Iterator[str]) -> Iterator[ModelResponseStream]:
    parser = StreamParser()
    for payload in iter_sse_data(lines):
        yield from parser.feed(payload)
