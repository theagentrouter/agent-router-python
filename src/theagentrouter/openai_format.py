"""OpenAI Chat Completions: build the request and read the response and the stream."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from . import exceptions
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

# Parameters passed through to the provider. Anything else is dropped, as LiteLLM does with drop_params;
# LiteLLM-only arguments such as fallbacks also warn once (transport.IGNORED).
PASSTHROUGH = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "seed",
    "response_format",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning_effort",
    "user",
    "modalities",
    "prediction",
    "audio",
    "service_tier",
    "web_search_options",
)


def build_body(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    stream: bool,
    params: dict[str, Any],
    provider: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [_with_thinking(_plain(m), provider == "bedrock") for m in messages],
    }
    if tools:
        body["tools"] = tools
    for key in PASSTHROUGH:
        if params.get(key) is not None:
            body[key] = params[key]
    if provider in ("openai", "azure") and "max_tokens" in body:
        # OpenAI replaced max_tokens with max_completion_tokens, and its newer models refuse max_tokens,
        # on Azure OpenAI too. If both are set, max_completion_tokens wins.
        limit = body.pop("max_tokens")
        body.setdefault("max_completion_tokens", limit)
    if params.get("extra_body"):
        body.update(params["extra_body"])
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True, **(params.get("stream_options") or {})}
    return body


def _with_thinking(message: dict[str, Any], keep: bool) -> dict[str, Any]:
    """An assistant's thinking_blocks carry Anthropic's signatures, and only Claude on Bedrock takes them back here.

    The gateway reads them as content parts, and turns them into Converse's reasoning content.
    """
    blocks = message.pop("thinking_blocks", None)
    if not keep or not blocks:
        return message
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") == "thinking" and block.get("signature"):
            parts.append({"type": "thinking", "text": block.get("thinking", ""), "signature": block["signature"]})
        elif block.get("type") == "redacted_thinking" and block.get("data"):
            parts.append({"type": "redacted_thinking", "redactedContent": block["data"]})
    content = message.get("content")
    if isinstance(content, list):
        parts.extend(content)
    elif content:
        parts.append({"type": "text", "text": content})
    message["content"] = parts or content
    return message


def response_format_of(value: Any) -> Any:
    """LiteLLM's response_format=SomeModel: a pydantic class as a strict JSON schema."""
    if not (isinstance(value, type) and hasattr(value, "model_json_schema")):
        return value
    return {
        "type": "json_schema",
        "json_schema": {"schema": _strict(value.model_json_schema()), "name": value.__name__, "strict": True},
    }


def _strict(schema: Any) -> Any:
    """What OpenAI's strict mode asks of a schema: closed objects, every property required, no null default."""
    if isinstance(schema, list):
        return [_strict(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _strict(v) for k, v in schema.items() if not (k == "default" and v is None)}
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def _plain(message: Any) -> dict[str, Any]:
    """Accept dicts or this SDK's own Message objects (frameworks pass back what they received)."""
    if isinstance(message, dict):
        out = dict(message)
    else:
        out = message.to_dict()
    calls = out.get("tool_calls")
    if calls:
        out["tool_calls"] = [dict(c) if isinstance(c, dict) else c.to_dict() for c in calls]
        for c in out["tool_calls"]:
            c.pop("index", None)
    return {k: v for k, v in out.items() if v is not None}


def _tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    if not raw:
        return None
    return [
        ToolCall(
            id=c.get("id") or "",
            type=c.get("type") or "function",
            index=c.get("index", i),
            function=Function(
                name=(c.get("function") or {}).get("name") or "",
                arguments=(c.get("function") or {}).get("arguments") or "",
            ),
        )
        for i, c in enumerate(raw)
    ]


def _usage(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    prompt = raw.get("prompt_tokens") or 0
    completion = raw.get("completion_tokens") or 0
    return Usage(
        prompt,
        completion,
        raw.get("total_tokens") or prompt + completion,
        _TokenDetails(raw["prompt_tokens_details"]) if raw.get("prompt_tokens_details") is not None else None,
        _TokenDetails(raw["completion_tokens_details"]) if raw.get("completion_tokens_details") is not None else None,
    )


def parse_response(data: dict[str, Any]) -> ModelResponse:
    choices = []
    for i, c in enumerate(data.get("choices") or []):
        m = c.get("message") or {}
        choices.append(
            Choices(
                index=c.get("index", i),
                finish_reason=c.get("finish_reason") or "stop",
                message=Message(
                    role=m.get("role") or "assistant",
                    content=m.get("content"),
                    tool_calls=_tool_calls(m.get("tool_calls")),
                    reasoning_content=m.get("reasoning_content") or m.get("reasoning"),
                    thinking_blocks=m.get("thinking_blocks"),
                    refusal=m.get("refusal"),
                    annotations=m.get("annotations"),
                    audio=m.get("audio"),
                ),
                logprobs=c.get("logprobs"),
            )
        )
    return ModelResponse(
        id=data.get("id") or "",
        model=data.get("model") or "",
        choices=choices,
        usage=_usage(data.get("usage")),
        created=data.get("created") or 0,
        system_fingerprint=data.get("system_fingerprint"),
        service_tier=data.get("service_tier"),
    )


def parse_chunk(data: dict[str, Any]) -> ModelResponseStream:
    choices = []
    for i, c in enumerate(data.get("choices") or []):
        d = c.get("delta") or {}
        choices.append(
            StreamingChoices(
                index=c.get("index", i),
                finish_reason=c.get("finish_reason"),
                delta=Delta(
                    role=d.get("role"),
                    content=d.get("content"),
                    tool_calls=_tool_calls(d.get("tool_calls")),
                    refusal=d.get("refusal"),
                    reasoning_content=d.get("reasoning_content") or d.get("reasoning"),
                    thinking_blocks=d.get("thinking_blocks"),
                    annotations=d.get("annotations"),
                    audio=d.get("audio"),
                ),
                logprobs=c.get("logprobs"),
            )
        )
    usage = _usage(data["usage"]) if data.get("usage") else None
    return ModelResponseStream(
        id=data.get("id") or "",
        model=data.get("model") or "",
        choices=choices,
        usage=usage,
        created=data.get("created") or 0,
        system_fingerprint=data.get("system_fingerprint"),
        service_tier=data.get("service_tier"),
    )


def iter_sse_data(lines: Iterator[str]) -> Iterator[str]:
    """Yield the data payloads of a server-sent event stream."""
    for line in lines:
        payload = sse_payload(line)
        if payload:
            yield payload


def sse_payload(line: str) -> str | None:
    if line.startswith("data:"):
        return line[5:].strip() or None
    return None


class StreamParser:
    """Turns one SSE data payload at a time into chunks."""

    done = False

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider

    def feed(self, payload: str) -> list[ModelResponseStream]:
        if payload == "[DONE]":
            self.done = True
            return []
        data = json.loads(payload)
        if data.get("error") is not None:
            error = data["error"]
            status = error.get("status_code", error.get("status", 500)) if isinstance(error, dict) else 500
            if not isinstance(status, int) or not 400 <= status <= 599:
                status = 500
            raise exceptions.from_http(status, payload, data.get("model"), None)
        chunk = parse_chunk(data)
        if self.provider == "bedrock":
            # The gateway streams thinking text apart from its signature. LiteLLM's shape has both as
            # thinking_blocks fragments, and stream_chunk_builder joins them into a block that can go back.
            for choice in chunk.choices:
                if choice.delta.reasoning_content and not choice.delta.thinking_blocks:
                    choice.delta.thinking_blocks = [{"type": "thinking", "thinking": choice.delta.reasoning_content}]
        return [chunk]


def parse_stream(lines: Iterator[str]) -> Iterator[ModelResponseStream]:
    parser = StreamParser()
    for payload in iter_sse_data(lines):
        yield from parser.feed(payload)
        if parser.done:
            return
