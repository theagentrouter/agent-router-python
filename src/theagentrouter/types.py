"""Response types shaped like LiteLLM's, so framework adapters read the same fields."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any


class _DictAccess:
    """LiteLLM's response objects also read like dicts (`resp.get("choices")`, `msg["content"]`).

    Frameworks use both styles, ADK among them, so the drop-in has to support both.
    """

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: str) -> bool:
        return getattr(self, key, None) is not None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def model_dump(self, *, mode: str = "python", exclude_none: bool = False) -> dict[str, Any]:
        """Serialize response fields; omit model fields whose values are None when exclude_none=True.

        JSON nulls inside payload dictionaries and lists remain data, not model fields.
        """
        if mode not in ("python", "json"):
            raise ValueError("mode must be 'python' or 'json'")
        return _dump(self, mode=mode, exclude_none=exclude_none)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Any:
        """pydantic's copy, which the OpenAI Agents SDK calls on stream chunks."""
        out = copy.deepcopy(self) if deep else copy.copy(self)
        for key, value in (update or {}).items():
            setattr(out, key, value)
        return out


@dataclass
class Function(_DictAccess):
    name: str
    arguments: str


@dataclass
class ToolCall(_DictAccess):
    id: str
    function: Function
    type: str = "function"
    index: int = 0


@dataclass
class Message(_DictAccess):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    # Claude's thinking blocks with their signatures, as LiteLLM keeps them. Send the message back
    # as it is: Anthropic needs them in a tool-use turn when thinking is on.
    thinking_blocks: list[dict[str, Any]] | None = None


@dataclass
class Delta(_DictAccess):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None


@dataclass
class Choices(_DictAccess):
    message: Message
    finish_reason: str = "stop"
    index: int = 0
    logprobs: Any = None


@dataclass
class StreamingChoices(_DictAccess):
    delta: Delta
    finish_reason: str | None = None
    index: int = 0
    logprobs: Any = None


class _TokenDetails(dict):
    """Provider token details, readable as both attributes and a dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


@dataclass
class Usage(_DictAccess):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Same fields as LiteLLM's Usage. None when the provider does not report them; the OpenAI
    # Agents SDK reads them for cached and reasoning tokens.
    prompt_tokens_details: Any = None
    completion_tokens_details: Any = None


@dataclass
class ModelResponse(_DictAccess):
    id: str
    model: str
    choices: list[Choices]
    usage: Usage = field(default_factory=Usage)
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"
    # Where the call went; LiteLLM keeps the same kind of detail in _hidden_params.
    _hidden_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponseStream(_DictAccess):
    id: str
    model: str
    choices: list[StreamingChoices]
    usage: Usage | None = None
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion.chunk"


def _dump(value: Any, *, mode: str, exclude_none: bool) -> Any:
    if isinstance(value, _DictAccess):
        return {
            item.name: _dump(getattr(value, item.name), mode=mode, exclude_none=exclude_none)
            for item in fields(value)
            if not item.name.startswith("_") and not (exclude_none and getattr(value, item.name) is None)
        }
    if isinstance(value, dict):
        return {key: _dump(item, mode=mode, exclude_none=exclude_none) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump(item, mode=mode, exclude_none=exclude_none) for item in value]
    if isinstance(value, tuple):
        items = [_dump(item, mode=mode, exclude_none=exclude_none) for item in value]
        return items if mode == "json" else tuple(items)
    return copy.deepcopy(value)


def _field(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _add_thinking(blocks: list[dict[str, Any]], fragments: list[Any]) -> None:
    """Join streamed thinking fragments into whole blocks: text adds to the open block, a signature closes it."""
    for fragment in fragments:
        fragment = dict(fragment)
        if fragment.get("type") != "thinking":
            blocks.append(fragment)  # redacted_thinking arrives whole
            continue
        if not blocks or blocks[-1].get("type") != "thinking" or blocks[-1].get("signature"):
            blocks.append({"type": "thinking", "thinking": ""})
        blocks[-1]["thinking"] += fragment.get("thinking") or ""
        if fragment.get("signature"):
            blocks[-1]["signature"] = fragment["signature"]


def stream_chunk_builder(chunks: list[Any], messages: list[Any] | None = None) -> ModelResponse | None:
    """LiteLLM's stream_chunk_builder: the streamed chunks put back together as one ModelResponse.

    Each choice (by its index) is rebuilt on its own: text and reasoning are joined, tool calls are
    joined by their index, thinking blocks are kept in order, and the last finish reason wins. The
    last usage wins. ``messages`` is accepted for LiteLLM's signature.
    """
    if not chunks:
        return None
    built: dict[int, dict[str, Any]] = {}
    usage = None
    for chunk in chunks:
        usage = _field(chunk, "usage") or usage
        for choice in _field(chunk, "choices") or []:
            index = _field(choice, "index") or 0
            one = built.setdefault(
                index, {"content": [], "reasoning": [], "thinking": [], "calls": {}, "finish": "stop"}
            )
            one["finish"] = _field(choice, "finish_reason") or one["finish"]
            delta = _field(choice, "delta")
            if delta is None:
                continue
            if _field(delta, "content"):
                one["content"].append(_field(delta, "content"))
            if _field(delta, "reasoning_content"):
                one["reasoning"].append(_field(delta, "reasoning_content"))
            _add_thinking(one["thinking"], _field(delta, "thinking_blocks") or [])
            for n, call in enumerate(_field(delta, "tool_calls") or []):
                at = _field(call, "index")
                slot = one["calls"].setdefault(n if at is None else at, {"id": None, "name": None, "arguments": []})
                fn = _field(call, "function")
                slot["id"] = slot["id"] or _field(call, "id")
                if fn is not None:
                    slot["name"] = slot["name"] or _field(fn, "name")
                    slot["arguments"].append(_field(fn, "arguments") or "")
    choices = []
    for index, one in sorted(built.items()):
        tool_calls = [
            ToolCall(
                id=c["id"] or "", function=Function(name=c["name"] or "", arguments="".join(c["arguments"])), index=i
            )
            for i, c in sorted(one["calls"].items())
        ] or None
        message = Message(
            content="".join(one["content"]) or None,
            tool_calls=tool_calls,
            reasoning_content="".join(one["reasoning"]) or None,
            thinking_blocks=one["thinking"] or None,
        )
        choices.append(Choices(message=message, finish_reason=one["finish"], index=index))
    if isinstance(usage, dict):
        usage = Usage(**{k: v for k, v in usage.items() if k in Usage.__dataclass_fields__})
    first = chunks[0]
    return ModelResponse(
        id=_field(first, "id") or "",
        model=_field(first, "model") or "",
        choices=choices or [Choices(message=Message())],
        usage=usage or Usage(),
        created=_field(first, "created") or int(time.time()),
    )
