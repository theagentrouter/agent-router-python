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
import re
import warnings
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
        "reasoning_effort",
        "response_format",
    )
)

# What each Claude model takes, as in LiteLLM's model list. The SDK's list has no field for it. A name on
# neither is left for the API to answer, as LiteLLM leaves it.
_CLAUDE = re.compile(r"claude-([a-z]+)-(\d)(?!\d)(?:[-._](\d{1,2})(?!\d))?")
# Temperature other than 1, top_p and top_k are refused; effort "xhigh" is taken.
_NEWER = frozenset(
    (
        ("opus", 4, 7),
        ("opus", 4, 8),
        ("opus", 5, 0),
        ("opus", 5, 5),
        ("sonnet", 5, 0),
        ("sonnet", 5, 5),
        ("haiku", 5, 5),
        ("fable", 5, 0),
        ("fable", 5, 1),
        ("mythos", 5, 0),
        ("mythos", 5, 1),
    )
)
# A forced tool call (tool_choice "required" or a named tool) is refused.
_NO_FORCED_TOOL = frozenset((("opus", 5, 5), ("sonnet", 5, 5), ("fable", 5, 1), ("mythos", 5, 1)))
# Thinking cannot be turned off.
_ALWAYS_THINKS = frozenset((("opus", 5, 5), ("sonnet", 5, 5)))
# speed= is taken.
_TAKES_SPEED = frozenset((("opus", 4, 6), ("opus", 4, 7), ("opus", 4, 8), ("opus", 5, 0), ("opus", 5, 5)))

# reasoning_effort as a token budget, on a model before adaptive thinking.
_BUDGETS = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096, "xhigh": 8192, "max": 16384}
_MIN_BUDGET = 1024

# Anthropic refuses these in a structured output's schema. As LiteLLM does, they become a note in the description.
_SCHEMA_NOTES = (
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "prefixItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "if",
    "then",
    "else",
    "not",
)
# Anthropic refuses metadata.user_id when it holds an email address or a phone number; LiteLLM leaves those out.
_NOT_A_USER_ID = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$|^\+?[\d\s()-]{7,}$")

# What LiteLLM refuses for a Bedrock model, unless drop_params is set. The gateway would ignore them.
_COHERE_REFUSES = ("logit_bias", "logprobs", "top_logprobs")
_BEDROCK_REFUSES = (
    *_COHERE_REFUSES,
    "n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "modalities",
    "prediction",
    "audio",
)

_warned: set[str] = set()

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
            image = part.get("image_url") or ""
            url = image.get("url", "") if isinstance(image, dict) else image
            if url.startswith("data:") and ";base64," in url:
                media, data = url[5:].split(";base64,", 1)
                source = {"type": "base64", "media_type": media, "data": data}
            else:
                source = {"type": "url", "url": url}
            blocks.append(_cached({"type": "image", "source": source}, part))
        elif kind in ("document", "image"):
            blocks.append(_cached(dict(part), part))
        elif kind == "file":
            file = part.get("file") or {}
            data = file.get("file_data") or ""
            if data.startswith("data:") and ";base64," in data:
                media, value = data[5:].split(";base64,", 1)
                source = {"type": "base64", "media_type": media, "data": value}
            elif data.startswith(("https://", "http://")):
                source = {"type": "url", "url": data}
            else:
                raise exceptions.UnsupportedParamsError(
                    "Claude file parts require a data URL or an HTTP URL in file_data"
                )
            blocks.append(_cached({"type": "document", "source": source}, part))
        else:
            raise exceptions.UnsupportedParamsError(f"Claude does not support content part {kind!r}; it was not sent")
    return blocks


def _tool_content(content: Any) -> str | list[dict[str, Any]]:
    """A tool result's content: text as a string; images and documents as blocks beside the text.

    cache_control stays off the inner blocks: the caller puts it on the tool_result itself.
    """
    if isinstance(content, list):
        if all(isinstance(p, dict) and p.get("type") == "text" for p in content):
            return _text_of(content)
        return [{k: v for k, v in b.items() if k != "cache_control"} for b in _user_blocks(content)]
    return _text_of(content) if isinstance(content, str) or content is None else str(content)


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
    except (ValueError, TypeError) as error:
        raise exceptions.BadRequestError(
            "Tool-call arguments must be valid JSON; the tool call was not sent"
        ) from error


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
                "content": _tool_content(m.get("content")),
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
        if t.get("type") not in (None, "function", "custom"):
            out.append(dict(t))  # one of Anthropic's own tools, such as web search
            continue
        f = t.get("function") or t
        tool = {
            "name": f.get("name"),
            "description": f.get("description", ""),
            "input_schema": f.get("parameters") or f.get("input_schema") or {"type": "object", "properties": {}},
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
        if choice.get("type") in ("auto", "any", "none"):
            return {"type": choice["type"]}
    return None


def _claude(model: str) -> tuple[str, int, int] | None:
    """A Claude model's family and version: ``us.anthropic.claude-opus-4-6-v1`` is ("opus", 4, 6)."""
    if "claude-mythos-preview" in model:
        return ("mythos", 5, 0)
    found = _CLAUDE.search(model)
    return (found[1], int(found[2]), int(found[3] or 0)) if found else None


def _adaptive(model: str) -> bool:
    """Claude 4.6 and later think adaptively: the model decides how much, and effort says how hard."""
    claude = _claude(model)
    return claude is not None and claude[1:] >= (4, 6)


def _refuses_sampling(model: str) -> bool:
    """Whether this Claude model refuses temperature other than 1, top_p and top_k."""
    claude = _claude(model)
    return claude is not None and (claude in _NEWER or claude[0] == "fable")


def _takes_xhigh(model: str) -> bool:
    from . import models

    key = models._key(model, "anthropic")
    return _claude(model) in _NEWER or "xhigh" in ((models.model_cost.get(key) or {}).get("reasoning_efforts") or ())


def _always_thinks(model: str) -> bool:
    claude = _claude(model)
    return claude is not None and (claude in _ALWAYS_THINKS or claude[0] in ("fable", "mythos"))


def _asked_max_tokens(params: dict[str, Any]) -> int | None:
    """The caller's limit. As in LiteLLM, max_completion_tokens wins, and 0 is a limit (a cache pre-warm)."""
    value = params.get("max_completion_tokens")
    return params.get("max_tokens") if value is None else value


def _max_output(model: str) -> int:
    """The default max_tokens: the model's own limit, as in LiteLLM, or 4096 for a name not on the model list."""
    from . import models

    key = models._key(model, "anthropic")
    info = models.model_cost.get(key) or {} if key else {}
    return info.get("max_output_tokens") or info.get("max_tokens") or DEFAULT_MAX_TOKENS


def _refused_sampling(model: str, params: dict[str, Any], gateway_decides: bool) -> list[str]:
    """The sampling options to leave out. Without drop_params they raise, as in LiteLLM."""
    refused = [
        key
        for key in ("temperature", "top_p", "top_k")
        if _refuses_sampling(model) and params.get(key) is not None and not (key == "temperature" and params[key] == 1)
    ]
    if refused and not params.get("drop_params"):
        if not gateway_decides:
            raise exceptions.UnsupportedParamsError(
                f"{model} does not support "
                + ", ".join(f"{key}={params[key]!r}" for key in refused)
                + ": it takes only temperature=1, and no top_p or top_k. Set drop_params=True to drop them.",
                model=model,
                llm_provider="anthropic",
            )
        return []
    return refused


def _downgrades_forced_tool(model: str, params: dict[str, Any], gateway_decides: bool) -> bool:
    """Whether a forced tool call becomes "auto". Without drop_params it raises, as in LiteLLM."""
    if _claude(model) not in _NO_FORCED_TOOL:
        return False
    if params.get("drop_params"):
        _warn_once(
            "forced_tool", f"{model} does not support a forced tool call; drop_params=True sends tool_choice 'auto'."
        )
        return True
    if not gateway_decides:
        raise exceptions.UnsupportedParamsError(
            f"{model} does not support a forced tool call (tool_choice 'required' or a named tool). Use "
            "tool_choice='auto' and say in the prompt when to call the tool, or set drop_params=True to use "
            "'auto'.",
            model=model,
            llm_provider="anthropic",
        )
    return False


def _warn_once(name: str, message: str) -> None:
    if name not in _warned:
        _warned.add(name)
        warnings.warn(message, UserWarning, stacklevel=5)


def _fit_budget(budget: int, max_tokens: int | None) -> dict[str, Any] | None:
    """A token budget below max_tokens, as Claude needs; None when not even the smallest budget fits."""
    if max_tokens is None:
        return {"type": "enabled", "budget_tokens": budget}
    if max_tokens <= _MIN_BUDGET:
        return None
    return {"type": "enabled", "budget_tokens": max_tokens - 1 if budget >= max_tokens else budget}


def _budget_to_effort(model: str, budget: Any) -> str:
    """A token budget as an effort level, for the models that refuse a budget. LiteLLM's steps."""
    try:
        budget = int(budget)
    except (TypeError, ValueError):
        budget = 0
    if budget >= _BUDGETS["xhigh"] and _takes_xhigh(model):
        return "xhigh"
    if budget >= _BUDGETS["high"]:
        return "high"
    return "medium" if budget >= _BUDGETS["medium"] else "low"


def _thinking(model: str, params: dict[str, Any], *, bedrock: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    """The thinking block and the effort level to send, by LiteLLM's rules for each model.

    thinking wins over reasoning_effort, which then only sets the effort level of a model that has one.
    """
    effort = params.get("reasoning_effort")
    if isinstance(effort, dict):  # as the OpenAI Responses API writes it
        effort = effort.get("effort")
    thinking, level = None, None
    adaptive = _adaptive(model)
    if isinstance(effort, str) and effort != "none":
        if effort not in _BUDGETS:
            raise _invalid_effort(model, effort)
        if adaptive:
            # Without display, Anthropic answers with empty thinking blocks.
            thinking = {"type": "adaptive", "display": "summarized"}
            level = "low" if effort == "minimal" else effort
        else:
            thinking = {"type": "enabled", "budget_tokens": _BUDGETS[effort]}
            asked = _asked_max_tokens(params)
            if asked and thinking["budget_tokens"] >= asked:
                # Claude needs max_tokens above the budget; keep the caller's max_tokens and think less.
                if asked <= _MIN_BUDGET:
                    raise exceptions.BadRequestError(
                        f"reasoning_effort needs max_tokens above {_MIN_BUDGET} for this model; got {asked}",
                        model=model,
                        llm_provider="anthropic",
                    )
                thinking["budget_tokens"] = asked - 1
    wanted = params.get("thinking")
    if wanted is True:
        wanted = {"type": "enabled", "budget_tokens": _BUDGETS["medium"]}
    if isinstance(wanted, dict):
        kind = wanted.get("type")
        if kind == "adaptive" and not adaptive:
            # A model before 4.6 refuses adaptive thinking, so it gets the "medium" budget, below max_tokens.
            thinking = _fit_budget(_BUDGETS["medium"], _asked_max_tokens(params))
        elif kind == "enabled" and adaptive and _claude(model)[1:] != (4, 6):
            # 4.7 and later refuse a fixed budget; 4.6 still takes one.
            shown = {"display": wanted["display"]} if wanted.get("display") in ("summarized", "omitted") else {}
            thinking = {"type": "adaptive", **shown}
            level = level or _budget_to_effort(model, wanted.get("budget_tokens") or 0)
        elif kind == "disabled" and _always_thinks(model):
            thinking = None  # these refuse "disabled"; leaving it out is what Anthropic asks for
        else:
            thinking = wanted
    if level == "xhigh" and not _takes_xhigh(model):
        if bedrock and _claude(model) == ("opus", 4, 6):
            level = "max"  # Bedrock's Opus 4.6 stops at max, and LiteLLM lowers it there
        else:
            raise exceptions.BadRequestError(
                f"effort='xhigh' is not supported by this model. Got model: {model}",
                model=model,
                llm_provider="anthropic",
            )
    return thinking, level


def _claude_schema(schema: Any) -> Any:
    """A JSON schema as Anthropic's structured output takes it, changed as LiteLLM and Anthropic's own SDK change it.

    What it refuses goes in the description, oneOf becomes anyOf, and every object closes with
    additionalProperties false. Schemas that refer to $defs stay that way.
    """
    if not isinstance(schema, dict):
        return schema
    notes = [f"{key}: {json.dumps(schema[key])}" for key in _SCHEMA_NOTES if key in schema and schema[key] is not False]
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SCHEMA_NOTES:
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            out[key] = {name: _claude_schema(item) for name, item in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _claude_schema(value)
        elif key in ("anyOf", "oneOf") and isinstance(value, list):
            out["anyOf"] = [*out.get("anyOf", []), *(_claude_schema(item) for item in value)]
        elif key == "allOf" and isinstance(value, list):
            out[key] = [_claude_schema(item) for item in value]
        else:
            out[key] = value
    if notes:
        note = "Note: " + ", ".join(notes) + "."
        out["description"] = f"{schema['description']} {note}" if schema.get("description") else note
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out


def with_betas(headers: dict[str, str], body: dict[str, Any]) -> dict[str, str]:
    """The anthropic-beta values a Messages request needs, as LiteLLM adds them, beside the caller's own."""
    names = []
    if body.get("compaction") is not None:
        names.append("compact-2026-09-04")
    management = body.get("context_management")
    edits = (management.get("edits") or ()) if isinstance(management, dict) else ()
    if any(edit.get("type") == "compact_20260112" for edit in edits):
        names.append("compact-2026-01-12")
    if any(edit.get("type") != "compact_20260112" for edit in edits):
        names.append("context-management-2025-06-27")
    if body.get("output_format") is not None or (body.get("output_config") or {}).get("format") is not None:
        names.append("structured-outputs-2025-11-13")
    if body.get("speed") == "fast":
        names.append("fast-mode-2026-02-01")
    if body.get("safeguards") is not None:
        names.append("dangerous-tool-use-2026-09-03")
    if not names:
        return headers
    out = {k: v for k, v in headers.items() if k.lower() != "anthropic-beta"}
    given = [piece.strip() for k, v in headers.items() if k.lower() == "anthropic-beta" for piece in v.split(",")]
    out["anthropic-beta"] = ",".join(dict.fromkeys([*filter(None, given), *names]))
    return out


def _invalid_effort(model: str, effort: Any) -> exceptions.BadRequestError:
    return exceptions.BadRequestError(
        f"Invalid reasoning_effort: {effort!r}. Must be one of: {', '.join(_BUDGETS)}, none",
        model=model,
        llm_provider="anthropic",
    )


def messages_params(
    model: str, params: dict[str, Any], max_tokens: int, gateway_decides: bool = False
) -> dict[str, Any]:
    """LiteLLM's rules for anthropic.messages: Anthropic's own fields, changed only where a model refuses them."""
    out = dict(params)
    effort = out.pop("reasoning_effort", None)
    if isinstance(out.get("metadata"), dict):
        out["metadata"] = {k: v for k, v in out["metadata"].items() if k == "user_id"}
    if "claude" not in model:
        if effort is not None:
            _warn_once("messages_effort", f"reasoning_effort is not sent for {model}, which is not a Claude name.")
        return out
    if not isinstance(out.get("output_config"), dict):
        out.pop("output_config", None)
    if out.get("speed") is not None and _claude(model) not in _TAKES_SPEED:
        if not params.get("drop_params") and not gateway_decides:
            raise exceptions.UnsupportedParamsError(
                f"{model} does not support speed={out['speed']!r}. Set drop_params=True to drop it.",
                model=model,
                llm_provider="anthropic",
            )
        if params.get("drop_params"):
            del out["speed"]
    adaptive = _adaptive(model)
    if isinstance(effort, str):
        if effort == "none":
            out.pop("thinking", None)
            out.pop("output_config", None)
        elif effort not in _BUDGETS:
            raise _invalid_effort(model, effort)
        else:
            mapped = (
                {"type": "adaptive", "display": "summarized"} if adaptive else _fit_budget(_BUDGETS[effort], max_tokens)
            )
            if mapped:
                # The caller's thinking and output_config win.
                out["thinking"] = out.get("thinking") or mapped
                if adaptive:
                    level = "low" if effort == "minimal" else effort
                    if level == "xhigh" and not _takes_xhigh(model):
                        raise exceptions.BadRequestError(
                            f"effort='xhigh' is not supported by this model. Got model: {model}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    out["output_config"] = {"effort": level, **(out.get("output_config") or {})}
    thinking = out.get("thinking")
    kind = thinking.get("type") if isinstance(thinking, dict) else None
    if kind == "disabled" and _always_thinks(model):
        out.pop("thinking")  # these refuse "disabled"; leaving it out is what Anthropic asks for
    elif kind == "enabled" and adaptive and _claude(model)[1:] != (4, 6):
        shown = {"display": thinking["display"]} if thinking.get("display") in ("summarized", "omitted") else {}
        out["thinking"] = {"type": "adaptive", **shown}
        level = _budget_to_effort(model, thinking.get("budget_tokens") or 0)
        out["output_config"] = {"effort": level, **(out.get("output_config") or {})}
    elif not adaptive:
        config = out.get("output_config")
        level = config.get("effort") if isinstance(config, dict) else None
        if level is not None or kind == "adaptive":
            if _claude(model) == ("opus", 4, 5) and (kind != "adaptive" or level in (None, "low", "medium", "high")):
                # Opus 4.5 takes an effort level, and refuses adaptive thinking.
                if kind == "adaptive":
                    out.pop("thinking")
            else:
                level = level or "medium"
                if level not in _BUDGETS:
                    raise _invalid_effort(model, level)
                out["thinking"] = _fit_budget(_BUDGETS[level], max_tokens)
                if isinstance(config, dict):
                    rest = {k: v for k, v in config.items() if k != "effort"}
                    out["output_config"] = rest or None
        thinking = out.get("thinking")
        config = out.get("output_config")
        thinks = isinstance(thinking, dict) and thinking.get("type") == "enabled"
        if out.get("temperature") not in (None, 1) and (
            thinks or (isinstance(config, dict) and config.get("effort") is not None)
        ):
            del out["temperature"]  # Anthropic takes only temperature=1 while thinking
    for key in _refused_sampling(model, out, gateway_decides):
        out.pop(key, None)
    return out


def bedrock_params(model: str, params: dict[str, Any], gateway_decides: bool = False) -> dict[str, Any]:
    """LiteLLM's rules for a Bedrock call, in the OpenAI format the gateway turns into Bedrock's Converse.

    The gateway reads temperature, top_p, max_tokens, tool_choice and thinking from that format, and
    has no field for an effort level, top_k, parallel_tool_calls or response_format.
    """
    out = dict(params)
    cohere = "cohere." in model
    refused = [
        key
        for key in (_COHERE_REFUSES if cohere else _BEDROCK_REFUSES)
        if params.get(key) is not None and not (key == "n" and params[key] == 1)
    ]
    if refused and not params.get("drop_params"):
        if not gateway_decides:
            raise exceptions.UnsupportedParamsError(
                f"Bedrock does not support these parameters: {', '.join(refused)}. Set drop_params=True to drop them.",
                model=model,
                llm_provider="bedrock",
            )
    else:
        for key in refused:
            del out[key]
    if out.get("response_format") is not None:
        _warn_once("bedrock_format", "The gateway cannot send response_format to Bedrock; it is not sent.")
    if out.get("tool_choice") == "none":
        # The gateway would read "none" as the name of a tool to force.
        if not params.get("drop_params") and not cohere:
            raise exceptions.UnsupportedParamsError(
                "Bedrock does not support tool_choice='none'. Set drop_params=True to drop it.",
                model=model,
                llm_provider="bedrock",
            )
        del out["tool_choice"]
    if out.get("parallel_tool_calls") is False:
        _warn_once("bedrock_parallel", "The gateway cannot send parallel_tool_calls=False to Bedrock; it is not sent.")
    if "claude" not in model and "arn:" not in model:  # an inference profile's ARN may hide a Claude model
        return out
    for key in _refused_sampling(model, params, gateway_decides):
        out.pop(key, None)
    choice = out.get("tool_choice")
    forced = choice == "required" or (
        isinstance(choice, dict) and ((choice.get("function") or {}).get("name") or choice.get("name"))
    )
    if forced and _downgrades_forced_tool(model, params, gateway_decides):
        out["tool_choice"] = "auto"
    if out.get("top_k") is not None:
        _warn_once("bedrock_top_k", "The gateway cannot send top_k to Bedrock; it is not sent.")
    thinking, level = _thinking(model, params, bedrock=True)
    out.pop("reasoning_effort", None)
    if level:
        _warn_once(
            "bedrock_effort",
            "The gateway cannot send an effort level to Bedrock, so Claude uses its default; thinking is still on.",
        )
    if thinking:
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int):
            thinking = {**thinking, "budget_tokens": max(budget, _MIN_BUDGET)}  # Bedrock's minimum, as LiteLLM sets
            if _asked_max_tokens(params) is None:
                out["max_tokens"] = thinking["budget_tokens"] + DEFAULT_MAX_TOKENS
        out["extra_body"] = {"thinking": thinking, **(out.get("extra_body") or {})}
    return out


def build_body(
    model: str,
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    stream: bool,
    params: dict[str, Any],
    gateway_decides: bool = False,
) -> dict[str, Any]:
    # gateway_decides: a gateway you set owns its model names, so what a name refuses is its call, not ours.
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
    refused = _refused_sampling(model, params, gateway_decides)
    system, converted = convert_messages(messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": converted,
        "max_tokens": _asked_max_tokens(params),
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = convert_tools(tools)
        choice = convert_tool_choice(params.get("tool_choice"))
        if choice and choice["type"] in ("any", "tool") and _downgrades_forced_tool(model, params, gateway_decides):
            choice = {"type": "auto"}
        if params.get("parallel_tool_calls") is not None:
            choice = choice or {"type": "auto"}
            if choice["type"] != "none":
                choice["disable_parallel_tool_use"] = not params["parallel_tool_calls"]
        if choice:
            body["tool_choice"] = choice
    for key in ("temperature", "top_p", "top_k"):
        if params.get(key) is not None and key not in refused:
            body[key] = params[key]
    stop = params.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    user = params.get("user")
    if isinstance(user, str) and user and not _NOT_A_USER_ID.match(user):
        body["metadata"] = {"user_id": user}
    thinking, level = _thinking(model, params)
    if thinking:
        body["thinking"] = thinking
    if level:
        body["output_config"] = {"effort": level}
    if body["max_tokens"] is None:
        budget = thinking.get("budget_tokens") if thinking else None
        # Claude needs room for the answer after thinking.
        body["max_tokens"] = budget + DEFAULT_MAX_TOKENS if isinstance(budget, int) else _max_output(model)
    response_format = params.get("response_format")
    if response_format:
        schema = response_format.get("response_schema") or (response_format.get("json_schema") or {}).get("schema")
        if schema is not None:
            body["output_config"] = {
                **(body.get("output_config") or {}),
                "format": {"type": "json_schema", "schema": _claude_schema(schema)},
            }
        elif response_format.get("type") == "json_object" and "json_object" not in _warned:
            _warned.add("json_object")
            warnings.warn(
                "Claude has no JSON mode without a schema: response_format json_object is not applied. "
                "Pass a JSON schema (type json_schema) to get JSON from Claude.",
                UserWarning,
                stacklevel=4,
            )
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
    details = _TokenDetails(cached_tokens=cached, cache_creation_tokens=raw.get("cache_creation_input_tokens") or 0)
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
        self.tool_arguments_seen: set[int] = set()

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
                if d.get("partial_json"):
                    self.tool_arguments_seen.add(event.get("index", 0))
                index = self.tool_index_by_block.get(event.get("index", 0), 0)
                return [
                    self._chunk(
                        Delta(
                            tool_calls=[ToolCall(id="", index=index, function=Function("", d.get("partial_json", "")))]
                        )
                    )
                ]
            return []
        if kind == "content_block_stop":
            block = event.get("index", 0)
            if block in self.tool_index_by_block and block not in self.tool_arguments_seen:
                return [
                    self._chunk(
                        Delta(
                            tool_calls=[
                                ToolCall(id="", index=self.tool_index_by_block[block], function=Function("", "{}"))
                            ]
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
