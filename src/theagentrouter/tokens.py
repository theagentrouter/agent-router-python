# SPDX-License-Identifier: Apache-2.0
"""LiteLLM's token_counter(): exact from the gateway where the provider can count, an estimate otherwise.

Agent Router counts tokens for OpenAI (``/responses/input_tokens``) and Anthropic
(``/messages/count_tokens``); the SDK asks it for those. Every other provider, a name with no
provider, and any count the gateway cannot give are estimated on this computer the way LiteLLM
does: its message rules, with tiktoken when it is installed (``pip install theagentrouter[tokens]``)
and about four characters a token without it.

``AGENTROUTER_TOKEN_COUNT=estimate`` always estimates, with no request: LiteLLM's behaviour, for
code that counts before every turn.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import httpx

from . import anthropic_format, exceptions, routing
from .openai_format import _plain

# Providers whose count the gateway can reach. Others have no count endpoint behind the local gateway.
_COUNTED = ("openai", "anthropic")
_IMAGE_TOKENS = 250  # LiteLLM's DEFAULT_IMAGE_TOKEN_COUNT; images are not downloaded to measure them
_TIMEOUT = 10.0

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, UserWarning, stacklevel=4)


def token_counter(
    model: str = "",
    custom_tokenizer: Any = None,
    text: str | list[str] | None = None,
    messages: list[Any] | None = None,
    count_response_tokens: bool | None = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    use_default_image_token_count: bool | None = False,
    default_token_count: int | None = None,
) -> int:
    """Count the tokens in ``text`` or ``messages`` (with ``tools``) for ``model``. LiteLLM's arguments."""
    if text is not None and messages is not None:
        raise ValueError("text and messages cannot both be set")
    if text is None and messages is None:
        raise ValueError("Either text or messages must be provided")
    if text is not None and (tools or tool_choice):
        raise ValueError("tools or tool_choice cannot be set if using text")
    if custom_tokenizer is not None:
        _warn_once(
            "custom_tokenizer", "theagentrouter ignores custom_tokenizer=; it counts with the gateway or tiktoken."
        )
    plain = [_plain(m) for m in messages] if messages is not None else None

    if not count_response_tokens and _ask_gateway(model):
        try:
            return _gateway_count(model, text, plain, tools, tool_choice)
        except (exceptions.APIError, httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            _warn_once(
                f"gateway:{model}",
                f"token_counter: the gateway did not count tokens for {model} ({e}); estimating instead.",
            )
    return _estimate(model, text, plain, tools, tool_choice, count_response_tokens, default_token_count)


# --- the gateway ---------------------------------------------------------------------------------


def _ask_gateway(model: str) -> bool:
    configured = routing.settings()
    if configured.get("token_count", "gateway") == "estimate":
        return False
    if configured.get("base_url"):
        return bool(model)  # a gateway you set: it decides; an error falls back to the estimate
    from . import routes

    if routes.get(model) is not None:
        return True
    try:
        full = routing.with_provider(model)
    except exceptions.BadRequestError:
        return False
    return full.split("/", 1)[0] in _COUNTED if "/" in full else False


def _gateway_count(model: str, text: Any, messages: list[dict[str, Any]] | None, tools: Any, tool_choice: Any) -> int:
    target = routing.resolve(model)
    if target.format == routing.ANTHROPIC:
        url = target.base_url.rstrip("/") + "/messages/count_tokens"
        body = _anthropic_body(target.model, text, messages, tools, tool_choice)
        headers = anthropic_format.headers(target.api_key)
        field = "input_tokens"
    else:
        url = target.base_url.rstrip("/") + "/responses/input_tokens"
        body = _responses_body(target.model, text, messages, tools, tool_choice)
        headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
        field = "input_tokens"
    headers.update({"content-type": "application/json", **target.headers})
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(url, json=body, headers=headers)
    if r.status_code >= 400:
        raise exceptions.from_http(r.status_code, r.text, model, target.provider)
    return int(r.json()[field])


def _join(text: Any) -> str:
    return "".join(t for t in text if isinstance(t, str)) if isinstance(text, list) else str(text)


def _anthropic_body(model: str, text: Any, messages: Any, tools: Any, tool_choice: Any) -> dict[str, Any]:
    system, converted = anthropic_format.convert_messages(
        messages if messages is not None else [{"role": "user", "content": _join(text)}]
    )
    body: dict[str, Any] = {"model": model, "messages": converted}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = anthropic_format.convert_tools(tools)
        choice = anthropic_format.convert_tool_choice(tool_choice)
        if choice:
            body["tool_choice"] = choice
    return body


def _responses_body(model: str, text: Any, messages: Any, tools: Any, tool_choice: Any) -> dict[str, Any]:
    """Chat Completions messages and tools, written as the Responses API's input items."""
    body: dict[str, Any] = {"model": model}
    if messages is None:
        body["input"] = _join(text)
        return body
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id", ""),
                    "output": _text(m.get("content")),
                }
            )
            continue
        content = m.get("content")
        if content:
            items.append({"role": role, "content": _parts(content, role)})
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                }
            )
    body["input"] = items
    if tools:
        body["tools"] = [
            {
                "type": "function",
                **{
                    k: v
                    for k, v in (t.get("function") or {}).items()
                    if k in ("name", "description", "parameters", "strict")
                },
            }
            for t in tools
        ]
        if isinstance(tool_choice, dict) and tool_choice.get("function"):
            body["tool_choice"] = {"type": "function", "name": tool_choice["function"].get("name")}
        elif tool_choice:
            body["tool_choice"] = tool_choice
    return body


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _parts(content: Any, role: str | None) -> Any:
    if isinstance(content, str):
        return content
    kind = "output_text" if role == "assistant" else "input_text"
    out = []
    for p in content:
        if isinstance(p, str):
            out.append({"type": kind, "text": p})
        elif p.get("type") == "text":
            out.append({"type": kind, "text": p.get("text", "")})
        elif p.get("type") == "image_url":
            image = p.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            out.append({"type": "input_image", "image_url": url})
    return out


# --- the estimate (LiteLLM's rules) --------------------------------------------------------------


def _count_function(model: str):
    try:
        import tiktoken
    except ImportError:
        _warn_once(
            "tiktoken",
            "token_counter: estimating about four characters a token; "
            "pip install theagentrouter[tokens] for LiteLLM's numbers.",
        )
        return lambda s: math.ceil(len(s) / 4)
    name = model.split("/", 1)[1] if "/" in model else model
    if "gpt-4o" in name:
        encoding = tiktoken.get_encoding("o200k_base")
    else:
        try:
            encoding = tiktoken.encoding_for_model(name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(encoding.encode(s, disallowed_special=()))


def _estimate(
    model: str,
    text: Any,
    messages: list[dict[str, Any]] | None,
    tools: Any,
    tool_choice: Any,
    count_response_tokens: bool | None,
    default_token_count: int | None,
) -> int:
    count = _count_function(model)
    if text is not None:
        return count(_join(text))
    total = 0
    for m in messages or []:
        total += 3  # tokens per message
        for key, value in m.items():
            if value is None:
                continue
            if key == "tool_calls":
                for call in value:
                    total += count(str((call.get("function") or {}).get("arguments", [])))
            elif isinstance(value, str):
                total += count(value)
                if key == "name":
                    total += 1
            elif key == "content" and isinstance(value, list):
                total += _content_list(count, value, default_token_count)
    if count_response_tokens:
        return total
    total += 3  # the reply is primed with <|start|>assistant<|message|>
    if tools:
        total += count(_functions(tools)) + 9
        if any(m.get("role") == "system" for m in messages or []):
            total -= 4
    if tool_choice == "none":
        total += 1
    elif isinstance(tool_choice, dict):
        total += 7 + count(str(tool_choice["function"]["name"]))
    return total


def _content_list(count, parts: list[Any], default_token_count: int | None) -> int:
    total = 0
    for p in parts:
        if isinstance(p, str):
            total += count(p)
        elif p.get("type") == "text":
            total += count(str(p.get("text", "")))
        elif p.get("type") == "image_url":
            total += _IMAGE_TOKENS
        elif p.get("type") == "thinking":
            total += count(str(p.get("thinking", "")))
        elif default_token_count is not None:
            total += default_token_count
    return total


def _functions(tools: list[dict[str, Any]]) -> str:
    """Tool definitions written the way OpenAI appears to read them (as LiteLLM counts them)."""
    lines = ["namespace functions {", ""]
    for tool in tools:
        fn = tool.get("function") or {}
        if fn.get("description"):
            lines.append(f"// {fn['description']}")
        params = fn.get("parameters") or {}
        if params.get("properties"):
            lines += [f"type {fn.get('name')} = (_: {{", _object(params, 0), "}) => any;"]
        else:
            lines.append(f"type {fn.get('name')} = () => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


def _object(params: dict[str, Any], indent: int) -> str:
    required = params.get("required") or []
    lines = []
    for key, props in (params.get("properties") or {}).items():
        if props.get("description"):
            lines.append(f"// {props['description']}")
        lines.append(f"{key}{'' if key in required else '?'}: {_type(props, indent)},")
    return "\n".join(" " * max(0, indent) + line for line in lines)


def _type(props: dict[str, Any], indent: int) -> str:
    kind = props.get("type")
    if kind == "string":
        return " | ".join(f'"{v}"' for v in props["enum"]) if "enum" in props else "string"
    if kind == "array":
        return f"{_type(props['items'], indent)}[]"
    if kind == "object":
        return f"{{\n{_object(props, indent + 2)}\n}}"
    if kind in ("integer", "number"):
        return " | ".join(f'"{v}"' for v in props["enum"]) if "enum" in props else "number"
    if kind in ("boolean", "null"):
        return kind
    return "any"
