"""What the app does, each written the way LiteLLM's docs show it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import litellm

ORDERS = {"A-100": "shipped", "A-200": "refunded"}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "order_status",
            "description": "The status of an order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def ask(model: str, question: str, **options: Any) -> str:
    resp = litellm.completion(model=model, messages=[{"role": "user", "content": question}], **options)
    return resp.choices[0].message.content


def stream(model: str, question: str, **options: Any) -> Iterator[str]:
    for chunk in litellm.completion(
        model=model, messages=[{"role": "user", "content": question}], stream=True, **options
    ):
        yield chunk.choices[0].delta.content or ""


async def astream(model: str, question: str, **options: Any) -> AsyncIterator[str]:
    response = await litellm.acompletion(
        model=model, messages=[{"role": "user", "content": question}], stream=True, **options
    )
    async for chunk in response:
        yield chunk.choices[0].delta.content or ""


def with_tools(model: str, question: str, **options: Any) -> tuple[str, list[dict]]:
    """One tool round: the model asks for order_status, the app answers it, the model replies."""
    messages: list[Any] = [{"role": "user", "content": question}]
    resp = litellm.completion(model=model, messages=messages, tools=TOOLS, **options)
    calls = resp.choices[0].message.tool_calls or []
    messages.append(resp.choices[0].message)
    for call in calls:
        args = json.loads(call.function.arguments)
        result = ORDERS.get(args.get("order_id") or args.get("query", ""), "unknown")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    final = litellm.completion(model=model, messages=messages, tools=TOOLS, **options)
    return final.choices[0].message.content, [json.loads(c.function.arguments) for c in calls]


def embed(model: str, texts: list[str], **options: Any) -> list[list[float]]:
    resp = litellm.embedding(model=model, input=texts, **options)
    return [item["embedding"] for item in resp.data]
