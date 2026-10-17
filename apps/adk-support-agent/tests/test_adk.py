import json
import sqlite3
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx
import pytest
from conftest import SDK

from adk_agent.agent import support_app

PROJECT = Path(__file__).resolve().parents[1]
STEPS = [
    ("tickets_get_ticket", {"ticket_id": "ACME-101"}),
    ("search_knowledge", {"query": "annual refund"}),
    ("save_report", {"filename": "resolution.md", "content": "Review refund [billing.md:L7]"}),
    ("tickets_add_ticket_note", {"ticket_id": "ACME-101", "body": "Resolution drafted from KB."}),
    ("tickets_get_ticket", {"ticket_id": "ACME-101"}),
]


def test_adapter_uses_the_sdk_under_test():
    from google.adk.models import lite_llm

    # ADK imports its SDK lazily when capabilities are first requested.
    _ = lite_llm.LiteLlm(model="openai/gpt-4.1-mini").capabilities
    if SDK == "theagentrouter":
        assert sys.modules["litellm"].__name__ == "theagentrouter.compat.litellm"
        assert lite_llm.acompletion.__module__.startswith("theagentrouter")
        with pytest.raises(PackageNotFoundError):
            version("litellm")
    else:
        assert sys.modules["litellm"].__name__ == "litellm"
        assert lite_llm.acompletion.__module__.startswith("litellm")
        assert version("litellm")


def reply(body, step, anthropic):
    call = STEPS[step] if step < len(STEPS) else None
    if anthropic:
        blocks = (
            [{"type": "tool_use", "id": f"call_{step}", "name": call[0], "input": call[1]}]
            if call
            else [
                {"type": "text", "text": "Support response drafted and ticket note saved."},
            ]
        )
        return {
            "id": f"msg_{step}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": blocks,
            "stop_reason": "tool_use" if call else "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }
    message = {
        "role": "assistant",
        "content": None if call else "Support response drafted and ticket note saved.",
    }
    if call:
        message["tool_calls"] = [
            {
                "id": f"call_{step}",
                "type": "function",
                "function": {
                    "name": call[0],
                    "arguments": json.dumps(call[1]),
                },
            }
        ]
    return {
        "id": f"chat_{step}",
        "object": "chat.completion",
        "created": 1,
        "model": body["model"],
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if call else "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def sse(data, anthropic, include_usage=False):
    def event(name, payload):
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    if anthropic:
        block = data["content"][0]
        start = {**data, "content": [], "stop_reason": None}
        stream = event("message_start", {"type": "message_start", "message": start})
        initial = (
            {**block, "input": {}} if block["type"] == "tool_use" else {"type": "text", "text": ""}
        )
        stream += event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": initial},
        )
        delta = (
            {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            if block["type"] == "tool_use"
            else {
                "type": "text_delta",
                "text": block["text"],
            }
        )
        stream += event(
            "content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}
        )
        stream += event("content_block_stop", {"type": "content_block_stop", "index": 0})
        stream += event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": data["stop_reason"]},
                "usage": {"output_tokens": 5},
            },
        )
        return stream + event("message_stop", {"type": "message_stop"})
    choice = data["choices"][0]
    delta = dict(choice["message"])
    if delta.get("tool_calls"):
        delta["tool_calls"][0]["index"] = 0
    first = {
        **data,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    last = {
        **data,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
    }
    chunks = [first, last]
    if include_usage:  # OpenAI sends usage on one more chunk with no choices, only when asked
        chunks.append(
            {**data, "object": "chat.completion.chunk", "choices": [], "usage": data["usage"]}
        )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.mark.parametrize("model", ["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4-5-20250929"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_adk_mcp_kb_and_report_round_trip(model, streaming, tmp_path, monkeypatch):
    requests = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")

    async def offline_send(client, request, **kwargs):
        body = json.loads(request.content)
        assert request.url.host == "fixture.test"
        anthropic = request.url.path.endswith("/messages")
        assert anthropic == model.startswith("anthropic/")
        data = reply(body, len(requests), anthropic)
        requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                request=request,
                content=sse(
                    data, anthropic, (body.get("stream_options") or {}).get("include_usage")
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, request=request, json=data)

    monkeypatch.setattr(httpx.AsyncClient, "send", offline_send)
    database = tmp_path / "tickets.db"
    async with support_app(
        PROJECT,
        model=model,
        api_base="http://fixture.test/v1",
        database=database,
        output_directory=tmp_path / "outputs",
    ) as app:
        result = await app.run(
            "Resolve ACME-101 and save a cited report and internal note", streaming=streaming
        )
        assert result.tool_calls == [name for name, _ in STEPS]
        assert "ticket note saved" in result.answer
        assert app.kb.validate_citations((tmp_path / "outputs/resolution.md").read_text()) == 1
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT body FROM notes").fetchall() == [
                ("Resolution drafted from KB.",)
            ]
        assert len(requests) == 6
        assert requests[0]["tools"]
