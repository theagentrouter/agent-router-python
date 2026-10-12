"""A local HTTP server that answers like OpenAI Chat Completions and Anthropic Messages.

Behaviour, for both formats:
- tools offered and the last message is not a tool result: the reply calls the first tool
  with {"query": <last user text>};
- otherwise: a text reply "echo: <last text>".
- model names starting with "err-" return errors: err-429, err-401, err-500, err-context, err-budget.
Every request is recorded in FakeProvider.requests as (path, headers, body).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ERRORS = {
    "err-429": (429, "Rate limit reached for requests"),
    "err-401": (401, "Incorrect API key provided"),
    "err-500": (500, "The server had an error"),
    "err-context": (400, "This model's maximum context length is 8192 tokens"),
    "err-budget": (429, "Budget exceeded for project payments"),
}


# The thinking block the fake streams when thinking is on; it must come back exactly like this.
THINKING_BLOCK = {
    "type": "thinking",
    "thinking": "Check the refunds.",
    "signature": "YWNtZS1zaWduYXR1cmUtMDAx",
}  # base64, as Anthropic sends it


def _last_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        c = m.get("content")
        if isinstance(c, str) and c:
            return c
        if isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    return part["text"]
                if part.get("type") == "tool_result":
                    return str(part.get("content"))
    return ""


class FakeProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                owner.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                model = body.get("model", "")
                bare = model.split("/")[-1]
                if bare in ERRORS:
                    status, message = ERRORS[bare]
                    return self._json(status, {"error": {"message": message, "type": "error"}})
                if self.path.endswith(("/count_tokens", "/input_tokens")):
                    if bare == "no-count":  # a provider without a count endpoint
                        return self._json(404, {"error": {"message": "not found", "type": "error"}})
                    return self._json(200, {"object": "response.input_tokens", "input_tokens": 42})
                if self.path.endswith("/responses"):
                    text = "ok:" + str(body.get("input"))[:20]
                    return self._json(
                        200,
                        {
                            "id": "resp_1",
                            "object": "response",
                            "model": bare,
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": text}],
                                }
                            ],
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                        },
                    )
                if self.path.endswith("/messages"):
                    return self._anthropic(body)
                if self.path.endswith("/embeddings"):
                    return self._embeddings(body)
                return self._openai(body)

            # -- helpers ---------------------------------------------------------------
            def _json(self, status: int, data: dict[str, Any]) -> None:
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _sse(self, events: list[str]) -> None:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for e in events:
                    self.wfile.write(e.encode())
                    self.wfile.flush()

            def _embeddings(self, body: dict[str, Any]) -> None:
                # One small vector per input, its first number the input's length; 3 tokens per input.
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                size = int(body.get("dimensions") or 3)
                data = [
                    {"object": "embedding", "index": i, "embedding": [float(len(str(x)))] + [0.5] * (size - 1)}
                    for i, x in enumerate(inputs)
                ]
                tokens = 3 * len(inputs)
                return self._json(
                    200,
                    {
                        "object": "list",
                        "model": body["model"],
                        "data": data,
                        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                    },
                )

            def _openai(self, body: dict[str, Any]) -> None:
                if body["model"].split("/")[-1] == "stream-error":
                    return self._sse(
                        [
                            'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n',
                            'data: {"error": {"message": "Rate limit reached", "status_code": 429}}\n\n',
                            "data: [DONE]\n\n",
                        ]
                    )
                messages = body.get("messages", [])
                text = _last_text(messages)
                call = body.get("tools") and messages and messages[-1].get("role") != "tool"
                if body.get("stream"):
                    events = []
                    if call:
                        name = body["tools"][0]["function"]["name"]
                        args = json.dumps({"query": text})
                        events.append(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": {"name": name, "arguments": ""},
                                                }
                                            ],
                                        },
                                    }
                                ]
                            }
                        )
                        for i in range(0, len(args), 7):
                            events.append(
                                {
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [{"index": 0, "function": {"arguments": args[i : i + 7]}}]
                                            },
                                        }
                                    ]
                                }
                            )
                        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
                    else:
                        for word in f"echo: {text}".split(" "):
                            events.append({"choices": [{"index": 0, "delta": {"content": word + " "}}]})
                        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    events.append(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 11,
                                "completion_tokens": 7,
                                "total_tokens": 18,
                                "prompt_tokens_details": {"cached_tokens": 5},
                                "completion_tokens_details": {"reasoning_tokens": 3},
                            },
                        }
                    )
                    if body["model"].split("/")[-1] == "response-details":
                        events.insert(
                            0,
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"reasoning_content": "thinking", "refusal": "refused"},
                                        "logprobs": {"content": []},
                                    }
                                ]
                            },
                        )
                    payload = [
                        f"data: {json.dumps({'id': 'chatcmpl-1', 'model': body['model'], **e})}\n\n" for e in events
                    ] + ["data: [DONE]\n\n"]
                    return self._sse(payload)
                message: dict[str, Any] = {"role": "assistant", "content": None if call else f"echo: {text}"}
                if body["model"].split("/")[-1] == "response-details":
                    message.update(refusal="refused", reasoning_content="thinking")
                if call:
                    message["tool_calls"] = [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": body["tools"][0]["function"]["name"],
                                "arguments": json.dumps({"query": text}),
                            },
                        }
                    ]
                return self._json(
                    200,
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "tool_calls" if call else "stop",
                                "logprobs": {"content": []},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                            "prompt_tokens_details": {"cached_tokens": 5},
                            "completion_tokens_details": {"reasoning_tokens": 3},
                        },
                    },
                )

            def _anthropic(self, body: dict[str, Any]) -> None:
                messages = body.get("messages", [])
                text = _last_text(messages)
                last_is_result = messages and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in (messages[-1].get("content") or [])
                )
                call = body.get("tools") and not last_is_result
                thinking = (body.get("thinking") or {}).get("type") == "enabled"
                if thinking and last_is_result:
                    # As Anthropic: with thinking on, the tool-use turn must carry its thinking block, unchanged.
                    sent = [
                        b
                        for m in messages
                        if m.get("role") == "assistant"
                        for b in m.get("content") or []
                        if isinstance(b, dict) and b.get("type") == "thinking"
                    ]
                    if sent != [THINKING_BLOCK]:
                        return self._json(
                            400,
                            {
                                "type": "error",
                                "error": {
                                    "type": "invalid_request_error",
                                    "message": f"thinking block changed: {sent}",
                                },
                            },
                        )
                if body.get("stream"):
                    ev = [
                        (
                            "message_start",
                            {
                                "type": "message_start",
                                "message": {
                                    "id": "msg_1",
                                    "model": body["model"],
                                    "usage": {"input_tokens": 11, "output_tokens": 1},
                                },
                            },
                        )
                    ]
                    first = 0
                    if call and thinking:  # a thinking block first: two fragments, then the signature alone
                        first = 1
                        ev.append(
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                },
                            )
                        )
                        half = len(THINKING_BLOCK["thinking"]) // 2
                        for piece in (THINKING_BLOCK["thinking"][:half], THINKING_BLOCK["thinking"][half:]):
                            ev.append(
                                (
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "thinking_delta", "thinking": piece},
                                    },
                                )
                            )
                        ev.append(
                            (
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "signature_delta", "signature": THINKING_BLOCK["signature"]},
                                },
                            )
                        )
                        ev.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
                    if call:
                        args = json.dumps({"query": text})
                        ev.append(
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": first,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": "toolu_1",
                                        "name": body["tools"][0]["name"],
                                        "input": {},
                                    },
                                },
                            )
                        )
                        for i in range(0, len(args), 7):
                            ev.append(
                                (
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": first,
                                        "delta": {"type": "input_json_delta", "partial_json": args[i : i + 7]},
                                    },
                                )
                            )
                        stop = "tool_use"
                    else:
                        ev.append(
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                        )
                        for word in f"echo: {text}".split(" "):
                            ev.append(
                                (
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": word + " "},
                                    },
                                )
                            )
                        stop = "end_turn"
                    ev.append(("content_block_stop", {"type": "content_block_stop", "index": first}))
                    ev.append(
                        (
                            "message_delta",
                            {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 7}},
                        )
                    )
                    ev.append(("message_stop", {"type": "message_stop"}))
                    return self._sse([f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in ev])
                content: list[dict[str, Any]]
                if call:
                    content = [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": body["tools"][0]["name"],
                            "input": {"query": text},
                        }
                    ]
                else:
                    content = [{"type": "text", "text": f"echo: {text}"}]
                return self._json(
                    200,
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": body["model"],
                        "content": content,
                        "stop_reason": "tool_use" if call else "end_turn",
                        "usage": {"input_tokens": 11, "output_tokens": 7},
                    },
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
