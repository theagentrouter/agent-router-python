"""Real providers through the real local gateway. Needs theagentrouter-local-gateway installed
(`make local-gateway AIGW=/path/to/aigw`) and provider keys; costs a few cents at most.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python3 scripts/smoke_local_gateway.py

Starts the local gateway (as the first call to a provider such as bedrock/ would), then checks,
per provider whose key is set: every call goes through the gateway, with a text reply, a tool
call and a streamed tool call with usage. Then another program uses the same gateway. Stops the
gateway at the end. State goes to a temporary AGENTROUTER_HOME, not your own.
Prints RESULT: PASS=n FAIL=n and exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

os.environ["AGENTROUTER_HOME"] = tempfile.mkdtemp(prefix="theagentrouter-smoke-")
for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_AUTOSTART"):
    os.environ.pop(var, None)

import theagentrouter  # noqa: E402
from theagentrouter import local_gateway  # noqa: E402

MODELS = {"openai": "openai/gpt-4o-mini", "anthropic": "anthropic/claude-haiku-4-5"}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_refunds",
            "description": "Find refunds for a day",
            "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
        },
    }
]
ASK = [{"role": "user", "content": "Use search_refunds to find refunds for yesterday."}]

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def run(model: str) -> None:
    print(model)
    try:
        r = theagentrouter.completion(
            model=model, messages=[{"role": "user", "content": "Say hello in three words."}], max_tokens=30
        )
        check("through the local gateway", r._hidden_params["route"] == "local gateway", r._hidden_params["sent_to"])
        check("text reply", bool(r.choices[0].message.content), r.choices[0].message.content or "")
    except Exception as e:
        check("text reply", False, f"{type(e).__name__}: {e}")
    try:
        r = theagentrouter.completion(model=model, messages=ASK, tools=TOOLS, tool_choice="required", max_tokens=200)
        call = (r.choices[0].message.tool_calls or [None])[0]
        check(
            "tool call",
            bool(call) and call.function.name == "search_refunds" and "day" in json.loads(call.function.arguments),
            call.function.arguments if call else "none",
        )
    except Exception as e:
        check("tool call", False, f"{type(e).__name__}: {e}")
    try:
        args, usage = "", None
        for chunk in theagentrouter.completion(
            model=model, messages=ASK, tools=TOOLS, tool_choice="required", max_tokens=200, stream=True
        ):
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                args += chunk.choices[0].delta.tool_calls[0].function.arguments or ""
            usage = chunk.usage or usage
        check(
            "streamed tool call with usage",
            "day" in json.loads(args or "{}") and bool(usage and usage.total_tokens),
            f"{args} usage={usage.total_tokens if usage else None}",
        )
    except Exception as e:
        check("streamed tool call with usage", False, f"{type(e).__name__}: {e}")


def main() -> int:
    if not local_gateway.installed():
        print("theagentrouter-local-gateway is not installed; run `make local-gateway AIGW=/path/to/aigw`")
        return 1
    providers = [p for p in MODELS if p in local_gateway.available(dict(os.environ))]
    if not providers:
        print("set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return 1
    try:
        state = local_gateway.ensure(MODELS[providers[0]])
        for provider in providers:
            run(MODELS[provider])
        print("another program")
        code = (
            "import theagentrouter; "
            f"r = theagentrouter.completion(model={MODELS[providers[0]]!r}, "
            "messages=[{'role': 'user', 'content': 'Say hi.'}], max_tokens=10); "
            "print(r._hidden_params['route'])"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        check("uses the same gateway", out.stdout.strip() == "local gateway", out.stdout.strip() or out.stderr[-200:])
        check("same gateway process", (local_gateway.running() or {}).get("pid") == state["pid"])
    finally:
        local_gateway.stop()
    print(f"RESULT: PASS={passed} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
