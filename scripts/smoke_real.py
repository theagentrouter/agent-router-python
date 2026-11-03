# SPDX-License-Identifier: Apache-2.0
"""Call real providers. Runs only for providers whose key is in the environment; costs a few cents at most.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python3 scripts/smoke_real.py

Checks, per provider: a text reply, a tool call, a streamed tool call with usage, and a bad
model name mapped to LiteLLM's NotFoundError or BadRequestError.
Prints RESULT: PASS=n FAIL=n and exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys

import theagentrouter
from theagentrouter import exceptions

MODELS = {
    "OPENAI_API_KEY": "openai/gpt-5.4-mini",
    "ANTHROPIC_API_KEY": "anthropic/claude-haiku-5-5",
    "GEMINI_API_KEY": "gemini/gemini-2.5-flash",
    "GROQ_API_KEY": "groq/llama-3.3-70b-versatile",
    "AWS_ACCESS_KEY_ID": "bedrock/us.anthropic.claude-haiku-5-5",  # with AWS_SECRET_ACCESS_KEY and AWS_REGION
    # With AZURE_API_BASE; the name after azure/ is your deployment (AGENTROUTER_SMOKE_AZURE_MODEL).
    "AZURE_API_KEY": os.environ.get("AGENTROUTER_SMOKE_AZURE_MODEL", "azure/gpt-5.1"),
}
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
    provider = model.split("/")[0]
    try:
        theagentrouter.completion(model=f"{provider}/no-such-model-xyz", messages=ASK, max_tokens=5)
        check("bad model name raises", False, "no error")
    except (exceptions.NotFoundError, exceptions.BadRequestError) as e:
        check("bad model name raises", True, type(e).__name__)
    except Exception as e:
        check("bad model name raises", False, f"{type(e).__name__}: {e}")


def main() -> int:
    available = [m for key, m in MODELS.items() if os.environ.get(key)]
    if not available:
        print("no provider keys set; set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        return 2
    # The bundled model list has no Bedrock or Azure models: register them so a running gateway can add them.
    theagentrouter.register_model(
        {
            m: {"litellm_provider": m.split("/")[0], "mode": "chat"}
            for m in available
            if m.startswith(("bedrock/", "azure/"))
        }
    )
    for model in available:
        run(model)
    print(f"RESULT: PASS={passed} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
