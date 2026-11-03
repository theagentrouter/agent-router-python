# SPDX-License-Identifier: Apache-2.0
"""A real Google ADK agent on real providers, through the SDK, with LiteLLM not installed.

Needs google-adk in the environment and OPENAI_API_KEY and/or ANTHROPIC_API_KEY; costs a few cents.

    python scripts/smoke_adk_real.py

For each model, with and without streaming: the agent must call search_refunds and answer with its result.
Prints RESULT: PASS=n FAIL=n and exits non-zero on any failure.
"""

# ruff: noqa: E402  (install_as_litellm must run before ADK is imported)

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys

LITELLM_INSTALLED = importlib.util.find_spec("litellm") is not None  # checked before install_as_litellm adds one

from theagentrouter.compat import install_as_litellm

install_as_litellm()

import google.adk
from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

MODELS = {"OPENAI_API_KEY": "openai/gpt-5.4-mini", "ANTHROPIC_API_KEY": "anthropic/claude-haiku-5-5"}
INSTRUCTION = (
    'Answer questions about refunds. Use search_refunds. Pass the day as the user wrote it, for example "yesterday".'
)

calls: list[str] = []
passed = failed = 0


def search_refunds(day: str) -> dict:
    """Find refunds for a day."""
    calls.append(day)
    return {"day": day, "refunds": [{"id": "r-1", "amount": 42}]}


def check(name: str, ok: bool, detail: str) -> None:
    global passed, failed
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")


async def run(model: str, stream: bool) -> None:
    calls.clear()
    agent = Agent(name="refunds", model=LiteLlm(model=model), instruction=INSTRUCTION, tools=[search_refunds])
    runner = InMemoryRunner(agent=agent, app_name="smoke")
    session = await runner.session_service.create_session(app_name="smoke", user_id="u")
    question = types.Content(role="user", parts=[types.Part(text="How much was refunded yesterday?")])
    config = RunConfig(streaming_mode=StreamingMode.SSE if stream else StreamingMode.NONE)
    text = ""
    async for event in runner.run_async(user_id="u", session_id=session.id, new_message=question, run_config=config):
        if event.content and event.content.parts and not event.partial:
            text += "".join(p.text or "" for p in event.content.parts)
    check(f"{model} stream={stream}", bool(calls) and "42" in text, f"tool calls={calls} reply={text.strip()[:80]!r}")


async def main() -> int:
    print(f"google-adk {google.adk.__version__}, LiteLLM installed: {LITELLM_INSTALLED}")
    models = [m for key, m in MODELS.items() if os.environ.get(key)]
    if not models:
        print("no provider keys set; set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        return 2
    for model in models:
        for stream in (False, True):
            try:
                await run(model, stream)
            except Exception as e:
                check(f"{model} stream={stream}", False, f"{type(e).__name__}: {e}")
    print(f"RESULT: PASS={passed} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
