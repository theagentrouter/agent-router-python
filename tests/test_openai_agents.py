"""A real OpenAI Agents SDK agent on theagentrouter, over HTTP to the fake provider.

The Agents SDK's LitellmModel calls `litellm.acompletion`; `install_as_litellm()` answers that import.
Skipped without openai-agents. Run in an environment that has openai-agents and not litellm.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from .fake_provider import FakeProvider

pytestmark = pytest.mark.skipif(importlib.util.find_spec("agents") is None, reason="openai-agents not installed")


@pytest.fixture
def fake(monkeypatch):
    server = FakeProvider()
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    yield server
    server.close()


def _agent(model: str, base_url: str):
    if importlib.util.find_spec("litellm") is not None and "litellm" not in sys.modules:
        pytest.skip("the real litellm is installed; this test needs an environment without it")
    from theagentrouter.compat import install_as_litellm

    install_as_litellm()

    from agents import Agent, function_tool, set_tracing_disabled
    from agents.extensions.models.litellm_model import LitellmModel

    set_tracing_disabled(True)  # no traces sent to OpenAI from a test

    @function_tool
    def search_refunds(query: str) -> dict:
        """Look up refunds matching a query."""
        return {"refunds": 3, "total_usd": 412.5, "query": query}

    return Agent(
        name="refunds",
        instructions="Answer questions about refunds. Use search_refunds.",
        model=LitellmModel(model=model, base_url=base_url),
        tools=[search_refunds],
    )


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "openai/gpt-4o-mini"])
async def test_agent_calls_tool_and_answers(fake, model):
    from agents import Runner

    result = await Runner.run(_agent(model, fake.url), "Summarise yesterday's refunds")
    assert "412.5" in result.final_output  # the model answered from the tool result
    paths = [p for p, _, _ in fake.requests]
    assert len(paths) == 2  # one call to ask for the tool, one to answer
    assert all(p.endswith("/messages" if model.startswith("anthropic") else "/chat/completions") for p in paths)


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "openai/gpt-4o-mini"])
async def test_agent_streaming(fake, model):
    from agents import Runner

    run = Runner.run_streamed(_agent(model, fake.url), "Summarise yesterday's refunds")
    tool_calls = 0
    async for event in run.stream_events():
        if event.type == "run_item_stream_event" and event.item.type == "tool_call_item":
            tool_calls += 1
    assert tool_calls == 1
    assert "412.5" in run.final_output
