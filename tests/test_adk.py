"""A real Google ADK agent on agentrouter, over HTTP to the fake provider. Skipped without google-adk.

Run in an environment that has google-adk and not litellm, so only agentrouter can answer
`import litellm`.
"""

from __future__ import annotations

import importlib.util

import pytest

from .fake_provider import FakeProvider


def _has_adk() -> bool:
    try:
        return importlib.util.find_spec("google.adk") is not None
    except ModuleNotFoundError:  # no "google" package at all
        return False


pytestmark = pytest.mark.skipif(not _has_adk(), reason="google-adk not installed")


def search_refunds(query: str) -> dict:
    """Look up refunds matching a query."""
    return {"refunds": 3, "total_usd": 412.5, "query": query}


@pytest.fixture
def fake(monkeypatch):
    server = FakeProvider()
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    yield server
    server.close()


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "openai/gpt-4o-mini"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_adk_agent_calls_tool_and_answers(fake, model, streaming):
    if importlib.util.find_spec("litellm") is not None:
        import sys

        if "litellm" not in sys.modules:
            pytest.skip("the real litellm is installed; this test needs an environment without it")
    from theagentrouter.compat import install_as_litellm

    install_as_litellm()

    from google.adk.agents import Agent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = Agent(
        name="refunds",
        model=LiteLlm(model=model, api_base=fake.url),
        instruction="Answer questions about refunds. Use search_refunds.",
        tools=[search_refunds],
    )
    runner = InMemoryRunner(agent=agent, app_name="acme")
    session = await runner.session_service.create_session(app_name="acme", user_id="maya")
    calls, results, texts = [], [], []
    async for event in runner.run_async(
        user_id="maya",
        session_id=session.id,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE if streaming else StreamingMode.NONE),
        new_message=types.Content(role="user", parts=[types.Part(text="Summarise yesterday's refunds")]),
    ):
        for part in (event.content.parts if event.content else []) or []:
            if part.function_call and not event.partial:
                calls.append(part.function_call)
            if part.function_response:
                results.append(part.function_response)
            if part.text and not event.partial:
                texts.append(part.text)

    assert [c.name for c in calls] == ["search_refunds"]
    assert calls[0].args == {"query": "Summarise yesterday's refunds"}
    assert results and results[0].response["refunds"] == 3
    assert texts and "412.5" in texts[-1]  # the model answered from the tool result
    paths = [p for p, _, _ in fake.requests]
    assert len(paths) == 2  # one call to ask for the tool, one to answer
    assert all(p.endswith("/messages" if model.startswith("anthropic") else "/chat/completions") for p in paths)


async def test_adk_streamed_thinking_goes_back_unchanged_with_the_tool_result(fake):
    """ADK joins streamed thinking fragments and attaches the signature from the chunk that has only it.

    The fake refuses the tool-result request, as Anthropic does, unless the thinking block comes back
    exactly as it was sent: same text, same signature.
    """
    if importlib.util.find_spec("litellm") is not None:
        import sys

        if "litellm" not in sys.modules:
            pytest.skip("the real litellm is installed; this test needs an environment without it")
    from theagentrouter.compat import install_as_litellm

    install_as_litellm()

    from google.adk.agents import Agent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from .fake_provider import THINKING_BLOCK

    agent = Agent(
        name="refunds",
        model=LiteLlm(
            model="anthropic/claude-sonnet-4-5",
            api_base=fake.url,
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        ),
        instruction="Answer questions about refunds. Use search_refunds.",
        tools=[search_refunds],
    )
    runner = InMemoryRunner(agent=agent, app_name="acme")
    session = await runner.session_service.create_session(app_name="acme", user_id="maya")
    texts = []
    async for event in runner.run_async(
        user_id="maya",
        session_id=session.id,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        new_message=types.Content(role="user", parts=[types.Part(text="Summarise yesterday's refunds")]),
    ):
        for part in (event.content.parts if event.content else []) or []:
            if part.text and not part.thought and not event.partial:
                texts.append(part.text)

    assert len(fake.requests) == 2
    second = fake.requests[1][2]
    assistant = next(m for m in second["messages"] if m["role"] == "assistant")
    assert assistant["content"][0] == THINKING_BLOCK
    assert texts and "412.5" in texts[-1]  # the fake accepted the tool result and answered
