"""A Google ADK agent on theagentrouter, with no LiteLLM installed.

pip install theagentrouter google-adk
ANTHROPIC_API_KEY=... python examples/agent_frameworks/google_adk.py
"""

import asyncio

from theagentrouter.compat import install_as_litellm

install_as_litellm()  # before importing ADK

from google.adk.agents import Agent  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402


def search_refunds(day: str) -> dict:
    """Find refunds for a day."""
    return {"day": day, "refunds": 3, "total_usd": 412.5}


async def main() -> None:
    agent = Agent(
        name="refunds",
        model=LiteLlm(model="anthropic/claude-sonnet-5-5"),
        instruction="Answer questions about refunds. Use search_refunds, passing the day as the user wrote it.",
        tools=[search_refunds],
    )
    runner = InMemoryRunner(agent=agent, app_name="acme")
    session = await runner.session_service.create_session(app_name="acme", user_id="maya")
    async for event in runner.run_async(
        user_id="maya",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="How many refunds yesterday?")]),
    ):
        for part in (event.content.parts if event.content else []) or []:
            if part.text:
                print(part.text)


asyncio.run(main())
