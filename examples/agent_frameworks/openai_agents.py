"""An OpenAI Agents SDK agent on theagentrouter, with no LiteLLM installed.

pip install theagentrouter openai-agents
ANTHROPIC_API_KEY=... python examples/agent_frameworks/openai_agents.py
"""

from theagentrouter.compat import install_as_litellm

install_as_litellm()  # before importing the Agents SDK's LiteLLM model

from agents import Agent, Runner, function_tool, set_tracing_disabled  # noqa: E402
from agents.extensions.models.litellm_model import LitellmModel  # noqa: E402


@function_tool
def search_refunds(day: str) -> dict:
    """Find refunds for a day."""
    return {"day": day, "refunds": 3, "total_usd": 412.5}


def main() -> None:
    set_tracing_disabled(True)  # the Agents SDK sends traces to OpenAI by default
    agent = Agent(
        name="refunds",
        instructions="Answer questions about refunds. Use search_refunds, passing the day as the user wrote it.",
        model=LitellmModel(model="anthropic/claude-sonnet-5-5"),
        tools=[search_refunds],
    )
    result = Runner.run_sync(agent, "How much was refunded yesterday?")
    print(result.final_output)


if __name__ == "__main__":
    main()
