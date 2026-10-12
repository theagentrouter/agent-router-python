# Agent frameworks

Many agent frameworks reach models through LiteLLM. For those, theagentrouter answers `import litellm` in its place, and the framework does not change.

| Framework | How | Status |
|---|---|---|
| Google ADK | `install_as_litellm()`, then ADK's `LiteLlm` model | Tested: tool calls and answers, with and without streaming, on OpenAI and Anthropic |
| OpenAI Agents SDK | `install_as_litellm()`, then the Agents SDK's `LitellmModel` | Tested: tool calls and answers, with and without streaming, on OpenAI and Anthropic |
| CrewAI, DSPy, LangChain (`langchain-litellm`), smolagents, Strands, Aider, OpenHands and others that import `litellm` | `install_as_litellm()`, then the framework's LiteLLM model | Not tested yet. The SDK has every LiteLLM function these call, counted from their source |

"Tested" means a real agent from that framework runs in a test against a local test server, and in a check against real OpenAI and Anthropic models, with LiteLLM not installed.

CI runs the tests against a test server with the latest Google ADK and OpenAI Agents SDK on every change.

Calls to Claude reject options this SDK cannot translate, including `response_format` and `reasoning_effort`. See [Parameters](usage.md#parameters) before enabling a framework's structured-output or reasoning settings. To turn on Claude's thinking, pass `extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}}`: in ADK to `LiteLlm(...)`, in the Agents SDK through `ModelSettings(extra_body=...)`.

## The one rule

Call `install_as_litellm()` **before** importing the framework:

```python
from theagentrouter.compat import install_as_litellm
install_as_litellm()

# now import the framework
```

LiteLLM does not need to be installed, and it is best not installed: every import of `litellm` after the call gets theagentrouter, but code that imported the real LiteLLM before the call keeps it. The tests run with LiteLLM not installed.

## Google ADK

```sh
pip install theagentrouter google-adk
```

```python
from theagentrouter.compat import install_as_litellm
install_as_litellm()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm


def search_refunds(day: str) -> dict:
    """Find refunds for a day."""
    return {"day": day, "total_usd": 42}


agent = Agent(
    name="refunds",
    model=LiteLlm(model="anthropic/claude-sonnet-4-5"),
    instruction="Answer questions about refunds. Use search_refunds, passing the day as the user wrote it.",
    tools=[search_refunds],
)
```

ADK's own extra, `google-adk[extensions]`, installs LiteLLM. Plain `google-adk` with theagentrouter does not need it.

Runnable example: [`examples/16_adk_agent.py`](../examples/16_adk_agent.py).

## OpenAI Agents SDK

```sh
pip install theagentrouter openai-agents
```

```python
from theagentrouter.compat import install_as_litellm
install_as_litellm()

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel


@function_tool
def search_refunds(day: str) -> dict:
    """Find refunds for a day."""
    return {"day": day, "total_usd": 42}


agent = Agent(
    name="refunds",
    instructions="Answer questions about refunds. Use search_refunds, passing the day as the user wrote it.",
    model=LitellmModel(model="anthropic/claude-sonnet-4-5"),
    tools=[search_refunds],
)
result = Runner.run_sync(agent, "How much was refunded yesterday?")
print(result.final_output)
```

The Agents SDK's own extra, `openai-agents[litellm]`, installs LiteLLM. Plain `openai-agents` with theagentrouter does not need it.

The Agents SDK sends traces to OpenAI by default. Turn that off with `set_tracing_disabled(True)` from `agents` if you do not want it.

Runnable example: [`examples/17_openai_agents.py`](../examples/17_openai_agents.py).

## Through a gateway

Frameworks need no change: every call already goes through the local gateway. Set `AGENTROUTER_BASE_URL` to send all calls to another gateway instead, such as your company's Agent Router.
