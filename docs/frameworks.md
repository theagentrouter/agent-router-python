# Agent frameworks

Many agent frameworks reach models through LiteLLM. For those, theagentrouter answers `import litellm` in its place, and the framework does not change.

| Framework | How | Status |
|---|---|---|
| Google ADK | `install_as_litellm()`, then ADK's `LiteLlm` model | Tested: tool calls and answers, with and without streaming, on OpenAI and Anthropic |
| OpenAI Agents SDK | `install_as_litellm()`, then the Agents SDK's `LitellmModel` | Tested: tool calls and answers, with and without streaming, on OpenAI and Anthropic |
| LangChain (`langchain-litellm`), LlamaIndex, Strands, CrewAI with `is_litellm=True` | Not yet | They stop on LiteLLM names the SDK does not have |
| DSPy, smolagents, Aider, OpenHands and others that import `litellm` | `install_as_litellm()`, then the framework's LiteLLM model | Not tested yet. A LiteLLM name the SDK does not have fails with `ImportError` or `AttributeError` |

"Tested" means a real agent from that framework runs in a test against a local test server, and in a check against real OpenAI and Anthropic models, with LiteLLM not installed.

CI runs the tests against a test server with the latest Google ADK and OpenAI Agents SDK on every change.

A framework's settings reach Claude as they do on a plain call. Newer Claude models refuse some of them, such as `temperature`, and with `drop_params` they are dropped: see [Parameters](usage.md#parameters). To turn on Claude's thinking, set `reasoning_effort`: in ADK `LiteLlm(model=..., reasoning_effort="low")`, in the Agents SDK `ModelSettings(reasoning=Reasoning(effort="low"))`. To drop what a model refuses, pass `drop_params=True`: in ADK to `LiteLlm(...)`, in the Agents SDK through `ModelSettings(extra_args={"drop_params": True})`.

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
    model=LiteLlm(model="anthropic/claude-sonnet-5-5"),
    instruction="Answer questions about refunds. Use search_refunds, passing the day as the user wrote it.",
    tools=[search_refunds],
)
```

ADK's own extra, `google-adk[extensions]`, installs LiteLLM. Plain `google-adk` with theagentrouter does not need it.

Runnable example: [`examples/agent_frameworks/google_adk.py`](../examples/agent_frameworks/google_adk.py).

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
    model=LitellmModel(model="anthropic/claude-sonnet-5-5"),
    tools=[search_refunds],
)
result = Runner.run_sync(agent, "How much was refunded yesterday?")
print(result.final_output)
```

The Agents SDK's own extra, `openai-agents[litellm]`, installs LiteLLM. Plain `openai-agents` with theagentrouter does not need it.

The Agents SDK sends traces to OpenAI by default. Turn that off with `set_tracing_disabled(True)` from `agents` if you do not want it.

Runnable example: [`examples/agent_frameworks/openai_agents.py`](../examples/agent_frameworks/openai_agents.py).

## Through a gateway

Frameworks need no change: every call already goes through the local gateway. Set `AGENTROUTER_BASE_URL` to send all calls to another gateway instead, such as your company's Agent Router.
