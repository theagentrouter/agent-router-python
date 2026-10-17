---
name: theagentrouter
description: Write Python code that calls AI models with theagentrouter, the Agent Router SDK (LiteLLM's API in a small install). Use when code calls OpenAI, Anthropic or other model providers through theagentrouter, or needs fallbacks, retries, request limits, embeddings or the local gateway.
---

# Using theagentrouter

`pip install theagentrouter` (or `theagentrouter-core`, the SDK alone, for a remote gateway), `import theagentrouter as ar`. The API is LiteLLM's: if you know `litellm.completion`, you know this.

## Calls

```python
import theagentrouter as ar

resp = ar.completion(model="anthropic/claude-sonnet-4-5", messages=[{"role": "user", "content": "hi"}])
resp.choices[0].message.content
resp._hidden_params["response_cost"]   # US dollars, from the bundled model list

for chunk in ar.completion(model="openai/gpt-4o-mini", messages=messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="")

resp = await ar.acompletion(model="openai/gpt-4o-mini", messages=messages)
vecs = ar.embedding(model="openai/text-embedding-3-small", input=["a", "b"])  # vecs.data[0]["embedding"]
resp = ar.responses(input="hi", model="openai/gpt-4o-mini")  # OpenAI models only; resp.output_text
resp = ar.anthropic.messages.create(model="anthropic/claude-sonnet-4-5", max_tokens=256, messages=messages)  # Claude only; resp["content"]
n = ar.token_counter(model="openai/gpt-4o-mini", messages=messages)  # the gateway counts OpenAI and Anthropic
```

- Model names are `provider/model`; a name on the model list (`gpt-4o-mini`) finds its provider, as in LiteLLM.
- Keys come from the provider's usual environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...). Never put a key in code.
- `tools`, `tool_choice`, `response_format`, `temperature`, `max_tokens`, `timeout`, `api_key`, `api_base`, `extra_headers` work as in LiteLLM.
- Errors have LiteLLM's names: `ar.exceptions.RateLimitError`, `AuthenticationError`, `ContextWindowExceededError`, ...

## Where calls go

The SDK never calls a provider itself; every call goes through Agent Router.

1. `api_base=` on a call: that endpoint, for that call only (LiteLLM's argument).
2. `AGENTROUTER_BASE_URL` set: every call goes to that gateway, with `AGENTROUTER_API_KEY`.
3. Otherwise the local gateway: Agent Router on this computer, installed with the SDK on macOS 13+ (Apple chips) and Linux, started by the first call. It reads provider keys from the environment, so `api_key=` without `api_base=` has no effect. No command to run.

## Fallbacks, retries, limits

Not arguments on `completion`. Define a route once; the local gateway runs it:

```python
ar.route(
    "chat",
    primary="openai/gpt-4o",
    fallbacks=["anthropic/claude-sonnet-4-5"],
    retry=ar.Retry(attempts=2),
    limit=ar.Limit(60, per="minute"),   # optional; extra requests get RateLimitError
)
resp = ar.completion(model="chat", messages=messages)
resp._hidden_params["served_by"]        # which model answered
```

Routes are served by the local gateway and shared by every program on the computer. LiteLLM's `Router` (`from theagentrouter.compat.litellm import Router`) works too.

## Agent frameworks

Google ADK, the OpenAI Agents SDK and other code that imports `litellm`: call this before importing the framework.

```python
from theagentrouter.compat import install_as_litellm
install_as_litellm()
```

## Not available

`image_generation`, `transcription`, `speech`, `text_completion`, LiteLLM's built-in logging integrations and its cache. Do not invent them; say they are not supported.

Full docs: `docs/usage.md`, `docs/frameworks.md` and `examples/` in the theagentrouter repository.
