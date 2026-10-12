---
name: migrate-from-litellm
description: Move a Python project from LiteLLM to theagentrouter, the Agent Router SDK. Use when the user asks to replace LiteLLM, drop the litellm dependency, or switch an agent framework (Google ADK, OpenAI Agents SDK) off LiteLLM.
---

# Migrate from LiteLLM to theagentrouter

theagentrouter keeps LiteLLM's API: `completion`, `acompletion`, `embedding`, `aembedding`, the same responses, the same error names and LiteLLM's `Router`. Most projects change one import. Do the steps in order and stop to report anything under "Not available".

## 1. Find every use of LiteLLM

```sh
grep -rnE "^\s*(import litellm|from litellm)" --include=*.py .
grep -rnE "litellm" pyproject.toml requirements*.txt setup.cfg setup.py 2>/dev/null
```

Also note frameworks that import LiteLLM themselves: Google ADK's `LiteLlm`, the OpenAI Agents SDK's `LitellmModel`, and tools built on LiteLLM.

## 2. Change the dependency

Replace `litellm` with `theagentrouter` in the project's dependencies:

```sh
pip install theagentrouter
```

On macOS 13+ (Apple chips) and Linux this also installs the local gateway, Agent Router on the computer, which every call goes through. For code that runs on servers, in containers, serverless functions or CI, use `theagentrouter-core` instead (the SDK alone, same import) with `AGENTROUTER_BASE_URL` pointing at a deployed Agent Router gateway, and on Windows or Intel Macs set `AGENTROUTER_BASE_URL` too; tell the user which applies.

Remove `litellm`, and `google-adk[extensions]` (which pulls LiteLLM in) in favour of plain `google-adk`.

## 3. Change the imports

Code that imports LiteLLM directly:

```python
# before
import litellm
# after
from theagentrouter.compat import litellm
```

`from litellm import completion, Router` becomes `from theagentrouter.compat.litellm import completion, Router`.

Code where a framework imports LiteLLM: call `install_as_litellm()` once, **before** the framework is imported, at the program's entry point:

```python
from theagentrouter.compat import install_as_litellm
install_as_litellm()

from google.adk.models.lite_llm import LiteLlm  # now gets theagentrouter
```

## 4. Fix model names

OpenAI and Anthropic names without a provider (`gpt-4o-mini`, `claude-sonnet-4-5`) work, as in LiteLLM. A bare `gemini-…` name meant Vertex AI in LiteLLM, which the local gateway does not serve yet: tell the user, and if they use the Gemini API, change it to `gemini/…`. With `AGENTROUTER_BASE_URL` set, the gateway takes its own names.

## 5. Move retries and fallbacks

`num_retries=`, `fallbacks=` and `max_retries=` on a single `completion` call are ignored with a warning. Two ways to keep them:

- Keep LiteLLM's `Router` as it is: `model_list`, `fallbacks` and `num_retries` work, served by the local gateway.
- Or define a route once and call it by name:

```python
import theagentrouter as ar

ar.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"], retry=ar.Retry(attempts=2))
ar.completion(model="chat", messages=messages)
```

A route can also cap requests: `limit=ar.Limit(60, per="minute")`.

## 6. Check what is not there

Search for these and report each use to the user instead of guessing a replacement:

| LiteLLM | In theagentrouter |
|---|---|
| `image_generation`, `transcription`, `speech` | Not available |
| `text_completion` | Use `completion` with messages |
| `success_callback`, `failure_callback`, `callbacks` | Your functions and loggers work; strings naming LiteLLM integrations (`"langfuse"`) warn and are skipped: the gateway's traces record calls |
| `responses`, `aresponses` | Work for OpenAI models; other providers raise `BadRequestError` |
| `anthropic.messages.create`, `acreate`, `anthropic_messages` | Work for Claude; other providers and routes raise `BadRequestError` |
| `token_counter` | Works; add `theagentrouter[tokens]` for LiteLLM's estimate on providers other than OpenAI and Anthropic |
| `caching`, `cache`, `metadata`, `tags`, `mock_response`, `custom_llm_provider`, `api_version` | Ignored with a warning |
| `embedding` with an `anthropic/` model | Anthropic has no embeddings API; LiteLLM fails too |
| The LiteLLM proxy server | Not part of the SDK; point `AGENTROUTER_BASE_URL` at a gateway instead |

## 7. Verify

```sh
python -W default -m pytest     # every ignored argument warns once; read the warnings
```

Keys stay in the same environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...). Report to the user: files changed, model names changed, warnings seen, and anything from step 6.

Full reference: `docs/coming-from-litellm.md` in the theagentrouter repository.
