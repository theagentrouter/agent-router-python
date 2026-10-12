"""Errors keep LiteLLM's names, so the same except clauses work. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/13_handling_errors.py
"""

import theagentrouter
from theagentrouter import exceptions

for model in ["openai/gpt-does-not-exist", "openai/gpt-4o-mini"]:
    try:
        resp = theagentrouter.completion(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
        print(f"{model}: {resp.choices[0].message.content}")
    except exceptions.NotFoundError as e:
        print(f"{model}: not found ({e.status_code}): {e}")
    except exceptions.RateLimitError:
        print(f"{model}: rate limited; try again later")
    except exceptions.ContextWindowExceededError:
        print(f"{model}: the conversation is too long for this model")
    except exceptions.APIError as e:
        print(f"{model}: {type(e).__name__} ({e.status_code}): {e}")
