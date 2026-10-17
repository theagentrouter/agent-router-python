"""The smallest program: one call, one answer. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/first_calls/one_call.py
"""

import theagentrouter

print(
    theagentrouter.completion(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    .choices[0]
    .message.content
)
