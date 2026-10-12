"""A route that retries: the local gateway tries the model again on rate limits and server errors.

Needs OPENAI_API_KEY.

    OPENAI_API_KEY=... python examples/08_retry.py
"""

import theagentrouter as ar

ar.route(
    "patient",
    primary="openai/gpt-4o-mini",
    retry=ar.Retry(attempts=3, on=(429, 500, 502, 503, 504), timeout=30, backoff=0.5),
)
resp = ar.completion(model="patient", messages=[{"role": "user", "content": "Say hello."}])
print(resp.choices[0].message.content)
