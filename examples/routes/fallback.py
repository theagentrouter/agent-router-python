"""A route with a fallback: when the primary fails, the next model answers.

The local gateway serves routes. The primary here is a model OpenAI does not have, so OpenAI
answers 404 and the gateway moves on to Anthropic. `on=` lists the status codes that count as a
failure; 404 is added here only because the model is made up. `attempts=0` means no retries, so
the gateway goes straight to the fallback.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/routes/fallback.py
"""

import theagentrouter as ar

ar.route(
    "chat",
    primary="openai/gpt-does-not-exist",
    fallbacks=["anthropic/claude-haiku-5-5"],
    retry=ar.Retry(attempts=0, on=(404, 429, 500, 502, 503, 504)),
)
resp = ar.completion(model="chat", messages=[{"role": "user", "content": "Say hello."}])
print(resp.choices[0].message.content)
print("answered by:", resp._hidden_params["served_by"])  # the SDK sent one request; the gateway fell back
