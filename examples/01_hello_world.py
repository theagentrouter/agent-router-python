"""A text reply and what it cost. Needs ANTHROPIC_API_KEY (or change the model and key)."""

import theagentrouter

resp = theagentrouter.completion(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
print("cost (USD):", theagentrouter.completion_cost(resp))  # from the model list that ships with the SDK
