"""Anthropic's Messages API with LiteLLM's `anthropic.messages.create`: prompt caching and thinking. Needs ANTHROPIC_API_KEY.

The request and the reply are in Anthropic's own format, as in Anthropic's SDK.

ANTHROPIC_API_KEY=... python examples/provider_apis/anthropic_messages.py
"""

import theagentrouter as ar

model = "anthropic/claude-sonnet-5-5"

# Prompt caching: mark a long, fixed part of the prompt with cache_control.
# Anthropic caches prompts of 1,024 tokens or more; the second call reads the cache.
policy = "Refunds are allowed within 30 days of purchase with a receipt. " * 150
system = [{"type": "text", "text": policy, "cache_control": {"type": "ephemeral"}}]
for question in ["Can I get a refund after 10 days?", "Can I get a refund after 40 days?"]:
    resp = ar.anthropic.messages.create(
        model=model, max_tokens=100, system=system, messages=[{"role": "user", "content": question}]
    )
    usage = resp["usage"]
    print(f"{usage.get('cache_read_input_tokens') or 0} tokens read from the cache: {resp['content'][0]['text']}")

# Thinking: the reply has a thinking block before the text.
resp = ar.anthropic.messages.create(
    model=model,
    max_tokens=2048,
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=[
        {"role": "user", "content": "A customer paid $120 and returns 2 of 5 equal items. How much is refunded?"}
    ],
)
for block in resp["content"]:
    print(f"{block['type']}: {block.get('thinking') or block.get('text')}")
print(f"cost: ${resp._hidden_params['response_cost']:.6f}")
