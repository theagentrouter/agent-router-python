"""OpenAI's Responses API with LiteLLM's `responses`: a reply, a follow-up that continues it, and what it cost. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/provider_apis/openai_responses.py
"""

import theagentrouter as ar

first = ar.responses(
    model="openai/gpt-4o-mini",
    instructions="Answer in one sentence.",
    input="What is a refund?",
)
print(first.output_text)

# previous_response_id continues the conversation; the provider keeps the earlier turns.
second = ar.responses(
    model="openai/gpt-4o-mini",
    input="And how is it different from a return?",
    previous_response_id=first.id,
)
print(second.output_text)
print(
    f"{second.usage.input_tokens} tokens in, {second.usage.output_tokens} out, ${second._hidden_params['response_cost']:.6f}"
)
