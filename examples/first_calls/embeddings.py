"""Embeddings with LiteLLM's arguments: one vector per text, and what the call cost. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/first_calls/embeddings.py
"""

import math

import theagentrouter as ar

texts = ["How do I get a refund?", "Can I return an item I bought?", "What time is it in Jakarta?"]
resp = ar.embedding(model="openai/text-embedding-3-small", input=texts)
vectors = [item["embedding"] for item in resp.data]


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


print(f"{len(vectors)} vectors of {len(vectors[0])} numbers, ${resp._hidden_params['response_cost']:.8f}")
print(f"refund vs return:  {similarity(vectors[0], vectors[1]):.2f}")
print(f"refund vs the time: {similarity(vectors[0], vectors[2]):.2f}")
