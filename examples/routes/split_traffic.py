"""Two models share the traffic of one route: each call goes to one of them at random.

Needs OPENAI_API_KEY and ANTHROPIC_API_KEY.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/routes/split_traffic.py
"""

from collections import Counter

import theagentrouter as ar

ar.route("shared", primary=["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"])
served = Counter(
    ar.completion(model="shared", messages=[{"role": "user", "content": "hi"}], max_tokens=5)._hidden_params[
        "served_by"
    ]
    for _ in range(6)
)
print(dict(served))
