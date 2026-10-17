"""Names of your own for models, so code says "fast" or "smart" and the choice lives in one place.

Routes are saved on this computer (~/.agentrouter/routes.json) and every program can use them,
so ar.list_routes() also shows routes that other programs or examples defined.
Needs OPENAI_API_KEY and ANTHROPIC_API_KEY.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/routes/your_own_names.py
"""

import theagentrouter as ar

ar.route("fast", primary="openai/gpt-4o-mini")
ar.route("smart", primary="anthropic/claude-sonnet-5-5")
print("fast and smart are saved:", {"fast", "smart"} <= set(ar.list_routes()))

for name in ("fast", "smart"):
    resp = ar.completion(model=name, messages=[{"role": "user", "content": "Name one colour."}], max_tokens=10)
    print(f"{name:6} -> {resp._hidden_params['served_by']:30} {resp.choices[0].message.content}")
