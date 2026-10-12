"""The same call to several providers: only the model name changes.

Every call goes through the local gateway, which reads each provider's key from the environment.
The Bedrock call runs only when AWS_PROFILE or AWS_ACCESS_KEY_ID is set, with AWS_REGION; other
ways of signing in to AWS (such as SSO without AWS_PROFILE) are not detected here.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/02_multi_provider.py
"""

import os

import theagentrouter

MODELS = ["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"]
if os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID"):
    MODELS.append("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0")

for model in MODELS:
    resp = theagentrouter.completion(
        model=model, messages=[{"role": "user", "content": "Name one colour."}], max_tokens=10
    )
    print(f"{model:55} {resp.choices[0].message.content}")
