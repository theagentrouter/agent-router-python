"""Send every call through a gateway your team or company runs, instead of the local gateway.

Needs AGENTROUTER_BASE_URL (the gateway's address, which takes the OpenAI format) and
AGENTROUTER_API_KEY (the gateway's key, not a provider's). No provider key is needed on this
computer: the gateway holds them.

    AGENTROUTER_BASE_URL=https://gateway.acme.example/v1 AGENTROUTER_API_KEY=... python examples/15_through_a_gateway.py
"""

import os
import sys
from urllib.parse import urlsplit

import theagentrouter

if not os.environ.get("AGENTROUTER_BASE_URL"):
    sys.exit("Set AGENTROUTER_BASE_URL to your gateway's address. Without it, calls go to the local gateway instead.")

resp = theagentrouter.completion(
    model="anthropic/claude-sonnet-4-5", messages=[{"role": "user", "content": "Say hello in five words."}]
)
print(resp.choices[0].message.content)
print("answered through:", urlsplit(os.environ["AGENTROUTER_BASE_URL"]).hostname)
