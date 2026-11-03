# SPDX-License-Identifier: Apache-2.0
"""The local gateway: Agent Router on this computer, started by the first call.

Every call goes through it; the SDK never calls a provider itself. The first call starts the
gateway (the very first start also downloads Envoy, which can take a minute). The gateway reads
the provider keys from the environment when it starts, and keeps running after this program ends,
so the next program on the computer uses the same one. Needs OPENAI_API_KEY.

    OPENAI_API_KEY=... python examples/gateways/local_gateway.py

To stop the gateway:

    python -c "from theagentrouter import local_gateway; local_gateway.stop()"
"""

import theagentrouter as ar

resp = ar.completion(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "Say hello in three words."}])
print(resp.choices[0].message.content)
print("went through:", resp._hidden_params["route"])  # "local gateway"
# The SDK sends the full name; the gateway passes OpenAI its own name, gpt-5.4-mini.
print("model name the SDK sent:", resp._hidden_params["model_sent"])
