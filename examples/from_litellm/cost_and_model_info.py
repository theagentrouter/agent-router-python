# SPDX-License-Identifier: Apache-2.0
"""What a call cost, and what a model can do, from the bundled model list. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/from_litellm/cost_and_model_info.py
"""

import theagentrouter

MODEL = "openai/gpt-5.4-mini"
info = theagentrouter.get_model_info(MODEL)
print(f"{MODEL}: {info['max_input_tokens']} input tokens, tools: {theagentrouter.supports_function_calling(MODEL)}")

resp = theagentrouter.completion(
    model=MODEL, messages=[{"role": "user", "content": "Write one sentence about refunds."}]
)
print(
    f"{resp.usage.prompt_tokens} in + {resp.usage.completion_tokens} out = ${theagentrouter.completion_cost(resp):.6f}"
)
