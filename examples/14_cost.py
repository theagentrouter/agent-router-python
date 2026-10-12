"""What a call cost, and what a model can do, from the bundled model list. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/14_cost.py
"""

import theagentrouter

MODEL = "openai/gpt-4o-mini"
info = theagentrouter.get_model_info(MODEL)
print(f"{MODEL}: {info['max_input_tokens']} input tokens, tools: {theagentrouter.supports_function_calling(MODEL)}")

resp = theagentrouter.completion(
    model=MODEL, messages=[{"role": "user", "content": "Write one sentence about refunds."}]
)
print(
    f"{resp.usage.prompt_tokens} in + {resp.usage.completion_tokens} out = ${theagentrouter.completion_cost(resp):.6f}"
)
