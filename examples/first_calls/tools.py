# SPDX-License-Identifier: Apache-2.0
"""A tool call and the answer that uses its result. Needs ANTHROPIC_API_KEY.

Tools are written in the OpenAI format for every provider; change MODEL to an openai/ model and
the code stays the same.
"""

import json

import theagentrouter

# A Claude model that accepts a forced tool call (tool_choice="required"); claude-sonnet-5-5 does not.
MODEL = "anthropic/claude-haiku-5-5"
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_refunds",
            "description": "Find refunds for a day",
            "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
        },
    }
]
messages = [{"role": "user", "content": "How many refunds were there yesterday?"}]

first = theagentrouter.completion(model=MODEL, messages=messages, tools=tools, tool_choice="required")
call = first.choices[0].message.tool_calls[0]
result = {"day": json.loads(call.function.arguments)["day"], "refunds": 3}  # your tool runs here

messages += [
    first.choices[0].message.to_dict(),
    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
]
final = theagentrouter.completion(model=MODEL, messages=messages, tools=tools)
print(final.choices[0].message.content)
