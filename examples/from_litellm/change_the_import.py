"""A typical LiteLLM script, moved to theagentrouter by changing one import.

Everything below the import is written for LiteLLM: a tool call, the answer, LiteLLM's error
names and completion_cost. Needs OPENAI_API_KEY (or change MODEL and the key).

    OPENAI_API_KEY=... python examples/from_litellm/change_the_import.py
"""

import json

from theagentrouter.compat import litellm  # was: import litellm

MODEL = "openai/gpt-5.4-mini"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_refunds",
            "description": "Find refunds for a day",
            "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
        },
    }
]


def search_refunds(day: str) -> dict:
    return {"day": day, "refunds": 3, "total_usd": 412.5}


def main() -> None:
    messages = [{"role": "user", "content": "How much was refunded yesterday?"}]
    try:
        resp = litellm.completion(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        message = resp.choices[0].message
        cost = litellm.completion_cost(resp)
        if message.tool_calls:
            call = message.tool_calls[0]
            result = search_refunds(**json.loads(call.function.arguments))
            messages.append(message.model_dump())
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            resp = litellm.completion(model=MODEL, messages=messages, tools=TOOLS)
            cost += litellm.completion_cost(resp)
        print(resp.choices[0].message.content)
        print(f"cost: ${cost:.6f}, context window: {litellm.get_model_info(MODEL)['max_input_tokens']} tokens")
    except litellm.RateLimitError:
        print("rate limited; try again later")
    except litellm.ContextWindowExceededError:
        print("the conversation is too long for this model")


if __name__ == "__main__":
    main()
