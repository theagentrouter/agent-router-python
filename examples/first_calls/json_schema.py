"""A reply that follows a JSON schema, with response_format. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/first_calls/json_schema.py
"""

import json

import theagentrouter

schema = {
    "type": "object",
    "properties": {"day": {"type": "string"}, "refunds": {"type": "integer"}},
    "required": ["day", "refunds"],
    "additionalProperties": False,
}
resp = theagentrouter.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Yesterday we had 3 refunds. Report it."}],
    response_format={"type": "json_schema", "json_schema": {"name": "report", "schema": schema, "strict": True}},
)
print(json.loads(resp.choices[0].message.content))
