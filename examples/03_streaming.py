"""Stream a reply as it is written. Needs OPENAI_API_KEY."""

import theagentrouter

for chunk in theagentrouter.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Count from one to ten."}],
    stream=True,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print(f"\n[{chunk.usage.total_tokens} tokens]")
