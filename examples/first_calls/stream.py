"""Stream a reply as it is written. Needs OPENAI_API_KEY."""

import theagentrouter

for chunk in theagentrouter.completion(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": "Count from one to ten."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
    if chunk.usage:  # on the last chunk
        print(f"\n[{chunk.usage.total_tokens} tokens]")
