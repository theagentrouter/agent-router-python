"""The app against real OpenAI and Anthropic, on whichever SDK is installed. Keys from the environment.

Prints PASS or FAIL per check and RESULT: PASS=n FAIL=n; exits non-zero on a failure.
"""

import sys

import assistant  # noqa: F401  replaces litellm first when the app runs on Agent Router's SDK
import litellm
from assistant import app

OPENAI, CLAUDE = "openai/gpt-5.4-mini", "anthropic/claude-haiku-5-5"
results = []


def check(name, fn):
    try:
        info = fn()
        results.append(True)
        print(f"PASS  {name}  {info if info is not None else ''}")
    except Exception as e:  # noqa: BLE001  report every failure, keep going
        results.append(False)
        print(f"FAIL  {name}  {type(e).__name__}: {str(e)[:200]}")


def stream_with_usage(model):
    chunks = list(
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert all(len(c.choices) == 1 for c in chunks) and chunks[-1].usage.total_tokens > 0
    return f"{len(chunks)} chunks"


print("SDK:", litellm.__name__)
for model in (OPENAI, CLAUDE):
    check(f"ask {model}", lambda m=model: app.ask(m, "Say hello in three words."))
    check(f"stream {model}", lambda m=model: "".join(app.stream(m, "Count to five.")))
    check(f"stream with usage {model}", lambda m=model: stream_with_usage(m))
    check(f"tools {model}", lambda m=model: app.with_tools(m, "What is the status of order A-100?")[1])
check(
    # Claude 5 thinks adaptively: a thinking request is accepted, and the model decides whether to think,
    # so the answer is checked, not reasoning_content.
    "claude thinking",
    lambda: (
        "391"
        in litellm.completion(
            model="anthropic/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
            max_tokens=2048,
        )
        .choices[0]
        .message.content
    ),
)
check(
    "claude drop_params",
    lambda: app.ask(CLAUDE, "Say hi.", seed=1, presence_penalty=0.0, drop_params=True),
)
check("embedding", lambda: len(app.embed("openai/text-embedding-3-small", ["refunds", "exports"])[0]))
print(f"RESULT: PASS={sum(results)} FAIL={len(results) - sum(results)}")
sys.exit(0 if all(results) else 1)
