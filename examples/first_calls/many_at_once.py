# SPDX-License-Identifier: Apache-2.0
"""Several calls at once with acompletion. Needs OPENAI_API_KEY.

OPENAI_API_KEY=... python examples/first_calls/many_at_once.py
"""

import asyncio

import theagentrouter


async def ask(question: str) -> str:
    resp = await theagentrouter.acompletion(
        model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": question}], max_tokens=30
    )
    return resp.choices[0].message.content


async def main() -> None:
    questions = ["Name a fruit.", "Name a river.", "Name a planet."]
    for question, answer in zip(questions, await asyncio.gather(*(ask(q) for q in questions)), strict=True):
        print(f"{question} -> {answer}")


asyncio.run(main())
