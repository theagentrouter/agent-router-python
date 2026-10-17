"""A route with a request limit: the local gateway lets 3 requests a minute through and answers the rest with 429.

Useful to stop an agent stuck in a loop from running up a bill. The limit counts requests, not
tokens, for every program on the computer that uses the route. The running gateway keeps the
count, so if you run this again within a minute, every request is over the limit.
Needs OPENAI_API_KEY.

    OPENAI_API_KEY=... python examples/routes/request_limit.py
"""

import theagentrouter as ar

ar.route("capped", primary="openai/gpt-4o-mini", retry=ar.Retry(attempts=0), limit=ar.Limit(3, per="minute"))

for n in range(1, 6):
    try:
        ar.completion(model="capped", messages=[{"role": "user", "content": "Say ok."}], max_tokens=5)
        print(f"request {n}: answered")
    except ar.exceptions.RateLimitError:
        print(f"request {n}: over the limit (429)")
