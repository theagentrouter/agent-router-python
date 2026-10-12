"""Routes on real providers through the real local gateway. Needs OPENAI_API_KEY, ANTHROPIC_API_KEY and
theagentrouter-local-gateway installed; costs a few cents at most.

    set -a; . ./.env; set +a; .venv/bin/python scripts/smoke_routes.py

Checks: a route answers through its primary; a primary that fails (a model OpenAI does not have)
falls back to Anthropic, through Anthropic's OpenAI-compatible endpoint, and the gateway does the
fallback (the SDK sends one request); LiteLLM's Router does the same; a tier of two models answers
from either. Uses a temporary AGENTROUTER_HOME and stops the
gateway at the end. Prints RESULT: PASS=n FAIL=n and exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ["AGENTROUTER_HOME"] = tempfile.mkdtemp(prefix="theagentrouter-smoke-routes-")
for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_AUTOSTART"):
    os.environ.pop(var, None)

import theagentrouter as ar  # noqa: E402
from theagentrouter import local_gateway  # noqa: E402
from theagentrouter.compat import litellm  # noqa: E402

ASK = [{"role": "user", "content": "Say hello in three words."}]
FALLBACK_ON = (404, 429, 500, 502, 503, 504)  # 404: the made-up model below

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ask(model: str):
    resp = ar.completion(model=model, messages=ASK, max_tokens=30)
    return resp, resp._hidden_params


def main() -> int:
    if not local_gateway.installed():
        print("theagentrouter-local-gateway is not installed; run `make local-gateway AIGW=/path/to/aigw`")
        return 1
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("set OPENAI_API_KEY and ANTHROPIC_API_KEY")
        return 1
    try:
        print("route with a working primary")
        ar.route("chat", primary="openai/gpt-4o-mini", fallbacks=["anthropic/claude-haiku-4-5"])
        try:
            resp, hp = ask("chat")
            check("answered", bool(resp.choices[0].message.content), resp.choices[0].message.content or "")
            check("by the primary", hp.get("served_by") == "openai/gpt-4o-mini", str(hp.get("served_by")))
            check("through the local gateway", hp.get("route") == "local gateway", hp.get("sent_to", ""))
        except Exception as e:
            check("answered", False, f"{type(e).__name__}: {e}")

        print("route whose primary fails")
        ar.route(
            "fallback",
            primary="openai/gpt-does-not-exist",
            fallbacks=["anthropic/claude-haiku-4-5"],
            retry=ar.Retry(attempts=0, on=FALLBACK_ON),
        )
        try:
            resp, hp = ask("fallback")
            check("answered", bool(resp.choices[0].message.content), resp.choices[0].message.content or "")
            check("by the fallback", hp.get("served_by") == "anthropic/claude-haiku-4-5", str(hp.get("served_by")))
            check("in one request from the SDK", hp.get("route_name") == "fallback" and "fallbacks_tried" not in hp)
        except Exception as e:
            check("answered", False, f"{type(e).__name__}: {e}")

        print("LiteLLM's Router")
        router = litellm.Router(
            model_list=[
                {"model_name": "main", "litellm_params": {"model": "openai/gpt-does-not-exist"}},
                {"model_name": "spare", "litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
            ],
            fallbacks=[{"main": ["spare"]}],
            num_retries=0,
        )
        try:
            resp = router.completion(model="main", messages=ASK, max_tokens=30)
            check("answered", bool(resp.choices[0].message.content), resp.choices[0].message.content or "")
            check("by the fallback", resp._hidden_params.get("served_by") == "anthropic/claude-haiku-4-5")
        except Exception as e:
            check("answered", False, f"{type(e).__name__}: {e}")

        print("a tier of two models")
        ar.route("shared", primary=["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"])
        served = set()
        for _ in range(6):
            try:
                served.add(ask("shared")[1].get("served_by"))
            except Exception as e:
                check("answered", False, f"{type(e).__name__}: {e}")
                break
        check(
            "answers from the tier",
            served <= {"openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"} and bool(served),
            ", ".join(sorted(s for s in served if s)),
        )
    finally:
        local_gateway.stop()
    print(f"RESULT: PASS={passed} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
