# SPDX-License-Identifier: Apache-2.0
"""Run every example in examples/ against real providers, one at a time; costs a few cents.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python3 scripts/smoke_examples.py

The examples share one local gateway in a fresh AGENTROUTER_HOME, as they would on a developer's
computer, so route examples restart a running gateway. gateways/your_gateway.py is pointed at that
gateway as "a gateway you set". An example whose framework is not installed is skipped. The
gateway is stopped at the end.

Prints PASS or FAIL per example, then RESULT: PASS=n FAIL=n SKIP=n, and exits non-zero on any failure.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The order of examples/README.md; a route example restarts the gateway an earlier one started.
EXAMPLES = [
    "from_litellm/change_the_import.py",
    "from_litellm/litellm_router.py",
    "from_litellm/litellm_errors.py",
    "from_litellm/cost_and_model_info.py",
    "first_calls/one_call.py",
    "first_calls/reply_and_cost.py",
    "first_calls/any_provider.py",
    "first_calls/stream.py",
    "first_calls/tools.py",
    "first_calls/many_at_once.py",
    "first_calls/json_schema.py",
    "first_calls/embeddings.py",
    "routes/fallback.py",
    "routes/retry.py",
    "routes/your_own_names.py",
    "routes/split_traffic.py",
    "routes/request_limit.py",
    "gateways/local_gateway.py",
    "gateways/your_gateway.py",
    "agent_frameworks/google_adk.py",
    "agent_frameworks/openai_agents.py",
    "provider_apis/openai_responses.py",
    "provider_apis/anthropic_messages.py",
]
NEEDS = {"agent_frameworks/google_adk.py": "google.adk", "agent_frameworks/openai_agents.py": "agents"}


def main() -> int:
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"set {' and '.join(missing)}: every example needs one of them")
        return 2
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTROUTER_")}
    home = tempfile.mkdtemp(prefix="smoke-examples-")
    env["AGENTROUTER_HOME"] = home
    secrets = [os.environ[k] for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(k)]
    passed = failed = skipped = 0
    try:
        for name in EXAMPLES:
            module = NEEDS.get(name)
            if module and importlib.util.find_spec(module.split(".")[0]) is None:
                print(f"  SKIP  {name}  ({module} not installed)")
                skipped += 1
                continue
            run_env = dict(env)
            if name == "gateways/your_gateway.py":
                state = json.loads((Path(home) / "local-gateway" / "state.json").read_text())
                run_env.update(
                    AGENTROUTER_BASE_URL=f"http://127.0.0.1:{state['port']}/v1", AGENTROUTER_API_KEY="unused"
                )
            try:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "examples" / name)],
                    cwd=ROOT,
                    env=run_env,
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                ok, output = result.returncode == 0, result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                ok, output = False, "timed out after 240s"
            for secret in secrets:
                output = output.replace(secret, "<redacted>")
            lines = [line for line in output.strip().splitlines() if line.strip()]
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({(lines[-1] if lines else '')[:120]})")
            passed, failed = passed + ok, failed + (not ok)
    finally:
        subprocess.run(
            [sys.executable, "-c", "from theagentrouter import local_gateway; local_gateway.stop()"],
            env=env,
            check=False,
        )
    print(f"RESULT: PASS={passed} FAIL={failed} SKIP={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
