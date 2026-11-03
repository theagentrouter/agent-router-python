#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# The apps against real providers, on real LiteLLM and on this SDK. Needs OPENAI_API_KEY and
# ANTHROPIC_API_KEY in the environment or in .env. Costs a few cents. Run `make apps` first: it
# builds the environments.
#   make apps-live [SDK=litellm|theagentrouter]
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
# One local gateway for both apps (one runs per computer), started fresh and stopped at the end.
export AGENTROUTER_HOME="$PWD/apps/.agentrouter-live"
stop_gateway() {
    [ -x .venv-apps-theagentrouter/bin/python ] && .venv-apps-theagentrouter/bin/python -c \
        "from theagentrouter import local_gateway; local_gateway.stop()" || true
}
stop_gateway
status=0
for sdk in ${SDK:-litellm theagentrouter}; do
    python=$PWD/.venv-apps-$sdk/bin/python
    echo "== litellm-assistant on $sdk"
    (cd apps/litellm-assistant && "$python" examples/live.py) || status=1
    for model in openai/gpt-5.4 anthropic/claude-sonnet-5-5; do
        for streaming in false true; do
            echo "== adk-support-agent on $sdk, $model, streaming=$streaming"
            log=$(mktemp)
            if (cd apps/adk-support-agent && ADK_MODEL=$model ADK_STREAMING=$streaming "$python" examples/support_agent.py >"$log" 2>&1); then
                echo "PASS"
            else
                echo "FAIL"; grep -v Warning "$log" | tail -5; status=1
            fi
            rm -f "$log"
        done
    done
done
stop_gateway
exit $status
