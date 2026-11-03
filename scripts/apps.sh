#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# The apps under apps/, offline against the fake provider, on real LiteLLM and on this SDK.
#   make apps                       both SDKs, both apps
#   make apps SDK=litellm APP=litellm-assistant
# One virtual environment per SDK (.venv-apps-litellm, .venv-apps-theagentrouter); never both SDKs in one.
set -eu
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
apps=${APP:-"adk-support-agent litellm-assistant"}
status=0
for sdk in ${SDK:-litellm theagentrouter}; do
    venv=.venv-apps-$sdk
    [ -x "$venv/bin/python" ] || "$PYTHON" -m venv "$venv"
    "$venv/bin/pip" install -q --upgrade pip
    for app in $apps; do
        if [ "$sdk" = litellm ]; then
            "$venv/bin/pip" install -q -e "apps/$app[litellm,test]"
        else
            gateway=""
            [ -x local-gateway/src/theagentrouter_local_gateway/bin/aigw ] && gateway="-e local-gateway"
            "$venv/bin/pip" install -q -e . $gateway -e "apps/$app[test]"
        fi
    done
    for app in $apps; do
        echo "== $app on $sdk"
        (cd "apps/$app" && APP_SDK=$sdk "../../$venv/bin/python" -m pytest -q -p no:cacheprovider) || status=1
    done
done
exit $status
