# SPDX-License-Identifier: Apache-2.0
"""Google ADK support agent.

On Agent Router's SDK, LiteLLM's imports are replaced before any ADK module loads; that call is the
whole migration. On LiteLLM itself, where the app started, nothing is replaced.
"""

try:
    from theagentrouter.compat import install_as_litellm
except ImportError:
    pass
else:
    install_as_litellm()
