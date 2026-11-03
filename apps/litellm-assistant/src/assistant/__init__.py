# SPDX-License-Identifier: Apache-2.0
"""A small assistant written against LiteLLM, with no agent framework.

On Agent Router's SDK, LiteLLM's imports are replaced before the app imports litellm; that call is
the whole migration. On LiteLLM itself nothing is replaced.
"""

try:
    from theagentrouter.compat import install_as_litellm
except ImportError:
    pass
else:
    install_as_litellm()
