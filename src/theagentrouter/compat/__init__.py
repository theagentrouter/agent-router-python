"""Compatibility surfaces for code written against other SDKs."""

from __future__ import annotations

import sys


def install_as_litellm() -> None:
    """Make `import litellm` (and `litellm.types.utils`) resolve to agentrouter's compat module.

    For frameworks that import LiteLLM by name, such as Google ADK's LiteLlm class and Aider.
    """
    from . import litellm

    sys.modules["litellm"] = litellm
    sys.modules["litellm.types"] = litellm.types
    sys.modules["litellm.types.utils"] = litellm.types.utils
    sys.modules["litellm.anthropic"] = litellm.anthropic
    sys.modules["litellm.anthropic.messages"] = litellm.anthropic.messages
