"""Compatibility surfaces for code written against other SDKs."""

from __future__ import annotations

import sys
import warnings
from types import ModuleType


def install_as_litellm() -> None:
    """Make `import litellm` (and `litellm.types.utils`) resolve to agentrouter's compat module.

    For frameworks that import LiteLLM by name, such as Google ADK's LiteLlm class and Aider.
    """
    from . import litellm

    previous = sys.modules.get("litellm")
    if previous is not None and previous is not litellm:
        warnings.warn(
            "install_as_litellm() replaces an already imported LiteLLM module; existing references still use the previous module.",
            UserWarning,
            stacklevel=2,
        )
        for name in list(sys.modules):
            if name.startswith("litellm."):
                del sys.modules[name]
    sys.modules["litellm"] = litellm
    sys.modules["litellm.types"] = litellm.types
    sys.modules["litellm.types.utils"] = litellm.types.utils
    sys.modules["litellm.anthropic"] = litellm.anthropic
    sys.modules["litellm.anthropic.messages"] = litellm.anthropic.messages
    sys.modules["litellm.exceptions"] = litellm.exceptions
    sys.modules["litellm.cost_calculator"] = litellm.cost_calculator
    litellm.utils = ModuleType("litellm.utils")
    for name in (
        "completion_cost",
        "cost_per_token",
        "token_counter",
        "get_llm_provider",
        "get_model_info",
        "get_supported_openai_params",
        "stream_chunk_builder",
        "ModelResponse",
        "ModelResponseStream",
        "Usage",
    ):
        setattr(litellm.utils, name, getattr(litellm, name))
    sys.modules["litellm.utils"] = litellm.utils
    litellm.integrations = ModuleType("litellm.integrations")
    litellm.integrations.__path__ = []
    custom_logger = ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = _CustomLogger
    litellm.integrations.custom_logger = custom_logger
    sys.modules["litellm.integrations"] = litellm.integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger


class _CustomLogger:
    """Subclassable callback interface compatible with LiteLLM's custom logger hooks."""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
