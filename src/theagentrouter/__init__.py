# SPDX-License-Identifier: Apache-2.0
"""theagentrouter: LiteLLM's API, with Agent Router underneath.

    import theagentrouter
    resp = theagentrouter.completion(model="anthropic/claude-sonnet-4-5", messages=[{"role": "user", "content": "hi"}])

Every call goes through a gateway; the SDK never calls a provider itself. By default that is the
local gateway, Agent Router on this computer, which the first call starts and which reads provider
keys from the environment. With AGENTROUTER_BASE_URL set, every call goes to that gateway instead.

Embeddings take LiteLLM's arguments too:

    theagentrouter.embedding(model="openai/text-embedding-3-small", input=["refunds", "returns"])

Routes give a model name of your own fallbacks and retries, served by the local gateway:

    theagentrouter.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"])
    theagentrouter.completion(model="chat", messages=[...])
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from . import anthropic, exceptions, mock, models, transport
from ._http import aclose, close
from .anthropic_messages import AnthropicMessagesResponse, anthropic_messages
from .embeddings import EmbeddingResponse, aembedding, embedding
from .models import (
    completion_cost,
    cost_per_token,
    get_model_info,
    model_cost,
    register_model,
    supports_function_calling,
    supports_prompt_caching,
    supports_reasoning,
    supports_response_schema,
    supports_vision,
)
from .responses import ResponsesAPIResponse, aresponses, responses
from .routes import Limit, Retry, Route, list_routes, remove_route, route
from .routing import validate_environment
from .tokens import token_counter
from .types import ModelResponse, ModelResponseStream, stream_chunk_builder

__version__ = "0.1.0a10"
drop_params = False  # as litellm.drop_params: True drops parameters the provider does not take, instead of raising


def completion(
    model: str, messages: list[Any], *, stream: bool = False, tools: list[dict[str, Any]] | None = None, **params: Any
) -> ModelResponse | Iterator[ModelResponseStream]:
    """Same arguments as litellm.completion. Returns a ModelResponse, or an iterator of chunks when streaming.

    Common params: api_key, api_base, timeout, temperature, max_tokens, tool_choice, response_format,
    extra_headers, extra_body, user.
    """
    if mock.requested(params):  # LiteLLM's mock_response: answered here, nothing is sent
        return mock.stream(model, params) if stream else mock.complete(model, params)
    if stream:
        return transport.stream(model, messages, tools, params)
    return transport.complete(model, messages, tools, params)


async def acompletion(
    model: str, messages: list[Any], *, stream: bool = False, tools: list[dict[str, Any]] | None = None, **params: Any
) -> ModelResponse | AsyncIterator[ModelResponseStream]:
    """Same arguments as litellm.acompletion."""
    if mock.requested(params):
        return await (mock.astream(model, params) if stream else mock.acomplete(model, params))
    if stream:
        return await transport.astream(model, messages, tools, params)
    return await transport.acomplete(model, messages, tools, params)


__all__ = [
    "close",
    "aclose",
    "EmbeddingResponse",
    "aembedding",
    "embedding",
    "Limit",
    "Retry",
    "Route",
    "list_routes",
    "remove_route",
    "route",
    "acompletion",
    "completion",
    "completion_cost",
    "cost_per_token",
    "exceptions",
    "get_model_info",
    "model_cost",
    "models",
    "register_model",
    "ModelResponse",
    "ModelResponseStream",
    "supports_function_calling",
    "supports_prompt_caching",
    "supports_reasoning",
    "supports_response_schema",
    "supports_vision",
    "token_counter",
    "responses",
    "aresponses",
    "ResponsesAPIResponse",
    "anthropic",
    "anthropic_messages",
    "AnthropicMessagesResponse",
    "stream_chunk_builder",
    "validate_environment",
]
