"""Model details: prices, context windows and capabilities, with LiteLLM's names.

The table is a snapshot of tetrateio/agent-router-models, refreshed with scripts/sync_models.py.
Keys are "provider/model"; lookups also accept a bare model name with custom_llm_provider.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from . import exceptions


def _load() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(resources.files(__package__).joinpath("model_catalog.json").read_text())
    except FileNotFoundError:
        return {}


# Same name LiteLLM uses, so code reading litellm.model_cost works unchanged.
model_cost: dict[str, dict[str, Any]] = _load()


def _key(model: str, custom_llm_provider: str | None = None) -> str | None:
    if model in model_cost:
        return model
    if custom_llm_provider and f"{custom_llm_provider}/{model}" in model_cost:
        return f"{custom_llm_provider}/{model}"
    matches = [k for k in model_cost if k.split("/", 1)[1] == model] if "/" not in model else []
    return matches[0] if matches else None


# Bare names LiteLLM itself resolves: OpenAI's and Anthropic's. Everything else needs provider/model there
# too, so code moved from LiteLLM already has the prefix.
_BARE = (
    ("openai", ("gpt-", "o1", "o3", "o4", "chatgpt-", "text-embedding-", "dall-e", "whisper", "tts-")),
    ("anthropic", ("claude-",)),
)


def provider_of(name: str) -> str | None:
    """The provider for a model name given without one (``gpt-4o-mini``), as LiteLLM finds it, or None.

    Only OpenAI and Anthropic names, as in LiteLLM. A bare ``gemini-...`` is refused: LiteLLM sends it
    to Google Cloud's Vertex AI, and sending it to the Gemini API instead would call another service
    with other keys.
    """
    if name.startswith(("gemini-", "gemma-")):
        raise exceptions.BadRequestError(
            f"model {name!r} has no provider. LiteLLM sends a bare Gemini name to Google Cloud's Vertex AI, "
            f"which the local gateway does not serve yet; for the Gemini API (GEMINI_API_KEY) use gemini/{name}",
            model=name,
        )
    for provider, prefixes in _BARE:
        if f"{provider}/{name}" in model_cost or name.startswith(prefixes):
            return provider
    return None


def get_model_info(model: str, custom_llm_provider: str | None = None) -> dict[str, Any]:
    key = _key(model, custom_llm_provider)
    if key is None:
        raise exceptions.NotFoundError(
            f"no model details for {model!r}; it may still work, the table only lists known models",
            model=model,
            llm_provider=custom_llm_provider,
        )
    return {"key": key, **model_cost[key]}


def _supports(model: str, field: str, custom_llm_provider: str | None) -> bool:
    key = _key(model, custom_llm_provider)
    return bool(key and model_cost[key].get(field))


def supports_function_calling(model: str, custom_llm_provider: str | None = None) -> bool:
    return _supports(model, "supports_function_calling", custom_llm_provider)


def supports_response_schema(model: str, custom_llm_provider: str | None = None) -> bool:
    return _supports(model, "supports_response_schema", custom_llm_provider)


def supports_reasoning(model: str, custom_llm_provider: str | None = None) -> bool:
    return _supports(model, "supports_reasoning", custom_llm_provider)


def supports_vision(model: str, custom_llm_provider: str | None = None) -> bool:
    return _supports(model, "supports_vision", custom_llm_provider)


def supports_prompt_caching(model: str, custom_llm_provider: str | None = None) -> bool:
    return _supports(model, "supports_prompt_caching", custom_llm_provider)


def cost_per_token(
    model: str, prompt_tokens: int = 0, completion_tokens: int = 0, custom_llm_provider: str | None = None
) -> tuple[float, float]:
    key = _key(model, custom_llm_provider)
    if key is None:
        return 0.0, 0.0
    info = model_cost[key]
    return (
        prompt_tokens * (info.get("input_cost_per_token") or 0.0),
        completion_tokens * (info.get("output_cost_per_token") or 0.0),
    )


def completion_cost(
    completion_response: Any = None,
    model: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    custom_llm_provider: str | None = None,
) -> float:
    """Cost of one call in US dollars, from the table. Zero when the model is not listed."""
    if completion_response is not None:
        hidden = getattr(completion_response, "_hidden_params", {}) or {}
        model = model or hidden.get("model_key") or completion_response.get("model")
        usage = completion_response.get("usage")
        if usage:
            # Chat Completions name the counts prompt/completion; the Responses API, input/output.
            def read(*names: str) -> int:
                for name in names:
                    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                    if value is not None:
                        return int(value)
                return 0

            prompt_tokens = read("prompt_tokens", "input_tokens")
            completion_tokens = read("completion_tokens", "output_tokens")
    if not model:
        return 0.0
    prompt, completion = cost_per_token(model, prompt_tokens, completion_tokens, custom_llm_provider)
    return prompt + completion
