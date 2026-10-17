"""Model details: prices, context windows and capabilities, with LiteLLM's names.

The table is a snapshot of tetrateio/agent-router-models, refreshed with scripts/sync_models.py.
Keys are "provider/model"; lookups also accept a bare model name with custom_llm_provider.
"""

from __future__ import annotations

import json
import warnings
from importlib import resources
from typing import Any

from . import exceptions

_unknown_prices: set[str] = set()


def _load() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(resources.files(__package__).joinpath("model_catalog.json").read_text())
    except FileNotFoundError:
        return {}


# Same name LiteLLM uses, so code reading litellm.model_cost works unchanged.
model_cost: dict[str, dict[str, Any]] = _load()


def _load_lifecycle() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(resources.files(__package__).joinpath("model_lifecycle.json").read_text())
    except FileNotFoundError:
        return {"retired": {}, "deprecated": {}}


# Models the list marks deprecated: retired ones (no longer offered) by name -> the list's reason, and
# still-offered ones by name -> their retirement date and reason. Not part of model_cost.
lifecycle: dict[str, dict[str, Any]] = _load_lifecycle()
_deprecation_warned: set[str] = set()
_table = model_cost  # register_model's argument has the same name, as in LiteLLM


# --- tetrateio/agent-router-models -------------------------------------------------------------------

# Providers whose context window counts the output too: the input limit is the window less the output.
_WINDOW_HOLDS_OUTPUT = frozenset({"openai"})


def _per_token(per_million: str | float | None) -> float | None:
    return None if per_million in (None, "") else float(per_million) / 1_000_000


def _from_catalog_record(record: dict[str, Any]) -> dict[str, Any]:
    """One model from tetrateio/agent-router-models, in LiteLLM's field names."""
    caps = set(record.get("capabilities") or [])
    metadata = record.get("metadata") or {}
    modal_in = set((record.get("modalities") or {}).get("input") or [])
    limits = record.get("limits") or {}
    window, output = record.get("contextWindow"), limits.get("max_output_tokens")
    max_input = limits.get("max_input_tokens") or (
        window - output if window and output and record.get("provider") in _WINDOW_HOLDS_OUTPUT else window
    )
    entry = {
        "litellm_provider": record.get("provider"),
        "mode": record.get("mode"),
        "max_input_tokens": max_input,
        "max_tokens": output,  # LiteLLM's max_tokens is the output limit, never the window
        "max_output_tokens": output,
        "input_cost_per_token": _per_token(record.get("inputTokensPricePerMillion")),
        "output_cost_per_token": _per_token(record.get("outputTokensPricePerMillion")),
        "cache_read_input_token_cost": _per_token(record.get("cachedTokensPricePerMillion")),
        "cache_creation_input_token_cost": _per_token(record.get("cachingTokensPricePerMillion")),
        "supports_function_calling": "tool_choice" in caps or "function_calling" in caps,
        "supports_tool_choice": "tool_choice" in caps,
        "supports_response_schema": "structured_outputs" in caps,
        "supports_reasoning": "reasoning" in caps,
        "supports_vision": "vision" in caps or "image" in modal_in,
        "supports_prompt_caching": "prompt_caching" in caps,
        "supports_pdf_input": "pdf_support" in caps,
        # LiteLLM's name for the date a model stops being served.
        "deprecation_date": metadata.get("retirement_date") if metadata.get("deprecated") else None,
        # Effort levels a model takes; for Claude, it also means adaptive thinking instead of a token budget.
        "reasoning_efforts": metadata.get("reasoning_efforts") or None,
    }
    return {k: v for k, v in entry.items() if v is not None}


def _catalog_key(record: dict[str, Any]) -> str:
    provider, model = record["provider"], record["model"]
    # Some catalogs already prefix the provider (DeepInfra: "deepinfra/Qwen/..."); add it once.
    return model if model.startswith(provider + "/") else f"{provider}/{model}"


def lifecycle_from_catalog(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The deprecated models of a tetrateio/agent-router-models file: retired, or still offered with a date."""
    out: dict[str, dict[str, Any]] = {"retired": {}, "deprecated": {}}
    for record in data.get("models", []):
        metadata = record.get("metadata") or {}
        if not metadata.get("deprecated"):
            continue
        date = metadata.get("retirement_date")
        reason = metadata.get("deprecated_reason") or (f"retires on {date}" if date else "deprecated")
        if record.get("isEnabled") is False:
            out["retired"][_catalog_key(record)] = reason
        else:
            out["deprecated"][_catalog_key(record)] = {"date": date, "reason": reason}
    return out


def retired_reason(model: str) -> str | None:
    """Why the model list no longer offers this model, or None when it is not retired there."""
    return lifecycle["retired"].get(model)


def warn_if_deprecated(model: str) -> None:
    """Warn once per program when the model list marks a model deprecated."""
    info = lifecycle["deprecated"].get(model)
    if info and model not in _deprecation_warned:
        _deprecation_warned.add(model)
        when = f" and retires on {info['date']}" if info.get("date") else ""
        warnings.warn(f"{model} is deprecated{when}: {info['reason']}", stacklevel=3)


def from_catalog(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A tetrateio/agent-router-models file (``{"version": ..., "models": [...]}``) as LiteLLM's model_cost entries."""
    table = {}
    for record in data.get("models", []):
        if record.get("isEnabled") is False:
            continue
        table[_catalog_key(record)] = _from_catalog_record(record)
    return table


# LiteLLM's provider names where the SDK's differ.
_PROVIDER_NAMES = {"together_ai": "together"}


def register_model(model_cost: dict[str, Any] | str) -> dict[str, dict[str, Any]]:
    """LiteLLM's register_model: add models, or change their details, in LiteLLM's format.

    ``model_cost`` is a dict of model name to details, or the URL of a JSON file of them, such as
    LiteLLM's ``model_prices_and_context_window.json``, or a file of tetrateio/agent-router-models
    (``https://raw.githubusercontent.com/tetrateio/agent-router-models/main/openai.json``). A name without a provider takes it from the
    entry's ``litellm_provider``. Entries that are not models (no ``mode``, no prices) are skipped. Registered models get prices and details, a rule in the local
    gateway when it next starts, and a rule in ``local_gateway.export_config()``. They last for this
    program only.
    """
    if isinstance(model_cost, str):
        import httpx

        response = httpx.get(model_cost, timeout=60, follow_redirects=True)
        response.raise_for_status()
        model_cost = response.json()
    if not isinstance(model_cost, dict):
        raise exceptions.BadRequestError("register_model takes a dict of model name to details, or a URL of one")
    if isinstance(model_cost.get("models"), list):  # a tetrateio/agent-router-models file
        for kind, models in lifecycle_from_catalog(model_cost).items():
            lifecycle[kind].update(models)
        model_cost = from_catalog(model_cost)
    for name, details in model_cost.items():
        if not isinstance(details, dict) or name == "sample_spec":  # LiteLLM's file documents its fields here
            continue
        if "/" in name:
            prefix, rest = name.split("/", 1)
            key = f"{_PROVIDER_NAMES.get(prefix, prefix)}/{rest}"
        elif details.get("litellm_provider"):
            provider = details["litellm_provider"]
            key = f"{_PROVIDER_NAMES.get(provider, provider)}/{name}"
        elif not any(k == "mode" or k.endswith("cost_per_token") for k in details):
            continue  # not a model: LiteLLM's file keeps other data at the top level too
        else:
            raise exceptions.BadRequestError(
                f"register_model: {name!r} has no provider; write provider/model or set litellm_provider", model=name
            )
        entry = _table.setdefault(key, {})
        entry.update(details)
        entry.setdefault("litellm_provider", key.split("/", 1)[0])
    return _table


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
        reason = retired_reason(model if "/" in model else f"{custom_llm_provider}/{model}")
        if reason is not None:
            raise exceptions.NotFoundError(
                f"{model} is no longer offered: {reason}", model=model, llm_provider=custom_llm_provider
            )
        raise exceptions.NotFoundError(
            f"no model details for {model!r}; it may still work, the table only lists known models",
            model=model,
            llm_provider=custom_llm_provider,
        )
    # As in LiteLLM, a limit the list does not know is None rather than missing.
    return {"key": key, "max_tokens": None, "max_input_tokens": None, "max_output_tokens": None, **model_cost[key]}


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
        if model not in _unknown_prices:
            _unknown_prices.add(model)
            warnings.warn(
                f"No price is known for {model!r}; the reported cost is an unpriced estimate of $0.",
                UserWarning,
                stacklevel=2,
            )
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
    prompt: str | None = None,
    completion: str | None = None,
) -> float:
    """Cost of one call in US dollars, from the table. Zero when the model is not listed."""
    cache_read = cache_write = 0
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
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            cache_read = read("cache_read_input_tokens") or int(details.get("cached_tokens") or 0)
            cache_write = read("cache_creation_input_tokens") or int(details.get("cache_creation_tokens") or 0)
    if not model:
        return 0.0
    if prompt is not None or completion is not None:
        from .tokens import _estimate

        if prompt is not None:
            prompt_tokens = _estimate(model, prompt, None, None, None, False, None)
        if completion is not None:
            completion_tokens = _estimate(model, completion, None, None, None, True, None)
    input_cost, output_cost = cost_per_token(model, prompt_tokens, completion_tokens, custom_llm_provider)
    key = _key(model, custom_llm_provider)
    if key:
        info = model_cost[key]
        ordinary = info.get("input_cost_per_token") or 0
        if cache_read:
            rate = info.get("cache_read_input_token_cost")
            input_cost += cache_read * ((ordinary if rate is None else rate) - ordinary)
        if cache_write:
            rate = info.get("cache_creation_input_token_cost")
            input_cost += cache_write * ((ordinary if rate is None else rate) - ordinary)
    return input_cost + output_cost
