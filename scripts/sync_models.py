"""Refresh the bundled model table from tetrateio/agent-router-models.

    python3 scripts/sync_models.py

MODELS_REPO=owner/name reads another copy of the list, for example a fork with a fix not yet merged.

Writes src/theagentrouter/model_catalog.json: one entry per "provider/model", in the field names
LiteLLM's model_cost uses, so code that reads litellm.model_cost keeps working.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

REPO = os.environ.get("MODELS_REPO", "tetrateio/agent-router-models")
SOURCE = "https://raw.githubusercontent.com/" + REPO + "/main/{name}.json"
CATALOGS = ["openai", "anthropic", "gemini", "groq", "xai", "deepinfra"]
OUT = Path(__file__).resolve().parent.parent / "src" / "theagentrouter" / "model_catalog.json"


def _per_token(per_million: str | float | None) -> float | None:
    if per_million in (None, ""):
        return None
    return float(per_million) / 1_000_000


def convert(record: dict) -> dict:
    caps = set(record.get("capabilities") or [])
    modal_in = set((record.get("modalities") or {}).get("input") or [])
    entry = {
        "litellm_provider": record.get("provider"),
        "mode": record.get("mode"),
        "max_input_tokens": record.get("contextWindow"),
        "max_tokens": record.get("maxOutputTokens") or record.get("contextWindow"),
        "max_output_tokens": record.get("maxOutputTokens"),
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
    }
    return {k: v for k, v in entry.items() if v is not None}


def main() -> None:
    table: dict[str, dict] = {}
    for name in CATALOGS:
        try:
            r = httpx.get(SOURCE.format(name=name), timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # a missing catalog should not stop the others
            print(f"skip {name}: {e}")
            continue
        for record in data.get("models", []):
            if record.get("isEnabled") is False:
                continue
            provider, model = record["provider"], record["model"]
            # Some catalogs already prefix the provider (DeepInfra: "deepinfra/Qwen/..."); add it once.
            key = model if model.startswith(provider + "/") else f"{provider}/{model}"
            table[key] = convert(record)
        print(f"{name}: {len(data.get('models', []))} models")
    OUT.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    print(f"wrote {len(table)} models to {OUT}")


if __name__ == "__main__":
    main()
