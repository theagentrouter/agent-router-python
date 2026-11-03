"""Refresh the bundled model table from theagentrouter/agent-router-models.

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

from theagentrouter.models import from_catalog, lifecycle_from_catalog

REPO = os.environ.get("MODELS_REPO", "theagentrouter/agent-router-models")
SOURCE = "https://raw.githubusercontent.com/" + REPO + "/main/{name}.json"
CATALOGS = ["openai", "anthropic", "gemini", "groq", "xai", "deepinfra"]
OUT = Path(__file__).resolve().parent.parent / "src" / "theagentrouter" / "model_catalog.json"
LIFECYCLE = OUT.with_name("model_lifecycle.json")  # deprecated and retired models, outside model_cost


def main() -> None:
    table: dict[str, dict] = {}
    lifecycle: dict[str, dict] = {"retired": {}, "deprecated": {}}
    for name in CATALOGS:
        try:
            r = httpx.get(SOURCE.format(name=name), timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # a missing catalog should not stop the others
            print(f"skip {name}: {e}")
            continue
        table.update(from_catalog(data))  # the same conversion register_model uses
        for kind, models in lifecycle_from_catalog(data).items():
            lifecycle[kind].update(models)
        print(f"{name}: {len(data.get('models', []))} models")
    OUT.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    print(f"wrote {len(table)} models to {OUT}")
    LIFECYCLE.write_text(json.dumps({k: dict(sorted(v.items())) for k, v in lifecycle.items()}, indent=1) + "\n")
    print(
        f"wrote {len(lifecycle['retired'])} retired and {len(lifecycle['deprecated'])} deprecated models to {LIFECYCLE}"
    )


if __name__ == "__main__":
    main()
