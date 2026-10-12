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

from theagentrouter.models import from_catalog

REPO = os.environ.get("MODELS_REPO", "tetrateio/agent-router-models")
SOURCE = "https://raw.githubusercontent.com/" + REPO + "/main/{name}.json"
CATALOGS = ["openai", "anthropic", "gemini", "groq", "xai", "deepinfra"]
OUT = Path(__file__).resolve().parent.parent / "src" / "theagentrouter" / "model_catalog.json"


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
        table.update(from_catalog(data))  # the same conversion register_model uses
        print(f"{name}: {len(data.get('models', []))} models")
    OUT.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    print(f"wrote {len(table)} models to {OUT}")


if __name__ == "__main__":
    main()
