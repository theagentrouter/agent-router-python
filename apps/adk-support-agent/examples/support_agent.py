# SPDX-License-Identifier: Apache-2.0
"""Run ADK → model adapter → MCP tickets + local KB with the configured provider."""

import asyncio
import json
import os
import sqlite3
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / ".env")
os.environ.setdefault("AGENTROUTER_HOME", str(PROJECT / "data" / "agentrouter"))

from adk_agent.agent import support_app  # noqa: E402

TASK = """Read MCP ticket ACME-101, including the invoice and usage facts. Search the KB
for refund eligibility and data export steps. Save a cited support-resolution draft
as adk-resolution.md. Then add an internal note to ACME-101 with your findings and the
actual saved report path. Read the ticket again to verify the note was persisted.
Do not promise refund approval. Cite KB policy documents; ticket facts come from MCP.
"""


def note_count(database: Path) -> int:
    if not database.exists():
        return 0
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM notes WHERE ticket_id = 'ACME-101'"
        ).fetchone()[0]


async def main():
    import litellm

    sdk = "theagentrouter" if litellm.__name__.startswith("theagentrouter") else "litellm"
    try:
        version("litellm")
    except PackageNotFoundError:
        litellm_installed = False
    else:
        litellm_installed = True
    if sdk == "theagentrouter" and litellm_installed:
        raise RuntimeError("The migrated example must run without LiteLLM installed")
    model = os.getenv("ADK_MODEL", "openai/gpt-5.4")
    provider = "ANTHROPIC" if model.startswith("anthropic/") else "OPENAI"
    if not os.getenv("AGENTROUTER_BASE_URL") and not os.getenv(f"{provider}_API_KEY"):
        raise SystemExit(f"Add {provider}_API_KEY to .env first")
    output = PROJECT / "outputs" / provider.lower()
    database = PROJECT / "data" / "tickets.db"
    before = note_count(database)
    async with support_app(PROJECT, model=model, output_directory=output) as app:
        result = await app.run(
            TASK, streaming=os.getenv("ADK_STREAMING", "false").lower() == "true"
        )
        saves = [
            item["response"]["saved_to"]
            for item in result.tool_results
            if item["name"] == "save_report" and item["response"].get("ok")
        ]
        persisted = note_count(database) - before
        if (
            not saves
            or persisted < 1
            or not {"tickets_get_ticket", "tickets_add_ticket_note"} <= set(result.tool_calls)
        ):
            raise RuntimeError("ADK did not complete the MCP + KB + report task")
        citations = app.kb.validate_citations(Path(saves[-1]).read_text(encoding="utf-8"))
        verification = {
            "framework": "google-adk",
            "framework_version": version("google-adk"),
            "sdk": sdk,
            "sdk_version": version("litellm" if sdk == "litellm" else "theagentrouter-core"),
            "litellm_distribution_installed": litellm_installed,
            "model": model,
            "tools": result.tool_calls,
            "total_tokens": result.total_tokens,
            "report": saves[-1],
            "valid_citations": citations,
            "persisted_ticket_notes": persisted,
        }
        (output / "verification.json").write_text(
            json.dumps(verification, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Tools:", ", ".join(result.tool_calls), flush=True)
        print(f"\n{result.answer}\n\nVerified {persisted} persisted ticket note(s).", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
