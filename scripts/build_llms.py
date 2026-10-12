"""Write llms.txt and llms-full.txt (https://llmstxt.org) from the README, docs, examples and skills.

    python scripts/build_llms.py          # write both files
    python scripts/build_llms.py --check  # fail if either is out of date (tests run this)

llms.txt is a short index of links for coding agents and AI search; llms-full.txt holds the whole
text, so an agent can read everything in one fetch. Links point at the repository's main branch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = [  # (path, what it covers), in reading order
    ("README.md", "install, a first call, fallbacks and retries, frameworks"),
    ("docs/usage.md", "every function, parameter, response field, error, embeddings, gateways and routes"),
    ("docs/coming-from-litellm.md", "what changes when moving from LiteLLM, and what is missing"),
    ("docs/frameworks.md", "Google ADK, the OpenAI Agents SDK and other frameworks"),
    ("docs/install.md", "installing, with and without the local gateway"),
    ("docs/local-gateway.md", "how the local gateway works and what it can and cannot do"),
]
SKILLS = ["skills/theagentrouter/SKILL.md", "skills/migrate-from-litellm/SKILL.md"]


def _project(field: str) -> str:
    """One string from pyproject.toml's [project] table (tomllib needs Python 3.11; the SDK supports 3.10)."""
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(rf'^{field}\s*=\s*"([^"]*)"', text, re.M)
    if not match:
        raise SystemExit(f"pyproject.toml has no {field}")
    return match.group(1)


def _repo_url() -> str:
    return _project("Homepage").rstrip("/") + "/blob/main/"


def _summary() -> str:
    return _project("description")


def _skill_description(path: str) -> str:
    match = re.search(r"^description:\s*(.+)$", (ROOT / path).read_text(), re.M)
    return match.group(1).strip() if match else path


def _example_docstring(path: Path) -> str:
    text = path.read_text()
    first = text.split('"""')[1] if text.startswith('"""') else ""
    return first.strip().splitlines()[0] if first.strip() else path.stem


def build() -> tuple[str, str]:
    url = _repo_url()
    examples = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
    index = [f"# theagentrouter\n\n> {_summary()}\n"]
    index.append(
        "LiteLLM's API (`completion`, `acompletion`, `embedding`, LiteLLM's errors and `Router`) in a small "
        "install. Model names are `provider/model`; keys come from the provider's environment variable. "
        "Fallbacks, retries and request limits are routes served by the local gateway (Agent Router).\n"
    )
    index.append("## Docs\n\n" + "\n".join(f"- [{p}]({url}{p}): {what}" for p, what in DOCS) + "\n")
    index.append(
        "## Examples\n\n"
        + "\n".join(f"- [{e.name}]({url}examples/{e.name}): {_example_docstring(e)}" for e in examples)
        + "\n"
    )
    index.append(
        "## Agent skills\n\n" + "\n".join(f"- [{s}]({url}{s}): {_skill_description(s)}" for s in SKILLS) + "\n"
    )
    index.append(f"## Optional\n\n- [llms-full.txt]({url}llms-full.txt): all of the above in one file\n")

    full = [f"# theagentrouter\n\n> {_summary()}\n"]
    for path, _ in DOCS:
        full.append(f"\n---\n\n<!-- {path} -->\n\n" + (ROOT / path).read_text().strip() + "\n")
    for s in SKILLS:
        full.append(f"\n---\n\n<!-- {s} -->\n\n" + (ROOT / s).read_text().strip() + "\n")
    for e in examples:
        full.append(f"\n---\n\n<!-- examples/{e.name} -->\n\n```python\n" + e.read_text().strip() + "\n```\n")
    return "\n".join(index), "".join(full)


def main() -> int:
    index, full = build()
    targets = {ROOT / "llms.txt": index, ROOT / "llms-full.txt": full}
    if "--check" in sys.argv:
        stale = [p.name for p, text in targets.items() if not p.exists() or p.read_text() != text]
        if stale:
            print(f"out of date: {', '.join(stale)}; run python scripts/build_llms.py")
            return 1
        return 0
    for path, text in targets.items():
        path.write_text(text)
        print(f"wrote {path.name} ({len(text):,} characters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
