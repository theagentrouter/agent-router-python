"""llms.txt and llms-full.txt stay in step with the docs, examples and skills they are built from."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_llms_files_are_up_to_date():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_llms.py"), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


def test_every_example_and_skill_is_listed():
    index = (ROOT / "llms.txt").read_text()
    for path in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        assert f"examples/{path.name}" in index
    for skill in ("skills/theagentrouter/SKILL.md", "skills/migrate-from-litellm/SKILL.md"):
        assert skill in index
