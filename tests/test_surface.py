"""The public surface must match surface.txt. After a deliberate change, run scripts/surface.py --write."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("surface_script", ROOT / "scripts" / "surface.py")
surface_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface_script)


def test_surface_matches_record():
    recorded = {line for line in (ROOT / "surface.txt").read_text().splitlines() if line}
    found = surface_script.surface()
    assert not recorded - found, f"missing or changed: {sorted(recorded - found)}"
    assert not found - recorded, f"new, not recorded (run scripts/surface.py --write): {sorted(found - recorded)}"


# The Enterprise SDK is `pip install tetrate-agentrouter`, imported as `tetrate.agentrouter`.
ENTERPRISE_NAMES = ("tetrate-agentrouter", "tetrate.agentrouter", "from tetrate import", "agentrouter_enterprise")


def test_open_source_sdk_never_refers_to_enterprise():
    package = ROOT / "src" / "theagentrouter"
    hits = [str(p) for p in package.rglob("*.py") if any(name in p.read_text() for name in ENTERPRISE_NAMES)]
    assert not hits, hits
