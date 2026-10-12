"""List the SDK's public surface: functions with their arguments, classes, values and settings.

    python3 scripts/surface.py            # print it
    python3 scripts/surface.py --write    # record it in surface.txt after a deliberate change

tests/test_surface.py fails when the package and surface.txt disagree. The Enterprise SDK
checks itself against the same file: everything listed must stay present and unchanged.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
import types
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "surface.txt"
MODULES = (
    "theagentrouter",
    "theagentrouter.exceptions",
    "theagentrouter.compat",
    "theagentrouter.compat.litellm",
)


def _signature(obj) -> str:
    try:
        return "(" + ", ".join(inspect.signature(obj).parameters) + ")"
    except (TypeError, ValueError):
        return ""


def _module(name: str) -> set[str]:
    mod = importlib.import_module(name)
    names = getattr(mod, "__all__", None) or [n for n in vars(mod) if not n.startswith("_")]
    out = set()
    for attr in names:
        obj = getattr(mod, attr)
        if (
            inspect.ismodule(obj)
            or attr == "annotations"
            or obj is types.ModuleType
            or getattr(typing, attr, None) is obj
        ):
            continue
        if inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
            out.add(f"{name}.{attr}{_signature(obj)}")
        elif inspect.isclass(obj):
            if not obj.__module__.startswith("theagentrouter"):
                continue  # imported for internal use (typing helpers and the like), not offered
            out.add(f"{name}.{attr} [class]")
        else:
            out.add(f"{name}.{attr} [value]")
    return out


def surface() -> set[str]:
    items: set[str] = set()
    for name in MODULES:
        items |= _module(name)
    package = Path(importlib.import_module("theagentrouter").__file__).parent
    for path in package.rglob("*.py"):
        items |= {f"setting {v}" for v in re.findall(r'"(AGENTROUTER_[A-Z_]+)"', path.read_text())}
    return items


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    found = sorted(surface())
    if "--write" in sys.argv:
        CONTRACT.write_text("\n".join(found) + "\n")
        print(f"wrote {len(found)} items to {CONTRACT.name}")
    else:
        print("\n".join(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
