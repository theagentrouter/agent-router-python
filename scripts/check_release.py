"""Check the three packages' versions and the release tag before building or publishing.

theagentrouter-core (the SDK, pyproject.toml), theagentrouter (packages/theagentrouter, no code) and
theagentrouter-local-gateway (local-gateway/) always share one version, and theagentrouter pins the
other two to it exactly.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _version(path: Path, name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*("[^"\n]+")', path.read_text(), re.MULTILINE)
    if not match:
        raise SystemExit(f"Missing {name} in {path}")
    return ast.literal_eval(match[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--print-version", action="store_true", help="print the version and stop")
    args = parser.parse_args()
    version = _version(ROOT / "pyproject.toml", "version")
    if args.print_version:
        print(version)
        return
    for path, name in (
        (ROOT / "src/theagentrouter/__init__.py", "__version__"),
        (ROOT / "local-gateway/pyproject.toml", "version"),
        (ROOT / "local-gateway/src/theagentrouter_local_gateway/__init__.py", "__version__"),
        (ROOT / "packages/theagentrouter/pyproject.toml", "version"),
    ):
        if _version(path, name) != version:
            raise SystemExit(f"Version mismatch in {path}")
    full = (ROOT / "packages/theagentrouter/pyproject.toml").read_text()
    for pin in (f"theagentrouter-core=={version}", f"theagentrouter-local-gateway=={version}"):
        if pin not in full:
            raise SystemExit(f"packages/theagentrouter does not pin {pin}")
    if f"theagentrouter-local-gateway=={version}" not in (ROOT / "pyproject.toml").read_text():
        raise SystemExit("theagentrouter-core's local-gateway extra does not pin the release version")
    if args.tag:
        if args.tag != f"v{version}":
            raise SystemExit(f"Tag {args.tag} does not match v{version}")
        if ".dev" in version or not re.search(
            rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
            (ROOT / "CHANGELOG.md").read_text(),
            re.MULTILINE,
        ):
            raise SystemExit("A publishable version needs a dated changelog entry")
    print(f"Release versions agree: {version}")


if __name__ == "__main__":
    main()
