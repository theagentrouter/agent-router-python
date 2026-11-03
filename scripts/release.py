# SPDX-License-Identifier: Apache-2.0
"""Make a release: set the version everywhere, date the changelog, check, commit and tag.

    python scripts/release.py 0.1.0a2            # a "Release 0.1.0a2" commit and the tag v0.1.0a2
    python scripts/release.py 0.1.0a2 --amend    # the same, folded into the last commit instead

It changes the version in the three pyproject.toml files, the exact pins between the packages and
both __version__ values, and turns "## Unreleased" in CHANGELOG.md into "## [0.1.0a2] - <today>".
Then it runs check_release.py, ruff and pytest; if any fails, the files are put back. It never
pushes: it prints the commands, because pushing the tag publishes to PyPI.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Every file that names the version.
FILES = (
    "pyproject.toml",
    "packages/theagentrouter/pyproject.toml",
    "local-gateway/pyproject.toml",
    "src/theagentrouter/__init__.py",
    "local-gateway/src/theagentrouter_local_gateway/__init__.py",
)
RELEASE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


def _run(*cmd: str, capture: bool = False) -> str:
    result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def current_version() -> str:
    match = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    if not match:
        raise SystemExit("No version in pyproject.toml")
    return match[1]


def set_version(version: str, today: str) -> None:
    """Write ``version`` everywhere the current one is, and date the changelog's Unreleased notes."""
    changelog = ROOT / "CHANGELOG.md"
    text = changelog.read_text()
    if not re.search(r"^## Unreleased$", text, re.MULTILINE):  # checked before anything is written
        raise SystemExit('CHANGELOG.md has no "## Unreleased" section to release')
    current = current_version()
    for name in FILES:
        path = ROOT / name
        path.write_text(
            path.read_text().replace(f'"{current}"', f'"{version}"').replace(f"=={current}", f"=={version}")
        )
    changelog.write_text(re.sub(r"^## Unreleased$", f"## [{version}] - {today}", text, count=1, flags=re.MULTILINE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="the new version, such as 1.2.3 or 1.2.3a1")
    parser.add_argument("--amend", action="store_true", help="fold into the last commit instead of a new one")
    args = parser.parse_args()
    version, tag = args.version, f"v{args.version}"

    if not RELEASE.match(version):
        raise SystemExit(f"{version} is not a release version such as 1.2.3 or 1.2.3a1")
    if version == current_version():
        raise SystemExit(f"The files already say {version}")
    if _run("git", "status", "--porcelain", capture=True):
        raise SystemExit("Commit or stash your changes first")
    if _run("git", "tag", "--list", tag, capture=True):
        raise SystemExit(f"Tag {tag} exists already")

    set_version(version, datetime.date.today().isoformat())
    try:
        _run(sys.executable, "scripts/check_release.py", "--tag", tag)
        _run(sys.executable, "-m", "ruff", "check", ".")
        _run(sys.executable, "-m", "ruff", "format", "--check", ".")
        _run(sys.executable, "-m", "pytest", "-q")
    except subprocess.CalledProcessError:
        _run("git", "checkout", "--", *FILES, "CHANGELOG.md")
        raise SystemExit("A check failed; the files are back as they were") from None

    _run("git", "add", *FILES, "CHANGELOG.md")
    _run("git", "commit", "-q", "-s", *(["--amend", "--no-edit"] if args.amend else ["-m", f"Release {version}"]))
    _run("git", "tag", "-a", tag, "-m", version)
    force = "--force-with-lease " if args.amend else ""
    print(f"\nTagged {tag} on {_run('git', 'rev-parse', '--short', 'HEAD', capture=True)}. To publish:\n")
    print(f"    git push {force}origin main")
    print(f"    git push origin {tag}    # starts release.yml, which publishes to PyPI")


if __name__ == "__main__":
    main()
