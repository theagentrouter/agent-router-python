"""Download the pinned upstream binary and verify its SHA-256 before packaging it."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    manifest = json.loads((ROOT / "local-gateway" / "aigw-release.json").read_text())
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "AMD64": "amd64"}.get(platform.machine())
    target = f"{platform.system().lower()}-{arch}"
    if target not in manifest["assets"]:
        raise SystemExit(f"No pinned aigw binary for {target}; use an external OpenAI-compatible gateway.")
    url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['version']}/aigw-{target}"
    folder = ROOT / "local-gateway" / "src" / "theagentrouter_local_gateway" / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
            output.close()
            if digest.hexdigest() != manifest["assets"][target]:
                raise SystemExit(f"SHA-256 mismatch for {url}")
            temporary.chmod(0o755)
            os.replace(temporary, folder / "aigw")
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified aigw {manifest['version']} for {target}")


if __name__ == "__main__":
    main()
