"""Build pinned aigw source with the reviewed patches in local-gateway/patches/ for local/CI checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "daa9f891a8afcb18576d4593ad18870d1d18abac"
# Each patch, in the order it applies, with the upstream test that checks it.
PATCHES = [
    ("aigw-v1.2.0-client-secrets.patch", "TestRunCmdContextRetainsClientAuthenticationSecrets"),
    ("aigw-v1.2.0-mcp-session-seed.patch", "Test_mustStartExtProc_mcpSessionEncryptionSeed|Test_doMain"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local-gateway/src/theagentrouter_local_gateway/bin/aigw")
    parser.add_argument(
        "--source", type=Path, help="Read objects from an existing upstream clone instead of downloading"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "local-gateway/aigw-release.json").read_text())
    if (manifest["repository"], manifest["version"]) != ("theagentrouter/agent-router", "v1.2.0"):
        raise SystemExit("Review/rebase the local gateway patch before changing the pinned upstream release")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    build_env = dict(os.environ)
    # Clang otherwise embeds the build host's macOS version in cgo's deployment target.
    if sys.platform == "darwin":
        build_env.setdefault("MACOSX_DEPLOYMENT_TARGET", "13.0")
        minimum = build_env["MACOSX_DEPLOYMENT_TARGET"]
        for key in ("CGO_CFLAGS", "CGO_LDFLAGS"):
            build_env[key] = (build_env.get(key, "") + f" -mmacosx-version-min={minimum}").strip()
    with tempfile.TemporaryDirectory(prefix="sdk-aigw-build-") as directory:
        source = Path(directory) / "source"
        if args.source:
            subprocess.run(
                ["git", "clone", "--no-hardlinks", "--no-checkout", str(args.source.resolve()), str(source)], check=True
            )
            subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=source, check=True)
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--branch=v1.2.0",
                    "https://github.com/theagentrouter/agent-router",
                    str(source),
                ],
                check=True,
            )
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual != COMMIT:
            raise SystemExit("The upstream tag no longer matches the reviewed source commit")
        for name, _ in PATCHES:
            patch = ROOT / "local-gateway/patches" / name
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        subprocess.run(
            ["go", "test", "./cmd/aigw", "-run", "|".join(test for _, test in PATCHES), "-count=1"],
            cwd=source,
            env=build_env,
            check=True,
        )
        subprocess.run(["go", "build", "-o", str(output), "./cmd/aigw"], cwd=source, env=build_env, check=True)
        if sys.platform == "linux":
            symbols = subprocess.check_output(["readelf", "--version-info", str(output)], text=True)
            required = {tuple(map(int, version.split("."))) for version in re.findall(r"GLIBC_([0-9.]+)", symbols)}
            if required and max(required) > (2, 36):
                raise SystemExit(
                    "The built gateway requires glibc newer than 2.36; rebuild on the minimum platform or with CGO_ENABLED=0 before packaging"
                )
    print(f"Built aigw {COMMIT} with {len(PATCHES)} patches: {output}")


if __name__ == "__main__":
    main()
