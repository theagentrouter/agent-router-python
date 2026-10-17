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
# Each aigw patch, in the order it applies, with the package and upstream tests that check it.
PATCHES = [
    ("aigw-v1.2.0-client-secrets.patch", "./cmd/aigw", "TestRunCmdContextRetainsClientAuthenticationSecrets"),
    ("aigw-v1.2.0-mcp-session-seed.patch", "./cmd/aigw", "Test_mustStartExtProc_mcpSessionEncryptionSeed|Test_doMain"),
    ("aigw-v1.2.0-reload-on-sighup.patch", "./cmd/aigw", "TestRunCmdContextReloadReplacesTheConfiguration|Test_doMain"),
    ("aigw-v1.2.0-loopback-listeners.patch", "./cmd/aigw", "Test_mustStartExtProc_listensOnLoopback"),
    ("aigw-v1.2.0-loopback-listeners.patch", "./cmd/extproc/mainlib", "Test_parseAndValidateFlags_adminAddr"),
]
# Envoy Gateway, which aigw v1.2.0 builds against, binds its xDS, wasm cache and health servers to every
# address; the patch moves them to loopback. aigw then builds against the patched copy.
ENVOY_GATEWAY = ("v1.9.2", "f9f92a112a4033abed5c8d43a5f7e37da257403c", "envoy-gateway-v1.9.2-loopback-listeners.patch")


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
        for name in dict.fromkeys(name for name, _, _ in PATCHES):
            _apply(ROOT / "local-gateway/patches" / name, source)
        tag, commit, patch = ENVOY_GATEWAY
        gateway = Path(directory) / "envoy-gateway"
        subprocess.run(
            ["git", "clone", "--depth=1", f"--branch={tag}", "https://github.com/envoyproxy/gateway", str(gateway)],
            check=True,
        )
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=gateway, text=True).strip() != commit:
            raise SystemExit(f"Envoy Gateway {tag} no longer matches the reviewed source commit")
        _apply(ROOT / "local-gateway/patches" / patch, gateway)
        subprocess.run(
            ["go", "mod", "edit", f"-replace=github.com/envoyproxy/gateway={gateway}"], cwd=source, check=True
        )
        for package in dict.fromkeys(package for _, package, _ in PATCHES):
            tests = "|".join(test for _, p, test in PATCHES if p == package)
            subprocess.run(["go", "test", package, "-run", tests, "-count=1"], cwd=source, env=build_env, check=True)
        subprocess.run(["go", "build", "-o", str(output), "./cmd/aigw"], cwd=source, env=build_env, check=True)
        if sys.platform == "linux":
            symbols = subprocess.check_output(["readelf", "--version-info", str(output)], text=True)
            required = {tuple(map(int, version.split("."))) for version in re.findall(r"GLIBC_([0-9.]+)", symbols)}
            if required and max(required) > (2, 36):
                raise SystemExit(
                    "The built gateway requires glibc newer than 2.36; rebuild on the minimum platform or with CGO_ENABLED=0 before packaging"
                )
    print(f"Built aigw {COMMIT} with its patches, on patched Envoy Gateway {ENVOY_GATEWAY[0]}: {output}")


def _apply(patch: Path, source: Path) -> None:
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)


if __name__ == "__main__":
    main()
