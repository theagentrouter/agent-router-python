"""Build one wheel per platform: the wheel carries the aigw program for the computer it was built on.

With ``AGENTROUTER_NO_BINARY_WHEEL=1`` it builds the fallback instead: a ``py3-none-any`` wheel with no
program. pip installs it only where no platform wheel fits (Intel macOS, macOS before 13, Alpine,
older glibc, Windows), so ``pip install theagentrouter`` still succeeds there and the SDK runs
without a local gateway.
"""

from __future__ import annotations

import os
import platform
import struct
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return
        bin_dir = Path(self.root, "src", "theagentrouter_local_gateway", "bin")
        binary = bin_dir / "aigw"
        if os.environ.get("AGENTROUTER_NO_BINARY_WHEEL") == "1":
            if bin_dir.exists() and any(bin_dir.iterdir()):
                raise RuntimeError(f"the fallback wheel must not carry a program; empty {bin_dir} first")
            build_data["tag"] = "py3-none-any"
            return
        if not binary.is_file():
            raise RuntimeError(f"no aigw program in {bin_dir}; run `make local-gateway AIGW=/path/to/aigw` first")
        build_data["pure_python"] = False
        tag = next(tags.sys_tags()).platform
        if platform.system() == "Darwin":
            # Read the executable's deployment target rather than guessing from the build host.
            with binary.open("rb") as handle:
                header = handle.read(32)
                if len(header) != 32 or struct.unpack_from("<I", header)[0] != 0xFEEDFACF:
                    raise RuntimeError("aigw is not a native 64-bit Mach-O executable")
                cpu = struct.unpack_from("<I", header, 4)[0]
                arch = {0x0100000C: "arm64", 0x01000007: "x86_64"}.get(cpu)
                if arch != platform.machine():
                    raise RuntimeError("aigw architecture does not match the wheel build host")
                minimum = None
                for _ in range(struct.unpack_from("<I", header, 16)[0]):
                    command, size = struct.unpack("<II", handle.read(8))
                    data = handle.read(size - 8)
                    if command == 0x32:
                        minimum = struct.unpack_from("<I", data, 4)[0]
                    elif command == 0x24:
                        minimum = struct.unpack_from("<I", data)[0]
                if minimum is None:
                    raise RuntimeError("aigw does not declare a macOS deployment target")
                tag = f"macosx_{minimum >> 16}_{(minimum >> 8) & 255}_{arch}"
        elif platform.system() == "Linux":
            with binary.open("rb") as handle:
                header = handle.read(64)
                machine = struct.unpack_from("<H", header, 18)[0] if len(header) == 64 else None
                expected = {"x86_64": 62, "aarch64": 183}.get(platform.machine())
                if header[:6] != b"\x7fELF\x02\x01" or expected is None or machine != expected:
                    raise RuntimeError("aigw is not a little-endian ELF64 executable for the wheel build host")
                offset = struct.unpack_from("<Q", header, 32)[0]
                size, count = struct.unpack_from("<HH", header, 54)
                for index in range(count):
                    handle.seek(offset + index * size)
                    kind = struct.unpack("<I", handle.read(4))[0]
                    if kind in (2, 3):  # PT_DYNAMIC or PT_INTERP: a new binary needs another compatibility audit
                        raise RuntimeError("Linux release wheels require a statically linked aigw executable")
            # The pinned launcher is static. Its downloaded Envoy is verified on glibc 2.36+.
            tag = "manylinux_2_36_" + platform.machine()
        else:
            raise RuntimeError("local gateway wheels are only available for macOS and Linux")
        build_data["tag"] = f"py3-none-{tag}"
