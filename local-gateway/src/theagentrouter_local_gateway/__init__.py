# SPDX-License-Identifier: Apache-2.0
"""Agent Router's aigw program, for theagentrouter's local gateway.

Installed with ``pip install theagentrouter``. The SDK starts the program itself. To run an exported
config on a server, ``binary_path()`` says where the program is. On a computer with no platform wheel, pip installs the fallback wheel,
which has no program: ``has_binary()`` is then False and the SDK runs without a local gateway.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

__version__ = "0.1.0a11"


def _path() -> Path:
    return Path(__file__).parent / "bin" / ("aigw.exe" if os.name == "nt" else "aigw")


def has_binary() -> bool:
    """False in the fallback wheel, which pip installs where no platform wheel fits."""
    return _path().is_file()


def binary_path() -> Path:
    """The aigw program in this package."""
    path = _path()
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; reinstall theagentrouter-local-gateway")
    if os.name != "nt" and not path.stat().st_mode & stat.S_IXUSR:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
