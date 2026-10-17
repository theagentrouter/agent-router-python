"""Shared by the apps' tests: which SDK is under test, and the repository's fake provider.

Every run names its SDK with APP_SDK, and the run stops when the environment holds a different
one. Without this check, a run meant for LiteLLM can quietly test this SDK, or the reverse.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SDKS = ("litellm", "theagentrouter")


def installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


def sdk_under_test() -> str:
    wanted = os.environ.get("APP_SDK")
    if wanted not in SDKS:
        raise RuntimeError("set APP_SDK=litellm or APP_SDK=theagentrouter")
    has = {"litellm": installed("litellm"), "theagentrouter": installed("theagentrouter-core")}
    other = SDKS[1 - SDKS.index(wanted)]
    if not has[wanted] or has[other]:
        raise RuntimeError(f"APP_SDK={wanted} needs {wanted} installed and {other} not installed; found {has}")
    return wanted


def fake_provider_class():
    """The fake provider the SDK's own tests use: OpenAI and Anthropic formats, errors by model name."""
    sys.path.insert(0, str(REPO / "tests"))
    from fake_provider import FakeProvider

    return FakeProvider
