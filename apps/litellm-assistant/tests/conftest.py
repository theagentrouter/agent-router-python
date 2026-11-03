# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from sdk import fake_provider_class, sdk_under_test  # noqa: E402

SDK = sdk_under_test()
import assistant  # noqa: E402,F401  the app replaces litellm first when it runs on Agent Router's SDK

os.environ["AGENTROUTER_AUTOSTART"] = "0"  # offline tests name the endpoint; gateway tests start their own
OPENAI = "openai/gpt-4o-mini"
CLAUDE = "anthropic/claude-sonnet-4-5"
MODELS = [OPENAI, CLAUDE]


def known_gap(reason: str):
    """Where this SDK differs from LiteLLM today. Strict: once fixed, the test fails until this mark goes."""
    return pytest.mark.xfail(SDK == "theagentrouter", reason=reason, strict=True)


only_agent_router = pytest.mark.skipif(SDK != "theagentrouter", reason="Agent Router's SDK only")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_API_KEY", "AGENTROUTER_PROJECT", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", str(tmp_path / "agentrouter"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture
def fake():
    server = fake_provider_class()()
    yield server
    server.close()


@pytest.fixture
def base(fake):
    return {"api_base": fake.url}
