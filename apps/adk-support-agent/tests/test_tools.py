# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from adk_agent.kb import KnowledgeBase
from adk_agent.tools import WorkspaceTools

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tools(tmp_path):
    kb = KnowledgeBase(PROJECT / "knowledge")
    yield WorkspaceTools(kb, tmp_path / "outputs")
    kb.close()


def test_search_and_reads_return_numbered_evidence(tools):
    hits = tools.search_knowledge("annual refund")
    assert hits["ok"]
    assert hits["passages"][0]["source"] == "billing.md"
    assert "L7:" in hits["passages"][0]["text"]
    assert tools.read_document("billing.md", 7, 7)["passage"]["text"].startswith("L7:")
    assert not tools.read_document("../.env")["ok"]


def test_saved_reports_have_valid_citations_and_unique_filenames(tools):
    content = "First-payment refund eligibility [billing.md:L7]"
    first = tools.save_report("resolution.md", content)
    second = tools.save_report("resolution.md", content)
    assert first["ok"] and second["ok"]
    assert Path(second["saved_to"]).name == "resolution-2.md"
    assert Path(first["saved_to"]).read_text() == content
    assert not tools.save_report("../escape.md", content)["ok"]
    assert not tools.save_report("missing.md", "No sources")["ok"]
    assert not tools.save_report("wrong.md", "[billing.md:L999]")["ok"]
