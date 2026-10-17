"""The app's own features, through the SDK under test, against the fake provider."""

import pytest
from assistant import app
from conftest import MODELS


@pytest.mark.parametrize("model", MODELS)
def test_ask(base, model):
    assert app.ask(model, "refunds yesterday", **base) == "echo: refunds yesterday"


@pytest.mark.parametrize("model", MODELS)
def test_stream(base, model):
    # The plain loop from LiteLLM's README (review H1: it crashed on the last chunk).
    assert "".join(app.stream(model, "refunds yesterday", **base)).strip() == "echo: refunds yesterday"


@pytest.mark.parametrize("model", MODELS)
async def test_astream(base, model):
    text = "".join([piece async for piece in app.astream(model, "refunds yesterday", **base)])
    assert text.strip() == "echo: refunds yesterday"


@pytest.mark.parametrize("model", MODELS)
def test_tool_round(base, fake, model):
    answer, calls = app.with_tools(model, "A-100", **base)
    assert calls == [{"query": "A-100"}]
    assert answer == "echo: shipped"


def test_embed(base):
    vectors = app.embed("openai/text-embedding-3-small", ["refunds", "exports"], **base)
    assert len(vectors) == 2 and all(vectors)
