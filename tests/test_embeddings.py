"""Embeddings: LiteLLM's embedding() and aembedding(), directly, through a gateway and through the real aigw."""

from __future__ import annotations

import pytest

import theagentrouter as ar
from theagentrouter import exceptions
from theagentrouter.compat import litellm

from .fake_provider import FakeProvider
from .test_local_gateway import needs_aigw, needs_xds_port

TEXTS = ["refunds yesterday", "returns"]


@pytest.fixture
def direct(monkeypatch):
    server = FakeProvider()
    for var in ("AGENTROUTER_BASE_URL", "AGENTROUTER_API_KEY", "AGENTROUTER_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTROUTER_HOME", "/nonexistent-agentrouter-home")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    yield server
    server.close()


def test_embedding_shape_matches_litellm(direct):
    r = ar.embedding(model="openai/text-embedding-3-small", input=TEXTS, api_base=direct.url)
    path, headers, body = direct.requests[-1]
    assert path == "/v1/embeddings"
    assert body == {"model": "text-embedding-3-small", "input": TEXTS}  # the provider gets its own name
    assert headers["authorization"] == "Bearer sk-test"
    assert r.object == "list" and len(r.data) == 2
    assert r.data[0]["embedding"][0] == len(TEXTS[0]) and r.data[1].embedding[0] == len(TEXTS[1])  # both styles
    assert [d["index"] for d in r.data] == [0, 1]
    assert r.usage.prompt_tokens == 6 and r.usage.total_tokens == 6
    assert r._hidden_params["route"] == "api_base"
    assert r._hidden_params["response_cost"] > 0  # text-embedding-3-small is in the bundled table


def test_one_string_and_provider_arguments(direct):
    r = ar.embedding(model="openai/text-embedding-3-small", input="hi", dimensions=5, user="u1", api_base=direct.url)
    _, _, body = direct.requests[-1]
    assert body["input"] == "hi" and body["dimensions"] == 5 and body["user"] == "u1"
    assert len(r.data) == 1 and len(r.data[0]["embedding"]) == 5


def test_anthropic_has_no_embeddings(direct):
    with pytest.raises(exceptions.BadRequestError, match="no embeddings API"):
        ar.embedding(model="anthropic/claude-sonnet-4-5", input="hi")
    assert direct.requests == []


def test_errors_keep_litellm_names(direct):
    with pytest.raises(exceptions.InternalServerError):
        ar.embedding(model="openai/err-500", input="hi", api_base=direct.url)


def test_through_a_set_gateway(direct, monkeypatch):
    monkeypatch.setenv("AGENTROUTER_BASE_URL", direct.url)
    monkeypatch.setenv("AGENTROUTER_API_KEY", "gw-key")
    r = ar.embedding(model="openai/text-embedding-3-small", input=TEXTS)
    path, headers, body = direct.requests[-1]
    assert path == "/v1/embeddings" and body["model"] == "openai/text-embedding-3-small"  # the gateway translates
    assert headers["authorization"] == "Bearer gw-key"
    assert r._hidden_params["route"] == "gateway"


def test_litellm_names(direct):
    r = litellm.embedding(model="openai/text-embedding-3-small", input=TEXTS, api_base=direct.url)
    assert isinstance(r, litellm.EmbeddingResponse) and len(r.data) == 2


async def test_async(direct):
    r = await litellm.aembedding(model="openai/text-embedding-3-small", input=TEXTS, api_base=direct.url)
    assert len(r.data) == 2


@needs_aigw
@needs_xds_port
def test_real_gateway_serves_embeddings(real_gateway):
    fake = real_gateway
    r = ar.embedding(model="acme/embed-small", input=TEXTS)
    path, headers, body = fake.requests[-1]
    assert path == "/v1/embeddings"
    assert body["model"] == "embed-small"  # aigw's rule passed on the provider's own name
    assert headers["authorization"] == "Bearer acme-secret"
    assert r._hidden_params["route"] == "local gateway" and len(r.data) == 2
