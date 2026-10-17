"""Agent Router's SDK only: the app's calls through gateways, with the fake provider behind them.

The local gateway tests start the real aigw (theagentrouter-local-gateway) and skip without it.
They cover the go-live readiness review's findings about the gateway; known_gap marks what is
still open.
"""

import json
import os
import socket

import httpx
import litellm
import pytest
from conftest import CLAUDE, MODELS, OPENAI, only_agent_router

pytestmark = only_agent_router
HELLO = [{"role": "user", "content": "refunds yesterday"}]


def _needs_local_gateway():
    from theagentrouter import local_gateway

    if not local_gateway.installed():
        pytest.skip("theagentrouter-local-gateway is not installed")
    if local_gateway._listening(local_gateway.XDS_PORT):
        pytest.skip("another aigw holds port 18000")
    return local_gateway


@pytest.fixture
def gateway(fake, monkeypatch):
    """The local gateway, started by the app's first call, with the fake provider as OpenAI and Anthropic."""
    local_gateway = _needs_local_gateway()
    monkeypatch.setenv("AGENTROUTER_AUTOSTART", "1")
    monkeypatch.setitem(local_gateway.PROVIDERS, "openai", {**local_gateway.PROVIDERS["openai"], "base": fake.url})
    anthropic = {**local_gateway.PROVIDERS["anthropic"], "base": fake.url.removesuffix("/v1")}
    monkeypatch.setitem(local_gateway.PROVIDERS, "anthropic", anthropic)
    yield local_gateway
    local_gateway.stop()


# ---- the app through the local gateway ----------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_stream_loop_through_the_local_gateway(gateway, model):
    from assistant import app

    assert "".join(app.stream(model, "refunds yesterday")).strip() == "echo: refunds yesterday"


def test_tool_round_through_the_local_gateway(gateway):
    from assistant import app

    assert app.with_tools(CLAUDE, "A-100")[0] == "echo: shipped"


def test_messages_api_through_the_local_gateway(gateway):
    resp = litellm.anthropic.messages.create(model=CLAUDE, messages=HELLO, max_tokens=50)
    assert resp["content"][0]["type"] == "text"


# ---- who can call the local gateway (review S1) -------------------------------------------------


def _network_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))  # picks the outgoing interface; sends nothing
        except OSError:
            return None
        address = s.getsockname()[0]
    return None if address.startswith("127.") else address


def test_local_gateway_refuses_other_computers(gateway, fake):
    address = _network_address()
    if address is None:
        pytest.skip("no network address besides loopback")
    litellm.completion(model=OPENAI, messages=HELLO)
    port = gateway.running()["port"]
    sent = len(fake.requests)
    body = {"model": OPENAI, "messages": HELLO}
    assert httpx.post(f"http://{address}:{port}/v1/chat/completions", json=body, timeout=20).status_code == 403
    assert len(fake.requests) == sent


# ---- which key goes where (review S3) -----------------------------------------------------------


def test_stored_key_stays_with_its_gateway(fake, tmp_path):
    home = tmp_path / "agentrouter"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(json.dumps({"base_url": "https://gw.acme.example/v1", "api_key": "acme-stored"}))
    os.environ["AGENTROUTER_BASE_URL"] = fake.url  # another gateway, named in the environment
    try:
        litellm.completion(model=OPENAI, messages=HELLO)
    finally:
        del os.environ["AGENTROUTER_BASE_URL"]
    assert "acme-stored" not in json.dumps(fake.requests[-1][1])


# ---- one gateway shared by every program (review H3, H4) ----------------------------------------


@pytest.mark.parametrize("params", [{"on": (42,)}, {"timeout": -1}, {"backoff": -1}, {"attempts": 1.5}])
def test_invalid_retry_is_refused_before_it_is_saved(monkeypatch, params):
    from theagentrouter import Retry, list_routes, local_gateway, route

    monkeypatch.setattr(local_gateway, "installed", lambda: True)
    with pytest.raises(ValueError):
        route("chat", primary=OPENAI, retry=Retry(**params))
    assert list_routes() == {}


def test_fractional_retry_timeout_works_through_the_gateway(gateway, monkeypatch):
    from theagentrouter import Retry, remove_route, route

    monkeypatch.setattr(gateway, "START_TIMEOUT", 15)
    try:
        route("chat", primary=OPENAI, retry=Retry(timeout=2.5, attempts=2))
        response = litellm.completion(model="chat", messages=HELLO)
        assert response.choices[0].message.content.strip() == "echo: refunds yesterday"
    finally:
        remove_route("chat")


def test_unknown_model_does_not_restart_the_gateway(gateway):
    litellm.completion(model=OPENAI, messages=HELLO)
    before = gateway.running()["pid"]
    with pytest.raises(litellm.APIError):
        litellm.completion(model="openai/gpt-4o-mini-typo", messages=HELLO)
    assert gateway.running()["pid"] == before
