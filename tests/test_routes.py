# SPDX-License-Identifier: Apache-2.0
"""Routes: fallbacks and retries done by the local gateway, and LiteLLM's Router on top of them.

The end-to-end tests start the real aigw against the fake provider and are skipped unless the
local-gateway package is installed and aigw's fixed internal ports are free (see test_local_gateway).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
import warnings

import pytest

import theagentrouter as ar
from theagentrouter import exceptions, local_gateway, routes
from theagentrouter.compat import litellm

from .test_local_gateway import HELLO, _free_port, _port, _write_state, needs_aigw, needs_xds_port


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: True)


# --- defining routes ------------------------------------------------------------------------------


def test_route_is_stored_and_listed(env, installed):
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"], retry=ar.Retry(attempts=2))
    assert r.tiers == (("openai/gpt-4o",), ("anthropic/claude-sonnet-4-5",))
    assert ar.list_routes() == {"chat": r}
    saved = json.loads((env / "routes.json").read_text())
    assert saved["chat"]["tiers"] == [["openai/gpt-4o"], ["anthropic/claude-sonnet-4-5"]]
    assert saved["chat"]["retry"]["attempts"] == 2


def test_redefining_replaces_and_says_so(env, installed, capsys):
    ar.route("chat", primary="openai/gpt-4o")
    ar.route("chat", primary="openai/gpt-4o")  # the same: nothing to say
    assert "replaced" not in capsys.readouterr().err
    ar.route("chat", primary="openai/gpt-4o-mini")
    assert "replaced route 'chat'" in capsys.readouterr().err
    assert ar.list_routes()["chat"].primary == "openai/gpt-4o-mini"


def test_remove_route(env, installed):
    ar.route("chat", primary="openai/gpt-4o")
    assert ar.remove_route("chat") is True
    assert ar.remove_route("chat") is False
    assert ar.list_routes() == {}


def test_a_list_in_a_tier_shares_traffic(env, installed):
    r = ar.route("chat", primary=["openai/gpt-4o", "openai/gpt-4o-mini"])
    assert r.tiers == (("openai/gpt-4o", "openai/gpt-4o-mini"),)


@pytest.mark.parametrize(
    ("primary", "error", "match"),
    [
        ("gpt-4o", exceptions.BadRequestError, "needs a provider prefix"),
        ("vertex/gemini-2.5-pro", exceptions.BadRequestError, "does not set up 'vertex'"),
    ],
)
def test_route_checks_each_model_now(env, installed, primary, error, match):
    with pytest.raises(error, match=match):
        ar.route("chat", primary=primary)
    assert ar.list_routes() == {}


def test_route_can_be_defined_for_export_without_a_local_gateway_or_keys(env, monkeypatch):
    monkeypatch.setattr(local_gateway, "installed", lambda: False)
    monkeypatch.delenv("OPENAI_API_KEY")
    ar.route("chat", primary="openai/gpt-4o")
    config = local_gateway.export_config(providers=["openai"])
    assert 'value: "chat"' in config and "${OPENAI_API_KEY}" in config
    with pytest.raises(exceptions.BadRequestError, match="needs the local gateway"):
        ar.completion(model="chat", messages=HELLO)


@pytest.fixture
def gateway_set(env, monkeypatch):
    """A gateway set with AGENTROUTER_BASE_URL, at the fake provider; nothing remembered from other tests."""
    from theagentrouter import routing

    from .fake_provider import FakeProvider

    server = FakeProvider()
    monkeypatch.setenv("AGENTROUTER_BASE_URL", server.url)
    monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")
    monkeypatch.setattr(routing, "_route_warned", set())
    monkeypatch.setattr(routes, "_for_gateway", {})
    yield server
    server.close()


def test_a_route_defined_with_a_gateway_set_is_called_by_name_there(env, gateway_set):
    # The same code as on a laptop: no error, nothing set up here, and the gateway's route of that name applies.
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"])
    assert r.name == "chat" and r.tiers == (("openai/gpt-4o",), ("anthropic/claude-sonnet-4-5",))
    assert ar.list_routes() == {} and not (env / "routes.json").exists()
    with pytest.warns(UserWarning, match="route 'chat' is defined in Python.*that gateway's own routes"):
        resp = ar.completion(model="chat", messages=HELLO)
    assert resp._hidden_params["route"] == "gateway" and gateway_set.requests[-1][2]["model"] == "chat"
    assert local_gateway.running() is None


def test_a_router_with_a_gateway_set_calls_its_groups_by_name(env, gateway_set):
    router = litellm.Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
        ],
        fallbacks=[{"chat": ["backup"]}],
    )
    assert router.routes["chat"].name == "chat" and ar.list_routes() == {}
    with pytest.warns(UserWarning, match="route 'chat' is defined in Python"):
        router.completion(model="chat", messages=HELLO)
    assert gateway_set.requests[-1][2]["model"] == "chat"


def test_a_router_made_here_sends_its_group_name_once_a_gateway_is_set(env, installed, monkeypatch):
    # Made on a laptop, then pointed at a gateway built with export_config(): that gateway knows the group's
    # name, not the hashed one this computer uses, so the name is chosen at each call.
    from theagentrouter import routing

    from .fake_provider import FakeProvider

    router = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}])
    assert router.routes["chat"].name.startswith("router-") and router.routes["chat"].name in ar.list_routes()
    server = FakeProvider()
    try:
        monkeypatch.setenv("AGENTROUTER_BASE_URL", server.url)
        monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")
        monkeypatch.setattr(routing, "_route_warned", set())
        monkeypatch.setattr(routes, "_for_gateway", {})
        with pytest.warns(UserWarning, match="route 'chat' is defined in Python.*that gateway's own routes"):
            router.completion(model="chat", messages=HELLO)
        assert server.requests[-1][2]["model"] == "chat"
    finally:
        server.close()


def test_a_router_made_with_a_gateway_set_sets_up_its_route_here_once_unset(env, gateway_set, monkeypatch):
    router = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}])
    assert router.routes["chat"].name == "chat" and ar.list_routes() == {}
    monkeypatch.delenv("AGENTROUTER_BASE_URL")
    name = router._target("chat")
    assert name.startswith("router-") and name.endswith(":chat") and name in ar.list_routes()


def test_export_names_router_routes_after_their_group(env, installed):
    router = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}])
    assert router.routes["chat"].name.startswith("router-")  # on this computer, the hashed name
    config = local_gateway.export_config(providers=["openai"])
    assert 'value: "chat"' in config and router.routes["chat"].name not in config  # the name code calls
    ar.route("chat", primary="openai/gpt-4o-mini")
    with pytest.raises(exceptions.BadRequestError, match="would both be 'chat'"):
        local_gateway.export_config(providers=["openai"])


# --- the gateway config ---------------------------------------------------------------------------


def _config(routes_: dict) -> str:
    return local_gateway.config_yaml(["anthropic", "openai"], dict(os.environ), port=1975, models=[], routes=routes_)


def _route_policies(config: str) -> list[str]:
    """The traffic policies of routes, without the gateway's own for failed connections."""
    docs = config.split("\n---\n")
    return [d for d in docs if "kind: BackendTrafficPolicy" in d and "name: retry-failed-connections" not in d]


def test_a_route_is_one_gateway_route_with_a_priority_per_tier(env, installed):
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["openai/gpt-4o-mini"], retry=ar.Retry(attempts=2))
    config = _config({"chat": r})
    assert config.count('value: "chat"') == 1
    assert 'modelNameOverride: "gpt-4o"\n          priority: 0' in config
    assert 'modelNameOverride: "gpt-4o-mini"\n          priority: 1' in config
    assert len(_route_policies(config)) == 1
    assert "numAttemptsPerPriority: 3" in config  # the first try and 2 retries in each tier
    assert "numRetries: 5" in config  # 3 tries in each of 2 tiers, minus the first
    assert "        - 503" in config


def test_models_in_one_tier_share_a_priority(env, installed):
    r = ar.route("chat", primary=["openai/gpt-4o", "openai/gpt-4o-mini"], fallbacks=["anthropic/claude-sonnet-4-5"])
    config = _config({"chat": r})
    assert config.count("priority: 0") == 2 and config.count("priority: 1") == 1


def test_route_without_retries_has_no_retry_policy(env, installed):
    r = ar.route("solo", primary="openai/gpt-4o", retry=ar.Retry(attempts=0))
    assert _route_policies(_config({"solo": r})) == []


def test_fallback_without_retries_still_moves_to_the_next_tier(env, installed):
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["openai/gpt-4o-mini"], retry=ar.Retry(attempts=0))
    config = _config({"chat": r})
    assert "numAttemptsPerPriority: 1" in config and "numRetries: 1" in config


def test_a_limit_goes_in_the_same_policy_as_the_retry(env, installed):
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["openai/gpt-4o-mini"], limit=ar.Limit(60, per="minute"))
    config = _config({"chat": r})
    policies = _route_policies(config)
    assert len(policies) == 1  # Envoy Gateway takes one per route
    policy = policies[0]
    assert "type: Local" in policy and "requests: 60" in policy and "unit: Minute" in policy
    assert "numAttemptsPerPriority:" in policy


def test_a_limit_alone_still_gets_a_policy(env, installed):
    r = ar.route("solo", primary="openai/gpt-4o", retry=ar.Retry(attempts=0), limit=ar.Limit(5, per="second"))
    config = _config({"solo": r})
    policy = _route_policies(config)[0]
    assert "unit: Second" in policy and "numAttemptsPerPriority" not in policy
    # A route's own policy replaces the gateway's, so it keeps the retry of a failed connection.
    assert "connect-failure" in policy and "retriable-status-codes" not in policy


def test_limit_is_stored_and_changes_the_fingerprint(env, installed):
    plain = ar.route("chat", primary="openai/gpt-4o")
    limited = ar.route("chat", primary="openai/gpt-4o", limit=ar.Limit(10, per="hour"))
    assert ar.list_routes()["chat"].limit == ar.Limit(10, per="hour")
    assert routes.fingerprint({"chat": plain}) != routes.fingerprint({"chat": limited})


@pytest.mark.parametrize(("args", "match"), [((0,), "from 1 to"), ((5, "week"), "must be one of")])
def test_limit_checks_its_values(args, match):
    with pytest.raises(ValueError, match=match):
        ar.Limit(*args)


def test_loopback_backends_are_written_as_a_hostname(env, installed, monkeypatch):
    # Under aigw run, IP backends of one rule cannot be told apart, so a fallback would get the
    # first backend's model name and key.
    monkeypatch.setitem(
        local_gateway.PROVIDERS, "acme", {"schema": "OpenAI", "base": "http://127.0.0.1:8001/v1", "keys": ["ACME_KEY"]}
    )
    monkeypatch.setenv("ACME_KEY", "x")
    config = local_gateway.config_yaml(["acme"], dict(os.environ), port=1975, models=["acme/m"])
    backends = "\n---\n".join(d for d in config.split("\n---\n") if "kind: Backend\n" in d)
    assert "hostname: localhost" in backends and "127.0.0.1" not in backends


def test_anthropic_in_a_route_uses_its_openai_compatible_endpoint(env, installed):
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"])
    config = _config({"chat": r})
    assert "name: anthropic-openai" in config
    assert 'modelNameOverride: "claude-sonnet-4-5"' in config
    backend = config.split("name: anthropic-openai\n  namespace: default\nspec:")[1].split("---")[0]
    assert "hostname: api.anthropic.com" in backend
    assert str(local_gateway._key_file("ANTHROPIC_API_KEY")) in config and "sk-ant-test" not in config


def test_a_route_name_takes_over_a_model_rule(env, installed):
    r = ar.route("openai/gpt-4o", primary="openai/gpt-4o", fallbacks=["openai/gpt-4o-mini"])
    config = local_gateway.config_yaml(
        ["openai"], dict(os.environ), port=1975, models=["openai/gpt-4o"], routes={r.name: r}
    )
    assert config.count('value: "openai/gpt-4o"\n') == 1  # the route's rule, not the plain model rule
    assert "priority: 1" in config


def test_backends_ready_needs_every_host_of_every_route(env, installed, monkeypatch):
    r = ar.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"])
    cluster = f"httproute/default/{local_gateway._k8s_name('chat')}/rule/0"
    openai_only = f"{cluster}::1.2.3.4:443::hostname::api.openai.com\n"
    both = openai_only + f"{cluster}::5.6.7.8:443::hostname::api.anthropic.com\n"

    class Reply:
        def __init__(self, text):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.text.encode()

    import urllib.request

    for text, ready in ((openai_only, False), (both, True)):
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, text=text, **k: Reply(text))
        assert local_gateway._backends_ready(1, {"chat": r}) is ready


# --- where a route name goes ----------------------------------------------------------------------


def _serve_route(env, fake, r):
    """Pretend the local gateway is running with this route; the fake provider stands in for it."""
    _write_state(env, _port(fake.url), ["anthropic", "openai"])
    state = json.loads((env / "local-gateway" / "state.json").read_text())
    state.update(routes=[r.name], routes_fingerprint=routes.fingerprint({r.name: r}))
    (env / "local-gateway" / "state.json").write_text(json.dumps(state))


@pytest.mark.parametrize(
    ("primary", "fallback"),
    [
        ("anthropic/claude-haiku-5-5", "openai/gpt-5.4-mini"),
        ("openai/gpt-5.4-mini", "anthropic/claude-haiku-5-5"),
        ("anthropic/claude-haiku-5-5", "azure/acme-deployment"),
    ],
)
def test_a_route_with_an_openai_model_sends_max_completion_tokens(env, installed, fake, primary, fallback):
    # Any model in the route may get the request, and OpenAI's newer models refuse max_tokens.
    _serve_route(env, fake, ar.route("chat", primary=primary, fallbacks=[fallback]))
    ar.completion(model="chat", messages=HELLO, max_tokens=50)
    body = fake.requests[-1][2]
    assert body["max_completion_tokens"] == 50 and "max_tokens" not in body


@pytest.mark.parametrize(
    ("primary", "fallback"),
    [
        ("anthropic/claude-haiku-5-5", "groq/acme-llama"),
        ("mistral/acme-large", "openai/gpt-5.4-mini"),
        ("openai/gpt-5.4-mini", "mistral/acme-large"),
    ],
)
def test_a_route_with_another_provider_sends_what_the_caller_wrote(env, installed, fake, primary, fallback):
    # Not every provider takes max_completion_tokens. If an OpenAI model refuses max_tokens, it says so.
    _serve_route(env, fake, ar.route("chat", primary=primary, fallbacks=[fallback]))
    ar.completion(model="chat", messages=HELLO, max_tokens=50)
    assert fake.requests[-1][2]["max_tokens"] == 50


def test_a_route_removes_what_claude_on_bedrock_cannot_take(env, installed, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "acme-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "acme-secret")
    newer = ar.route("newer", primary="openai/gpt-5.4", fallbacks=["bedrock/us.anthropic.claude-sonnet-5-5"])
    older = ar.route("older", primary="openai/gpt-5.4", fallbacks=["bedrock/anthropic.claude-sonnet-4-5"])
    other = ar.route("other", primary="openai/gpt-5.4", fallbacks=["bedrock/amazon.nova-pro-v1:0"])
    providers = ["bedrock", "openai"]
    config = local_gateway.config_yaml(providers, dict(os.environ), port=1975, models=[], routes={"r": newer})
    assert (
        "remove: [inferenceConfig.temperature, inferenceConfig.topP, additionalModelRequestFields.reasoning_config]"
        in config
    )
    config = local_gateway.config_yaml(providers, dict(os.environ), port=1975, models=[], routes={"r": older})
    assert "remove: [additionalModelRequestFields.reasoning_config]" in config
    config = local_gateway.config_yaml(providers, dict(os.environ), port=1975, models=[], routes={"r": other})
    assert "bodyMutation" not in config


def test_a_route_removes_what_a_claude_model_refuses_at_that_model(env, installed):
    config = _config({"chat": ar.route("chat", primary="openai/gpt-5.4", fallbacks=["anthropic/claude-sonnet-5-5"])})
    assert config.count("bodyMutation") == 1
    openai, claude = config.split("name: anthropic-openai\n          namespace: default\n          modelNameOverride")
    assert "bodyMutation" not in openai and "remove: [temperature, top_p]" in claude
    older = ar.route("older", primary="openai/gpt-5.4", fallbacks=["anthropic/claude-sonnet-4-6"])
    assert "bodyMutation" not in _config({"older": older})


def test_route_name_goes_through_the_local_gateway(env, installed, fake, monkeypatch):
    r = ar.route("chat", primary="openai/gpt-4o-mini", fallbacks=["anthropic/claude-sonnet-4-5"])
    _serve_route(env, fake, r)
    monkeypatch.setattr(local_gateway, "_start", lambda *a, **k: pytest.fail("restarted a gateway that is current"))

    resp = ar.completion(model="chat", messages=HELLO)
    path, headers, body = fake.requests[-1]
    assert path == "/v1/chat/completions" and body["model"] == "chat"  # the gateway picks the model
    assert len(fake.requests) == 1  # retries and fallbacks are the gateway's, not the SDK's
    assert "authorization" not in headers
    assert resp._hidden_params["route"] == "local gateway"
    assert resp._hidden_params["route_name"] == "chat"


@pytest.mark.parametrize(
    ("reply", "served"),
    [
        ("gpt-4o-mini-2024-07-18", "openai/gpt-4o-mini"),
        ("gpt-4o-2024-08-06", "openai/gpt-4o"),
        ("claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4-5"),
        ("something-else", None),
        (None, None),
    ],
)
def test_served_by_is_read_from_the_reply_model(reply, served):
    from theagentrouter.transport import served_by

    members = ("openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5")
    assert served_by(reply, members) == served


def test_changed_routes_restart_the_gateway(env, installed, monkeypatch):
    calls = []
    old = {
        "pid": 1,
        "port": 1975,
        "providers": ["openai"],
        "models": [],
        "routes": ["chat"],
        "routes_fingerprint": "x",
        "versions": local_gateway.versions(),
    }
    monkeypatch.setattr(local_gateway, "running", lambda: old)
    monkeypatch.setattr(local_gateway, "stop", lambda state=None: calls.append("stop"))
    monkeypatch.setattr(
        local_gateway, "_start", lambda providers, models, routes=None: calls.append(("start", sorted(routes))) or {}
    )
    ar.route("chat", primary="openai/gpt-4o")
    local_gateway.ensure("chat")
    assert calls == ["stop", ("start", ["chat"])]


def test_unknown_name_says_how_to_define_a_route(env):
    with pytest.raises(exceptions.BadRequestError, match=r"theagentrouter\.route\(\)"):
        ar.completion(model="chat", messages=HELLO)


def test_fallbacks_argument_points_to_routes(env, fake):
    with pytest.warns(UserWarning, match="define a route"):
        ar.completion(model="openai/gpt-4o-mini", messages=HELLO, api_base=fake.url, fallbacks=["anthropic/x"])


# --- LiteLLM's Router -----------------------------------------------------------------------------


def test_router_turns_groups_and_fallbacks_into_routes(env, installed):
    router = litellm.Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}},
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o-mini"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
        ],
        fallbacks=[{"chat": ["backup"]}],
        num_retries=2,
    )
    chat = router.routes["chat"]
    assert chat.name.startswith("router-") and chat.name.endswith(":chat")
    assert ar.list_routes()[chat.name] == chat
    assert chat.tiers == (("openai/gpt-4o", "openai/gpt-4o-mini"), ("anthropic/claude-sonnet-4-5",))
    assert chat.retry.attempts == 2
    assert 404 in chat.retry.on and 401 in chat.retry.on  # LiteLLM's Router falls back on these too
    assert router.routes["backup"].tiers == (("anthropic/claude-sonnet-4-5",),)
    assert set(router.routes) == {"chat", "backup"}
    assert "chat" not in ar.list_routes()  # the group name stays free for other programs


def test_routers_in_two_programs_keep_their_own_routes(env, installed):
    one = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}])
    two = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
    again = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}])
    assert one.routes["chat"].name != two.routes["chat"].name
    assert again.routes["chat"].name == one.routes["chat"].name  # the same definition shares one route
    two._target("chat")  # `again` removed two's route; two's next call puts it back
    saved = ar.list_routes()
    assert saved[one.routes["chat"].name].primary == "openai/gpt-4o"
    assert saved[two.routes["chat"].name].primary == "openai/gpt-4o-mini"


def test_a_new_router_definition_removes_the_route_an_earlier_one_left(env, installed):
    # Otherwise the old route stays saved, and the gateway would not start without its provider's key.
    old = litellm.Router(
        model_list=[{"model_name": "chat", "litellm_params": {"model": "anthropic/claude-sonnet-4-5"}}]
    )
    new = litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}])
    assert set(ar.list_routes()) == {new.routes["chat"].name}
    # A program still running the old Router puts its route back on its next call, and keeps the new one.
    assert old._target("chat") == old.routes["chat"].name
    assert set(ar.list_routes()) == {old.routes["chat"].name, new.routes["chat"].name}


def test_a_router_group_named_after_a_model_does_not_take_it_over(env, installed):
    router = litellm.Router(model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
    assert "gpt-4o" not in ar.list_routes() and "openai/gpt-4o" not in ar.list_routes()
    assert router._target("gpt-4o") == router.routes["gpt-4o"].name


@pytest.mark.parametrize("requests", [2.5, True, "10", 2**32])
def test_limit_checks_its_types(requests):
    with pytest.raises(ValueError, match="Limit.requests"):
        ar.Limit(requests)


def test_router_takes_keys_from_the_environment(env, installed):
    with pytest.warns(UserWarning, match="api_key for chat is not used"):
        litellm.Router(
            model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"}}]
        )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        litellm.Router(
            model_list=[
                {
                    "model_name": "chat",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "os.environ/OPENAI_API_KEY"},
                }
            ]
        )
    assert not [w for w in seen if "api_key" in str(w.message)]


def test_router_shares_a_groups_traffic_by_weight_rpm_or_tpm(env, installed):
    # LiteLLM's default simple-shuffle: the first of weight, rpm and tpm in litellm_params that any
    # deployment sets; a deployment without it gets no traffic. rpm and tpm are not limits there.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        router = litellm.Router(
            model_list=[
                {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "rpm": 10, "tpm": 50000}},
                {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o-mini", "rpm": 30}},
                {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4.1", "tpm": 9}},
                {"model_name": "backup", "litellm_params": {"model": "anthropic/claude-sonnet-4-5"}, "rpm": 5},
            ],
            fallbacks=[{"chat": ["backup"]}],
        )
    chat = router.routes["chat"]
    assert chat.weights == ((10, 30, 0), (1,))  # rpm decides; a top-level rpm is not a weight
    assert ar.list_routes()[chat.name].weights == chat.weights  # saved
    config = local_gateway.config_yaml(
        ["openai", "anthropic"], dict(os.environ), port=1975, models=[], routes={"chat": chat}
    )
    assert [line.strip() for line in config.splitlines() if line.strip().startswith("weight:")] == [
        "weight: 10",
        "weight: 30",
        "weight: 0",
        "weight: 1",
    ]
    weight = litellm.Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "weight": 0.25, "rpm": 9}},
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o-mini", "weight": 1}},
        ]
    )
    assert weight.routes["chat"].weights == ((1, 4),)  # weight wins over rpm; a fraction keeps its ratio
    equal = litellm.Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "rpm": 5}},
            {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o-mini", "rpm": 5}},
        ]
    )
    assert equal.routes["chat"].weights == ()  # equal shares need no weights
    with pytest.raises(TypeError, match="Deployment rpm must be numeric"):
        litellm.Router(model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "rpm": "10"}}])


def _shares(*weights):
    return litellm._weights([{"litellm_params": {"weight": w}} for w in weights])


def test_router_weights_keep_their_ratio():
    # The gateway takes whole numbers up to 1,000,000; LiteLLM's weights can be any number.
    assert _shares(0.01, 100) == (1, 10000)  # 1 in 10,001 of the traffic, as in LiteLLM, not 1 in 1,001
    assert _shares(0.1, 0.2, 0.7) == (1, 2, 7)  # 0.1 as one tenth, not its nearest binary fraction
    assert _shares(10, 30, 0) == (10, 30, 0)  # whole numbers that fit are kept as written
    # A ratio too wide to fit: the largest becomes 1,000,000 and a positive weight keeps at least 1.
    assert _shares(1e-9, 1) == (1, 1_000_000)
    assert _shares(3_000_000, 1, 0) == (1_000_000, 1, 0)


def test_router_says_what_it_does_not_use(env, installed):
    with pytest.warns(
        UserWarning, match=r"ignores routing_strategy='usage-based-routing-v2'.*does not hold deployments to rpm"
    ):
        litellm.Router(
            model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "rpm": 10}}],
            routing_strategy="usage-based-routing-v2",
        )
    with pytest.warns(
        UserWarning, match=r"does not use these litellm_params: api_base \(for chat\); timeout \(for chat\)"
    ):
        litellm.Router(
            model_list=[
                {
                    "model_name": "chat",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_base": "https://acme.openai.azure.com",
                        "timeout": 5,
                    },
                }
            ]
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # what the gateway reads where it starts, and the default strategy
        litellm.Router(
            model_list=[
                {
                    "model_name": "chat",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_key": "os.environ/AZURE_API_KEY",
                        "api_base": "os.environ/AZURE_API_BASE",
                        "api_version": "os.environ/AZURE_API_VERSION",
                    },
                    "model_info": {"id": "a"},
                }
            ],
            routing_strategy="simple-shuffle",
        )


def test_router_rejects_unknown_fallback_groups(env, installed):
    with pytest.raises(exceptions.BadRequestError, match="not in model_list"):
        litellm.Router(
            model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}}],
            fallbacks=[{"chat": ["nope"]}],
        )


# --- the real aigw --------------------------------------------------------------------------------


@pytest.fixture
def second_fake(real_gateway, monkeypatch):
    """A second provider at its own address: "acme2" answers from here, "acme" from the first."""
    from .fake_provider import FakeProvider

    server = FakeProvider()
    monkeypatch.setitem(
        local_gateway.PROVIDERS, "acme2", {"schema": "OpenAI", "base": server.url, "keys": ["ACME2_KEY"]}
    )
    monkeypatch.setenv("ACME2_KEY", "acme2-secret")
    yield server
    server.close()


@needs_aigw
@needs_xds_port
def test_real_gateway_falls_back_to_the_next_model(real_gateway, second_fake):
    first = real_gateway
    ar.route("chat", primary="acme/err-500", fallbacks=["acme2/acme-small"], retry=ar.Retry(attempts=1))

    resp = ar.completion(model="chat", messages=HELLO)
    assert resp.choices[0].message.content == "echo: refunds yesterday"
    # Each backend gets its own model name and key: the two tries of the primary, then the fallback.
    assert [(b["model"], h.get("authorization")) for _, h, b in first.requests] == [
        ("err-500", "Bearer acme-secret")
    ] * 2
    assert [(b["model"], h.get("authorization")) for _, h, b in second_fake.requests] == [
        ("acme-small", "Bearer acme2-secret")
    ]
    assert resp._hidden_params["served_by"] == "acme2/acme-small"


@needs_aigw
@needs_xds_port
def test_real_gateway_removes_what_a_fallback_refuses(real_gateway, monkeypatch):
    fake = real_gateway
    monkeypatch.setenv("ANTHROPIC_API_KEY", "acme-anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake.url)
    ar.route("chat", primary="acme/err-500", fallbacks=["anthropic/claude-sonnet-5-5"], retry=ar.Retry(attempts=0))

    resp = ar.completion(model="chat", messages=HELLO, temperature=0.2, top_p=0.9)
    assert resp.choices[0].message.content == "echo: refunds yesterday"
    first, second = [body for _, _, body in fake.requests][-2:]
    assert first["model"] == "err-500" and first["temperature"] == 0.2 and first["top_p"] == 0.9
    assert second["model"] == "claude-sonnet-5-5" and "temperature" not in second and "top_p" not in second


@needs_aigw
@needs_xds_port
def test_real_gateway_passes_on_what_a_model_says_about_max_tokens(real_gateway):
    fake = real_gateway
    ar.route("chat", primary="acme/refuses-max-tokens", retry=ar.Retry(attempts=0))
    resp = ar.completion(model="chat", messages=HELLO, max_tokens=50)
    assert resp.choices[0].message.content == "echo: refunds yesterday"
    assert [(b.get("max_tokens"), b.get("max_completion_tokens")) for _, _, b in fake.requests] == [
        (50, None),
        (None, 50),
    ]


@needs_aigw
@needs_xds_port
@pytest.mark.parametrize("server_key", ["acme-server-secret", None, ""])
def test_exported_config_serves_a_team_gateway(real_gateway, second_fake, env, monkeypatch, server_key):
    """aigw run on the exported config, as a team runs it on a server, serves the same calls and routes."""
    if server_key is None:
        monkeypatch.delenv("AGENTROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AGENTROUTER_API_KEY", server_key)
    ar.route("chat", primary="acme/err-500", fallbacks=["acme2/acme-small"], retry=ar.Retry(attempts=1))
    port = _free_port()
    server = env.parent / "server"
    server.mkdir()
    (server / "config.yaml").write_text(
        local_gateway.export_config(port=port, providers=["acme", "acme2", "anthropic"], models=["acme/acme-large"])
    )
    aigw_env = dict(
        os.environ,
        ENVOY_VERSION=local_gateway.ENVOY_VERSION,
        AIGW_CONFIG_HOME=str(server / "config-home"),
        AIGW_STATE_HOME=str(server / "state"),
        AIGW_RUNTIME_DIR=str(local_gateway._runtime_dir()),
    )
    with open(server / "aigw.log", "wb") as log:
        proc = subprocess.Popen(
            [local_gateway._binary(), "run", str(server / "config.yaml"), f"--admin-port={_free_port()}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=aigw_env,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 120
        while not (
            "listening on" in (server / "aigw.log").read_text(errors="replace") and local_gateway._listening(port)
        ):
            assert proc.poll() is None and time.monotonic() < deadline, (server / "aigw.log").read_text()[-2000:]
            time.sleep(0.2)
        monkeypatch.setenv("AGENTROUTER_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")
        warnings.simplefilter("ignore", UserWarning)  # "chat" is a route here too; the server has the same one

        import httpx

        request = {"model": "acme/acme-large", "messages": HELLO}
        before = len(real_gateway.requests)
        with httpx.Client(trust_env=False) as client:
            if not server_key:
                for headers in ({}, {"Authorization": "Bearer wrong"}, {"x-api-key": "wrong"}):
                    response = client.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions", json=request, headers=headers
                    )
                    assert response.status_code == 403
                assert len(real_gateway.requests) == before
                return
            missing_key = client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=request)
            assert missing_key.status_code == 403, (missing_key.text, (server / "aigw.log").read_text()[-4000:])
            assert (
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=request,
                    headers={"Authorization": "Bearer wrong"},
                ).status_code
                == 403
            )
        assert len(real_gateway.requests) == before

        r = ar.completion(model="acme/acme-large", messages=HELLO)
        assert r.choices[0].message.content == "echo: refunds yesterday" and r._hidden_params["route"] == "gateway"
        r = ar.completion(model="claude-sonnet-4-5", messages=HELLO)
        assert r._hidden_params["format"] == "anthropic" and real_gateway.requests[-1][0].endswith("/messages")
        assert "acme-server-secret" not in json.dumps(real_gateway.requests)
        # The route's fallback, once Envoy has looked up the backends' hostnames: the server's route
        # tries acme/err-500, then answers from acme2. (The SDK does not know a remote gateway's routes,
        # so it cannot say which model answered.)
        while not second_fake.requests:
            assert time.monotonic() < deadline
            with contextlib.suppress(exceptions.APIError):
                r = ar.completion(model="chat", messages=HELLO)
            time.sleep(0.5)
        assert r.choices[0].message.content == "echo: refunds yesterday"
        assert second_fake.requests[-1][2]["model"] == "acme-small"
        assert local_gateway.running() is None  # the SDK started nothing on this computer
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)


@contextlib.contextmanager
def _team_gateway(folder, config: str, port: int):
    """aigw run on an exported config, as a team runs it on a server."""
    folder.mkdir()
    (folder / "config.yaml").write_text(config)
    aigw_env = dict(
        os.environ,
        ENVOY_VERSION=local_gateway.ENVOY_VERSION,
        AIGW_CONFIG_HOME=str(folder / "config-home"),
        AIGW_STATE_HOME=str(folder / "state"),
        AIGW_RUNTIME_DIR=str(local_gateway._runtime_dir()),
    )
    with open(folder / "aigw.log", "wb") as log:
        proc = subprocess.Popen(
            [local_gateway._binary(), "run", str(folder / "config.yaml"), f"--admin-port={_free_port()}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=aigw_env,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 120
        while not (
            "listening on" in (folder / "aigw.log").read_text(errors="replace") and local_gateway._listening(port)
        ):
            assert proc.poll() is None and time.monotonic() < deadline, (folder / "aigw.log").read_text()[-2000:]
            time.sleep(0.2)
        yield
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)


@needs_aigw
@needs_xds_port
def test_router_code_runs_unchanged_against_an_exported_gateway(real_gateway, second_fake, env, monkeypatch):
    """The same LiteLLM Router code on a laptop, then against a team gateway made with export_config()."""

    def build():
        return litellm.Router(
            model_list=[
                {"model_name": "main", "litellm_params": {"model": "acme/err-500"}},
                {"model_name": "spare", "litellm_params": {"model": "acme2/acme-small"}},
            ],
            fallbacks=[{"main": ["spare"]}],
            num_retries=0,
        )

    laptop = build()
    monkeypatch.setenv("AGENTROUTER_API_KEY", "acme-server-secret")
    port = _free_port()
    config = local_gateway.export_config(port=port, providers=["acme", "acme2"])
    with _team_gateway(env.parent / "server", config, port):
        monkeypatch.setenv("AGENTROUTER_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("AGENTROUTER_AUTOSTART", "0")
        monkeypatch.setattr(routes, "_for_gateway", {})
        warnings.simplefilter("ignore", UserWarning)  # "the gateway's route of that name applies"
        router = build()
        # Once Envoy has looked up the backends' hostnames, the server's "main" falls back to acme2.
        deadline = time.monotonic() + 60
        while not second_fake.requests:
            assert time.monotonic() < deadline
            with contextlib.suppress(exceptions.APIError):
                r = router.completion(model="main", messages=HELLO)
            time.sleep(0.5)
        assert r.choices[0].message.content == "echo: refunds yesterday"
        assert second_fake.requests[-1][2]["model"] == "acme-small"
        # The Router made on the laptop, before AGENTROUTER_BASE_URL was set, sends the group's name too.
        assert laptop.completion(model="main", messages=HELLO).choices[0].message.content == "echo: refunds yesterday"
        assert local_gateway.running() is None  # the SDK started nothing on this computer
        # A client key the server does not have: Envoy refuses it in plain text, and the error names the setting.
        monkeypatch.setenv("AGENTROUTER_API_KEY", "not-the-server-key")
        with pytest.raises(exceptions.PermissionDeniedError, match="AGENTROUTER_API_KEY on its server"):
            router.completion(model="main", messages=HELLO)


@needs_aigw
@needs_xds_port
def test_real_gateway_tries_the_primary_first_every_time(real_gateway, second_fake):
    ar.route("chat", primary="acme/acme-large", fallbacks=["acme2/acme-small"])
    for _ in range(6):
        assert ar.completion(model="chat", messages=HELLO)._hidden_params["served_by"] == "acme/acme-large"
    assert second_fake.requests == []


@needs_aigw
@needs_xds_port
def test_real_gateway_retries_then_gives_up(real_gateway):
    fake = real_gateway
    ar.route("flaky", primary="acme/err-500", retry=ar.Retry(attempts=2))
    with pytest.raises(exceptions.InternalServerError):
        ar.completion(model="flaky", messages=HELLO)
    assert sum(1 for _, _, body in fake.requests if body["model"] == "err-500") == 3


@needs_aigw
@needs_xds_port
def test_real_gateway_does_not_retry_what_is_not_listed(real_gateway):
    fake = real_gateway
    ar.route("strict", primary="acme/err-401", retry=ar.Retry(attempts=2))
    with pytest.raises(exceptions.AuthenticationError):
        ar.completion(model="strict", messages=HELLO)
    assert sum(1 for _, _, body in fake.requests if body["model"] == "err-401") == 1


@needs_aigw
@needs_xds_port
def test_real_gateway_serves_litellm_router(real_gateway, second_fake):
    router = litellm.Router(
        model_list=[
            {"model_name": "main", "litellm_params": {"model": "acme/err-429"}},
            {"model_name": "spare", "litellm_params": {"model": "acme2/acme-small"}},
        ],
        fallbacks=[{"main": ["spare"]}],
        num_retries=0,
    )
    resp = router.completion(model="main", messages=HELLO)
    assert resp.choices[0].message.content == "echo: refunds yesterday"
    assert [b["model"] for _, _, b in real_gateway.requests] == ["err-429"]
    assert [b["model"] for _, _, b in second_fake.requests] == ["acme-small"]


@needs_aigw
@needs_xds_port
def test_real_gateway_shares_a_router_group_by_rpm(real_gateway, second_fake):
    # As in LiteLLM's simple-shuffle, a deployment with no rpm, beside one with rpm, gets no traffic.
    router = litellm.Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "acme/acme-large", "rpm": 5}},
            {"model_name": "chat", "litellm_params": {"model": "acme2/acme-small"}},
        ],
        num_retries=0,
    )
    for _ in range(6):
        router.completion(model="chat", messages=HELLO)
    assert [b["model"] for _, _, b in real_gateway.requests] == ["acme-large"] * 6
    assert second_fake.requests == []


@needs_aigw
@needs_xds_port
def test_real_gateway_enforces_a_request_limit(real_gateway):
    fake = real_gateway
    ar.route("capped", primary="acme/acme-large", retry=ar.Retry(attempts=0), limit=ar.Limit(2, per="minute"))
    ar.completion(model="capped", messages=HELLO)
    ar.completion(model="capped", messages=HELLO)
    with pytest.raises(exceptions.RateLimitError):
        ar.completion(model="capped", messages=HELLO)
    assert sum(1 for _, _, body in fake.requests if body["model"] == "acme-large") == 2  # the third never left
