"""Decide where a call goes. The SDK talks only to a gateway; it never calls a provider by itself.

One rule, first match wins:

1. ``api_base`` on the call (LiteLLM's argument): the endpoint the developer named, for that call
   only, in the provider's format (Anthropic Messages for ``anthropic/``, the OpenAI format
   otherwise) and with the provider's own model name.
2. A gateway is set (``AGENTROUTER_BASE_URL``, or ``base_url`` in ``AGENTROUTER_HOME/config.json``):
   every call goes there in the OpenAI format, with the model name unchanged
   (``anthropic/claude-sonnet-4-5``). The gateway translates.
3. Otherwise the local gateway (Agent Router's ``aigw`` on this computer), started by the first
   call that needs it (``AGENTROUTER_AUTOSTART=0`` turns that off). Claude goes to its Anthropic
   Messages endpoint, so prompt caching and thinking survive; everything else in the OpenAI format.

Provider addresses and keys live only in the gateway's config: the local gateway reads keys from
the environment when it starts. A format belongs in the SDK only if Agent Router serves it.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import exceptions, local_gateway

OPENAI = "openai"  # Chat Completions
ANTHROPIC = "anthropic"  # Messages


def home() -> Path:
    return Path(os.environ.get("AGENTROUTER_HOME") or Path.home() / ".agentrouter")


def settings() -> dict[str, str]:
    """Settings from the environment, falling back to AGENTROUTER_HOME/config.json.

    The file lets a setting outlive one shell: a sign-in tool writes it, and every new process
    reads it. Environment variables always win over the file.
    """
    try:
        stored = json.loads((home() / "config.json").read_text())
    except (OSError, ValueError):
        stored = {}
    out = {}
    for name, key in (
        ("base_url", "AGENTROUTER_BASE_URL"),
        ("api_key", "AGENTROUTER_API_KEY"),
        ("project", "AGENTROUTER_PROJECT"),
        ("token_count", "AGENTROUTER_TOKEN_COUNT"),
    ):
        value = os.environ.get(key) or stored.get(name)
        if value:
            out[name] = str(value)
    return out


@dataclass
class Target:
    route: str  # "direct", "gateway" or "local gateway"
    format: str  # OPENAI or ANTHROPIC
    base_url: str
    api_key: str | None
    model: str  # the model name sent in the request body
    provider: str
    headers: dict[str, str] = field(default_factory=dict)
    members: tuple[str, ...] = ()  # for a route: its provider/model names, to tell which one answered

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + ("/messages" if self.format == ANTHROPIC else "/chat/completions")


def with_provider(model: str) -> str:
    """``gpt-4o-mini`` becomes ``openai/gpt-4o-mini``, as in LiteLLM: the provider comes from the model list."""
    if "/" in model:
        return model
    from . import models

    provider = models.provider_of(model)
    return f"{provider}/{model}" if provider else model


def validate_environment(
    model: str | None = None, api_key: str | None = None, api_base: str | None = None
) -> dict[str, Any]:
    """LiteLLM's validate_environment: are the keys this model needs set?

    With a gateway set, the gateway holds the provider keys, so only AGENTROUTER_API_KEY is checked.
    Otherwise the local gateway reads the provider's usual variable, so that one is checked; for a
    route, every provider in it.
    """
    if api_key:
        return {"keys_in_environment": True, "missing_keys": []}
    if not model:
        return {"keys_in_environment": False, "missing_keys": []}
    if not api_base and settings().get("base_url"):
        if settings().get("api_key"):
            return {"keys_in_environment": True, "missing_keys": []}
        return {"keys_in_environment": False, "missing_keys": ["AGENTROUTER_API_KEY"]}
    from . import routes

    r = routes.get(model)
    names = list(r.models) if r is not None else [with_provider(model)]
    missing: list[str] = []
    known = False
    env = dict(os.environ)
    for name in names:
        provider = name.split("/", 1)[0] if "/" in name else ""
        if provider not in local_gateway.PROVIDERS:
            continue
        known = True
        # The same check the local gateway makes before it starts: Bedrock needs a secret and a region too.
        lacking = local_gateway.missing(provider, env)
        for variable in (lacking or "").split(" and "):
            variable = variable.split(" or ")[0]
            if variable and variable not in missing:
                missing.append(variable)
    return {"keys_in_environment": known and not missing, "missing_keys": missing}


def split_model(model: str) -> tuple[str, str]:
    provider, sep, name = with_provider(model).partition("/")
    if not sep or not name:
        raise exceptions.BadRequestError(
            f"model {model!r} has no provider prefix and no route by that name; use provider/model, "
            "for example openai/gpt-4o-mini, or define it with theagentrouter.route()",
            model=model,
        )
    return provider, name


def resolve(
    model: str, *, api_base: str | None = None, api_key: str | None = None, extra_headers: dict[str, str] | None = None
) -> Target:
    headers = dict(extra_headers or {})
    if api_base:
        return _named_endpoint(model, api_base, api_key, headers)

    configured = settings()
    gateway = configured.get("base_url")
    if gateway:
        # The gateway owns its model names: `provider/model` or its own, such as `claude-sonnet-5`.
        _warn_route_unused(model, gateway)
        key = api_key or configured.get("api_key")
        provider = model.split("/", 1)[0] if "/" in model else ""
        return Target("gateway", OPENAI, gateway, key, model, provider, headers)

    from . import routes

    r = routes.get(model)
    if r is not None:
        return _through_route(r, headers)

    model = with_provider(model)
    provider, _ = split_model(model)
    if api_key is not None:
        _warn_key_unused()
    local = local_gateway.running()
    if local and provider in local["providers"] and local_gateway.serves(local, model, routes.list_routes()):
        return _through_local(local, model, provider, headers)
    return _through_local(_start_local(model, provider), model, provider, headers)


_key_warned = False
_route_warned: set[str] = set()


def _warn_route_unused(model: str, gateway: str) -> None:
    """A route defined in Python only programs the local gateway; a gateway you set has its own."""
    if model in _route_warned:
        return
    from . import routes

    if routes.get(model) is not None:
        _route_warned.add(model)
        warnings.warn(
            f"route {model!r} is defined in Python, but AGENTROUTER_BASE_URL is set: routes apply to the local "
            f"gateway only, so {model!r} goes to {gateway} as a model name, and that gateway's own routes and "
            "fallbacks apply.",
            UserWarning,
            stacklevel=5,
        )


def _warn_key_unused() -> None:
    global _key_warned
    if not _key_warned:
        _key_warned = True
        warnings.warn(
            "theagentrouter ignores api_key= without api_base=: the local gateway reads provider keys from the "
            "environment when it starts. Set the provider's variable (for example OPENAI_API_KEY) instead.",
            UserWarning,
            stacklevel=5,
        )


def _named_endpoint(model: str, api_base: str, api_key: str | None, headers: dict[str, str]) -> Target:
    """LiteLLM's api_base: an endpoint the developer names, in the provider's format, with its own model name."""
    provider, name = split_model(model)
    spec = local_gateway.PROVIDERS.get(provider, {})
    fmt = ANTHROPIC if spec.get("format") == ANTHROPIC else OPENAI
    variable = (spec.get("keys") or [None])[0]
    key = api_key or (os.environ.get(variable) if variable else None)
    if variable and not key:
        raise exceptions.AuthenticationError(
            f"set {variable} or pass api_key to call {api_base}", model=model, llm_provider=provider
        )
    return Target("api_base", fmt, api_base, key, name, provider, headers)


def _through_local(state: dict, model: str, provider: str, headers: dict[str, str]) -> Target:
    # The full name, as to any gateway; the gateway's rule for it passes the provider its own name.
    fmt = local_gateway.PROVIDERS.get(provider, {}).get("format", OPENAI)
    return Target("local gateway", fmt, local_gateway.url(state, provider), None, model, provider, headers)


def _through_route(r, headers: dict[str, str]) -> Target:
    """A route always goes through the local gateway, in the OpenAI format; the gateway does the retries and fallbacks."""
    if not local_gateway.installed():
        raise exceptions.BadRequestError(
            f"route {r.name!r} needs the local gateway, and {local_gateway.MISSING}.", model=r.name
        )
    if local_gateway.autostart_allowed():
        state = local_gateway.ensure(r.name)
    else:
        from . import routes

        # Starting is off, but a running gateway that already has this route can still serve it.
        state = local_gateway.running()
        if not local_gateway.serves(state, r.name, routes.list_routes()):
            raise exceptions.BadRequestError(
                f"route {r.name!r} needs the local gateway with the current routes, and starting it is turned off "
                "(AGENTROUTER_AUTOSTART=0).",
                model=r.name,
            )
    provider = r.primary.split("/", 1)[0]
    return Target(
        "local gateway",
        OPENAI,
        local_gateway.url(state),
        None,
        r.name,
        provider,
        headers,
        r.models,
    )


def _start_local(model: str, provider: str) -> dict:
    """Start the local gateway (or restart it with this model), or say what is missing."""
    elsewhere = f"or set AGENTROUTER_BASE_URL to a gateway that reaches {provider!r}"
    if not local_gateway.installed():
        message = f"{model} needs the local gateway, and {local_gateway.MISSING}."
    elif not local_gateway.autostart_allowed():
        message = (
            f"{model} needs the local gateway, and starting it is turned off (AGENTROUTER_AUTOSTART=0); {elsewhere}."
        )
    elif provider not in local_gateway.PROVIDERS:
        message = f"the local gateway does not set up {provider!r} yet; {elsewhere}."
    elif lacking := local_gateway.missing(provider, dict(os.environ)):
        raise exceptions.AuthenticationError(
            f"set {lacking} to use {provider} through the local gateway", model=model, llm_provider=provider
        )
    else:
        return local_gateway.ensure(model)
    raise exceptions.BadRequestError(message, model=model, llm_provider=provider)
