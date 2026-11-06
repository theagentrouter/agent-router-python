# SPDX-License-Identifier: Apache-2.0
"""The local gateway: Agent Router's ``aigw``, run by the SDK on the developer's computer.

``pip install theagentrouter`` also installs ``theagentrouter-local-gateway`` on macOS 13+ (Apple chips)
and Linux; it holds the ``aigw`` program. Every call goes through a gateway, and with none set this
is the one. Nothing else is run by hand:

1. The first call starts ``aigw run`` in the background.
2. The SDK writes the gateway's config from the provider keys in the environment. The config holds
   no key values: each key is in a private file that ``aigw`` reads when it starts.
3. Later calls, from any program on the computer, find the running gateway through
   ``AGENTROUTER_HOME/local-gateway/state.json`` and use it.

The SDK sends the model name as written (``bedrock/anthropic.claude-sonnet-4-5``, or a name without
a provider such as ``gpt-4o-mini``), as it does to any gateway. ``aigw`` routes on the name as it is
sent and cannot strip a ``provider/`` prefix, so the config has one route rule per model: it matches
the full name, and for OpenAI and Anthropic the name without a provider too, and passes the provider
its own name with ``modelNameOverride``. The rules cover the bundled model list for the providers whose keys are
set, the model that starts the gateway, and models added with ``register_model``. Adding one to a running
gateway restarts it, since ``aigw`` reads its config once.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

from . import _files, exceptions

PACKAGE = "theagentrouter_local_gateway"
# The local gateway comes with `pip install theagentrouter` on the platforms it is built for.
MISSING = (
    "the local gateway is not installed. pip install theagentrouter adds it on macOS 13+ (Apple chips) and "
    "Linux with glibc 2.36+ (x86-64, Arm); with theagentrouter-core alone, or elsewhere, set "
    "AGENTROUTER_BASE_URL to an Agent Router gateway"
)
HOST = "127.0.0.1"
# 0: a free port chosen at each start and kept in state.json, so another Agent Router gateway on 1975
# (a plain aigw run) never blocks this one. Tests set fixed ports.
PORT = 0
ADMIN_PORT = 0
# Fixed inside aigw and Envoy Gateway; another aigw on the computer holds them (see _busy_internal).
EXTPROC_PORT = 1061
XDS_PORT = 18000
# The Envoy that aigw downloads and runs. Without it aigw takes the newest release, which can need a
# newer OS (Envoy 1.40.0 does not start on macOS 14). Envoy Gateway v1.9, inside aigw, targets 1.39.
# ENVOY_VERSION in the environment overrides it.
ENVOY_VERSION = "1.39.4"
START_ATTEMPTS = 3  # a free port can be taken between choosing it and aigw binding it
START_TIMEOUT = 180.0  # seconds; the first start may fetch Envoy
LOCK_TIMEOUT = 30.0  # a start holds the lock for several seconds
# Gateway API allows at most 16 rules in one route, and aigw adds one rule of its own to each
# AIGatewayRoute, so the per-model rules are spread over several routes of at most 15.
RULES_PER_ROUTE = 15
LOOPBACK = ("127.0.0.1", "::1")
BACKENDS_READY_TIMEOUT = 15.0  # seconds to wait for Envoy to look up every route's backends
AZURE_API_VERSION = "2024-10-21"  # Azure OpenAI's generally available version; AZURE_API_VERSION overrides it

# Providers the local gateway sets up. "keys": all must be set; "any": at least one must be set.
PROVIDERS: dict[str, dict] = {
    "openai": {"schema": "OpenAI", "base": "https://api.openai.com/v1", "keys": ["OPENAI_API_KEY"]},
    # aigw takes Anthropic Messages as they are, under /anthropic (no translation from the OpenAI format).
    "anthropic": {
        "schema": "Anthropic",
        "base": "https://api.anthropic.com",
        "keys": ["ANTHROPIC_API_KEY"],
        "format": "anthropic",
        "path": "/anthropic/v1",
        # Routes go through the OpenAI format, which aigw does not translate to Anthropic Messages;
        # Anthropic's own OpenAI-compatible endpoint takes them instead.
        "openai_base": "https://api.anthropic.com/v1",
    },
    "gemini": {
        "schema": "OpenAI",
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "keys": ["GEMINI_API_KEY"],
    },
    "groq": {"schema": "OpenAI", "base": "https://api.groq.com/openai/v1", "keys": ["GROQ_API_KEY"]},
    "mistral": {"schema": "OpenAI", "base": "https://api.mistral.ai/v1", "keys": ["MISTRAL_API_KEY"]},
    "deepseek": {"schema": "OpenAI", "base": "https://api.deepseek.com/v1", "keys": ["DEEPSEEK_API_KEY"]},
    "together": {"schema": "OpenAI", "base": "https://api.together.xyz/v1", "keys": ["TOGETHER_API_KEY"]},
    "bedrock": {"schema": "AWSBedrock", "any": ["AWS_PROFILE", "AWS_ACCESS_KEY_ID"]},
    # LiteLLM's settings. The model name after azure/ is the deployment, which aigw puts in the path.
    "azure": {"schema": "AzureOpenAI", "keys": ["AZURE_API_KEY", "AZURE_API_BASE"], "base_variable": "AZURE_API_BASE"},
}


# A provider's name can resolve to an address that does not answer from this network. No request was
# sent on a connection that failed, so the gateway tries the next address, as an HTTP client does.
_RETRY_FAILED_CONNECTIONS = """  retry:
    numRetries: 2
    retryOn:
      triggers:
        - connect-failure"""


def installed() -> bool:
    """The local gateway package is installed and holds a program for this computer.

    Where no platform wheel fits, pip installs the fallback wheel, which has none.
    """
    if importlib.util.find_spec(PACKAGE) is None:
        return False
    has_binary = getattr(importlib.import_module(PACKAGE), "has_binary", None)
    return has_binary() if has_binary else True


def autostart_allowed() -> bool:
    return os.environ.get("AGENTROUTER_AUTOSTART", "1") != "0"


def folder() -> Path:
    from .routing import home

    return home() / "local-gateway"


def _aws_region(env: dict[str, str]) -> str | None:
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    if region and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", region):
        raise exceptions.BadRequestError("AWS_REGION must be an AWS region name, for example us-east-1")
    return region


def _azure_version(env: dict[str, str]) -> str:
    version = env.get("AZURE_API_VERSION") or AZURE_API_VERSION
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-preview)?", version):
        raise exceptions.BadRequestError(
            "AZURE_API_VERSION must be an Azure OpenAI API version, for example 2024-10-21"
        )
    return version


def _base_variable(provider: str) -> str:
    return PROVIDERS[provider].get("base_variable") or f"{provider.upper()}_BASE_URL"


def missing(provider: str, env: dict[str, str]) -> str | None:
    """What the environment lacks for this provider, or None when it can be set up."""
    spec = PROVIDERS[provider]
    if "any" in spec:
        if not any(env.get(k) for k in spec["any"]):
            return " or ".join(spec["any"])
        if provider == "bedrock":
            if env.get("AWS_ACCESS_KEY_ID") and not env.get("AWS_PROFILE") and not env.get("AWS_SECRET_ACCESS_KEY"):
                return "AWS_SECRET_ACCESS_KEY"
            if not _aws_region(env):
                return "AWS_REGION"
        return None
    absent = [k for k in spec["keys"] if not env.get(k)]
    return " and ".join(absent) or None


def available(env: dict[str, str]) -> list[str]:
    """Providers whose keys are in the environment."""
    return sorted(p for p in PROVIDERS if missing(p, env) is None)


def routed_models(providers: list[str], extra: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """The full model names the gateway gets a rule for: the model list's, plus ``extra``."""
    from .models import model_cost

    names = {m for m in model_cost if m.split("/", 1)[0] in providers}
    names |= {m for m in extra if m.split("/", 1)[0] in providers}
    return sorted(names)


# --- config ---------------------------------------------------------------------------------------


def _endpoint(base: str) -> tuple[str, int, bool, str]:
    """(host, port, tls, version path) for a base URL."""
    parts = urlsplit(base)
    tls = parts.scheme == "https"
    port = parts.port or (443 if tls else 80)
    return parts.hostname or "", port, tls, parts.path.strip("/")


def _is_ip(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False


def _route_backend(model: str) -> tuple[str, str]:
    """(backend name, model name the provider gets) for one model in a route, which speaks the OpenAI format."""
    provider, own = model.split("/", 1)
    own = _provider_name(provider, own)
    if PROVIDERS[provider].get("format") == "anthropic":
        return f"{provider}-openai", own
    return provider, own


def _provider_name(provider: str, own: str) -> str:
    """The name the provider gets. As in LiteLLM, bedrock/converse/... and bedrock/invoke/... name a Bedrock API."""
    if provider == "bedrock":
        return own.removeprefix("converse/").removeprefix("invoke/")
    return own


def _names(model: str) -> list[str]:
    """The names a model's rule matches: the full name, and the name without a provider where the
    model list gives that name this provider (``gpt-4o-mini``, ``claude-sonnet-4-5``), as LiteLLM and
    Agent Router Enterprise write them."""
    from . import models

    provider, own = model.split("/", 1)
    try:
        bare = models.provider_of(own) == provider
    except exceptions.BadRequestError:
        bare = False
    return [model, own] if bare else [model]


def _k8s_name(name: str) -> str:
    """A resource name for a route: lowercase letters, digits and dashes, unique per route name."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "route"
    return f"theagentrouter-route-{slug}-{hashlib.sha256(name.encode()).hexdigest()[:6]}"


def config_yaml(
    providers: list[str],
    env: dict[str, str],
    port: int = PORT,
    models: list[str] | None = None,
    routes: dict | None = None,
    local: bool = True,
    strip_client_keys: bool = False,
) -> str:
    """The resources ``aigw run`` reads. Key values never appear; ``${VAR}`` is filled in by aigw.

    ``local``: answer only callers on this computer. Envoy Gateway binds every listener to all
    addresses and has no setting for that, so a policy refuses other callers with 403; without it,
    anyone on the network could use the developer's provider keys.

    ``strip_client_keys``: remove the caller's ``Authorization`` and ``x-api-key`` before a request goes to
    a provider, which then gets its own key. Used when an exported gateway checks a client key.

    One route rule per full model name in ``models`` (default: ``routed_models(providers)``), and one
    route per entry in ``routes`` (name -> routes.Route): its models in priority order, plus a retry policy.
    """
    docs = [
        """apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: theagentrouter
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller""",
        f"""apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: theagentrouter
  namespace: default
spec:
  gatewayClassName: theagentrouter
  listeners:
    - name: http
      protocol: HTTP
      port: {port}""",
        """apiVersion: gateway.envoyproxy.io/v1alpha1
kind: ClientTrafficPolicy
metadata:
  name: client-buffer-limit
  namespace: default
spec:
  targetRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: theagentrouter
  connection:
    bufferLimit: 50Mi""",
        f"""apiVersion: gateway.envoyproxy.io/v1alpha1
kind: BackendTrafficPolicy
metadata:
  name: retry-failed-connections
  namespace: default
spec:
  targetRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: theagentrouter
{_RETRY_FAILED_CONNECTIONS}""",
    ]
    if local:
        # The Host check stops a web page from reaching the gateway through DNS rebinding: the browser
        # connects from this computer, but sends the page's own name as the Host.
        docs.append(
            f"""apiVersion: gateway.envoyproxy.io/v1alpha1
kind: SecurityPolicy
metadata:
  name: this-computer-only
  namespace: default
spec:
  targetRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: theagentrouter
  authorization:
    defaultAction: Deny
    rules:
      - name: loopback
        action: Allow
        principal:
          clientCIDRs:
            - 127.0.0.1/32
            - ::1/128
          headers:
            - name: host
              values:
                - "127.0.0.1:{port}"
                - "localhost:{port}"
                - "[::1]:{port}\""""
        )
    routes = routes or {}
    backends = [(name, PROVIDERS[name]) for name in providers]
    for r in routes.values():
        for model in r.models:
            backend, _ = _route_backend(model)
            if backend not in dict(backends):
                spec = PROVIDERS[model.split("/", 1)[0]]
                backends.append((backend, {**spec, "schema": "OpenAI", "base": spec["openai_base"], "format": None}))
    for name, spec in backends:
        if spec["schema"] == "AWSBedrock":
            region = _aws_region(env)
            host, ep_port, tls, version = f"bedrock-runtime.{region}.amazonaws.com", 443, True, ""
        else:
            variable = _base_variable(name.split("-")[0])
            base = env.get(variable) or spec.get("base", "")
            parts = urlsplit(base)
            if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
                raise exceptions.BadRequestError(f"{variable} needs an HTTP(S) URL without credentials")
            host, ep_port, tls, version = _endpoint(base)
            if host in LOOPBACK:
                # A hostname, not an IP: aigw run cannot tell IP backends of one rule apart (see _route_docs).
                host = "localhost"
            if spec["schema"] == "OpenAI" and version == "v1":
                version = ""
            if spec["schema"] == "Anthropic":
                version = ""
            if spec["schema"] == "AzureOpenAI":
                version = _azure_version(env)  # aigw writes the whole path, /openai/deployments/...
        address = f"ip:\n        address: {host}" if _is_ip(host) else f"fqdn:\n        hostname: {host}"
        docs.append(
            f"""apiVersion: gateway.envoyproxy.io/v1alpha1
kind: Backend
metadata:
  name: {name}
  namespace: default
spec:
  endpoints:
    - {address}
        port: {ep_port}"""
        )
        if tls:
            docs.append(
                f"""apiVersion: gateway.networking.k8s.io/v1
kind: BackendTLSPolicy
metadata:
  name: {name}-tls
  namespace: default
spec:
  targetRefs:
    - group: gateway.envoyproxy.io
      kind: Backend
      name: {name}
  validation:
    wellKnownCACertificates: System
    hostname: {host}"""
            )
        schema = f"    name: {spec['schema']}" + (f'\n    version: "{version}"' if version else "")
        # Removed before the provider's own key is added, so a client key never reaches a provider.
        mutation = "\n  headerMutation:\n    remove: [authorization, x-api-key]" if strip_client_keys else ""
        docs.append(
            f"""apiVersion: aigateway.envoyproxy.io/v1beta1
kind: AIServiceBackend
metadata:
  name: {name}
  namespace: default
spec:
  timeouts:
    request: 10m
  schema:
{schema}
  backendRef:
    name: {name}
    kind: Backend
    group: gateway.envoyproxy.io
    namespace: default{mutation}"""
        )
        target = f"""  targetRefs:
    - group: aigateway.envoyproxy.io
      kind: AIServiceBackend
      name: {name}"""
        if spec["schema"] == "AWSBedrock":
            # No credentials file: aigw uses the AWS default chain (AWS_PROFILE, AWS_ACCESS_KEY_ID, ...).
            policy = f"""  type: AWSCredentials
  awsCredentials:
    region: {_aws_region(env)}"""
        else:
            kind, field = {
                "Anthropic": ("AnthropicAPIKey", "anthropicAPIKey"),
                "AzureOpenAI": ("AzureAPIKey", "azureAPIKey"),
            }.get(spec["schema"], ("APIKey", "apiKey"))
            policy = f"""  type: {kind}
  {field}:
    secretRef:
      name: {name}-key"""
            if local:
                # A private file, so the config holds no key value.
                source = f"""  annotations:
    substitution.aigw.run/file/apiKey: {json.dumps(str(_key_file(spec["keys"][0])))}
type: Opaque
stringData:
  apiKey: file"""
            else:
                source = f"""type: Opaque
stringData:
  apiKey: ${{{spec["keys"][0]}}}"""
            docs.append(
                f"""apiVersion: v1
kind: Secret
metadata:
  name: {name}-key
  namespace: default
{source}"""
            )
        docs.append(
            f"""apiVersion: aigateway.envoyproxy.io/v1beta1
kind: BackendSecurityPolicy
metadata:
  name: {name}-auth
  namespace: default
spec:
{target}
{policy}"""
        )
    rules = []
    for model in routed_models(providers) if models is None else models:
        provider, own = model.split("/", 1)
        if provider not in providers or model in routes:  # a route by the same name takes the name
            continue
        matches = "".join(
            f"""
        - headers:
            - type: Exact
              name: x-ai-eg-model
              value: {json.dumps(name)}"""
            for name in _names(model)
            if name not in routes
        )
        rules.append(
            f"""    - matches:{matches}
      backendRefs:
        - name: {provider}
          namespace: default
          modelNameOverride: {json.dumps(_provider_name(provider, own))}
      timeouts:
        request: 10m"""
        )
    for n in range(0, max(len(rules), 1), RULES_PER_ROUTE):
        part = rules[n : n + RULES_PER_ROUTE]
        if not part:
            break
        docs.append(
            f"""apiVersion: aigateway.envoyproxy.io/v1beta1
kind: AIGatewayRoute
metadata:
  name: theagentrouter-{n // RULES_PER_ROUTE + 1}
  namespace: default
spec:
  parentRefs:
    - name: theagentrouter
      kind: Gateway
      group: gateway.networking.k8s.io
  rules:
"""
            + "\n".join(part)
            + """
  llmRequestCosts:
    - metadataKey: llm_input_token
      type: InputToken
    - metadataKey: llm_output_token
      type: OutputToken"""
        )
    for r in routes.values():
        docs.extend(_route_docs(r))
    return "\n---\n".join(docs) + "\n"


def export_config(
    port: int = 1975,
    providers: list[str] | None = None,
    models: list[str] | None = None,
    *,
    authentication: bool = True,
    api_key_env: str = "AGENTROUTER_API_KEY",
) -> str:
    """The local gateway's config, for a team to run the same gateway on a server: ``aigw run config.yaml``.

    Developers then set ``AGENTROUTER_BASE_URL`` to that server, and the same code works there as with
    the local gateway. The config is the one the SDK writes for the local gateway, listening on a
    fixed ``port``: one rule per model in the model list for ``providers`` (default: those whose keys
    are set here), plus ``models`` (full names the list lacks, such as a fine-tuned model), and this
    computer's routes. Key values never appear: ``aigw`` fills in
    ``${OPENAI_API_KEY}``-style references from the server's environment. Authentication is on by
    default: set ``api_key_env`` on the server, and clients send that key in Authorization: Bearer
    or x-api-key. Keys are removed before forwarding. ``authentication=False`` explicitly exports
    an open endpoint. Terminate TLS in front of the server when callers use an untrusted network.
    """
    from . import routes as routes_mod

    env = dict(os.environ)
    chosen = sorted(available(env) if providers is None else set(providers))
    unknown = [p for p in chosen if p not in PROVIDERS]
    if unknown:
        raise exceptions.BadRequestError(
            f"unknown providers: {', '.join(unknown)}. The gateway config can hold {', '.join(sorted(PROVIDERS))}"
        )
    if not chosen:
        raise exceptions.BadRequestError(
            "no provider keys are set here, so no providers to put in the config; "
            "name them, for example export_config(providers=['openai', 'anthropic'])"
        )
    if "bedrock" in chosen and not _aws_region(env):
        raise exceptions.BadRequestError(
            "bedrock needs AWS_REGION or AWS_DEFAULT_REGION set: the config names the region"
        )
    if "azure" in chosen and not env.get("AZURE_API_BASE"):
        raise exceptions.BadRequestError("azure needs AZURE_API_BASE set: the config names the endpoint")
    extra = set(models or ())
    bare = sorted(m for m in extra if "/" not in m)
    if bare:
        raise exceptions.BadRequestError(f"models need their provider, for example openai/{bare[0]}")
    current = _as_called(routes_mod.list_routes())
    used = {m.split("/", 1)[0] for r in current.values() for m in r.models} | {m.split("/", 1)[0] for m in extra}
    outside = sorted(used - set(chosen))
    if outside:
        raise exceptions.BadRequestError(
            f"the routes or models use providers not in the config: {', '.join(outside)}; add them to providers"
        )
    config = config_yaml(
        chosen, env, port, routed_models(chosen, extra), current, local=False, strip_client_keys=authentication
    )
    if authentication:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        config += "---\n" + _api_key_config(api_key_env)
    return config


def _as_called(current: dict) -> dict:
    """Routes under the names code calls them by: a LiteLLM Router's ``router-1a2b3c4d:chat`` becomes ``chat``."""
    import dataclasses

    out: dict = {}
    origin: dict[str, str] = {}
    for name, r in sorted(current.items()):
        group = re.fullmatch(r"router-[0-9a-f]{8}:(.+)", name)
        called = group.group(1) if group else name
        if called in out:
            raise exceptions.BadRequestError(
                f"routes {origin[called]!r} and {name!r} would both be {called!r} in the config; "
                "remove one with theagentrouter.remove_route()"
            )
        out[called] = dataclasses.replace(r, name=called)
        origin[called] = name
    return out


def _api_key_config(variable: str) -> str:
    # aigw fills in ${VAR} across the whole file. Missing or empty, the expected value becomes a line break,
    # which no header can carry, so a server started without the variable refuses every request.
    return f"""apiVersion: gateway.envoyproxy.io/v1alpha1
kind: SecurityPolicy
metadata:
  name: theagentrouter-client-auth
  namespace: default
spec:
  targetRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: theagentrouter
  authorization:
    defaultAction: Deny
    rules:
      - name: client-key
        action: Allow
        principal:
          headers:
            - name: x-api-key
              values: ["${{{variable}:-\\n}}"]
      - name: client-bearer
        action: Allow
        principal:
          headers:
            - name: authorization
              values: ["Bearer ${{{variable}:-\\n}}"]
"""


def _route_docs(r) -> list[str]:
    """One AIGatewayRoute for a route: its models as backends at priority 0, 1, ... (one per tier), and a
    retry policy that tries each tier ``attempts + 1`` times before moving to the next.

    This is Agent Router's provider fallback, as Agent Router Enterprise renders it. Every backend
    must be a hostname: under ``aigw run`` an IP backend's endpoints come over EDS, and the AI Gateway
    extension server then labels the whole rule with its first backend, so a fallback is sent the
    first backend's model name and key and the priorities are lost (``theagentrouter/agent-router``
    ``internal/extensionserver/post_translate_modify.go``, the ``LoadAssignment == nil`` case).

    One request goes to every model in the route, so the gateway removes what a newer Claude refuses
    before that model gets it. Otherwise a fallback to it fails on a temperature the first model took.
    The removal runs on the request as the backend gets it, so Bedrock's fields have Converse's names.
    """
    from .anthropic_format import _refuses_sampling

    name = _k8s_name(r.name)
    refs = []
    for priority, tier in enumerate(r.tiers):
        for index, model in enumerate(tier):
            weight = r.weights[priority][index] if r.weights else 1
            backend, own = _route_backend(model)
            removed = []
            if model.startswith("anthropic/") and _refuses_sampling(own):
                removed = ["temperature", "top_p"]
            elif model.startswith("bedrock/") and "claude" in own:
                removed = ["additionalModelRequestFields.reasoning_config"]
                if _refuses_sampling(own):
                    removed = ["inferenceConfig.temperature", "inferenceConfig.topP", *removed]
            mutation = f"\n          bodyMutation:\n            remove: [{', '.join(removed)}]" if removed else ""
            refs.append(
                f"""        - name: {backend}
          namespace: default
          modelNameOverride: {json.dumps(own)}
          priority: {priority}
          weight: {weight}{mutation}"""
            )
    docs = [
        f"""apiVersion: aigateway.envoyproxy.io/v1beta1
kind: AIGatewayRoute
metadata:
  name: {name}
  namespace: default
spec:
  parentRefs:
    - name: theagentrouter
      kind: Gateway
      group: gateway.networking.k8s.io
  rules:
    - matches:
        - headers:
            - type: Exact
              name: x-ai-eg-model
              value: {json.dumps(r.name)}
      backendRefs:
"""
        + "\n".join(refs)
        + """
      timeouts:
        request: 10m
  llmRequestCosts:
    - metadataKey: llm_input_token
      type: InputToken
    - metadataKey: llm_output_token
      type: OutputToken"""
    ]
    per_tier = r.retry.attempts + 1
    retries = per_tier * len(r.tiers) - 1
    policy = []
    if r.limit:
        # Envoy's local rate limit: counted in the gateway, no Redis; it answers 429 before any retry.
        policy.append(
            f"""  rateLimit:
    type: Local
    local:
      rules:
        - limit:
            requests: {r.limit.requests}
            unit: {r.limit.per.capitalize()}"""
        )
    if retries > 0:
        codes = "\n".join(f"        - {c}" for c in r.retry.on)
        from .routes import _duration

        timeout = f"\n      timeout: {_duration(r.retry.timeout)}" if r.retry.timeout is not None else ""
        policy.append(
            f"""  retry:
    numRetries: {retries}
    numAttemptsPerPriority: {per_tier}
    perRetry:
      backOff:
        baseInterval: {_duration(r.retry.backoff)}
        maxInterval: {_duration(max(10, r.retry.backoff))}{timeout}
    retryOn:
      httpStatusCodes:
{codes}
      triggers:
        - connect-failure
        - reset
        - retriable-status-codes"""
        )
    elif r.limit:
        policy.append(_RETRY_FAILED_CONNECTIONS)  # a route's own policy replaces the gateway's
    if policy:
        # One policy per route: Envoy Gateway takes only one BackendTrafficPolicy for an HTTPRoute.
        docs.append(
            f"""apiVersion: gateway.envoyproxy.io/v1alpha1
kind: BackendTrafficPolicy
metadata:
  name: {name}
  namespace: default
spec:
  targetRefs:
    - group: gateway.networking.k8s.io
      kind: HTTPRoute
      name: {name}
"""
            + "\n".join(policy)
        )
    return docs


def _envoy_admin(log_path: Path) -> int | None:
    """Envoy's admin port, from aigw's start line ("listening on ... (admin http://localhost:N)")."""
    import re

    m = re.search(r"\(admin http://[^:]+:(\d+)\)", log_path.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def _backend_address(backend: str) -> tuple[str, int]:
    """(hostname, port) a route backend is written with in the config."""
    provider = backend.removesuffix("-openai")
    spec = PROVIDERS[provider]
    if spec["schema"] == "AWSBedrock":
        return f"bedrock-runtime.{_aws_region(dict(os.environ))}.amazonaws.com", 443
    base = os.environ.get(_base_variable(provider)) or (
        spec["openai_base"] if backend.endswith("-openai") else spec.get("base", "")
    )
    host, port, _, _ = _endpoint(base)
    return ("localhost" if host in LOOPBACK else host), port


def _backends_ready(admin: int, routes: dict) -> bool:
    """Whether Envoy has looked up every backend host of every route.

    Envoy resolves hostnames after it starts listening; until it has, a fallback has nowhere to go.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{HOST}:{admin}/clusters", timeout=2) as resp:
            lines = resp.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return False
    for r in routes.values():
        prefix = f"httproute/default/{_k8s_name(r.name)}/rule/0::"
        seen = set()
        for line in lines:  # <cluster>::<ip>:<port>::hostname::<hostname>
            parts = line.split("::")
            if line.startswith(prefix) and len(parts) == 4 and parts[2] == "hostname":
                seen.add((parts[3], int(parts[1].rsplit(":", 1)[1])))
        if not {_backend_address(_route_backend(m)[0]) for m in r.models} <= seen:
            return False
    return True


def _alive(pid: int) -> bool:
    if pid <= 1:
        return False
    # Reap our own exited child before kill(0), which still succeeds for zombies.
    with contextlib.suppress(ChildProcessError):
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise exceptions.APIConnectionError(f"refusing unsafe local gateway directory: {path}")
    path.chmod(0o700)


def _stopped(pid: int) -> bool:
    if not _alive(pid):
        return True
    # A child of another SDK process can exit without that parent reaping it yet.
    # Query process status only during shutdown, never on the request path.
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=1)
        return result.stdout.strip().startswith("Z")
    return False


def _write_state(state: dict) -> None:
    directory = folder()
    _private_dir(directory)
    descriptor, name = tempfile.mkstemp(prefix=".state-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(state, handle)
        os.replace(name, directory / "state.json")
    finally:
        Path(name).unlink(missing_ok=True)


def _key_file(variable: str) -> Path:
    return folder() / "keys" / variable


def _write_keys(providers: list[str], env: dict[str, str]) -> None:
    """Each provider's API key from this environment, in a private file aigw reads when it starts."""
    directory = folder() / "keys"
    _private_dir(directory)
    for provider in providers:
        spec = PROVIDERS.get(provider, {})
        if spec.get("schema") == "AWSBedrock" or not spec.get("keys") or not env.get(spec["keys"][0]):
            continue
        descriptor, name = tempfile.mkstemp(prefix=".key-", dir=directory)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(env[spec["keys"][0]])
            os.replace(name, directory / spec["keys"][0])
        finally:
            Path(name).unlink(missing_ok=True)


def _credential_fingerprints(providers: list[str], env: dict[str, str]) -> dict[str, str]:
    fingerprints = {}
    for provider in providers:
        spec = PROVIDERS.get(provider, {})
        variables = list(spec.get("keys", []))
        if provider == "bedrock":
            variables = [
                "AWS_PROFILE",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
            ]
        if any(env.get(key) for key in variables):
            value = json.dumps({key: env.get(key) for key in variables}, sort_keys=True)
            fingerprints[provider] = hashlib.sha256(value.encode()).hexdigest()
    return fingerprints


def check_credentials(state: dict) -> None:
    """Report rotated credentials rather than silently continuing to use the old key."""
    current = _credential_fingerprints(state.get("providers", []), dict(os.environ))
    previous = state.get("credential_fingerprints", {})
    if any(provider in previous and previous[provider] != value for provider, value in current.items()):
        raise exceptions.AuthenticationError(
            "provider credentials changed since the local gateway started; "
            "call local_gateway.stop() when other programs have finished, then retry to load the new keys"
        )


def _listening(port: int, timeout: float = 0.5) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((HOST, port), timeout=timeout):
        return True
    return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _port_owner(port: int) -> str | None:
    """The name of the program listening on a port, where lsof can tell."""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fc"], capture_output=True, text=True, timeout=3
        ).stdout
        names = [line[1:] for line in out.splitlines() if line.startswith("c")]
        if names:
            return names[0]
    return None


def _busy_internal() -> None:
    """aigw's internal ports cannot move yet; say which program holds them."""
    for port, what in ((EXTPROC_PORT, "internal port"), (XDS_PORT, "xDS port")):
        if not _listening(port):
            continue
        owner = _port_owner(port)
        held = f"is in use by {owner}" if owner else "is already in use"
        hint = (
            " Perhaps another Agent Router gateway, and two cannot run on one computer yet:"
            " stop that program, or set AGENTROUTER_BASE_URL to a gateway."
        )
        raise exceptions.APIConnectionError(
            f"the local gateway cannot start: aigw's {what} {port} on {HOST} {held}.{hint}"
        )


# When this process last found each gateway (pid, port) listening; a call does not probe again within this time.
_listening_seen: dict[tuple[int, int], float] = {}
LISTENING_RECHECK = 30.0


def running() -> dict | None:
    """The running local gateway's state, or None. Any program on the computer finds the same one."""
    state = _files.read_json(folder() / "state.json")
    if not isinstance(state, dict) or state.get("starting") or not _alive(int(state.get("pid", 0))):
        return None
    key = (int(state["pid"]), int(state.get("port", PORT)))
    seen = _listening_seen.get(key)
    if seen is None or time.monotonic() - seen > LISTENING_RECHECK:
        if not _listening(key[1]):
            _listening_seen.pop(key, None)
            return None
        _listening_seen[key] = time.monotonic()
    return state


def url(state: dict, provider: str | None = None) -> str:
    path = PROVIDERS.get(provider or "", {}).get("path", "/v1")
    return f"http://{HOST}:{state['port']}{path}"


@contextlib.contextmanager
def _lock() -> Iterator[None]:
    """One program at a time starts or restarts the gateway."""
    _private_dir(folder())
    descriptor = os.open(folder() / "lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "a+") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            raise exceptions.APIConnectionError("refusing unsafe local gateway lock file")
        os.fchmod(handle.fileno(), 0o600)
        try:
            import fcntl
        except ImportError:  # Windows: no lock yet
            yield
            return
        deadline = time.monotonic() + LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise exceptions.Timeout("timed out waiting for the local gateway startup lock") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _binary() -> str:
    gateway = importlib.import_module(PACKAGE)
    return str(gateway.binary_path())


def _runtime_dir() -> Path:
    # aigw puts Unix sockets here; keep the path short, and separate per AGENTROUTER_HOME.
    tag = hashlib.sha256(str(folder()).encode()).hexdigest()[:8]
    base = Path("/tmp") if os.name == "posix" else Path(tempfile.gettempdir())
    return base / f"theagentrouter-{os.getuid() if hasattr(os, 'getuid') else 0}-{tag}"


def stop(state: dict | None = None, timeout: float = 15.0) -> None:
    """Stop the local gateway, if one is running."""
    if state is None:
        try:
            state = json.loads((folder() / "state.json").read_text())
        except (OSError, ValueError):
            return
    if not state:
        return
    pid = int(state["pid"])
    if pid <= 1 or pid == os.getpid():
        raise exceptions.APIConnectionError("refusing to stop an invalid local gateway process")
    # A gateway that is gone may have left its process number to another program: leave that one alone.
    if _is_gateway(pid):
        _terminate(pid, timeout)
    (folder() / "state.json").unlink(missing_ok=True)
    for path in (folder() / "aigw-state", folder() / "keys", _runtime_dir()):
        if path.exists():
            _private_dir(path)
            shutil.rmtree(path)


def _is_gateway(pid: int) -> bool:
    """Whether the process is still aigw. Asked only when stopping, never on the request path."""
    try:
        # -ww: without a terminal, ps on Linux cuts the line at 80 columns, before a long path's aigw.
        result = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=1
        )
    except (OSError, subprocess.SubprocessError):
        return True  # no ps, as on Windows: the stored number is all there is to go on
    command = result.stdout.strip()
    if not command or "<defunct>" in command:
        return True  # gone, or a zombie: a signal cannot reach another program
    return re.search(r"(^|[/\\])aigw(\.exe)?(\s|$)", command) is not None


def _terminate(pid: int, timeout: float) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while not _stopped(pid) and time.monotonic() < deadline:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        time.sleep(0.1)
    if not _stopped(pid) and hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        # Reap the killed child as well. Never delete key files while it is still running.
        deadline = time.monotonic() + 2
        while not _stopped(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if not _stopped(pid):
        raise exceptions.Timeout("the local gateway did not stop; its state and key files were retained")


def _start(providers: list[str], models: list[str], routes: dict | None = None) -> dict:
    from . import routes as routes_mod

    routes = {} if routes is None else routes
    env = dict(os.environ)
    home = folder()
    _private_dir(home)
    if PORT and _listening(PORT):
        raise exceptions.APIConnectionError(
            f"the local gateway cannot start: port {PORT} on {HOST} is already in use, perhaps by another gateway. "
            "Set AGENTROUTER_BASE_URL to use that gateway instead."
        )
    _busy_internal()
    runtime = _runtime_dir()
    _private_dir(runtime)
    env.setdefault("ENVOY_VERSION", ENVOY_VERSION)
    env.update(
        AIGW_CONFIG_HOME=str(home / "config-home"),
        AIGW_STATE_HOME=str(home / "aigw-state"),
        AIGW_RUNTIME_DIR=str(runtime),
    )
    _write_keys(providers, env)
    log_path = home / "aigw.log"
    for attempt in range(START_ATTEMPTS):
        port, admin_port = PORT or _free_port(), ADMIN_PORT or _free_port()
        while admin_port == port:  # two calls can return the same port
            admin_port = _free_port()
        try:
            proc = _launch(home, env, log_path, providers, models, routes, port, admin_port)
            break
        except exceptions.APIConnectionError as error:
            # Something took a chosen port before aigw bound it, another program or Envoy itself: choose again.
            taken = (
                isinstance(error, _PortTaken)
                or "address already in use" in log_path.read_text(errors="replace").lower()
            )
            if not taken or attempt == START_ATTEMPTS - 1 or (PORT and ADMIN_PORT):
                raise
            _busy_internal()
    state = {
        "pid": proc.pid,
        "port": port,
        "providers": providers,
        "models": models,
        "routes": sorted(routes),
        "routes_fingerprint": routes_mod.fingerprint(routes),
        "versions": versions(),
        "credential_fingerprints": _credential_fingerprints(providers, env),
    }
    try:
        admin = _envoy_admin(log_path) if routes else None
        if admin:
            ready_by = time.monotonic() + BACKENDS_READY_TIMEOUT
            while time.monotonic() < ready_by and not _backends_ready(admin, routes):
                time.sleep(0.2)
        _write_state(state)
    except BaseException:
        stop({"pid": proc.pid})
        raise
    # aigw outlives this program on purpose. Without this, Python warns "subprocess N is still running"
    # when proc goes; _alive() and _stopped() still collect it if it exits while this program runs.
    proc.returncode = 0
    print(
        f"theagentrouter: started the local gateway on http://{HOST}:{port} "
        f"(providers from your environment: {', '.join(providers)}).",
        file=sys.stderr,
    )
    return state


class _PortTaken(exceptions.APIConnectionError):
    """A port chosen for the gateway was taken before the gateway bound it."""


def _launch(
    home: Path,
    env: dict,
    log_path: Path,
    providers: list[str],
    models: list[str],
    routes: dict,
    port: int,
    admin_port: int,
) -> subprocess.Popen:
    config = home / "config.yaml"
    config.write_text(config_yaml(providers, env, port, models, routes))
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as log:
        proc = subprocess.Popen(
            [_binary(), "run", str(config), f"--admin-port={admin_port}"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=home,
            env=env,
            start_new_session=True,  # outlives this program; later calls from any program use it
        )
    try:
        _write_state({"pid": proc.pid, "port": port, "starting": True})
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            lines = log_path.read_text(errors="replace").strip().splitlines()
            if proc.poll() is not None:
                errors = [line for line in lines if re.search(r"error|fatal|invalid|failed", line, re.IGNORECASE)]
                detail = " | ".join((errors or lines)[-5:])
                raise exceptions.APIConnectionError(
                    f"the local gateway stopped while starting (log: {log_path}): {detail}"
                )
            if "listening on" in "\n".join(lines) and _listening(port):
                # Envoy gets its admin page's port from the operating system as it starts, and that can be the
                # port chosen for the gateway, free until the listener binds it. The admin page would answer every call.
                if _envoy_admin(log_path) == port:
                    raise _PortTaken(f"Envoy's admin page took port {port}, chosen for the local gateway")
                return proc
            time.sleep(0.2)
        raise exceptions.Timeout(f"the local gateway did not start within {START_TIMEOUT:.0f}s (log: {log_path})")
    except BaseException:
        stop({"pid": proc.pid})
        proc.wait(timeout=2)
        raise


def versions() -> dict[str, str]:
    """The SDK's and the local gateway package's versions, kept in state.json by the start."""
    from . import __version__

    out = {"sdk": __version__}
    with contextlib.suppress(ImportError, AttributeError):
        out["local_gateway"] = importlib.import_module(PACKAGE).__version__
    return out


def serves(state: dict | None, model: str, current: dict) -> bool:
    """The running gateway has the right rule for this name: the current route, or a plain model rule.

    A name that was a route when the gateway started keeps that route's fallbacks, retries and
    limit until a restart, so it counts as served only while it is still the same route. Existing
    rules can be reused across SDK versions; changing a different version's config requires an explicit stop.
    """
    from . import routes as routes_mod

    if not state:
        return False
    if model in current:
        return state.get("routes_fingerprint") == routes_mod.fingerprint(current)
    return model in state.get("models", []) and model not in state.get("routes", [])


def ensure(model: str) -> dict:
    """The running gateway with a rule for this model name or route, starting or restarting it.

    A route counts as present only while the running gateway has this computer's current routes.
    """
    from . import routes as routes_mod

    with _lock():
        state = running()
        current = routes_mod.list_routes()
        if serves(state, model, current):
            check_credentials(state)
            return state
        if state and state.get("versions") != versions():
            raise exceptions.BadRequestError(
                "the local gateway was started by another SDK version; stop it explicitly when other programs "
                "have finished before changing its configuration",
                model=model,
            )
        providers = available(dict(os.environ))
        if state and set(state.get("providers", [])) - set(providers):
            raise exceptions.AuthenticationError(
                "changing the shared gateway requires credentials for every provider already running; "
                "its current process was left running",
                model=model,
            )
        needed = {m.split("/", 1)[0] for route in current.values() for m in route.models}
        if needed - set(providers):
            raise exceptions.AuthenticationError(
                "set credentials for all providers in the saved routes before starting"
            )
        seen = set(state.get("models", [])) if state else set()
        models = routed_models(providers, seen | ({model} if model not in current else set()))
        if state:
            # aigw reads its config once at start: a new provider, model or route means a restart.
            stop(state)
        return _start(providers, models, current)
