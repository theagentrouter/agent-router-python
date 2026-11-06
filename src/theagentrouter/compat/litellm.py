# SPDX-License-Identifier: Apache-2.0
"""LiteLLM-compatible module: `from theagentrouter.compat import litellm`, or install_as_litellm().

Covers what agent frameworks use from LiteLLM: completion and acompletion, embedding and aembedding, the response
types, the error names, the model table, token_counter, and LiteLLM's Router for fallbacks, retries and
load sharing, served by the local gateway. LiteLLM's cache and batch are not here.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import Any, TypedDict

# Re-exported under LiteLLM's names; frameworks call litellm.completion, litellm.get_model_info, ...
from .. import (  # noqa: F401
    EmbeddingResponse,
    anthropic,
    anthropic_messages,
    completion_cost,
    cost_per_token,
    exceptions,
    get_model_info,
    model_cost,
    register_model,
    stream_chunk_builder,
    supports_function_calling,
    supports_prompt_caching,
    supports_reasoning,
    supports_response_schema,
    supports_vision,
    validate_environment,
)
from .. import acompletion as _acompletion
from .. import aembedding as _aembedding
from .. import aresponses as _aresponses
from .. import completion as _completion
from .. import embedding as _embedding
from .. import responses as _responses
from ..exceptions import *  # noqa: F401,F403  (LiteLLM exposes its errors at top level too)
from ..local_gateway import PROVIDERS as _PROVIDERS
from ..models import model_cost as _model_cost
from ..responses import ResponsesAPIResponse
from ..tokens import token_counter as _token_counter
from ..types import Choices, Delta, Message, ModelResponse, ModelResponseStream, StreamingChoices, Usage

drop_params = False  # as in LiteLLM: True drops parameters the provider does not take, instead of raising
api_base = None
api_key = None
num_retries = None
request_timeout = None
modify_params = False
# Settings frameworks set on import; accepted, and have no effect here.
suppress_debug_info = True
telemetry = False
cache = None
set_verbose = False
disable_token_counter = False  # True makes token_counter return 0, as in LiteLLM

add_function_to_prompt = False  # ADK sets this

# LiteLLM's callback lists. Functions and logger objects in them are called after each completion;
# LiteLLM's built-in integrations, named by string ("langfuse"), are not here: the gateway's
# OpenTelemetry traces and metrics replace them.
success_callback: list[Any] = []
failure_callback: list[Any] = []
callbacks: list[Any] = []
_async_success_callback: list[Any] = []

# Model names without a provider, as in LiteLLM's model_list.
model_list: list[str] = sorted({k.split("/", 1)[1] for k in _model_cost})

# LiteLLM's litellm.cost_calculator path.
cost_calculator = ModuleType("litellm.cost_calculator")
cost_calculator.completion_cost = completion_cost


def _defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    import warnings

    if modify_params:
        warnings.warn(
            "theagentrouter does not modify invalid messages; modify_params is unsupported.", UserWarning, stacklevel=3
        )
    return {
        **{
            name: value
            for name in ("api_base", "api_key", "num_retries", "request_timeout")
            if (value := globals()[name]) is not None
        },
        **kwargs,
    }


def completion(model: str, messages: list[Any], **kwargs: Any):
    """litellm.completion, calling the functions in success_callback, failure_callback and callbacks."""
    from . import _callbacks

    return _callbacks.run(_completion, model, messages, _defaults(kwargs), globals())


async def acompletion(model: str, messages: list[Any], **kwargs: Any):
    """litellm.acompletion, calling the callbacks; async loggers get their async_log_* methods."""
    from . import _callbacks

    return await _callbacks.arun(_acompletion, model, messages, _defaults(kwargs), globals())


def embedding(model: str, input: str | list[Any], **kwargs: Any):
    return _embedding(model=model, input=input, **_defaults(kwargs))


async def aembedding(model: str, input: str | list[Any], **kwargs: Any):
    return await _aembedding(model=model, input=input, **_defaults(kwargs))


def responses(model: str, input: Any, **kwargs: Any):
    return _responses(model=model, input=input, **_defaults(kwargs))


async def aresponses(model: str, input: Any, **kwargs: Any):
    return await _aresponses(model=model, input=input, **_defaults(kwargs))


def token_counter(*args: Any, **kwargs: Any) -> int:
    """LiteLLM's token_counter; returns 0 when disable_token_counter is set, as LiteLLM does."""
    if disable_token_counter:
        return 0
    return _token_counter(*args, **kwargs)


# LiteLLM's `litellm.types.utils` path, for frameworks that import types from there.
types = ModuleType("litellm.types")
types.utils = ModuleType("litellm.types.utils")
for _t in (ModelResponse, ModelResponseStream, EmbeddingResponse, Choices, Message, Delta, StreamingChoices, Usage):
    setattr(types.utils, _t.__name__, _t)
types.utils.ResponsesAPIResponse = ResponsesAPIResponse

# Request-side message shapes. In LiteLLM these are TypedDicts: calling one returns a plain dict.
OpenAIMessageContent = str | Iterable[dict]


class _AttrDict(dict):
    """A dict that also reads by attribute. LiteLLM's tool-call objects work both ways."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


class ChatCompletionToolCallFunctionChunk(_AttrDict):
    """`name` and `arguments`."""


class ChatCompletionMessageToolCall(_AttrDict):
    """`id`, `type`, `function`, and `index` when streamed. ADK reads `.type` and `["id"]` alike."""


class ChatCompletionDeltaToolCall(_AttrDict):
    """A tool call in a stream chunk: `index`, `id`, `type`, `function`."""


class CostPerToken(TypedDict, total=False):
    input_cost_per_token: float
    output_cost_per_token: float


for _t in (ChatCompletionMessageToolCall, ChatCompletionDeltaToolCall, CostPerToken):
    setattr(types.utils, _t.__name__, _t)


class ChatCompletionSystemMessage(TypedDict, total=False):
    role: str
    content: Any


class ChatCompletionUserMessage(TypedDict, total=False):
    role: str
    content: Any


class ChatCompletionAssistantMessage(TypedDict, total=False):
    role: str
    content: Any
    tool_calls: list
    reasoning_content: Any


class ChatCompletionToolMessage(TypedDict, total=False):
    role: str
    content: Any
    tool_call_id: str


class CustomStreamWrapper:
    """LiteLLM's stream type. Frameworks only name it in type hints; the mock streams generators."""


async def acreate_file(*args: Any, **kwargs: Any):
    raise exceptions.BadRequestError("file upload is not supported: Agent Router does not serve a files API")


# Providers this SDK reaches without a gateway; LiteLLM exposes the same idea as provider_list.
provider_list: list[str] = sorted(_PROVIDERS)


def get_supported_openai_params(model: str, custom_llm_provider: str | None = None, **_: Any) -> list[str]:
    from .. import anthropic_format, routing

    params = [
        "messages",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "stop",
        "seed",
        "stream",
        "stream_options",
        "response_format",
        "user",
    ]
    if supports_function_calling(model, custom_llm_provider):
        params += ["tools", "tool_choice", "parallel_tool_calls"]
    if supports_reasoning(model, custom_llm_provider):
        params += ["reasoning_effort"]
    provider = custom_llm_provider or routing.provider_for(model)
    if provider == "anthropic":  # Claude goes in Anthropic Messages to every gateway
        params = [
            p
            for p in params
            if p in anthropic_format.SUPPORTED_PARAMS
            or p
            in (
                "messages",
                "tools",
                "stream",
                "stream_options",
            )
        ]
    return params


def get_llm_provider(
    model: str,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    **_: Any,
) -> tuple[str, str, str | None, str | None]:
    """Same return shape as LiteLLM: (model name, provider, api key, api base)."""
    if custom_llm_provider:
        return model, custom_llm_provider, api_key, api_base
    from ..routing import with_provider

    if "/" not in model and model.startswith(("gemini-", "gemma-")):
        return model, "vertex_ai", api_key, api_base  # LiteLLM's answer; calls to it are refused with the reason
    provider, sep, name = with_provider(model).partition("/")
    if not sep:
        raise exceptions.BadRequestError(
            f"model {model!r} has no provider prefix and is not in the model list", model=model
        )
    return name, provider, api_key, api_base


def _route_name(group: str, models: list[str], fallbacks: list[list[str]], retry: Any, weights: Any = ()) -> str:
    """The route a Router group becomes: the same definition gets the same name in every program."""
    import hashlib
    import json
    from dataclasses import asdict

    parts = [group, models, fallbacks, asdict(retry)]
    if weights:  # only when set, so a group without weights keeps the name it had before weights existed
        parts.append([list(w) for w in weights])
    definition = json.dumps(parts, sort_keys=True)
    return f"router-{hashlib.sha256(definition.encode()).hexdigest()[:8]}:{group}"


# Settings a deployment's litellm_params can hold that the Router uses: the model, and weight, rpm or
# tpm for its share of the group's traffic.
_USED = ("model", "api_key", "weight", "rpm", "tpm")
# A setting the gateway reads from where it starts, so os.environ/<that variable> asks for nothing new.
_GATEWAY_VARIABLES = {"api_base": "AZURE_API_BASE", "api_version": "AZURE_API_VERSION"}
_WEIGHT_MAX = 1_000_000  # Gateway API's largest backend weight


def _weights(deployments: list[dict[str, Any]]) -> tuple[int, ...]:
    """A group's traffic shares, by LiteLLM's simple-shuffle rule: the first of weight, rpm and tpm that any
    deployment sets; a deployment without it gets none. Empty when the shares are equal."""
    for metric in ("weight", "rpm", "tpm"):
        values = []
        for d in deployments:
            value = (d.get("litellm_params") or {}).get(metric)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise TypeError(f"Deployment {metric} must be numeric")  # LiteLLM's error
            values.append(max(float(value or 0), 0.0))
        largest = max(values, default=0.0)
        if largest <= 0:
            continue
        if len(set(values)) == 1:
            return ()
        if all(v.is_integer() for v in values) and largest <= _WEIGHT_MAX:
            return tuple(int(v) for v in values)
        return _whole_weights(values)
    return ()


def _whole_weights(values: list[float]) -> tuple[int, ...]:
    """Whole-number weights in the same ratio, for the gateway, which takes 0 to 1,000,000.

    Exact when the ratio fits: 0.01 and 100 become 1 and 10000, 0.25 and 1 become 1 and 4. When it does
    not, the largest becomes 1,000,000 and the others are rounded to it, a positive one to at least 1.
    """
    from fractions import Fraction
    from math import lcm

    exact = [Fraction(str(v)) for v in values]  # 0.1 as 1/10, not as its nearest binary fraction
    scale = lcm(*(f.denominator for f in exact))
    whole = [int(f * scale) for f in exact]
    largest = max(whole)
    if largest <= _WEIGHT_MAX:
        return tuple(whole)
    return tuple(max(1, round(w * _WEIGHT_MAX / largest)) if w > 0 else 0 for w in whole)


def _remove_earlier(group: str, keep: str) -> None:
    """Remove the routes earlier definitions of a group left, so their providers' keys stop being needed."""
    import re

    from .. import routes

    earlier = re.compile(r"router-[0-9a-f]{8}:" + re.escape(group))
    for name in routes.list_routes():
        if name != keep and earlier.fullmatch(name):
            routes.remove_route(name)


class Router:
    """LiteLLM's Router: model groups with fallbacks and retries. The local gateway does the work.

        router = litellm.Router(
            model_list=[
                {"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o"}},
                {"model_name": "chat-backup", "litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
            ],
            fallbacks=[{"chat": ["chat-backup"]}],
            num_retries=2,
        )
        router.completion(model="chat", messages=[...])

    Each model_name becomes a route (theagentrouter.route) named after the group and its definition,
    such as ``router-1a2b3c4d:chat``, so two programs with a group of the same name keep separate
    routes, and a group named after a model does not take over that model's calls in other programs.
    A new definition of a group removes the routes earlier ones left; a Router still using one adds it
    back on its next call. ``export_config()`` names each of these routes after its group, ``chat``.
    With AGENTROUTER_BASE_URL set, nothing is set up here: calls send the group name to that gateway,
    and its route of that name applies.
    The Router's methods take the group name. Several deployments with one model_name share its traffic,
    by weight, rpm or tpm in litellm_params as LiteLLM's default simple-shuffle shares it. Keys come from
    the environment: litellm_params api_key values other than os.environ/<the provider's usual variable>
    are not used, and say so once. Other litellm_params, such as api_base, are not used either, and say
    so; so does a routing_strategy other than simple-shuffle.
    """

    def __init__(
        self,
        model_list: list[dict[str, Any]] | None = None,
        fallbacks: list[dict[str, list[str]]] | None = None,
        num_retries: int | None = None,
        **kwargs: Any,
    ) -> None:
        import warnings

        from .. import routes
        from ..local_gateway import PROVIDERS
        from ..routing import settings

        self.model_list = list(model_list or [])
        groups: dict[str, list[str]] = {}
        deployments: dict[str, list[dict[str, Any]]] = {}
        unused: dict[str, list[str]] = {}  # a litellm_params setting the gateway does not take -> its groups
        for d in self.model_list:
            params = d.get("litellm_params") or {}
            model = params.get("model")
            if not d.get("model_name") or not model:
                raise exceptions.BadRequestError("each model_list entry needs model_name and litellm_params.model")
            for setting, value in sorted(params.items()):
                if value is None or setting in _USED or value == f"os.environ/{_GATEWAY_VARIABLES.get(setting)}":
                    continue
                if d["model_name"] not in unused.setdefault(setting, []):
                    unused[setting].append(d["model_name"])
            key = params.get("api_key")
            provider = model.split("/", 1)[0]
            usual = (PROVIDERS.get(provider) or {}).get("keys", [None])[0]
            if key and key != f"os.environ/{usual}":
                warnings.warn(
                    f"theagentrouter's Router takes keys from the environment ({usual}); "
                    f"the api_key for {d['model_name']} is not used.",
                    UserWarning,
                    stacklevel=2,
                )
            groups.setdefault(d["model_name"], []).append(model)
            deployments.setdefault(d["model_name"], []).append(d)
        chains: dict[str, list[str]] = {}
        for item in fallbacks or []:
            for name, rest in item.items():
                chains.setdefault(name, []).extend(rest)
        for name in chains:
            missing = [n for n in [name, *chains[name]] if n not in groups]
            if missing:
                raise exceptions.BadRequestError(f"fallbacks name model groups not in model_list: {missing}")
        ignored = sorted(k for k, v in kwargs.items() if v and k not in ("routing_strategy", "set_verbose"))
        if ignored:
            warnings.warn(f"theagentrouter's Router ignores {', '.join(ignored)}.", UserWarning, stacklevel=2)
        strategy = kwargs.get("routing_strategy")
        strategy = getattr(strategy, "value", strategy)
        if strategy and strategy != "simple-shuffle":
            # LiteLLM's other strategies choose a deployment in the program; usage-based ones also hold
            # each deployment to its rpm and tpm. The gateway shares traffic as simple-shuffle does.
            warnings.warn(
                f"theagentrouter's Router ignores routing_strategy={strategy!r}: the gateway shares a group's "
                "traffic by weight, rpm or tpm, as with 'simple-shuffle', and does not hold deployments to rpm or "
                "tpm. For a request limit, define a route with limit=Limit(...).",
                UserWarning,
                stacklevel=2,
            )
        if unused:
            listed = "; ".join(f"{k} (for {', '.join(groups_)})" for k, groups_ in sorted(unused.items()))
            warnings.warn(
                f"theagentrouter's Router does not use these litellm_params: {listed}. "
                "The gateway takes provider addresses and settings from where it starts.",
                UserWarning,
                stacklevel=2,
            )
        # LiteLLM's Router falls back on nearly any error, including a wrong key or an unknown model.
        on = (401, 403, 404, 408, *routes.RETRY_ON)
        retry = routes.Retry(attempts=num_retries if num_retries is not None else 1, on=on)
        self.routes = {}
        gateway = settings().get("base_url")
        for name, models in groups.items():
            order = [name, *chains.get(name, [])]
            later = [groups[f] for f in order[1:]]
            shares = [_weights(deployments[g]) for g in order]
            weights = (
                tuple(w or (1,) * len(groups[g]) for w, g in zip(shares, order, strict=True)) if any(shares) else ()
            )
            tiers = tuple(tuple(groups[g]) for g in order)
            # The gateway you set has its own route by the group's name, as export_config() names it.
            route_name = name if gateway else _route_name(name, models, later, retry, weights)
            self.routes[name] = routes._define(routes.Route(route_name, tiers, retry, None, weights))
            if not gateway:
                _remove_earlier(name, keep=route_name)

    def _target(self, model: str) -> str:
        from dataclasses import replace

        from .. import routes
        from ..routing import settings

        r = self.routes.get(model)
        if r is None:
            return model
        # Decided at each call, not when the Router was made: AGENTROUTER_BASE_URL can be set or unset in between.
        if settings().get("base_url"):
            # The gateway you set has its own route by the group's name, as export_config() names it.
            if not routes.defined_for_gateway(model):
                routes._define(replace(r, name=model))
            return model
        local = _route_name(model, list(r.tiers[0]), [list(t) for t in r.tiers[1:]], r.retry, r.weights)
        if routes.get(local) is None:  # made with a gateway set, or a newer definition of this group removed it
            routes._define(replace(r, name=local))
        return local

    def completion(self, model: str, messages: list[Any], **kwargs: Any):
        return completion(model=self._target(model), messages=messages, **kwargs)

    async def acompletion(self, model: str, messages: list[Any], **kwargs: Any):
        return await acompletion(model=self._target(model), messages=messages, **kwargs)

    def embedding(self, model: str, input: str | list[Any], **kwargs: Any):
        return embedding(model=self._target(model), input=input, **kwargs)

    async def aembedding(self, model: str, input: str | list[Any], **kwargs: Any):
        return await aembedding(model=self._target(model), input=input, **kwargs)

    def get_model_list(self) -> list[dict[str, Any]]:
        import copy

        return copy.deepcopy(self.model_list)
