"""Embeddings: LiteLLM's embedding() and aembedding(), sent in the OpenAI format.

The call goes where a completion for the same model would go (see routing): to the local gateway,
a gateway you set, or an api_base you name. Anthropic has no embeddings API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import exceptions, models, routing
from .transport import _raise_for, _timeout, _warn_ignored, _wrap, served_by
from .types import Usage, _DictAccess

# LiteLLM's embedding arguments that are passed to the provider as they are.
_PASSED = ("dimensions", "encoding_format", "user")


class Embedding(dict):
    """One vector. Reads like LiteLLM's: `item["embedding"]` and `item.embedding` both work."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


@dataclass
class EmbeddingResponse(_DictAccess):
    model: str
    data: list[Embedding]
    usage: Usage = field(default_factory=Usage)
    object: str = "list"
    _hidden_params: dict[str, Any] = field(default_factory=dict)


def _prepare(model: str, input: str | list[Any], params: dict[str, Any]) -> tuple[routing.Target, dict, dict, str]:
    _warn_ignored(params)
    if model.startswith("anthropic/"):
        raise exceptions.BadRequestError(
            "anthropic has no embeddings API; use an embedding model of another provider, "
            "for example openai/text-embedding-3-small",
            model=model,
            llm_provider="anthropic",
        )
    target = routing.resolve(
        model,
        api_base=params.get("api_base") or params.get("base_url"),
        api_key=params.get("api_key"),
        extra_headers=params.get("extra_headers"),
    )
    body: dict[str, Any] = {"model": target.model, "input": input}
    body.update({k: params[k] for k in _PASSED if params.get(k) is not None})
    body.update(params.get("extra_body") or {})
    headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
    headers.update({"content-type": "application/json", **target.headers})
    url = target.base_url.rstrip("/") + "/embeddings"
    return target, body, headers, url


def _parse(target: routing.Target, data: dict[str, Any], url: str) -> EmbeddingResponse:
    raw = data.get("usage") or {}
    prompt = int(raw.get("prompt_tokens") or 0)
    usage = Usage(prompt_tokens=prompt, completion_tokens=0, total_tokens=int(raw.get("total_tokens") or prompt))
    items = [
        Embedding(object=d.get("object", "embedding"), index=d.get("index", i), embedding=d.get("embedding"))
        for i, d in enumerate(data.get("data") or [])
    ]
    gateway = target.route in ("gateway", "local gateway")
    key = target.model if gateway else f"{target.provider}/{target.model}"
    resp = EmbeddingResponse(model=data.get("model") or target.model, data=items, usage=usage)
    resp._hidden_params = {
        "route": target.route,
        "sent_to": url,
        "llm_provider": target.provider,
        "model_sent": target.model,
        "model_key": key,
    }
    if target.members:  # a route: priced as the model that answered
        resp._hidden_params["route_name"] = target.model
        served = served_by(resp.model, target.members)
        if served:
            resp._hidden_params["served_by"] = served
            resp._hidden_params["model_key"] = key = served
    resp._hidden_params["response_cost"] = models.completion_cost(model=key, prompt_tokens=prompt)
    return resp


def embedding(model: str, input: str | list[Any], **params: Any) -> EmbeddingResponse:
    target, body, headers, url = _prepare(model, input, params)
    try:
        with httpx.Client(timeout=_timeout(params)) as client:
            r = client.post(url, json=body, headers=headers)
    except httpx.HTTPError as e:
        raise _wrap(e, target, model) from e
    _raise_for(r, r.text, target, model)
    try:
        return _parse(target, r.json(), url)
    except ValueError as e:
        raise exceptions.APIResponseValidationError(
            f"{target.provider}: reply is not JSON", model=model, llm_provider=target.provider
        ) from e


async def aembedding(model: str, input: str | list[Any], **params: Any) -> EmbeddingResponse:
    # In a worker thread: preparing can start the local gateway, which must not block the event loop.
    target, body, headers, url = await asyncio.to_thread(_prepare, model, input, params)
    try:
        async with httpx.AsyncClient(timeout=_timeout(params)) as client:
            r = await client.post(url, json=body, headers=headers)
    except httpx.HTTPError as e:
        raise _wrap(e, target, model) from e
    _raise_for(r, r.text, target, model)
    try:
        return _parse(target, r.json(), url)
    except ValueError as e:
        raise exceptions.APIResponseValidationError(
            f"{target.provider}: reply is not JSON", model=model, llm_provider=target.provider
        ) from e
