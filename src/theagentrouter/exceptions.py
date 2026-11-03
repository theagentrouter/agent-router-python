# SPDX-License-Identifier: Apache-2.0
"""Errors with LiteLLM's class names, so code that catches them by name keeps working."""

from __future__ import annotations

import json
from typing import Any


class APIError(Exception):
    status_code: int | None = 500

    def __init__(
        self,
        *args: Any,
        message: str | None = None,
        model: str | None = None,
        llm_provider: str | None = None,
        status_code: int | None = None,
        response: Any = None,
        body: Any = None,
        **kwargs: Any,
    ):
        provider_first = {
            "AuthenticationError",
            "RateLimitError",
            "PermissionDeniedError",
            "InternalServerError",
            "BadGatewayError",
            "ServiceUnavailableError",
            "APIConnectionError",
            "APIResponseValidationError",
            "UnsupportedParamsError",
        }
        order = (
            ["message", "llm_provider", "model"]
            if type(self).__name__ in provider_first
            else ["message", "model", "llm_provider"]
        )
        if type(self).__name__ == "APIError" and args and isinstance(args[0], int):
            order = ["status_code", "message", "llm_provider", "model", "request"]
        elif type(self).__name__ == "UnsupportedParamsError":
            order += ["status_code", "response"]
        elif type(self).__name__ == "APIConnectionError":
            order += ["request"]
        elif type(self).__name__ in ("Timeout", "APIResponseValidationError"):
            order += ["litellm_debug_info"]
        else:
            order += ["response"]
        if order[-1] != "litellm_debug_info":
            order.append("litellm_debug_info")
        order += ["max_retries", "num_retries", "body"]
        if len(args) > len(order):
            raise TypeError("too many positional exception arguments")
        values = {
            "message": message,
            "model": model,
            "llm_provider": llm_provider,
            "status_code": status_code,
            "response": response,
            "body": body,
            **kwargs,
        }
        values.update(zip(order, args, strict=False))
        message = str(values.pop("message") or "")
        super().__init__(message)
        self.message = message
        for name, value in values.items():
            if name != "status_code" or value is not None:
                setattr(self, name, value)
        self.headers = getattr(self, "headers", None) or (self.response.headers if self.response is not None else {})

    def __str__(self) -> str:
        return self.message


class BadRequestError(APIError):
    status_code = 400


class UnsupportedParamsError(BadRequestError):
    pass


class ContextWindowExceededError(BadRequestError):
    pass


class ContentPolicyViolationError(BadRequestError):
    pass


class AuthenticationError(APIError):
    status_code = 401


class PermissionDeniedError(APIError):
    status_code = 403


class NotFoundError(APIError):
    status_code = 404


class Timeout(APIError):
    status_code = 408


class UnprocessableEntityError(APIError):
    status_code = 422


class RateLimitError(APIError):
    status_code = 429


class BudgetExceededError(RateLimitError):
    """A gateway refused the call because a budget is used up."""

    def __init__(
        self,
        current_cost=None,
        max_budget=None,
        message=None,
        llm_provider=None,
        entity_type=None,
        entity_id=None,
        **kwargs,
    ):
        if isinstance(current_cost, str) and message is None:
            message, current_cost = current_cost, None
        super().__init__(
            message=message or f"Budget exceeded: current cost {current_cost}, maximum budget {max_budget}",
            llm_provider=llm_provider,
            **kwargs,
        )
        self.current_cost = current_cost
        self.max_budget = max_budget
        self.entity_type = entity_type
        self.entity_id = entity_id


class InternalServerError(APIError):
    status_code = 500


class BadGatewayError(APIError):
    status_code = 502


class ServiceUnavailableError(APIError):
    status_code = 503


class APIConnectionError(APIError):
    status_code = None  # no response arrived


class APIResponseValidationError(APIError):
    status_code = None


_BY_STATUS: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: Timeout,
    413: BadRequestError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
    504: Timeout,
    529: ServiceUnavailableError,  # Anthropic "overloaded"
}

_CONTEXT_HINTS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "context window",
    "input tokens exceed the configured limit",  # OpenAI's newer models; on LiteLLM's list too
)
_POLICY_HINTS = ("content policy", "content_policy", "content_filter", "safety system")
# A 429 is a budget block only when it says so: Agent Router sends code "budget_exceeded", and a
# provider's "token budget per minute exceeded" is an ordinary rate limit that is worth retrying.
_BUDGET_CODES = ("budget_exceeded",)
_BUDGET_HINTS = ("budget exceeded", "budget has been exceeded")


def _code_of(body: Any) -> str:
    err = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return ""
    return f"{err.get('code') or ''} {err.get('type') or ''}".lower()


def _message_of(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or fallback)
        if isinstance(err, str):
            return err
    return fallback


def from_http(status: int, text: str, model: str | None, provider: str | None) -> APIError:
    """Map a provider's or gateway's HTTP error to the matching LiteLLM error class."""
    try:
        body: Any = json.loads(text)
    except ValueError:
        body = None
    message = _message_of(body, text.strip()[:500] or f"HTTP {status}")
    lower = message.lower()
    cls = _BY_STATUS.get(status, InternalServerError if status >= 500 else APIError)
    if cls is BadRequestError and any(h in lower for h in _CONTEXT_HINTS):
        cls = ContextWindowExceededError
    elif cls is BadRequestError and any(h in lower for h in _POLICY_HINTS):
        cls = ContentPolicyViolationError
    elif cls is RateLimitError and (
        any(c in _code_of(body).split() for c in _BUDGET_CODES) or any(h in lower for h in _BUDGET_HINTS)
    ):
        cls = BudgetExceededError
    prefix = f"{provider}: " if provider else ""
    return cls(prefix + message, model=model, llm_provider=provider, status_code=status, body=body)


__all__ = [name for name, obj in list(globals().items()) if isinstance(obj, type) and issubclass(obj, Exception)]
