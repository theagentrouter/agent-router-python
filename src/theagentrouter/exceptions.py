"""Errors with LiteLLM's class names, so code that catches them by name keeps working."""

from __future__ import annotations

import json
from typing import Any


class APIError(Exception):
    status_code: int | None = 500

    def __init__(
        self,
        message: str,
        model: str | None = None,
        llm_provider: str | None = None,
        status_code: int | None = None,
        response: Any = None,
        body: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        self.response = response
        self.body = body
        if status_code is not None:
            self.status_code = status_code

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
)
_POLICY_HINTS = ("content policy", "content_policy", "content_filter", "safety system")
_BUDGET_HINTS = ("budget",)


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
    elif cls is RateLimitError and any(h in lower for h in _BUDGET_HINTS):
        cls = BudgetExceededError
    prefix = f"{provider}: " if provider else ""
    return cls(prefix + message, model=model, llm_provider=provider, status_code=status, body=body)


__all__ = [name for name, obj in list(globals().items()) if isinstance(obj, type) and issubclass(obj, Exception)]
