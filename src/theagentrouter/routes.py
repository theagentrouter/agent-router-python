# SPDX-License-Identifier: Apache-2.0
"""Routes: a model name of your own, served by the local gateway with fallbacks and retries.

    import theagentrouter as ar
    ar.route("chat", primary="openai/gpt-4o", fallbacks=["anthropic/claude-sonnet-4-5"], retry=ar.Retry(attempts=2))
    ar.completion(model="chat", messages=[{"role": "user", "content": "hi"}])

The local gateway (Agent Router's ``aigw``) does the retries and the fallbacks: a route becomes one
gateway route with its models at priority 0, 1, ..., as Agent Router Enterprise does it. Routes are
kept in ``AGENTROUTER_HOME/routes.json`` because the local gateway is shared by every program on
the computer, so its routes are too. Defining a route again replaces it, and the
gateway restarts with the change on the next call that needs it.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation

from . import _files, exceptions

RETRY_ON = (429, 500, 502, 503, 504)
LIMIT_PER = ("second", "minute", "hour", "day")


def _milliseconds(value: float, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number of seconds")
    try:
        milliseconds = Decimal(str(value)) * 1000
        if not milliseconds.is_finite() or milliseconds <= 0 or milliseconds != milliseconds.to_integral_value():
            raise ValueError(f"{name} must be positive, finite and expressible in whole milliseconds")
        if milliseconds >= 100000 * 3600000:
            raise ValueError(f"{name} exceeds the gateway's maximum duration")
        return int(milliseconds)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a finite duration") from error


def _duration(value: float) -> str:
    remaining = _milliseconds(value, "duration")
    parts = []
    for unit, size in (("h", 3600000), ("m", 60000), ("s", 1000), ("ms", 1)):
        count, remaining = divmod(remaining, size)
        if count:
            parts.append(f"{count}{unit}")
    return "".join(parts)


@dataclass(frozen=True)
class Retry:
    """How the local gateway retries a route.

    attempts: retries in each tier of the route before the gateway moves to the next tier.
    on: HTTP status codes that count as a failure worth retrying. Connection failures always count.
    timeout: seconds each try may take, or None for no limit of its own.
    backoff: seconds before the first retry; it doubles up to the larger of 10 seconds and backoff.
    """

    attempts: int = 1
    on: tuple[int, ...] = RETRY_ON
    timeout: float | None = None
    backoff: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or not 0 <= self.attempts < 2**32 - 1:
            raise ValueError("Retry.attempts must be a non-negative integer below 4294967295")
        try:
            codes = tuple(self.on)
        except TypeError as error:
            raise ValueError("Retry.on must contain HTTP status codes from 100 to 599") from error
        if any(isinstance(c, bool) or not isinstance(c, int) or not 100 <= c <= 599 for c in codes):
            raise ValueError("Retry.on must contain integer HTTP status codes from 100 to 599")
        object.__setattr__(self, "on", codes)
        if self.timeout is not None:
            _milliseconds(self.timeout, "Retry.timeout")
        _milliseconds(self.backoff, "Retry.backoff")


@dataclass(frozen=True)
class Limit:
    """A cap on how many requests a route takes, counted by the local gateway.

    requests: how many requests in each period; the gateway answers the rest with 429 (RateLimitError).
    per: "second", "minute", "hour" or "day".

    The count covers the whole route, for every program on the computer that uses it. It counts
    requests, not tokens: Agent Router counts tokens only with a Redis-backed rate-limit service,
    which the local gateway (aigw run) does not have.
    """

    requests: int
    per: str = "minute"

    def __post_init__(self) -> None:
        if isinstance(self.requests, bool) or not isinstance(self.requests, int) or not 1 <= self.requests < 2**32:
            raise ValueError("Limit.requests must be a whole number from 1 to 4294967295")
        if self.per not in LIMIT_PER:
            raise ValueError(f"Limit.per must be one of {', '.join(LIMIT_PER)}")


@dataclass(frozen=True)
class Route:
    """A name of your own, and the models behind it in order: tiers[0] first, then each fallback.

    Models in one tier share the traffic between them; the next tier is used when they fail.
    """

    name: str
    tiers: tuple[tuple[str, ...], ...]
    retry: Retry = field(default_factory=Retry)
    limit: Limit | None = None

    @property
    def primary(self) -> str:
        return self.tiers[0][0]

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(m for tier in self.tiers for m in tier)


def _path():
    from .routing import home

    return home() / "routes.json"


def _load() -> dict[str, Route]:
    raw = _files.read_json(_path())
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, r in raw.items():
        retry = r.get("retry") or {}
        limit = r.get("limit")
        try:
            out[name] = Route(
                name,
                tuple(tuple(t) for t in r["tiers"]),
                Retry(**{**retry, "on": tuple(retry.get("on", RETRY_ON))}),
                Limit(**limit) if limit else None,
            )
        except (ValueError, TypeError, KeyError):
            warnings.warn(
                f"saved route {name!r} is invalid and disabled; redefine it or remove its entry from routes.json",
                UserWarning,
                stacklevel=2,
            )
    return out


def _dump(routes: dict[str, Route]) -> dict:
    return {
        n: {
            "tiers": [list(t) for t in r.tiers],
            "retry": asdict(r.retry),
            "limit": asdict(r.limit) if r.limit else None,
        }
        for n, r in sorted(routes.items())
    }


def _save(routes: dict[str, Route]) -> None:
    """Write the whole file at once: a reader sees the old routes or the new ones, never half a file."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".routes-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(_dump(routes), indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def _editing() -> Iterator[None]:
    """One program at a time reads, changes and writes the routes, so two definitions both stay."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / "routes.lock", "a+") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # Windows: no lock yet, as for the gateway's start lock
            pass
        yield


def list_routes() -> dict[str, Route]:
    """Every route on this computer, by name."""
    return _load()


def get(name: str) -> Route | None:
    return _load().get(name)


def _tier(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    models = (value,) if isinstance(value, str) else tuple(value)
    if not models:
        raise exceptions.BadRequestError("a route needs at least one model in each tier")
    return models


def _check(name: str, tiers: tuple[tuple[str, ...], ...]) -> None:
    """Fail now, with the fix in the message, rather than on the first call."""
    from . import local_gateway
    from .routing import settings

    if not name or name != name.strip():
        raise exceptions.BadRequestError(f"route name {name!r} must be a non-empty name without spaces at the ends")
    gateway = settings().get("base_url")
    if gateway:
        raise exceptions.BadRequestError(
            f"routes are served by the local gateway, and AGENTROUTER_BASE_URL sends calls to {gateway}. "
            "Set up fallbacks in that gateway instead, or unset AGENTROUTER_BASE_URL."
        )
    for model in (m for tier in tiers for m in tier):
        provider, sep, own = model.partition("/")
        if not sep or not own:
            raise exceptions.BadRequestError(
                f"{model!r} in route {name!r} needs a provider prefix, for example openai/gpt-4o-mini", model=model
            )
        if provider not in local_gateway.PROVIDERS:
            raise exceptions.BadRequestError(
                f"the local gateway does not set up {provider!r} yet, so {model!r} cannot be in a route", model=model
            )


def route(
    name: str,
    primary: str | list[str],
    fallbacks: list[str | list[str]] | None = None,
    retry: Retry | None = None,
    limit: Limit | None = None,
) -> Route:
    """Define (or replace) a route: call it with completion(model=name, ...).

    primary: a provider/model name, or a list of them that share the traffic.
    fallbacks: tried in order when the primary fails; each item a model or a list of them.
    retry: how many times each model is retried first, and on which errors. Default: Retry().
    limit: at most this many requests to the route in a period, for example Limit(60, per="minute").
    """
    tiers = (_tier(primary), *(_tier(f) for f in fallbacks or []))
    _check(name, tiers)
    new = Route(name, tiers, retry or Retry(), limit)
    if (new.retry.attempts + 1) * len(tiers) - 1 >= 2**32:
        raise ValueError("the total retries across route tiers must fit an unsigned 32-bit integer")
    with _editing():
        routes = _load()
        old = routes.get(name)
        if old != new:
            routes[name] = new
            _save(routes)
    if old is not None and old != new:
        print(f"theagentrouter: replaced route {name!r}.", file=sys.stderr)
    return new


def remove_route(name: str) -> bool:
    """Remove a route. Returns False if there was none by that name."""
    with _editing():
        routes = _load()
        if routes.pop(name, None) is None:
            return False
        _save(routes)
    return True


def fingerprint(routes: dict[str, Route]) -> str:
    """Changes whenever any route changes; the running gateway is restarted when it differs."""
    import hashlib

    return hashlib.sha256(json.dumps(_dump(routes), sort_keys=True).encode()).hexdigest()[:16]
