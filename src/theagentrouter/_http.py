"""Reuse HTTP connections, with loopback requests kept out of environment proxies."""

from __future__ import annotations

import asyncio
import atexit
import ipaddress
import os
import threading
import weakref
from urllib.parse import urlsplit

import httpx

_sync: dict[tuple[int, bool], httpx.Client] = {}
_async: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def _trust_env(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def client(url: str) -> httpx.Client:
    key = (os.getpid(), _trust_env(url))
    with _lock:
        if key not in _sync or _sync[key].is_closed:
            _sync[key] = httpx.Client(trust_env=key[1])
        return _sync[key]


def async_client(url: str) -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    clients = _async.setdefault(loop, {})
    trust = _trust_env(url)
    if trust not in clients or clients[trust].is_closed:
        clients[trust] = httpx.AsyncClient(trust_env=trust)
    return clients[trust]


def close() -> None:
    with _lock:
        for value in _sync.values():
            value.close()
        _sync.clear()


async def aclose() -> None:
    for value in _async.pop(asyncio.get_running_loop(), {}).values():
        await value.aclose()
    close()


atexit.register(close)
