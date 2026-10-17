"""Small JSON files the SDK reads on every call: settings, routes and the local gateway's state.

Each is read again only when it changes (its size, modification time or inode), so a call through
the local gateway costs a stat per file rather than an open and a parse.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

_cache: dict[str, tuple[tuple[int, int, int], Any]] = {}


def read_json(path: Path) -> Any:
    """The file's JSON value, or None when it is missing or not JSON. Callers get their own copy."""
    try:
        st = os.stat(path)
    except OSError:
        _cache.pop(str(path), None)
        return None
    key = (st.st_mtime_ns, st.st_size, st.st_ino)
    hit = _cache.get(str(path))
    if hit is None or hit[0] != key:
        try:
            hit = (key, json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            return None
        _cache[str(path)] = hit
    return copy.deepcopy(hit[1])
