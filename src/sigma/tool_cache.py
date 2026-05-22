"""Tool result caching — avoid redundant tool calls with LRU cache.

Keys are (tool_name, params_json) tuples. Values are the tool result dict.
Integrates with existing CacheConfig for LRU eviction.

Usage:
    cache = ToolCache(max_size=256, ttl_seconds=300)
    result = cache.get_or_run("freecad_mass_extractor", {"path": "model.FCStd"}, runner)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from sigma.log import get_logger

_log = get_logger("sigma.tool_cache")


@dataclass
class ToolCacheEntry:
    result: dict
    timestamp: float = field(default_factory=time.time)


class ToolCache:
    """LRU cache for tool call results.

    Features:
      - LRU eviction when max_size reached
      - Optional TTL (time-to-live) per entry
      - Cache key: (tool_name, sorted JSON of params)
      - Thread-safe for sync usage
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 0):
        """
        Args:
            max_size: Max cache entries. Oldest LRU entry evicted when exceeded.
            ttl_seconds: Entry lifetime in seconds. 0 = no expiry.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, ToolCacheEntry] = OrderedDict()

    def _make_key(self, tool_name: str, params: dict) -> str:
        """Build a deterministic cache key from tool name and params."""
        params_str = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        # Hash for compact keys
        h = hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()[:16]
        return f"{tool_name}:{h}"

    def get(self, tool_name: str, params: dict) -> dict | None:
        """Get cached tool result, or None if not found or expired."""
        key = self._make_key(tool_name, params)
        entry = self._store.get(key)
        if entry is None:
            return None
        if self.ttl_seconds > 0 and (time.time() - entry.timestamp) > self.ttl_seconds:
            del self._store[key]
            return None
        # LRU: move to end (most recently used)
        self._store.move_to_end(key)
        return entry.result

    def put(self, tool_name: str, params: dict, result: dict) -> None:
        """Store a tool result in the cache."""
        key = self._make_key(tool_name, params)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = ToolCacheEntry(result=result)
            return
        if len(self._store) >= self.max_size:
            # Evict oldest (first in OrderedDict)
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = ToolCacheEntry(result=result)

    def get_or_run(
        self,
        tool_name: str,
        params: dict,
        runner: Callable[[], dict],
    ) -> dict:
        """Get cached result or run the tool and cache the result."""
        cached = self.get(tool_name, params)
        if cached is not None:
            _log.debug("Cache hit: %s", tool_name)
            return cached
        result = runner()
        self.put(tool_name, params, result)
        return result

    async def aget_or_run(
        self,
        tool_name: str,
        params: dict,
        runner: Callable[[], Any],
    ) -> dict:
        """Async variant: get cached or run and cache."""
        import asyncio
        cached = self.get(tool_name, params)
        if cached is not None:
            return cached
        coro = runner()
        if asyncio.iscoroutine(coro):
            result = await coro
        else:
            result = coro
        self.put(tool_name, params, result)
        return result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Not tracked in-memory; returns 0. Use external monitoring."""
        return 0.0
