"""Tests for sigma.tool_cache — LRU tool result caching."""

import time
import pytest
from sigma.tool_cache import ToolCache, ToolCacheEntry


class TestToolCacheEntry:
    def test_defaults(self):
        entry = ToolCacheEntry(result={"ok": True})
        assert entry.result == {"ok": True}
        assert entry.timestamp > 0

    def test_custom_timestamp(self):
        entry = ToolCacheEntry(result={}, timestamp=100.0)
        assert entry.timestamp == 100.0


class TestToolCache:
    def test_put_and_get(self):
        cache = ToolCache()
        cache.put("my_tool", {"x": 1}, {"result": "ok"})
        result = cache.get("my_tool", {"x": 1})
        assert result == {"result": "ok"}

    def test_miss(self):
        cache = ToolCache()
        assert cache.get("nonexistent", {}) is None

    def test_different_params_different_key(self):
        cache = ToolCache()
        cache.put("tool", {"a": 1}, {"v": 1})
        cache.put("tool", {"a": 2}, {"v": 2})
        assert cache.get("tool", {"a": 1}) == {"v": 1}
        assert cache.get("tool", {"a": 2}) == {"v": 2}

    def test_same_params_same_key(self):
        cache = ToolCache()
        cache.put("tool", {"a": 1, "b": 2}, {"v": "first"})
        # Order shouldn't matter (sorted keys)
        cache.put("tool", {"b": 2, "a": 1}, {"v": "second"})
        result = cache.get("tool", {"a": 1, "b": 2})
        assert result == {"v": "second"}

    def test_lru_eviction(self):
        cache = ToolCache(max_size=3)
        cache.put("a", {"i": 1}, {"v": 1})
        cache.put("b", {"i": 2}, {"v": 2})
        cache.put("c", {"i": 3}, {"v": 3})
        # Cache is full. Adding a 4th should evict "a" (oldest)
        cache.put("d", {"i": 4}, {"v": 4})
        assert cache.get("a", {"i": 1}) is None
        assert cache.get("d", {"i": 4}) == {"v": 4}

    def test_lru_access_refreshes(self):
        cache = ToolCache(max_size=3)
        cache.put("a", {"i": 1}, {"v": 1})
        cache.put("b", {"i": 2}, {"v": 2})
        cache.put("c", {"i": 3}, {"v": 3})
        # Access "a" to make it most recently used
        cache.get("a", {"i": 1})
        # Now "b" is the oldest
        cache.put("d", {"i": 4}, {"v": 4})
        assert cache.get("b", {"i": 2}) is None
        assert cache.get("a", {"i": 1}) == {"v": 1}

    def test_ttl_expiry(self):
        cache = ToolCache(ttl_seconds=0.01)
        cache.put("tool", {}, {"v": 1})
        time.sleep(0.02)
        assert cache.get("tool", {}) is None

    def test_ttl_not_expired(self):
        cache = ToolCache(ttl_seconds=60)
        cache.put("tool", {}, {"v": 1})
        assert cache.get("tool", {}) == {"v": 1}

    def test_get_or_run_hit(self):
        cache = ToolCache()
        cache.put("tool", {}, {"cached": True})
        call_count = 0

        def runner():
            nonlocal call_count
            call_count += 1
            return {"cached": False}

        result = cache.get_or_run("tool", {}, runner)
        assert result == {"cached": True}
        assert call_count == 0

    def test_get_or_run_miss(self):
        cache = ToolCache()

        def runner():
            return {"fresh": True}

        result = cache.get_or_run("tool", {"x": 1}, runner)
        assert result == {"fresh": True}
        assert cache.get("tool", {"x": 1}) == {"fresh": True}

    def test_clear(self):
        cache = ToolCache()
        cache.put("a", {}, {"v": 1})
        cache.put("b", {}, {"v": 2})
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a", {}) is None

    def test_len(self):
        cache = ToolCache()
        assert len(cache) == 0
        cache.put("a", {}, {"v": 1})
        assert len(cache) == 1

    def test_hit_rate(self):
        cache = ToolCache()
        assert cache.hit_rate == 0.0

    def test_make_key_deterministic(self):
        cache = ToolCache()
        k1 = cache._make_key("tool", {"a": 1, "b": 2})
        k2 = cache._make_key("tool", {"b": 2, "a": 1})
        assert k1 == k2

    def test_make_key_different_tools(self):
        cache = ToolCache()
        k1 = cache._make_key("tool_a", {"x": 1})
        k2 = cache._make_key("tool_b", {"x": 1})
        assert k1 != k2


class TestAsyncGetOrRun:
    @pytest.mark.asyncio
    async def test_async_get_or_run_hit(self):
        cache = ToolCache()
        cache.put("tool", {}, {"cached": True})

        async def runner():
            return {"fresh": False}

        result = await cache.aget_or_run("tool", {}, runner)
        assert result == {"cached": True}

    @pytest.mark.asyncio
    async def test_async_get_or_run_miss_async(self):
        cache = ToolCache()

        async def runner():
            return {"async": True}

        result = await cache.aget_or_run("tool", {}, runner)
        assert result == {"async": True}

    @pytest.mark.asyncio
    async def test_async_get_or_run_miss_sync_runner(self):
        cache = ToolCache()
        # Sync runner that returns a coroutine is handled
        result = await cache.aget_or_run("tool", {}, lambda: {"sync": True})
        assert result == {"sync": True}
