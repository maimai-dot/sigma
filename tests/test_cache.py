"""Tests for LLM response caching — CachedLLMBackend and AsyncCachedLLMBackend."""

import time
import pytest
from unittest import mock

from sigma.llm import UniversalBackend, AsyncUniversalBackend, LLMResponse
from sigma.cache import CacheConfig, CachedLLMBackend, AsyncCachedLLMBackend


# ── Sync Cache Tests ──────────────────────────────────────────────

class TestCacheConfig:
    def test_defaults(self):
        cc = CacheConfig()
        assert cc.enabled is False
        assert cc.ttl_seconds == 300
        assert cc.max_entries == 512

    def test_custom(self):
        cc = CacheConfig(enabled=True, ttl_seconds=60, max_entries=100)
        assert cc.enabled is True
        assert cc.ttl_seconds == 60
        assert cc.max_entries == 100


class TestCachedLLMBackend:
    @pytest.fixture
    def mock_backend(self):
        be = mock.Mock(spec=UniversalBackend)
        be.chat.return_value = LLMResponse(content="cached response", input_tokens=10, output_tokens=5)
        be.retry = mock.Mock()
        return be

    @pytest.fixture
    def cache(self, mock_backend):
        return CachedLLMBackend(mock_backend, CacheConfig(enabled=True, ttl_seconds=300, max_entries=100))

    def test_cache_miss_delegates_to_backend(self, cache, mock_backend):
        resp = cache.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "cached response"
        assert mock_backend.chat.call_count == 1

    def test_cache_hit_avoids_backend_call(self, cache, mock_backend):
        msgs = [{"role": "user", "content": "hi"}]
        r1 = cache.chat(messages=msgs, model="test")
        r2 = cache.chat(messages=msgs, model="test")
        assert r1 is r2  # same object returned
        assert mock_backend.chat.call_count == 1

    def test_different_messages_different_keys(self, cache, mock_backend):
        cache.chat(messages=[{"role": "user", "content": "A"}], model="test")
        cache.chat(messages=[{"role": "user", "content": "B"}], model="test")
        assert mock_backend.chat.call_count == 2

    def test_different_models_different_keys(self, cache, mock_backend):
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="m1")
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="m2")
        assert mock_backend.chat.call_count == 2

    def test_different_temperature_different_keys(self, cache, mock_backend):
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="t", temperature=0.0)
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="t", temperature=1.0)
        assert mock_backend.chat.call_count == 2

    def test_max_tokens_different_keys(self, cache, mock_backend):
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="t", max_tokens=100)
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="t", max_tokens=200)
        assert mock_backend.chat.call_count == 2

    def test_ttl_expiry(self, mock_backend):
        cc = CacheConfig(enabled=True, ttl_seconds=0, max_entries=100)  # immediate expiry
        cache = CachedLLMBackend(mock_backend, cc)
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert mock_backend.chat.call_count == 2  # both miss

    def test_lru_eviction(self, mock_backend):
        cc = CacheConfig(enabled=True, ttl_seconds=9999, max_entries=2)
        cache = CachedLLMBackend(mock_backend, cc)
        cache.chat(messages=[{"role": "user", "content": "A"}], model="t")
        cache.chat(messages=[{"role": "user", "content": "B"}], model="t")
        cache.chat(messages=[{"role": "user", "content": "C"}], model="t")
        # A should be evicted, B and C remain
        assert cache.stats["size"] == 2
        # Accessing A should be a miss (reaches backend)
        cache.chat(messages=[{"role": "user", "content": "A"}], model="t")
        assert mock_backend.chat.call_count == 4

    def test_cache_stats(self, cache, mock_backend):
        cache.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert cache.stats["size"] == 1
        assert cache.stats["ttl"] == 300

    def test_delegates_retry_attribute(self, cache, mock_backend):
        assert cache.retry is mock_backend.retry


# ── Async Cache Tests ─────────────────────────────────────────────

class TestAsyncCachedLLMBackend:
    @pytest.fixture
    def mock_async_backend(self):
        be = mock.AsyncMock(spec=AsyncUniversalBackend)
        be.chat.return_value = LLMResponse(content="async cached", input_tokens=10, output_tokens=5)
        be.retry = mock.Mock()
        return be

    @pytest.fixture
    def async_cache(self, mock_async_backend):
        return AsyncCachedLLMBackend(mock_async_backend, CacheConfig(enabled=True, ttl_seconds=300, max_entries=100))

    @pytest.mark.asyncio
    async def test_cache_miss_delegates(self, async_cache, mock_async_backend):
        resp = await async_cache.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "async cached"
        assert mock_async_backend.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_call(self, async_cache, mock_async_backend):
        msgs = [{"role": "user", "content": "hi"}]
        r1 = await async_cache.chat(messages=msgs, model="test")
        r2 = await async_cache.chat(messages=msgs, model="test")
        assert r1 is r2
        assert mock_async_backend.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys(self, async_cache, mock_async_backend):
        await async_cache.chat(messages=[{"role": "user", "content": "A"}], model="test")
        await async_cache.chat(messages=[{"role": "user", "content": "B"}], model="test")
        assert mock_async_backend.chat.call_count == 2
