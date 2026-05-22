"""LLM response caching with LRU eviction and TTL expiry.

Wraps any LLMBackend or AsyncLLMBackend transparently.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from sigma.llm import LLMBackend, AsyncLLMBackend, LLMResponse


@dataclass
class CacheConfig:
    """Configuration for the LLM response cache."""
    enabled: bool = False
    ttl_seconds: int = 300
    max_entries: int = 512


class CachedLLMBackend:
    """Transparent caching wrapper for sync LLMBackend."""

    def __init__(self, backend: LLMBackend, config: CacheConfig):
        self._backend = backend
        self._config = config
        self._cache: OrderedDict[str, tuple[LLMResponse, float]] = OrderedDict()

    @property
    def retry(self):
        return self._backend.retry

    @property
    def client(self):
        return self._backend.client

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        key = self._cache_key(model, messages, max_tokens, temperature, tools)
        self._evict_expired()

        if key in self._cache:
            resp, _ = self._cache[key]
            self._cache.move_to_end(key)
            return resp

        resp = self._backend.chat(messages, model, max_tokens, temperature, tools)
        self._set(key, resp)
        return resp

    def _cache_key(self, model: str, messages: list, max_tokens: int, temperature: float, tools: list[dict] | None = None) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "tools": tools},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _set(self, key: str, resp: LLMResponse) -> None:
        if len(self._cache) >= self._config.max_entries:
            self._cache.popitem(last=False)  # LRU eviction
        self._cache[key] = (resp, time.monotonic())

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._config.ttl_seconds]
        for k in expired:
            del self._cache[k]

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "max": self._config.max_entries, "ttl": self._config.ttl_seconds}


class AsyncCachedLLMBackend:
    """Transparent caching wrapper for async AsyncLLMBackend."""

    def __init__(self, backend: AsyncLLMBackend, config: CacheConfig):
        self._backend = backend
        self._config = config
        self._cache: OrderedDict[str, tuple[LLMResponse, float]] = OrderedDict()

    @property
    def retry(self):
        return self._backend.retry

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        key = _cache_key(model, messages, max_tokens, temperature, tools)
        _evict_expired(self._cache, self._config.ttl_seconds)

        if key in self._cache:
            resp, _ = self._cache[key]
            self._cache.move_to_end(key)
            return resp

        resp = await self._backend.chat(messages, model, max_tokens, temperature, tools)
        _set(self._cache, key, resp, self._config.max_entries)
        return resp

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "max": self._config.max_entries, "ttl": self._config.ttl_seconds}


def _cache_key(model: str, messages: list, max_tokens: int, temperature: float, tools: list[dict] | None = None) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "tools": tools},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _set(cache: OrderedDict, key: str, resp: LLMResponse, max_entries: int) -> None:
    if len(cache) >= max_entries:
        cache.popitem(last=False)
    cache[key] = (resp, time.monotonic())


def _evict_expired(cache: OrderedDict, ttl_seconds: int) -> None:
    now = time.monotonic()
    expired = [k for k, (_, ts) in cache.items() if now - ts > ttl_seconds]
    for k in expired:
        del cache[k]
