"""Tests for LLM backend — UniversalBackend init, lazy client, chat, retry logic."""

import os
import time
import pytest
from unittest import mock
from sigma.llm import (
    UniversalBackend, AsyncUniversalBackend, LLMResponse, LLMBackend, RetryConfig,
    RateLimitError, APITimeoutError,
    AuthenticationError,
)


API_KEY = os.environ.get("SIGMA_TEST_API_KEY", "sk-test")


class TestLLMResponse:
    """LLMResponse dataclass."""

    def test_creation(self):
        resp = LLMResponse(content="hello", input_tokens=10, output_tokens=5)
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.retries == 0

    def test_defaults(self):
        resp = LLMResponse(content="")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.retries == 0

    def test_retries_field(self):
        resp = LLMResponse(content="ok", retries=2)
        assert resp.retries == 2


class TestRetryConfig:
    """RetryConfig defaults and overrides."""

    def test_defaults(self):
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.delay == 1.0
        assert rc.backoff == 2.0

    def test_custom(self):
        rc = RetryConfig(max_retries=5, delay=0.5, backoff=3.0)
        assert rc.max_retries == 5
        assert rc.delay == 0.5
        assert rc.backoff == 3.0


class TestLLMBackendProtocol:
    """LLMBackend is a runtime-checkable Protocol."""

    def test_openai_backend_is_llm_backend(self):
        be = UniversalBackend(api_key="test", base_url="https://test.com")
        assert isinstance(be, LLMBackend)


class TestUniversalBackendInit:
    """UniversalBackend initialization and lazy client."""

    def test_explicit_api_key(self):
        be = UniversalBackend(api_key="sk-test", base_url="https://test.com")
        assert be._api_key == "sk-test"
        assert be._base_url == "https://test.com"

    def test_env_api_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        be = UniversalBackend()
        assert be._api_key == "sk-env"

    def test_openai_env_fallback(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("SIGMA_API_KEY", "sk-openai")
        be = UniversalBackend()
        assert be._api_key == "sk-openai"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("SIGMA_BASE_URL", raising=False)
        be = UniversalBackend(api_key="sk-test")
        assert be._base_url == "https://api.deepseek.com"

    def test_client_is_lazy(self):
        be = UniversalBackend(api_key="sk-test", base_url="https://test.com")
        assert be._client is None

    def test_client_created_on_access(self):
        be = UniversalBackend(api_key="sk-test", base_url="https://test.com")
        client = be.client
        assert client is not None
        assert be._client is client  # cached

    def test_default_retry_config(self):
        be = UniversalBackend(api_key="sk-test")
        assert isinstance(be.retry, RetryConfig)
        assert be.retry.max_retries == 3

    def test_custom_retry_config(self):
        rc = RetryConfig(max_retries=1, delay=0.1, backoff=1.0)
        be = UniversalBackend(api_key="sk-test", retry=rc)
        assert be.retry is rc


class TestUniversalBackendChat:
    """Chat method — real API calls to DeepSeek."""

    def test_simple_chat(self):
        backend = UniversalBackend(api_key=API_KEY)
        resp = backend.chat(
            messages=[{"role": "user", "content": "Say 'hello' in one word, no punctuation."}],
            model="deepseek-chat",
            max_tokens=10,
            temperature=0.0,
        )
        assert isinstance(resp, LLMResponse)
        assert len(resp.content) > 0
        assert "hello" in resp.content.lower()
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0
        assert resp.retries == 0

    def test_chinese_response(self):
        backend = UniversalBackend(api_key=API_KEY)
        resp = backend.chat(
            messages=[{"role": "user", "content": "用两个字回答：中国的首都是哪里？"}],
            model="deepseek-chat",
            max_tokens=10,
            temperature=0.0,
        )
        assert len(resp.content) > 0
        assert "北京" in resp.content

    def test_invalid_api_key(self):
        backend = UniversalBackend(api_key="invalid-key", base_url="https://api.deepseek.com")
        resp = backend.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-chat",
            max_tokens=10,
        )
        assert "[LLM_ERROR" in resp.content
        assert resp.retries == 0  # Auth errors should not retry

    def test_empty_model_defaults(self):
        backend = UniversalBackend(api_key=API_KEY)
        resp = backend.chat(
            messages=[{"role": "user", "content": "respond with just the word OK"}],
            model="deepseek-chat",
            max_tokens=5,
            temperature=0.0,
        )
        assert len(resp.content) > 0


class TestRetryBehavior:
    """Retry logic tests using mocks."""

    def test_success_on_first_attempt(self):
        be = UniversalBackend(api_key="sk-test", base_url="https://test.com")
        be._client = mock.Mock()
        be._client.chat.completions.create.return_value = mock.Mock(
            usage=None, choices=[mock.Mock(message=mock.Mock(content="ok"))],
        )
        resp = be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "ok"
        assert resp.retries == 0
        assert be._client.chat.completions.create.call_count == 1

    def test_retry_on_rate_limit(self):
        be = UniversalBackend(
            api_key="sk-test", base_url="https://test.com",
            retry=RetryConfig(max_retries=2, delay=0.0, backoff=1.0),
        )
        be._client = mock.Mock()
        be._client.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=mock.Mock(), body=None),
            mock.Mock(usage=None, choices=[mock.Mock(message=mock.Mock(content="ok"))]),
        ]
        resp = be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "ok"
        assert resp.retries == 1
        assert be._client.chat.completions.create.call_count == 2

    def test_retry_exhausted(self):
        be = UniversalBackend(
            api_key="sk-test", base_url="https://test.com",
            retry=RetryConfig(max_retries=2, delay=0.0, backoff=1.0),
        )
        be._client = mock.Mock()
        be._client.chat.completions.create.side_effect = APITimeoutError("timeout")
        resp = be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert "[LLM_ERROR" in resp.content
        assert resp.retries == 2
        assert be._client.chat.completions.create.call_count == 3

    def test_no_retry_on_auth_error(self):
        be = UniversalBackend(api_key="sk-test", base_url="https://test.com")
        be._client = mock.Mock()
        be._client.chat.completions.create.side_effect = AuthenticationError(
            "invalid key", response=mock.Mock(), body=None,
        )
        resp = be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert "[LLM_ERROR" in resp.content
        assert resp.retries == 0
        assert be._client.chat.completions.create.call_count == 1

    def test_retry_on_connection_error(self):
        be = UniversalBackend(
            api_key="sk-test", base_url="https://test.com",
            retry=RetryConfig(max_retries=1, delay=0.0, backoff=1.0),
        )
        be._client = mock.Mock()
        be._client.chat.completions.create.side_effect = [
            APITimeoutError("connection timeout"),
            mock.Mock(usage=None, choices=[mock.Mock(message=mock.Mock(content="recovered"))]),
        ]
        resp = be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "recovered"
        assert resp.retries == 1


# ── Async Backend ──────────────────────────────────────────────

API_KEY_ASYNC = API_KEY


class TestAsyncLLMBackendProtocol:
    """AsyncLLMBackend is a runtime-checkable Protocol."""

    def test_async_openai_backend_is_async_llm_backend(self):
        from sigma.llm import AsyncLLMBackend
        be = AsyncUniversalBackend(api_key="test", base_url="https://test.com")
        assert isinstance(be, AsyncLLMBackend)


class TestAsyncUniversalBackendInit:
    """AsyncUniversalBackend initialization and lazy client."""

    def test_explicit_api_key(self):
        be = AsyncUniversalBackend(api_key="sk-test", base_url="https://test.com")
        assert be._api_key == "sk-test"
        assert be._base_url == "https://test.com"

    def test_env_api_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-async")
        be = AsyncUniversalBackend()
        assert be._api_key == "sk-env-async"

    def test_client_is_lazy(self):
        be = AsyncUniversalBackend(api_key="sk-test", base_url="https://test.com")
        assert be._client is None

    def test_default_retry_config(self):
        be = AsyncUniversalBackend(api_key="sk-test")
        assert isinstance(be.retry, RetryConfig)
        assert be.retry.max_retries == 3

    def test_custom_retry_config(self):
        rc = RetryConfig(max_retries=1, delay=0.1, backoff=1.0)
        be = AsyncUniversalBackend(api_key="sk-test", retry=rc)
        assert be.retry is rc


class TestAsyncUniversalBackendChat:
    """Async chat — real API calls to DeepSeek."""

    @pytest.mark.asyncio
    async def test_simple_chat(self):
        backend = AsyncUniversalBackend(api_key=API_KEY_ASYNC)
        resp = await backend.chat(
            messages=[{"role": "user", "content": "Say 'hello' in one word, no punctuation."}],
            model="deepseek-chat",
            max_tokens=10,
            temperature=0.0,
        )
        assert isinstance(resp, LLMResponse)
        assert len(resp.content) > 0
        assert "hello" in resp.content.lower()
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0
        assert resp.retries == 0

    @pytest.mark.asyncio
    async def test_chinese_response(self):
        backend = AsyncUniversalBackend(api_key=API_KEY_ASYNC)
        resp = await backend.chat(
            messages=[{"role": "user", "content": "用两个字回答：中国的首都是哪里？"}],
            model="deepseek-chat",
            max_tokens=10,
            temperature=0.0,
        )
        assert len(resp.content) > 0
        assert "北京" in resp.content

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        backend = AsyncUniversalBackend(api_key="invalid-key", base_url="https://api.deepseek.com")
        resp = await backend.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-chat",
            max_tokens=10,
        )
        assert "[LLM_ERROR" in resp.content
        assert resp.retries == 0


class TestAsyncRetryBehavior:
    """Retry logic tests using mocks for async backend."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        be = AsyncUniversalBackend(api_key="sk-test", base_url="https://test.com")
        be._client = mock.AsyncMock()
        be._client.chat.completions.create.return_value = mock.Mock(
            usage=None, choices=[mock.Mock(message=mock.Mock(content="ok"))],
        )
        resp = await be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "ok"
        assert resp.retries == 0

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        be = AsyncUniversalBackend(
            api_key="sk-test", base_url="https://test.com",
            retry=RetryConfig(max_retries=2, delay=0.0, backoff=1.0),
        )
        be._client = mock.AsyncMock()
        be._client.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=mock.Mock(), body=None),
            mock.Mock(usage=None, choices=[mock.Mock(message=mock.Mock(content="ok"))]),
        ]
        resp = await be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "ok"
        assert resp.retries == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        be = AsyncUniversalBackend(
            api_key="sk-test", base_url="https://test.com",
            retry=RetryConfig(max_retries=2, delay=0.0, backoff=1.0),
        )
        be._client = mock.AsyncMock()
        be._client.chat.completions.create.side_effect = APITimeoutError("timeout")
        resp = await be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert "[LLM_ERROR" in resp.content
        assert resp.retries == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        be = AsyncUniversalBackend(api_key="sk-test", base_url="https://test.com")
        be._client = mock.AsyncMock()
        be._client.chat.completions.create.side_effect = AuthenticationError(
            "invalid key", response=mock.Mock(), body=None,
        )
        resp = await be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert "[LLM_ERROR" in resp.content
        assert resp.retries == 0

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        be = AsyncUniversalBackend(
            api_key="sk-test", base_url="https://test.com",
            retry=RetryConfig(max_retries=1, delay=0.0, backoff=1.0),
        )
        be._client = mock.AsyncMock()
        be._client.chat.completions.create.side_effect = [
            APITimeoutError("connection timeout"),
            mock.Mock(usage=None, choices=[mock.Mock(message=mock.Mock(content="recovered"))]),
        ]
        resp = await be.chat(messages=[{"role": "user", "content": "hi"}], model="test")
        assert resp.content == "recovered"
        assert resp.retries == 1
