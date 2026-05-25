"""LLM backend abstraction for Sigma.

Supports both sync/async backends with retry, streaming, rate limiting,
per-agent LLM configuration, protocol-based typing, and multimodal content.
"""

import asyncio
import base64
import os
import re
import threading
import time
from collections.abc import Generator, AsyncGenerator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Callable, TypedDict

from openai import (
    OpenAI,
    AsyncOpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
)


def _sanitize_error(err: Exception | str) -> str:
    """Sanitize error messages to prevent API key leakage into persisted output.

    OpenAI/DeepSeek SDK AuthenticationError messages may include the raw API
    key (e.g. "Incorrect API key provided: sk-xxxxxxxxxxxxx").  Since error
    strings flow into LLMResponse.content and are persisted to disk (REPORT.md,
    result.json, checkpoint.json), we must redact them.
    """
    msg = str(err)
    msg = re.sub(r'sk-[A-Za-z0-9]{20,}', '[REDACTED]', msg)
    msg = re.sub(r'org-[A-Za-z0-9]{20,}', '[REDACTED]', msg)
    msg = re.sub(r'"api_key":\s*"[^"]*"', '"api_key": "[REDACTED]"', msg)
    return msg


def _fmt_error(err: Exception | str) -> str:
    """Format a sanitized error string for LLMResponse.content."""
    return f"[LLM_ERROR: {_sanitize_error(err)}]"


TRANSIENT_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


@dataclass
class RetryConfig:
    """Retry policy for LLM calls."""
    max_retries: int = 3
    delay: float = 1.0         # seconds before first retry
    backoff: float = 2.0        # multiplier for each subsequent retry


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0             # how many retries were used
    tool_calls: list[dict] | None = None  # OpenAI function calling tool_calls


# ── Multimodal Content Types ────────────────────────────────────────


class TextContent(TypedDict):
    """Text block in a multimodal message."""
    type: str          # "text"
    text: str


class ImageUrlDetail(TypedDict, total=False):
    """Optional detail level for image_url content."""
    detail: str        # "auto", "low", "high"


class ImageUrlContent(TypedDict):
    """Image block in a multimodal message."""
    type: str          # "image_url"
    image_url: dict    # {"url": "https://...", "detail": "auto"}


ContentPart = TextContent | ImageUrlContent
"""A single content part in a multimodal message."""


def build_message(
    role: str,
    content: str | list[ContentPart],
) -> dict:
    """Build a chat message dict with text or multimodal content.

    Args:
        role: "system", "user", or "assistant"
        content: Either a plain string or a list of ContentPart dicts

    Returns:
        A message dict ready for the OpenAI-compatible messages list.

    Example:
        >>> build_message("user", "Hello")
        {'role': 'user', 'content': 'Hello'}

        >>> msg = build_message("user", [
        ...     text_content("Describe this diagram:"),
        ...     image_url("https://example.com/diagram.png"),
        ... ])
    """
    return {"role": role, "content": content}


def text_content(text: str) -> TextContent:
    """Create a text content part for multimodal messages.

    Example:
        >>> text_content("What is in this image?")
        {'type': 'text', 'text': 'What is in this image?'}
    """
    return TextContent(type="text", text=text)


def image_url(url: str, detail: str = "auto") -> ImageUrlContent:
    """Create an image_url content part for multimodal messages.

    Args:
        url: Image URL or base64 data URI (data:image/...;base64,...)
        detail: "auto", "low", or "high" (controls resolution tier)

    Example:
        >>> image_url("https://example.com/photo.jpg")
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/photo.jpg', 'detail': 'auto'}}
    """
    return ImageUrlContent(
        type="image_url",
        image_url={"url": url, "detail": detail},
    )


def encode_image_base64(path: str) -> str:
    """Read an image file and return a base64 data URI.

    Supports PNG, JPEG, GIF, and WebP. The resulting data URI can be
    passed directly to image_url().

    Args:
        path: Path to the image file.

    Returns:
        A data URI string: "data:image/<type>;base64,<encoded>"

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not a supported image type.
    """
    import pathlib
    ext = pathlib.Path(path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext)
    if mime is None:
        raise ValueError(
            f"Unsupported image format: {ext}. Supported: {', '.join(mime_map.keys())}"
        )
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def is_multimodal(content: object) -> bool:
    """Check if message content is a multimodal content part list."""
    return isinstance(content, list) and all(
        isinstance(p, dict) and "type" in p for p in content
    )


# ── Rate Limiting ──────────────────────────────────────────────────


class RateLimiter:
    """Sliding-window rate limiter for API calls.

    Tracks call timestamps in a thread-safe deque.
    When max_rpm is reached, sleeps until capacity is available.
    """

    def __init__(self, max_rpm: int = 0):
        """
        Args:
            max_rpm: Max requests per minute. 0 or negative = no limit.
        """
        self.max_rpm = max_rpm
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        if self.max_rpm <= 0:
            return
        with self._lock:
            now = time.time()
            window_start = now - 60.0
            # Purge expired timestamps
            self._timestamps = [t for t in self._timestamps if t > window_start]
            if len(self._timestamps) >= self.max_rpm:
                # Sleep until the oldest timestamp exits the window
                oldest = self._timestamps[0]
                sleep_time = (oldest + 60.0) - now + 0.05  # 50ms buffer
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Recheck timestamps after sleeping
                now = time.time()
                window_start = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > window_start]
            self._timestamps.append(now)

    @property
    def current_rpm(self) -> int:
        with self._lock:
            window_start = time.time() - 60.0
            return len([t for t in self._timestamps if t > window_start])


class AsyncRateLimiter:
    """Async sliding-window rate limiter for API calls."""

    def __init__(self, max_rpm: int = 0):
        self.max_rpm = max_rpm
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        if self.max_rpm <= 0:
            return
        async with self._lock:
            now = time.time()
            window_start = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > window_start]
            if len(self._timestamps) >= self.max_rpm:
                oldest = self._timestamps[0]
                sleep_time = (oldest + 60.0) - now + 0.05
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.time()
                window_start = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > window_start]
            self._timestamps.append(now)


# ── Per-Agent LLM Config ───────────────────────────────────────────


@dataclass
class AgentLLMConfig:
    """Per-agent LLM configuration override.

    Attach to AgentSpec to use a different model, temperature, or token
    budget for a specific agent. All fields are optional — only set fields
    override the global default.
    """
    model: str = ""
    """Model name override (e.g. 'deepseek-chat' vs 'deepseek-reasoner')."""
    max_tokens: int = 0
    """Max output tokens override. 0 = use global default."""
    temperature: float | None = None
    """Temperature override. None = use global default."""
    max_rpm: int = 0
    """Per-agent RPM limit. 0 = use global/shared limiter."""


# Type alias for a chat message dict that may contain multimodal content.
# Content is either a string or a list of TextContent/ImageUrlContent dicts.
ChatMessage = dict[str, object]


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends. Implement to use any LLM provider."""

    def chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a chat completion and return the response."""
        ...


@runtime_checkable
class AsyncLLMBackend(Protocol):
    """Protocol for async LLM backends."""

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send an async chat completion and return the response."""
        ...


class UniversalBackend:
    """LLM backend for any OpenAI-compatible API (DeepSeek, 智谱, 月之暗面, 通义千问, etc).

    Defaults to DeepSeek — no dependency on OpenAI the company.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
        max_rpm: int = 0,
    ):
        """
        Args:
            api_key: API key. Falls back to SIGMA_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY env vars.
            base_url: API base URL. Falls back to SIGMA_BASE_URL / OPENAI_API_BASE env var or DeepSeek default.
            retry: Retry policy. Defaults to 3 retries with exponential backoff.
            max_rpm: Max requests per minute. 0 = no limit.
        """
        key = api_key or os.getenv(
            "SIGMA_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        )
        url = base_url or os.getenv(
            "SIGMA_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
        )
        self._api_key = key
        self._base_url = url
        self._client: OpenAI | None = None
        self.retry = retry or RetryConfig()
        self._limiter = RateLimiter(max_rpm)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        last_error = None
        delay = self.retry.delay

        for attempt in range(self.retry.max_retries + 1):
            self._limiter.acquire()
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = self.client.chat.completions.create(**kwargs)
            except TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self.retry.max_retries:
                    time.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(
                    content=_fmt_error(e),
                    retries=attempt,
                )
            except (AuthenticationError, PermissionDeniedError, BadRequestError) as e:
                return LLMResponse(
                    content=_fmt_error(e),
                    retries=0,
                )
            except Exception as e:
                last_error = e
                if attempt < self.retry.max_retries:
                    time.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(
                    content=_fmt_error(e),
                    retries=attempt,
                )
            else:
                input_tokens = resp.usage.prompt_tokens if resp.usage else 0
                output_tokens = resp.usage.completion_tokens if resp.usage else 0
                choice = resp.choices[0] if resp.choices else None
                content = choice.message.content if choice else ""
                tool_calls = None
                if choice and choice.message.tool_calls:
                    try:
                        tool_calls = [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ]
                    except (TypeError, AttributeError):
                        tool_calls = None
                return LLMResponse(
                    content=content or "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retries=attempt,
                    tool_calls=tool_calls,
                )

        # Should not reach here, but safety fallback
        return LLMResponse(
            content=_fmt_error(last_error),
            retries=self.retry.max_retries,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        on_chunk: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Stream a chat completion, yielding chunks via callback."""
        last_error = None
        delay = self.retry.delay
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for attempt in range(self.retry.max_retries + 1):
            collected.clear()
            tool_call_buf: dict[int, dict] = {}  # index → {id, name, arguments}
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                stream = self.client.chat.completions.create(**kwargs)
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        text = delta.content or ""
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            text = delta.reasoning_content + text
                        if text:
                            collected.append(text)
                            if on_chunk:
                                on_chunk(text)
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_call_buf:
                                    tool_call_buf[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                                if tc.id:
                                    tool_call_buf[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_call_buf[idx]["function"]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        tool_call_buf[idx]["function"]["arguments"] += tc.function.arguments
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens or 0
                        output_tokens = chunk.usage.completion_tokens or 0
                tool_calls = [tool_call_buf[i] for i in sorted(tool_call_buf)] if tool_call_buf else None
                return LLMResponse(
                    content="".join(collected),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retries=attempt,
                    tool_calls=tool_calls,
                )
            except TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self.retry.max_retries:
                    time.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(content=_fmt_error(e), retries=attempt)
            except (AuthenticationError, PermissionDeniedError, BadRequestError) as e:
                return LLMResponse(content=_fmt_error(e), retries=0)
            except Exception as e:
                last_error = e
                if attempt < self.retry.max_retries:
                    time.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(content=_fmt_error(e), retries=attempt)

        return LLMResponse(
            content=_fmt_error(last_error),
            retries=self.retry.max_retries,
        )


class AsyncUniversalBackend:
    """Async backend for any OpenAI-compatible API with retry + rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
        max_rpm: int = 0,
    ):
        """
        Args:
            api_key: API key. Falls back to SIGMA_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY env vars.
            base_url: API base URL. Falls back to SIGMA_BASE_URL / OPENAI_API_BASE env var or DeepSeek default.
            retry: Retry policy. Defaults to 3 retries with exponential backoff.
            max_rpm: Max requests per minute. 0 = no limit.
        """
        key = api_key or os.getenv(
            "SIGMA_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        )
        url = base_url or os.getenv(
            "SIGMA_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
        )
        self._api_key = key
        self._base_url = url
        self._client: AsyncOpenAI | None = None
        self.retry = retry or RetryConfig()
        self._limiter = AsyncRateLimiter(max_rpm)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        delay = self.retry.delay

        for attempt in range(self.retry.max_retries + 1):
            await self._limiter.acquire()
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = await self.client.chat.completions.create(**kwargs)
            except TRANSIENT_ERRORS as e:
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(
                    content=_fmt_error(e),
                    retries=attempt,
                )
            except (AuthenticationError, PermissionDeniedError, BadRequestError) as e:
                return LLMResponse(
                    content=_fmt_error(e),
                    retries=0,
                )
            except Exception as e:
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(
                    content=_fmt_error(e),
                    retries=attempt,
                )
            else:
                input_tokens = resp.usage.prompt_tokens if resp.usage else 0
                output_tokens = resp.usage.completion_tokens if resp.usage else 0
                choice = resp.choices[0] if resp.choices else None
                content = choice.message.content if choice else ""
                tool_calls = None
                if choice and choice.message.tool_calls:
                    try:
                        tool_calls = [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ]
                    except (TypeError, AttributeError):
                        tool_calls = None
                return LLMResponse(
                    content=content or "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retries=attempt,
                    tool_calls=tool_calls,
                )

        return LLMResponse(
            content=_fmt_error("retry exhausted"),
            retries=self.retry.max_retries,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        on_chunk: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Stream an async chat completion, yielding chunks via callback."""
        delay = self.retry.delay
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for attempt in range(self.retry.max_retries + 1):
            collected.clear()
            tool_call_buf: dict[int, dict] = {}
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        text = delta.content or ""
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            text = delta.reasoning_content + text
                        if text:
                            collected.append(text)
                            if on_chunk:
                                on_chunk(text)
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_call_buf:
                                    tool_call_buf[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                                if tc.id:
                                    tool_call_buf[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_call_buf[idx]["function"]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        tool_call_buf[idx]["function"]["arguments"] += tc.function.arguments
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens or 0
                        output_tokens = chunk.usage.completion_tokens or 0
                tool_calls = [tool_call_buf[i] for i in sorted(tool_call_buf)] if tool_call_buf else None
                return LLMResponse(
                    content="".join(collected),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retries=attempt,
                    tool_calls=tool_calls,
                )
            except TRANSIENT_ERRORS as e:
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(content=_fmt_error(e), retries=attempt)
            except (AuthenticationError, PermissionDeniedError, BadRequestError) as e:
                return LLMResponse(content=_fmt_error(e), retries=0)
            except Exception as e:
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.retry.backoff
                    continue
                return LLMResponse(content=_fmt_error(e), retries=attempt)

        return LLMResponse(
            content=_fmt_error("retry exhausted"),
            retries=self.retry.max_retries,
        )


# ═══════════════════════════════════════════════════════════════════════
# Backward-compatible aliases (deprecated since 0.2.0)
# ═══════════════════════════════════════════════════════════════════════

import warnings as _warnings


class OpenAIBackend(UniversalBackend):
    """Deprecated alias. Use UniversalBackend instead."""

    def __init__(self, *args, **kwargs):
        _warnings.warn(
            "OpenAIBackend is deprecated. Use UniversalBackend instead.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class AsyncOpenAIBackend(AsyncUniversalBackend):
    """Deprecated alias. Use AsyncUniversalBackend instead."""

    def __init__(self, *args, **kwargs):
        _warnings.warn(
            "AsyncOpenAIBackend is deprecated. Use AsyncUniversalBackend instead.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)
