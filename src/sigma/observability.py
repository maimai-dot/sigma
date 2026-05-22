"""Observability — optional OpenTelemetry integration for Sigma/Tau.

Design:
  - Zero hard dependency on opentelemetry packages (all imports are lazy)
  - Decorator-based: @traced wraps any function in an OTel span
  - Context manager: with tracing_span("name"): ... for inline spans
  - TauOrchestrator integration via config flag or env var
  - SigmaOrchestrator integration via hook injection

Usage:
    from sigma.observability import init_tracing, traced

    init_tracing(service_name="sigma", exporter="console")
    # or set SIGMA_TRACING=true in env

    @traced
    def my_function(): ...

    with tracing_span("my_operation"):
        do_work()
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════════
# Tracing state
# ═══════════════════════════════════════════════════════════════════════

_tracer = None
_initialized = False


def _lazy_import_opentelemetry():
    """Lazy import opentelemetry packages — raises if not installed."""
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        return otel_trace, TracerProvider, BatchSpanProcessor, ConsoleSpanExporter, Resource
    except ImportError:
        raise ImportError(
            "OpenTelemetry packages not installed. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )


def init_tracing(
    service_name: str = "sigma",
    exporter: str = "console",
    otlp_endpoint: str = "",
    enabled: bool = True,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces
        exporter: "console" or "otlp"
        otlp_endpoint: OTLP collector endpoint (when exporter="otlp")
        enabled: Set False to disable tracing even when packages are installed

    Returns True if tracing was successfully initialized.
    """
    global _tracer, _initialized

    if _initialized or not enabled:
        return _initialized

    try:
        otel_trace, TracerProvider, BatchSpanProcessor, ConsoleSpanExporter, Resource = (
            _lazy_import_opentelemetry()
        )

        resource = Resource(attributes={"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if exporter == "otlp" and otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except ImportError:
                pass  # Fall back to console

        if exporter == "console" or not _tracer:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        otel_trace.set_tracer_provider(provider)
        _tracer = otel_trace.get_tracer(service_name)
        _initialized = True
        return True

    except ImportError:
        return False


def get_tracer():
    """Get the current tracer, or None if tracing is not initialized."""
    return _tracer


def is_initialized() -> bool:
    return _initialized


# ═══════════════════════════════════════════════════════════════════════
# Decorator-based tracing
# ═══════════════════════════════════════════════════════════════════════

def traced(fn=None, *, name: str = "", attributes: dict[str, Any] | None = None):
    """Decorator: wrap a function in an OpenTelemetry span.

    Usage:
        @traced
        def my_func(): ...

        @traced(name="custom_name", attributes={"key": "val"})
        def my_func(): ...
    """
    def _decorator(f):
        span_name = name or f.__qualname__
        attrs = attributes or {}

        def wrapper(*args, **kwargs):
            if _tracer is None:
                return f(*args, **kwargs)
            with _tracer.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    t0 = time.perf_counter()
                    result = f(*args, **kwargs)
                    span.set_attribute("duration_ms", (time.perf_counter() - t0) * 1000)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(2, str(e))  # Error
                    raise
        wrapper.__name__ = f.__name__
        return wrapper

    if fn is not None:
        return _decorator(fn)
    return _decorator


# ═══════════════════════════════════════════════════════════════════════
# Context manager for inline spans
# ═══════════════════════════════════════════════════════════════════════

@contextmanager
def tracing_span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for an inline OpenTelemetry span.

    Usage:
        with tracing_span("decompose_task", {"agents": 4}):
            result = decompose(...)
    """
    if _tracer is None:
        yield
        return

    with _tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        try:
            t0 = time.perf_counter()
            yield span
            span.set_attribute("duration_ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            span.record_exception(e)
            span.set_status(2, str(e))
            raise


# ═══════════════════════════════════════════════════════════════════════
# Auto-initialization from env
# ═══════════════════════════════════════════════════════════════════════

_auto_initialized = False


def _auto_init():
    """Auto-initialize tracing from environment if SIGMA_TRACING=true."""
    global _auto_initialized
    if _auto_initialized:
        return
    _auto_initialized = True

    if os.environ.get("SIGMA_TRACING", "").lower() in ("1", "true", "yes"):
        svc = os.environ.get("OTEL_SERVICE_NAME", "sigma")
        exporter = os.environ.get("SIGMA_TRACING_EXPORTER", "console")
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        init_tracing(service_name=svc, exporter=exporter, otlp_endpoint=endpoint)
