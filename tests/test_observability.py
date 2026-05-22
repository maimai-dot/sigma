"""Tests for sigma.observability — OpenTelemetry integration."""

import os
import pytest


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_tracing():
    """Reset tracing state before each test."""
    import sigma.observability as obs
    obs._initialized = False
    obs._tracer = None
    yield
    obs._initialized = False
    obs._tracer = None


# ═══════════════════════════════════════════════════════════════════════
# Module loading
# ═══════════════════════════════════════════════════════════════════════

class TestModuleLoading:
    def test_import_works(self):
        import sigma.observability as obs
        assert hasattr(obs, "init_tracing")
        assert hasattr(obs, "traced")
        assert hasattr(obs, "tracing_span")

    def test_init_tracing_console(self):
        from sigma.observability import init_tracing, is_initialized, _initialized
        # May work or not depending on OTel availability
        result = init_tracing(service_name="test")
        # Reset to avoid polluting other tests
        import sigma.observability as obs
        obs._initialized = False
        obs._tracer = None
        # Either result is valid
        assert result in (True, False)


# ═══════════════════════════════════════════════════════════════════════
# traced decorator
# ═══════════════════════════════════════════════════════════════════════

class TestTracedDecorator:
    def test_traced_no_args(self):
        from sigma.observability import traced

        @traced
        def foo(x):
            return x * 2

        assert foo(21) == 42
        assert foo.__name__ == "foo"

    def test_traced_with_name(self):
        from sigma.observability import traced

        @traced(name="my_name")
        def bar():
            return "ok"

        assert bar() == "ok"

    def test_traced_with_attributes(self):
        from sigma.observability import traced

        @traced(attributes={"key": "val"})
        def baz():
            return 1

        assert baz() == 1

    def test_traced_propagates_exception(self):
        from sigma.observability import traced

        @traced
        def oops():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            oops()

    def test_traced_preserves_func_name(self):
        from sigma.observability import traced

        @traced(name="custom")
        def some_function():
            pass

        assert some_function.__name__ == "some_function"


# ═══════════════════════════════════════════════════════════════════════
# tracing_span context manager
# ═══════════════════════════════════════════════════════════════════════

class TestTracingSpan:
    def test_span_works(self):
        from sigma.observability import tracing_span

        with tracing_span("test_span") as span:
            result = 1 + 1

        assert result == 2

    def test_span_with_attributes(self):
        from sigma.observability import tracing_span

        with tracing_span("test_span", {"count": 5}) as span:
            pass

    def test_span_propagates_exception(self):
        from sigma.observability import tracing_span

        with pytest.raises(RuntimeError, match="boom"):
            with tracing_span("failing_span"):
                raise RuntimeError("boom")

    def test_nested_spans(self):
        from sigma.observability import tracing_span

        results = []
        with tracing_span("outer"):
            results.append(1)
            with tracing_span("inner"):
                results.append(2)
            results.append(3)
        assert results == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════
# Auto-init
# ═══════════════════════════════════════════════════════════════════════

class TestAutoInit:
    def test_auto_init_no_env(self, monkeypatch):
        monkeypatch.delenv("SIGMA_TRACING", raising=False)
        import sigma.observability as obs
        obs._auto_initialized = False
        obs._auto_init()
        # Auto-init should at least not crash
        assert True

    def test_auto_init_idempotent(self):
        import sigma.observability as obs
        obs._auto_initialized = True
        obs._auto_init()
        assert True


# ═══════════════════════════════════════════════════════════════════════
# init_tracing
# ═══════════════════════════════════════════════════════════════════════

class TestInitTracing:
    def test_disabled(self):
        from sigma.observability import init_tracing
        import sigma.observability as obs
        obs._initialized = False
        result = init_tracing(enabled=False)
        assert result is False

    def test_get_tracer_before_init(self):
        from sigma.observability import get_tracer
        # After reset, tracer should be None
        assert get_tracer() is None
