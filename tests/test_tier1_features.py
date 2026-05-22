"""Tests for rate limiting, per-agent LLM config, and task-level hooks."""

import time
import pytest

from sigma.llm import RateLimiter, AsyncRateLimiter, AgentLLMConfig, RetryConfig
from sigma.hooks import HookSystem, HookPoint


# ── RateLimiter ────────────────────────────────────────────────────

class TestRateLimiter:

    def test_no_limit_by_default(self):
        rl = RateLimiter()
        assert rl.max_rpm <= 0
        t0 = time.perf_counter()
        for _ in range(100):
            rl.acquire()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0  # Should not block

    def test_zero_max_rpm_no_limit(self):
        rl = RateLimiter(max_rpm=0)
        t0 = time.perf_counter()
        for _ in range(50):
            rl.acquire()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5

    def test_current_rpm_tracks_calls(self):
        rl = RateLimiter(max_rpm=1000)
        for _ in range(10):
            rl.acquire()
        assert rl.current_rpm == 10

    def test_basic_throttling(self):
        rl = RateLimiter(max_rpm=30)
        t0 = time.perf_counter()
        for _ in range(31):
            rl.acquire()
        elapsed = time.perf_counter() - t0
        # With 30 RPM, the 31st call should be throttled slightly
        assert elapsed > 0.01

    def test_high_rpm_no_throttling(self):
        rl = RateLimiter(max_rpm=10000)
        t0 = time.perf_counter()
        for _ in range(100):
            rl.acquire()
        elapsed = time.perf_counter() - t0
        # 100 calls at 10000 RPM should not throttle
        assert elapsed < 2.0


# ── AgentLLMConfig ─────────────────────────────────────────────────

class TestAgentLLMConfig:

    def test_defaults(self):
        cfg = AgentLLMConfig()
        assert cfg.model == ""
        assert cfg.max_tokens == 0
        assert cfg.temperature is None
        assert cfg.max_rpm == 0

    def test_full_override(self):
        cfg = AgentLLMConfig(
            model="deepseek-reasoner",
            max_tokens=4096,
            temperature=0.8,
            max_rpm=60,
        )
        assert cfg.model == "deepseek-reasoner"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.8
        assert cfg.max_rpm == 60

    def test_partial_override(self):
        cfg = AgentLLMConfig(model="custom-model")
        assert cfg.model == "custom-model"
        assert cfg.max_tokens == 0
        assert cfg.temperature is None


# ── AgentSpec.get_llm_params ───────────────────────────────────────

class TestAgentLLMParams:

    def test_no_config_returns_defaults(self):
        from sigma.protocol import AgentSpec
        spec = AgentSpec(
            name="test", role="Test", goal="test", backstory="test",
        )
        defaults = {"model": "default", "max_tokens": 1024, "temperature": 0.2}
        result = spec.get_llm_params(defaults)
        assert result == defaults

    def test_full_override(self):
        from sigma.protocol import AgentSpec
        spec = AgentSpec(
            name="test", role="Test", goal="test", backstory="test",
            llm_config=AgentLLMConfig(model="overridden", max_tokens=2048, temperature=0.9),
        )
        defaults = {"model": "default", "max_tokens": 1024, "temperature": 0.2}
        result = spec.get_llm_params(defaults)
        assert result["model"] == "overridden"
        assert result["max_tokens"] == 2048
        assert result["temperature"] == 0.9

    def test_partial_override(self):
        from sigma.protocol import AgentSpec
        spec = AgentSpec(
            name="test", role="Test", goal="test", backstory="test",
            llm_config=AgentLLMConfig(model="custom"),
        )
        defaults = {"model": "default", "max_tokens": 1024, "temperature": 0.2}
        result = spec.get_llm_params(defaults)
        assert result["model"] == "custom"
        assert result["max_tokens"] == 1024  # unchanged
        assert result["temperature"] == 0.2  # unchanged

    def test_empty_model_not_override(self):
        from sigma.protocol import AgentSpec
        spec = AgentSpec(
            name="test", role="Test", goal="test", backstory="test",
            llm_config=AgentLLMConfig(model="", max_tokens=0),
        )
        defaults = {"model": "default", "max_tokens": 2048, "temperature": 0.1}
        result = spec.get_llm_params(defaults)
        assert result["model"] == "default"  # empty string doesn't override
        assert result["max_tokens"] == 2048  # 0 doesn't override


# ── Task-Level Hooks ───────────────────────────────────────────────

class TestTaskHooks:

    def test_before_task_hook_exists(self):
        assert HookPoint.BEFORE_TASK.value == "before_task"

    def test_after_task_hook_exists(self):
        assert HookPoint.AFTER_TASK.value == "after_task"

    def test_on_task_error_hook_exists(self):
        assert HookPoint.ON_TASK_ERROR.value == "on_task_error"

    def test_before_task_fires(self):
        hooks = HookSystem()
        fired = []

        def on_before(**ctx):
            fired.append(ctx)

        hooks.register(HookPoint.BEFORE_TASK, on_before)
        hooks.trigger(HookPoint.BEFORE_TASK, task_id="t1", task_desc="do X",
                       agent_names=["Alice"])
        assert len(fired) == 1
        assert fired[0]["task_id"] == "t1"
        assert fired[0]["task_desc"] == "do X"

    def test_after_task_fires(self):
        hooks = HookSystem()
        fired = []

        hooks.register(HookPoint.AFTER_TASK, lambda **ctx: fired.append(ctx))
        hooks.trigger(HookPoint.AFTER_TASK, task_id="t2", result="ok", duration_ms=150.0)
        assert len(fired) == 1
        assert fired[0]["duration_ms"] == 150.0

    def test_on_task_error_fires(self):
        hooks = HookSystem()
        fired = []

        hooks.register(HookPoint.ON_TASK_ERROR, lambda **ctx: fired.append(ctx))
        hooks.trigger(HookPoint.ON_TASK_ERROR, task_id="t3", error="timeout", attempt=1)
        assert len(fired) == 1
        assert fired[0]["error"] == "timeout"

    def test_task_hooks_do_not_interfere_with_aerc_hooks(self):
        hooks = HookSystem()
        before_plan_calls = []
        task_calls = []

        hooks.register(HookPoint.BEFORE_PLAN, lambda **ctx: before_plan_calls.append(ctx))
        hooks.register(HookPoint.BEFORE_TASK, lambda **ctx: task_calls.append(ctx))

        hooks.trigger(HookPoint.BEFORE_TASK, task_id="x")
        hooks.trigger(HookPoint.BEFORE_PLAN)

        assert len(task_calls) == 1
        assert len(before_plan_calls) == 1

    def test_hook_count(self):
        # 13 AERC + 3 task-level = 16
        assert len(HookPoint) == 16
