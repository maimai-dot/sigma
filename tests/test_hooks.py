"""Tests for HookSystem — register, trigger, priority ordering, ON_ERROR behavior."""

import pytest
from sigma.hooks import HookSystem, HookPoint


class TestHookPoint:
    """HookPoint enum covers all AERC lifecycle events."""

    def test_all_points_exist(self):
        points = {p.value for p in HookPoint}
        assert "on_start" in points
        assert "on_round_start" in points
        assert "before_plan" in points
        assert "after_plan" in points
        assert "before_do" in points
        assert "after_do" in points
        assert "before_check" in points
        assert "after_check" in points
        assert "before_act" in points
        assert "after_act" in points
        assert "on_round_end" in points
        assert "on_error" in points
        assert "on_complete" in points

    def test_count(self):
        assert len(HookPoint) == 16


class TestHookSystemRegister:
    """Registering callbacks."""

    def test_register_single(self):
        hs = HookSystem()
        called = []

        def cb(**ctx):
            called.append(1)

        hs.register(HookPoint.ON_START, cb)
        hs.trigger(HookPoint.ON_START)
        assert called == [1]

    def test_register_multiple_same_point(self):
        hs = HookSystem()
        called = []

        hs.register(HookPoint.AFTER_PLAN, lambda **ctx: called.append("a"))
        hs.register(HookPoint.AFTER_PLAN, lambda **ctx: called.append("b"))
        hs.trigger(HookPoint.AFTER_PLAN)
        assert called == ["a", "b"]

    def test_register_different_points(self):
        hs = HookSystem()
        results = []

        hs.register(HookPoint.ON_START, lambda **ctx: results.append("start"))
        hs.register(HookPoint.ON_COMPLETE, lambda **ctx: results.append("complete"))
        hs.trigger(HookPoint.ON_START)
        hs.trigger(HookPoint.ON_COMPLETE)
        assert results == ["start", "complete"]

    def test_no_callbacks_no_error(self):
        hs = HookSystem()
        result = hs.trigger(HookPoint.BEFORE_PLAN)
        assert result == {}


class TestHookSystemTrigger:
    """Trigger execution and context passing."""

    def test_context_passed_to_callback(self):
        hs = HookSystem()
        captured = {}

        def cb(**ctx):
            captured.update(ctx)

        hs.register(HookPoint.ON_START, cb)
        hs.trigger(HookPoint.ON_START, instruction="test", state={"round": 1})
        assert captured["instruction"] == "test"
        assert captured["state"] == {"round": 1}

    def test_trigger_returns_final_context(self):
        hs = HookSystem()
        result = hs.trigger(HookPoint.ON_ROUND_END, round_num=3, verdict="converged")
        assert result["round_num"] == 3
        assert result["verdict"] == "converged"

    def test_callback_can_update_context(self):
        hs = HookSystem()

        def add_field(**ctx):
            return {"added_by_hook": True}

        hs.register(HookPoint.AFTER_PLAN, add_field)
        result = hs.trigger(HookPoint.AFTER_PLAN, plan="data")
        assert result["plan"] == "data"
        assert result["added_by_hook"] is True

    def test_callback_can_override_context(self):
        hs = HookSystem()

        def override(**ctx):
            return {"status": "modified"}

        hs.register(HookPoint.BEFORE_DO, override)
        result = hs.trigger(HookPoint.BEFORE_DO, status="original")
        assert result["status"] == "modified"

    def test_callback_returning_none_does_not_change_context(self):
        hs = HookSystem()

        def noop(**ctx):
            return None

        hs.register(HookPoint.ON_START, noop)
        result = hs.trigger(HookPoint.ON_START, key="value")
        assert result["key"] == "value"


class TestHookSystemPriority:
    """Callbacks execute in priority order (lower first)."""

    def test_priority_ordering(self):
        hs = HookSystem()
        order = []

        hs.register(HookPoint.ON_START, lambda **ctx: order.append("c"), priority=30)
        hs.register(HookPoint.ON_START, lambda **ctx: order.append("a"), priority=10)
        hs.register(HookPoint.ON_START, lambda **ctx: order.append("b"), priority=20)
        hs.trigger(HookPoint.ON_START)
        assert order == ["a", "b", "c"]

    def test_default_priority_zero(self):
        hs = HookSystem()
        order = []

        hs.register(HookPoint.ON_START, lambda **ctx: order.append("second"), priority=5)
        hs.register(HookPoint.ON_START, lambda **ctx: order.append("first"))  # default 0
        hs.trigger(HookPoint.ON_START)
        assert order == ["first", "second"]

    def test_same_priority_stable(self):
        hs = HookSystem()
        order = []

        hs.register(HookPoint.ON_START, lambda **ctx: order.append("x"), priority=0)
        hs.register(HookPoint.ON_START, lambda **ctx: order.append("y"), priority=0)
        hs.trigger(HookPoint.ON_START)
        assert order == ["x", "y"]  # insertion order preserved by insort_right


class TestHookSystemOnError:
    """ON_ERROR hook fires when a callback raises, then re-raises."""

    def test_on_error_fires_then_reraises(self):
        hs = HookSystem()
        errors_caught = []

        hs.register(HookPoint.ON_ERROR, lambda **ctx: errors_caught.append(ctx.get("error")))

        def bad_callback(**ctx):
            raise ValueError("boom")

        hs.register(HookPoint.BEFORE_PLAN, bad_callback)
        with pytest.raises(ValueError, match="boom"):
            hs.trigger(HookPoint.BEFORE_PLAN)

        assert len(errors_caught) == 1
        assert isinstance(errors_caught[0], ValueError)

    def test_on_error_receives_context(self):
        hs = HookSystem()
        captured = {}

        hs.register(HookPoint.ON_ERROR, lambda **ctx: captured.update(ctx))

        def bad_callback(**ctx):
            raise RuntimeError("fail")

        hs.register(HookPoint.AFTER_CHECK, bad_callback)
        with pytest.raises(RuntimeError):
            hs.trigger(HookPoint.AFTER_CHECK, round_num=2, state="active")

        assert captured["round_num"] == 2
        assert captured["state"] == "active"
        assert isinstance(captured["error"], RuntimeError)

    def test_on_error_itself_raising_does_not_loop(self):
        hs = HookSystem()

        hs.register(HookPoint.ON_ERROR, lambda **ctx: (_ for _ in ()).throw(ValueError("error hook failed")))

        def bad_callback(**ctx):
            raise RuntimeError("original")

        hs.register(HookPoint.BEFORE_DO, bad_callback)
        with pytest.raises(ValueError, match="error hook failed"):
            hs.trigger(HookPoint.BEFORE_DO)


class TestHookSystemClear:
    """Clear removes all registered hooks."""

    def test_clear_removes_all(self):
        hs = HookSystem()
        called = []

        hs.register(HookPoint.ON_START, lambda **ctx: called.append(1))
        hs.register(HookPoint.AFTER_PLAN, lambda **ctx: called.append(2))
        hs.clear()
        hs.trigger(HookPoint.ON_START)
        hs.trigger(HookPoint.AFTER_PLAN)
        assert called == []

    def test_clear_then_register(self):
        hs = HookSystem()
        hs.register(HookPoint.ON_START, lambda **ctx: None)
        hs.clear()
        called = []

        hs.register(HookPoint.ON_START, lambda **ctx: called.append("new"))
        hs.trigger(HookPoint.ON_START)
        assert called == ["new"]


class TestHookSystemContextChaining:
    """Callbacks can chain modifications through context."""

    def test_chained_modifications(self):
        hs = HookSystem()

        def step1(**ctx):
            return {"pipeline": ctx.get("pipeline", []) + ["step1"]}

        def step2(**ctx):
            return {"pipeline": ctx.get("pipeline", []) + ["step2"]}

        hs.register(HookPoint.ON_START, step1, priority=0)
        hs.register(HookPoint.ON_START, step2, priority=1)
        result = hs.trigger(HookPoint.ON_START)
        assert result["pipeline"] == ["step1", "step2"]

    def test_real_world_auth_hook(self):
        hs = HookSystem()
        authorized = {"allowed": False}

        def auth_check(**ctx):
            instruction = ctx.get("instruction", "")
            if "机密" in instruction:
                return {"allowed": False}
            return {"allowed": True}

        hs.register(HookPoint.ON_START, auth_check)
        result = hs.trigger(HookPoint.ON_START, instruction="检查火箭设计")
        assert result["allowed"] is True

        result2 = hs.trigger(HookPoint.ON_START, instruction="查看机密文件")
        assert result2["allowed"] is False
