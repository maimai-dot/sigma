"""Tests for sigma.guardrails — output validation system."""

import pytest
from sigma.guardrails import (
    GuardResult, GuardrailReport, Severity,
    RangeCheck, CrossParamCheck, SetCheck, CustomCheck,
    GuardrailSet, Guardrail,
    rocket_knsb_guardrails,
)


# ═══════════════════════════════════════════════════════════════════════
# GuardResult
# ═══════════════════════════════════════════════════════════════════════

class TestGuardResult:
    def test_pass_is_ok(self):
        r = GuardResult(Severity.PASS)
        assert r.ok

    def test_warn_is_ok(self):
        r = GuardResult(Severity.WARN)
        assert r.ok

    def test_block_is_not_ok(self):
        r = GuardResult(Severity.BLOCK)
        assert not r.ok

    def test_to_dict(self):
        r = GuardResult(Severity.WARN, message="test", guard_name="g1",
                        param_key="p1", actual_value=42.0)
        d = r.to_dict()
        assert d["severity"] == "WARN"
        assert d["message"] == "test"
        assert d["guard_name"] == "g1"
        assert d["param_key"] == "p1"
        assert d["actual_value"] == 42.0


# ═══════════════════════════════════════════════════════════════════════
# GuardrailReport
# ═══════════════════════════════════════════════════════════════════════

class TestGuardrailReport:
    def test_empty_report_all_ok(self):
        r = GuardrailReport()
        assert r.all_ok
        assert not r.blocked
        assert r.warnings == []
        assert r.summary == "No guardrails defined"

    def test_all_pass(self):
        r = GuardrailReport(results=[
            GuardResult(Severity.PASS, guard_name="g1"),
            GuardResult(Severity.PASS, guard_name="g2"),
        ])
        assert r.all_ok
        assert not r.blocked
        assert r.summary == "2 PASS"

    def test_mixed_severity(self):
        r = GuardrailReport(results=[
            GuardResult(Severity.PASS, guard_name="g1"),
            GuardResult(Severity.WARN, message="warn1", guard_name="g2"),
            GuardResult(Severity.BLOCK, message="block1", guard_name="g3"),
        ])
        assert not r.all_ok
        assert r.blocked
        assert len(r.warnings) == 1
        assert "1 BLOCK" in r.summary
        assert "1 WARN" in r.summary
        assert "1 PASS" in r.summary

    def test_to_dict(self):
        r = GuardrailReport(results=[
            GuardResult(Severity.PASS, guard_name="g1"),
            GuardResult(Severity.WARN, message="w", guard_name="g2"),
        ])
        d = r.to_dict()
        assert len(d["results"]) == 2
        assert d["blocked"] == False
        assert "1 WARN" in d["summary"]


# ═══════════════════════════════════════════════════════════════════════
# RangeCheck
# ═══════════════════════════════════════════════════════════════════════

class TestRangeCheck:
    def test_value_in_range(self):
        c = RangeCheck("thrust", min_val=10, max_val=1000)
        r = c.check({"thrust": 500}, {})
        assert r.severity == Severity.PASS

    def test_value_below_min(self):
        c = RangeCheck("thrust", min_val=10)
        r = c.check({"thrust": 5}, {})
        assert r.severity == Severity.WARN
        assert "below" in r.message

    def test_value_above_max(self):
        c = RangeCheck("thrust", max_val=1000)
        r = c.check({"thrust": 5000}, {})
        assert r.severity == Severity.WARN
        assert "above" in r.message

    def test_negative_value_block(self):
        c = RangeCheck("mass_kg", min_val=0.01)
        r = c.check({"mass_kg": -5}, {})
        assert r.severity == Severity.BLOCK

    def test_exclude_zero_blocks(self):
        c = RangeCheck("isp_s", exclude_zero=True)
        r = c.check({"isp_s": 0}, {})
        assert r.severity == Severity.BLOCK

    def test_param_missing_passes(self):
        c = RangeCheck("thrust", min_val=10)
        r = c.check({}, {})
        assert r.severity == Severity.PASS

    def test_no_bounds_all_ok(self):
        c = RangeCheck("thrust")
        r = c.check({"thrust": 99999}, {})
        assert r.severity == Severity.PASS


# ═══════════════════════════════════════════════════════════════════════
# CrossParamCheck
# ═══════════════════════════════════════════════════════════════════════

class TestCrossParamCheck:
    def test_ratio_in_range(self):
        c = CrossParamCheck("thrust", "mass", min_ratio=1, max_ratio=100)
        r = c.check({"thrust": 500, "mass": 50}, {})
        assert r.severity == Severity.PASS

    def test_ratio_below_min(self):
        c = CrossParamCheck("thrust", "mass", min_ratio=1)
        r = c.check({"thrust": 5, "mass": 100}, {})
        assert r.severity == Severity.WARN

    def test_ratio_above_max(self):
        c = CrossParamCheck("thrust", "mass", max_ratio=100)
        r = c.check({"thrust": 5000, "mass": 1}, {})
        assert r.severity == Severity.WARN

    def test_param_missing_passes(self):
        c = CrossParamCheck("thrust", "mass", min_ratio=1)
        r = c.check({"thrust": 500}, {})
        assert r.severity == Severity.PASS

    def test_less_than_pass(self):
        c = CrossParamCheck("a", "b", check_type="less_than")
        r = c.check({"a": 5, "b": 10}, {})
        assert r.severity == Severity.PASS

    def test_less_than_fails(self):
        c = CrossParamCheck("a", "b", check_type="less_than")
        r = c.check({"a": 15, "b": 10}, {})
        assert r.severity == Severity.WARN

    def test_greater_than_pass(self):
        c = CrossParamCheck("a", "b", check_type="greater_than")
        r = c.check({"a": 15, "b": 10}, {})
        assert r.severity == Severity.PASS

    def test_greater_than_fails(self):
        c = CrossParamCheck("a", "b", check_type="greater_than")
        r = c.check({"a": 5, "b": 10}, {})
        assert r.severity == Severity.WARN


# ═══════════════════════════════════════════════════════════════════════
# SetCheck
# ═══════════════════════════════════════════════════════════════════════

class TestSetCheck:
    def test_value_in_valid_set(self):
        c = SetCheck("grade", valid_set={1, 2, 3})
        r = c.check({"grade": 2}, {})
        assert r.severity == Severity.PASS

    def test_value_not_in_valid_set(self):
        c = SetCheck("grade", valid_set={1, 2, 3})
        r = c.check({"grade": 5}, {})
        assert r.severity == Severity.WARN

    def test_value_in_invalid_set(self):
        c = SetCheck("grade", invalid_set={0, -1})
        r = c.check({"grade": 0}, {})
        assert r.severity == Severity.BLOCK


# ═══════════════════════════════════════════════════════════════════════
# CustomCheck
# ═══════════════════════════════════════════════════════════════════════

class TestCustomCheck:
    def test_custom_pass(self):
        c = CustomCheck("always_pass", lambda params, ctx: GuardResult(Severity.PASS))
        r = c.check({}, {})
        assert r.severity == Severity.PASS

    def test_custom_block(self):
        c = CustomCheck("always_block", lambda params, ctx:
            GuardResult(Severity.BLOCK, message="custom reason"))
        r = c.check({}, {})
        assert r.severity == Severity.BLOCK


# ═══════════════════════════════════════════════════════════════════════
# GuardrailSet
# ═══════════════════════════════════════════════════════════════════════

class TestGuardrailSet:
    def test_empty_set(self):
        gs = GuardrailSet()
        assert len(gs) == 0
        assert not bool(gs)
        r = gs.check_all({"x": 1})
        assert r.all_ok
        assert r.summary == "No guardrails defined"

    def test_all_pass(self):
        gs = GuardrailSet([
            RangeCheck("a", min_val=0),
            RangeCheck("b", min_val=0),
        ])
        r = gs.check_all({"a": 5, "b": 10})
        assert r.all_ok

    def test_one_blocks(self):
        gs = GuardrailSet([
            RangeCheck("a", min_val=0),
            RangeCheck("b", min_val=0, exclude_zero=True),
        ])
        r = gs.check_all({"a": 5, "b": 0})
        assert r.blocked
        assert not r.all_ok

    def test_fluent_add(self):
        gs = (GuardrailSet()
              .add_range("thrust", min_val=0, max_val=10000)
              .add_ratio("thrust", "mass", min_ratio=1)
              .add_custom("always", lambda p, c: GuardResult(Severity.PASS)))
        assert len(gs) == 3

    def test_guardrail_error_is_warn(self):
        class BrokenCheck(Guardrail):
            name = "broken"
            def check(self, params, context):
                raise RuntimeError("boom")

        gs = GuardrailSet([BrokenCheck()])
        r = gs.check_all({"x": 1})
        assert len(r.results) == 1
        assert r.results[0].severity == Severity.WARN
        assert "boom" in r.results[0].message


# ═══════════════════════════════════════════════════════════════════════
# Rocket KNSB Preset
# ═══════════════════════════════════════════════════════════════════════

class TestRocketKNSBPreset:
    def test_has_guardrails(self):
        gs = rocket_knsb_guardrails()
        assert len(gs) > 5

    def test_valid_params_pass(self):
        gs = rocket_knsb_guardrails()
        r = gs.check_all({
            "thrust_n": 1500,
            "isp_s": 155,
            "chamber_pressure_bar": 50,
            "mass_kg": 5.0,
            "throat_diameter_mm": 15,
        })
        assert r.all_ok

    def test_negative_thrust_blocked(self):
        gs = rocket_knsb_guardrails()
        r = gs.check_all({"thrust_n": -100, "isp_s": 150})
        assert r.blocked

    def test_isp_zero_blocked(self):
        gs = rocket_knsb_guardrails()
        r = gs.check_all({"thrust_n": 1500, "isp_s": 0})
        assert r.blocked

    def test_twr_insane_warned(self):
        """TWR 300:1 is physically impossible for amateur rocket."""
        gs = rocket_knsb_guardrails()
        r = gs.check_all({"thrust_n": 150000, "mass_kg": 5.0})
        # thrust_n=150000 above max 50000 → WARN
        assert any(rr.severity == Severity.WARN for rr in r.results
                   if "above maximum" in rr.message)


# ═══════════════════════════════════════════════════════════════════════
# Executor Integration
# ═══════════════════════════════════════════════════════════════════════

class TestExecutorGuardrails:
    def test_subtask_blocked_by_guardrails(self):
        """Executor with guardrails should mark subtask as failed on BLOCK."""
        from sigma.tau.types import SubTask, TaskGraph
        from sigma.tau.executor import IndependentExecutor

        gs = GuardrailSet([
            RangeCheck("thrust_n", min_val=0.1, exclude_zero=True),
        ])

        executor = IndependentExecutor(guardrails=gs)
        graph = TaskGraph(
            instruction="test",
            subtasks=[
                SubTask(id="st_0", description="分析", assigned_agents=["A"],
                        interface_params=["thrust_n"]),
            ],
        )
        agents = {
            "A": __import__("sigma.protocol", fromlist=["AgentSpec"]).AgentSpec(
                name="A", role="test", goal="test", backstory="test",
            ),
        }

        # Mock LLM returns a clearly invalid value
        def mock_llm(system, user):
            return "分析结果: PARAM:thrust_n=0"  # Zero thrust → BLOCK

        results = executor.run_all(graph, agents, {}, mock_llm, verbose=False)
        assert not results["st_0"].success
        gr = results["st_0"].guardrails_report
        assert gr is not None
        assert gr["blocked"] is True

    def test_subtask_passes_with_good_params(self):
        """Executor with guardrails passes on valid output."""
        from sigma.tau.types import SubTask, TaskGraph
        from sigma.tau.executor import IndependentExecutor

        gs = GuardrailSet([
            RangeCheck("thrust_n", min_val=0.1),
        ])

        executor = IndependentExecutor(guardrails=gs)
        graph = TaskGraph(
            instruction="test",
            subtasks=[
                SubTask(id="st_0", description="分析", assigned_agents=["A"],
                        interface_params=["thrust_n"]),
            ],
        )
        agents = {
            "A": __import__("sigma.protocol", fromlist=["AgentSpec"]).AgentSpec(
                name="A", role="test", goal="test", backstory="test",
            ),
        }

        def mock_llm(system, user):
            return "分析结果: PARAM:thrust_n=1500"

        results = executor.run_all(graph, agents, {}, mock_llm, verbose=False)
        assert results["st_0"].success
