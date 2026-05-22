"""Tests for trigger system — knowledge gaps, tool abnormalities, data deviations."""

import pytest
from sigma.triggers import Trigger, TriggerSystem


class TestTriggerDataclass:
    """Trigger data class."""

    def test_trigger_creation(self):
        t = Trigger(
            trigger_type="knowledge_gap", source="Agent A",
            message="missing data", severity=5.0, context={"key": "val"},
        )
        assert t.trigger_type == "knowledge_gap"
        assert t.source == "Agent A"
        assert t.severity == 5.0
        assert t.context == {"key": "val"}
        assert not t.handled

    def test_trigger_handled_flag(self):
        t = Trigger(
            trigger_type="tool_abnormal", source="tool1",
            message="fail", severity=8.0, context={}, handled=True,
        )
        assert t.handled


class TestKnowledgeGap:
    """Knowledge gap detection."""

    def test_detects_chinese_gap(self):
        ts = TriggerSystem()
        analyses = {"Agent A": "这个参数我不确定，需要查一下资料"}
        triggers = ts.check_knowledge_gap(analyses)
        assert len(triggers) >= 1
        assert any(t.trigger_type == "knowledge_gap" for t in triggers)

    def test_detects_english_gap(self):
        ts = TriggerSystem()
        analyses = {"Agent B": "I am uncertain about this value, need to verify"}
        triggers = ts.check_knowledge_gap(analyses)
        assert len(triggers) >= 1

    def test_no_gap_in_confident_text(self):
        ts = TriggerSystem()
        analyses = {"Agent A": "The thrust is 100N exactly, confirmed by test data"}
        triggers = ts.check_knowledge_gap(analyses)
        assert len(triggers) == 0

    def test_multiple_agents(self):
        ts = TriggerSystem()
        analyses = {
            "Agent A": "数据分析需要确认",
            "Agent B": "This is certain",
        }
        triggers = ts.check_knowledge_gap(analyses)
        assert len(triggers) == 1

    def test_snippet_in_context(self):
        ts = TriggerSystem()
        analyses = {"Agent A": "这个值不确定，建议搜索NASA CEA数据来验证"}
        triggers = ts.check_knowledge_gap(analyses)
        assert len(triggers) >= 1
        assert "snippet" in triggers[0].context


class TestToolAbnormal:
    """Tool abnormality detection."""

    def test_failed_tool(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", {
            "success": False, "error": "connection timeout",
        })
        assert len(triggers) == 1
        assert triggers[0].trigger_type == "tool_abnormal"
        assert triggers[0].severity == 8.0

    def test_simulated_result_not_abnormal(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", {
            "success": "simulated", "performance": {"Isp_s": 200.0},
        })
        assert len(triggers) == 0

    def test_out_of_range_value(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", {
            "success": True, "performance": {"Isp_s": 9999.0},
        })
        assert len(triggers) == 1
        assert "异常值" in triggers[0].message

    def test_in_range_value(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", {
            "success": True, "performance": {"Isp_s": 200.0},
        })
        assert len(triggers) == 0

    def test_unknown_key_no_range_check(self):
        """Keys without a matching range prefix are skipped."""
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", {
            "success": True, "performance": {"unknown_key": 99999.0},
        })
        assert len(triggers) == 0

    def test_non_numeric_performance_skipped(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", {
            "success": True, "performance": {"Isp_s": "not a number"},
        })
        assert len(triggers) == 0

    def test_non_dict_result(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("sim", "not a dict")
        assert len(triggers) == 0

    def test_mass_out_of_range(self):
        ts = TriggerSystem()
        triggers = ts.check_tool_abnormal("tool", {
            "success": True, "performance": {"mass_kg": -5.0},
        })
        assert len(triggers) == 1


class TestDataDeviation:
    """Data deviation detection."""

    def test_large_deviation(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"Isp": 200.0}, actual={"Isp": 280.0},
        )
        assert len(triggers) == 1
        assert triggers[0].trigger_type == "data_deviation"

    def test_small_deviation(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"Isp": 200.0}, actual={"Isp": 210.0},
        )
        assert len(triggers) == 0

    def test_boundary_30_percent(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"Isp": 200.0}, actual={"Isp": 260.0},  # 30%
        )
        assert len(triggers) == 0  # > 0.30, so exactly 0.30 is not flagged

    def test_just_above_30_percent(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"Isp": 200.0}, actual={"Isp": 260.1},
        )
        assert len(triggers) == 1

    def test_key_not_in_actual(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"Isp": 200.0, "mass": 10.0},
            actual={"Isp": 210.0},
        )
        assert len(triggers) == 0

    def test_non_numeric_skipped(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"name": "aluminum"}, actual={"name": "steel"},
        )
        assert len(triggers) == 0

    def test_rate_zero_prev(self):
        ts = TriggerSystem()
        triggers = ts.check_data_deviation(
            expected={"Isp": 0.0}, actual={"Isp": 10.0},
        )
        assert len(triggers) == 1  # denominator uses 0.001, so 10/0.001 = 10000 > 0.30


class TestTriggerHelpers:
    """needs_skill_crafter and needs_tool_retry."""

    def test_needs_skill_crafter_true(self):
        ts = TriggerSystem()
        triggers = [
            Trigger("knowledge_gap", "A", "gap", severity=5.0, context={}),
        ]
        assert ts.needs_skill_crafter(triggers)

    def test_needs_skill_crafter_low_severity(self):
        ts = TriggerSystem()
        triggers = [
            Trigger("knowledge_gap", "A", "gap", severity=1.0, context={}),
        ]
        assert not ts.needs_skill_crafter(triggers)

    def test_needs_skill_crafter_wrong_type(self):
        ts = TriggerSystem()
        triggers = [
            Trigger("tool_abnormal", "tool", "fail", severity=8.0, context={}),
        ]
        assert not ts.needs_skill_crafter(triggers)

    def test_needs_tool_retry(self):
        ts = TriggerSystem()
        triggers = [
            Trigger("tool_abnormal", "sim", "fail", severity=8.0, context={}),
            Trigger("knowledge_gap", "A", "gap", severity=5.0, context={}),
        ]
        retry = ts.needs_tool_retry(triggers)
        assert retry == ["sim"]

    def test_needs_tool_retry_empty(self):
        ts = TriggerSystem()
        assert ts.needs_tool_retry([]) == []
