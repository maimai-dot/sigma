"""Tests for cost tracking — RoundCost, CostTracker, pricing calculations."""

import pytest
from sigma.cost_tracker import (
    RoundCost,
    CostTracker,
    DEEPSEEK_PRICING,
)


class TestRoundCost:
    """RoundCost dataclass."""

    def test_defaults(self):
        rc = RoundCost(round_num=1)
        assert rc.round_num == 1
        assert rc.input_tokens == 0
        assert rc.output_tokens == 0
        assert rc.calls == 0
        assert rc.estimated_cost_rmb == 0.0
        assert rc.total_tokens == 0

    def test_total_tokens(self):
        rc = RoundCost(round_num=1, input_tokens=500, output_tokens=300)
        assert rc.total_tokens == 800

    def test_field_assignment(self):
        rc = RoundCost(round_num=2)
        rc.input_tokens = 1000
        rc.output_tokens = 500
        rc.calls = 3
        rc.estimated_cost_rmb = 0.005
        assert rc.total_tokens == 1500


class TestCostTrackerInit:
    """CostTracker initialization."""

    def test_default_pricing(self):
        ct = CostTracker()
        assert ct.pricing == DEEPSEEK_PRICING
        assert ct.rounds == []

    def test_custom_pricing(self):
        custom = {"input": 2.0, "output": 8.0, "cached_input": 0.5}
        ct = CostTracker(pricing=custom)
        assert ct.pricing == custom


class TestCostTrackerRound:
    """CostTracker round lifecycle."""

    def test_start_round(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        assert cost.round_num == 1
        assert len(ct.rounds) == 1
        assert ct.rounds[0] is cost

    def test_multiple_rounds(self):
        ct = CostTracker()
        r1 = ct.start_round(1)
        r2 = ct.start_round(2)
        r3 = ct.start_round(3)
        assert len(ct.rounds) == 3
        assert ct.rounds == [r1, r2, r3]


class TestCostTrackerRecordCall:
    """record_call and record_estimated_call."""

    def test_record_call_accumulates(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_call(cost, input_tokens=500, output_tokens=200)
        assert cost.input_tokens == 500
        assert cost.output_tokens == 200
        assert cost.calls == 1

    def test_record_call_cost_calculation(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_call(cost, input_tokens=1_000_000, output_tokens=1_000_000)
        # input: 1M * ¥1/M = ¥1, output: 1M * ¥4/M = ¥4, total = ¥5
        assert cost.estimated_cost_rmb == pytest.approx(5.0)

    def test_record_call_zero_tokens(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_call(cost, input_tokens=0, output_tokens=0)
        assert cost.estimated_cost_rmb == 0.0
        assert cost.calls == 1

    def test_record_estimated_call(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_estimated_call(cost, input_tokens=1000, output_tokens=500)
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500
        assert cost.calls == 1

    def test_multiple_calls_accumulate(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_call(cost, input_tokens=100, output_tokens=50)
        ct.record_call(cost, input_tokens=200, output_tokens=100)
        assert cost.input_tokens == 300
        assert cost.output_tokens == 150
        assert cost.calls == 2

    def test_small_tokens_cost(self):
        """Verify cost for small token counts is not zero."""
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_call(cost, input_tokens=1000, output_tokens=1000)
        # input: 1000/1M * 1 = 0.001, output: 1000/1M * 4 = 0.004, total = 0.005
        assert cost.estimated_cost_rmb == pytest.approx(0.005)


class TestCostTrackerSummary:
    """Round and total summaries."""

    def test_round_summary(self):
        ct = CostTracker()
        cost = ct.start_round(1)
        ct.record_call(cost, input_tokens=5000, output_tokens=3000)
        summary = ct.round_summary(cost)
        assert "8,000" in summary or "8000" in summary
        assert "¥" in summary
        assert "调用 1 次" in summary

    def test_total_summary(self):
        ct = CostTracker()
        r1 = ct.start_round(1)
        ct.record_call(r1, input_tokens=1000, output_tokens=500)
        r2 = ct.start_round(2)
        ct.record_call(r2, input_tokens=2000, output_tokens=1000)
        summary = ct.total_summary()
        assert "4,500" in summary  # 1000+500+2000+1000
        assert "¥" in summary
        assert "LLM 调用: 2 次" in summary
        assert "轮次: 2" in summary

    def test_empty_summary(self):
        ct = CostTracker()
        summary = ct.total_summary()
        assert "总 Token: 0" in summary
