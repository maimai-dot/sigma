"""Tests for convergence judge — numeric convergence, oscillation detection, verdicts."""

import pytest
from sigma.convergence import ConvergenceJudge, Verdict, JudgeResult
from sigma.state import SharedState, StateManager, Conflict


def make_state(**overrides) -> SharedState:
    s = SharedState(task_instruction="test")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_conflict(conflict_id: str, severity: float, owners=None) -> Conflict:
    return Conflict(
        id=conflict_id,
        description=f"conflict {conflict_id}",
        owners=owners or ["Agent A", "Agent B"],
        severity=severity,
        trend="new",
        quantitative=True,
        values={},
    )


class TestQuantitativeConvergence:
    """Numeric parameter convergence checks."""

    def test_all_converged(self):
        judge = ConvergenceJudge()
        prev = make_state(task_params={"mass_kg": 10.0, "Isp_s": 200.0})
        curr = make_state(task_params={"mass_kg": 10.2, "Isp_s": 205.0})
        # 10.0→10.2 Δ2%, 200→205 Δ2.5% — both < 5%
        converged, remaining = judge._check_quantitative(prev, curr)
        assert len(converged) == 2
        assert len(remaining) == 0

    def test_one_converged_one_not(self):
        judge = ConvergenceJudge()
        prev = make_state(task_params={"mass_kg": 10.0, "Isp_s": 200.0})
        curr = make_state(task_params={"mass_kg": 10.9, "Isp_s": 250.0})
        # 10.0→10.9 Δ9%, 200→250 Δ25%
        converged, remaining = judge._check_quantitative(prev, curr)
        assert len(converged) == 0
        assert len(remaining) == 2

    def test_threadhold_boundary(self):
        judge = ConvergenceJudge()
        prev = make_state(task_params={"x": 100.0})
        curr = make_state(task_params={"x": 104.9})  # Δ4.9% < 5%
        converged, _ = judge._check_quantitative(prev, curr)
        assert len(converged) == 1

        curr2 = make_state(task_params={"x": 105.1})  # Δ5.1% >= 5%
        _, remaining = judge._check_quantitative(prev, curr2)
        assert len(remaining) == 1

    def test_new_param_skipped(self):
        judge = ConvergenceJudge()
        prev = make_state(task_params={"mass_kg": 10.0})
        curr = make_state(task_params={"mass_kg": 10.2, "new_param": 5.0})
        converged, remaining = judge._check_quantitative(prev, curr)
        assert len(converged) == 1  # mass_kg only
        assert len(remaining) == 0  # new_param skipped (no prev value)

    def test_non_numeric_skipped(self):
        judge = ConvergenceJudge()
        prev = make_state(task_params={"name": "aluminum", "count": 5})
        curr = make_state(task_params={"name": "steel", "count": 5})
        converged, remaining = judge._check_quantitative(prev, curr)
        assert len(converged) == 1  # count: 5→5 Δ0%
        assert len(remaining) == 0  # name skipped (str)


class TestQualitativeConvergence:
    """Qualitative conflict convergence checks."""

    def test_conflict_resolved_when_gone(self):
        judge = ConvergenceJudge()
        prev = make_state()
        prev.active_conflicts = [
            make_conflict("c1", 7.0), make_conflict("c2", 3.0),
        ]
        curr = make_state()
        curr.active_conflicts = [make_conflict("c2", 2.0)]
        resolved, remaining = judge._check_qualitative(prev, curr)
        assert "c1" in resolved
        assert len(resolved) == 1

    def test_severity_drops_below_threshold(self):
        judge = ConvergenceJudge()
        prev = make_state()
        prev.active_conflicts = [make_conflict("c1", 5.0)]
        curr = make_state()
        curr.active_conflicts = [make_conflict("c1", 1.0)]  # < 2.0
        resolved, remaining = judge._check_qualitative(prev, curr)
        assert len(resolved) == 1
        assert len(remaining) == 0

    def test_severity_stays_high(self):
        judge = ConvergenceJudge()
        prev = make_state()
        prev.active_conflicts = [make_conflict("c1", 5.0)]
        curr = make_state()
        curr.active_conflicts = [make_conflict("c1", 4.0)]  # still >= 2.0
        resolved, remaining = judge._check_qualitative(prev, curr)
        assert len(resolved) == 0
        assert len(remaining) == 1

    def test_severity_not_shrinking(self):
        judge = ConvergenceJudge()
        prev = make_state()
        prev.active_conflicts = [make_conflict("c1", 3.0)]
        curr = make_state()
        curr.active_conflicts = [make_conflict("c1", 5.0)]  # grew
        resolved, remaining = judge._check_qualitative(prev, curr)
        assert len(resolved) == 0
        assert len(remaining) == 1


class TestOscillationDetection:
    """Oscillation detection logic."""

    def test_no_oscillation_with_improvement(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": [{"severity": 10.0}]},
            {"entries": [{"severity": 7.0}]},
            {"entries": [{"severity": 4.0}]},
        ])
        assert not judge.detect_oscillation(state)

    def test_oscillation_detected(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": [{"severity": 5.0}]},
            {"entries": [{"severity": 6.0}]},
            {"entries": [{"severity": 7.0}]},
        ])
        assert judge.detect_oscillation(state)

    def test_stall_detected(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": [{"severity": 5.0}]},
            {"entries": [{"severity": 5.0}]},
            {"entries": [{"severity": 5.0}]},
        ])
        assert judge.detect_oscillation(state)

    def test_insufficient_data(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": [{"severity": 5.0}]},
        ])
        assert not judge.detect_oscillation(state)


class TestConvergenceRate:
    """Convergence rate calculation."""

    def test_rate_with_improvement(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": [{"severity": 10.0}]},
            {"entries": [{"severity": 7.0}]},
        ])
        assert judge.convergence_rate(state) == 0.3  # (10-7)/10

    def test_rate_first_round(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": [{"severity": 5.0}]},
        ])
        assert judge.convergence_rate(state) == 1.0

    def test_rate_zero_prev(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": []},
            {"entries": [{"severity": 5.0}]},
        ])
        assert judge.convergence_rate(state) == 0.0

    def test_rate_zero_prev_and_curr(self):
        judge = ConvergenceJudge()
        state = make_state(convergence_log=[
            {"entries": []},
            {"entries": []},
        ])
        assert judge.convergence_rate(state) == 1.0  # both zero


class TestJudgeVerdict:
    """Full judge verdict tests."""

    def test_full_convergence(self):
        judge = ConvergenceJudge()
        prev = make_state(task_params={"x": 10.0, "y": 20.0})
        curr = make_state(
            task_params={"x": 10.1, "y": 20.2},
            convergence_log=[
                {"entries": [{"severity": 1.0}]},
                {"entries": [{"severity": 0.5}]},
            ],
        )
        result = judge.judge(prev, curr)
        assert result.verdict == Verdict.CONVERGED
        assert result.should_stop

    def test_oscillation_trumps(self):
        judge = ConvergenceJudge()
        prev = make_state(active_conflicts=[make_conflict("c1", 5.0)])
        curr = make_state(
            active_conflicts=[make_conflict("c1", 6.0)],
            convergence_log=[
                {"entries": [{"severity": 2.0}]},
                {"entries": [{"severity": 3.0}]},
                {"entries": [{"severity": 4.0}]},
            ],
        )
        result = judge.judge(prev, curr)
        assert result.verdict == Verdict.OSCILLATING
        assert result.should_stop

    def test_max_rounds_stall(self):
        judge = ConvergenceJudge()
        prev = make_state(active_conflicts=[make_conflict("c1", 5.0)])
        curr = make_state(
            round_num=4, max_rounds=4,
            active_conflicts=[make_conflict("c1", 5.0)],
            convergence_log=[{"entries": [{"severity": 5.0}]}],
        )
        result = judge.judge(prev, curr)
        assert result.verdict == Verdict.STALLED
        assert result.should_stop

    def test_slow_convergence(self):
        judge = ConvergenceJudge()
        prev = make_state(active_conflicts=[make_conflict("c1", 10.0)])
        curr = make_state(
            round_num=1, max_rounds=4,
            active_conflicts=[make_conflict("c1", 9.5)],
            convergence_log=[
                {"entries": [{"severity": 10.0}]},
                {"entries": [{"severity": 9.5}]},
            ],
        )
        result = judge.judge(prev, curr)
        assert result.verdict == Verdict.SLOW  # 5% rate < 10%
        assert not result.should_stop

    def test_normal_converging(self):
        judge = ConvergenceJudge()
        prev = make_state(active_conflicts=[make_conflict("c1", 10.0)])
        curr = make_state(
            round_num=1, max_rounds=4,
            active_conflicts=[make_conflict("c1", 5.0)],
            convergence_log=[
                {"entries": [{"severity": 10.0}]},
                {"entries": [{"severity": 5.0}]},
            ],
        )
        result = judge.judge(prev, curr)
        assert result.verdict == Verdict.CONVERGING  # 50% rate >= 10%
        assert not result.should_stop


class TestPhysicalImpossible:
    """Physical impossibility detection."""

    def test_detects_physical_claim(self):
        judge = ConvergenceJudge()
        analyses = {
            "Agent A": "这在物理上不可能实现，违反了热力学定律",
            "Agent B": "这个方案是可行的",
        }
        alarms = judge.check_physical_impossible(analyses)
        assert len(alarms) == 1
        assert alarms[0]["agent"] == "Agent A"
        assert alarms[0]["flag_type"] == "physical_limit"

    def test_no_false_positive(self):
        judge = ConvergenceJudge()
        analyses = {"Agent A": "这个设计完全可行，没有问题"}
        alarms = judge.check_physical_impossible(analyses)
        assert len(alarms) == 0

    def test_english_keywords(self):
        judge = ConvergenceJudge()
        analyses = {"Agent B": "This is physically impossible under current constraints"}
        alarms = judge.check_physical_impossible(analyses)
        assert len(alarms) == 1
