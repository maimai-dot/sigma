"""Tests for shared state — data classes, StateManager, serialization, delta."""

import json
import tempfile
from pathlib import Path

import pytest
from sigma.state import (
    SharedState,
    StateManager,
    Conflict,
    Decision,
    AlarmFlag,
    ConsensusEstimate,
    RoundRecord,
    ComplexityAssessment,
    ComplexityTier,
)


def make_conflict(conflict_id="c1", severity=5.0, owners=None) -> Conflict:
    return Conflict(
        id=conflict_id,
        description=f"conflict {conflict_id}",
        owners=owners or ["Agent A", "Agent B"],
        severity=severity,
        trend="new",
        quantitative=True,
        values={},
    )


class TestSharedState:
    """SharedState creation and defaults."""

    def test_defaults(self):
        s = SharedState(task_instruction="test")
        assert s.task_instruction == "test"
        assert s.task_params == {}
        assert s.round_num == 0
        assert s.max_rounds == 4
        assert s.complexity_tier == "standard"
        assert s.complexity_assessment == {}
        assert s.convergence_log == []
        assert s.decisions == []
        assert s.alarm_flags == []
        assert s.dependency_graph == {}
        assert s.active_conflicts == []
        assert s.cost_summary == {
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "calls": 0,
        }
        assert s.history == []

    def test_field_override(self):
        s = SharedState(
            task_instruction="design engine",
            task_params={"mass_kg": 10.0},
            round_num=2,
            max_rounds=3,
            complexity_tier="rigorous",
        )
        assert s.task_params == {"mass_kg": 10.0}
        assert s.round_num == 2
        assert s.max_rounds == 3
        assert s.complexity_tier == "rigorous"


class TestSharedStateToContext:
    """to_context() serialization."""

    def test_minimal_context(self):
        s = SharedState(task_instruction="test")
        ctx = s.to_context()
        assert "## 共享工程状态" in ctx
        assert "任务: test" in ctx
        assert "当前轮次: 0/4" in ctx

    def test_context_with_params(self):
        s = SharedState(
            task_instruction="test",
            task_params={"mass_kg": 10.0, "Isp_s": 200.0},
        )
        ctx = s.to_context()
        assert "设计参数" in ctx
        assert "mass_kg: 10.0" in ctx
        assert "Isp_s: 200.0" in ctx

    def test_context_with_decisions(self):
        s = SharedState(task_instruction="test")
        s.decisions = [
            Decision(round_num=1, domain="propulsion", decision="Use KNSB",
                     reason="safe for amateur", made_by="Propulsion Chief"),
        ]
        ctx = s.to_context()
        assert "已做决策" in ctx
        assert "KNSB" in ctx

    def test_context_with_conflicts(self):
        s = SharedState(task_instruction="test")
        s.active_conflicts = [make_conflict("c1", 7.0)]
        ctx = s.to_context()
        assert "当前冲突" in ctx
        assert "严重度: 7.0" in ctx

    def test_context_with_alarms(self):
        s = SharedState(task_instruction="test")
        s.alarm_flags = [
            AlarmFlag(flag_type="physical_limit", message="violates physics", round_num=1),
        ]
        ctx = s.to_context()
        assert "告警" in ctx
        assert "physical_limit" in ctx

    def test_context_resolved_alarm_skipped(self):
        s = SharedState(task_instruction="test")
        s.alarm_flags = [
            AlarmFlag(flag_type="physical_limit", message="violates physics",
                      round_num=1, resolved=True),
        ]
        ctx = s.to_context()
        assert "physical_limit" not in ctx

    def test_context_truncates_decisions_to_5(self):
        s = SharedState(task_instruction="test")
        s.decisions = [
            Decision(round_num=i, domain="d", decision=f"dec{i}",
                     reason="r", made_by="Agent")
            for i in range(10)
        ]
        ctx = s.to_context()
        assert "[R0]" not in ctx
        assert "[R9]" in ctx


class TestDataClasses:
    """Data class construction and attributes."""

    def test_conflict_creation(self):
        c = Conflict(
            id="c1", description="thrust mismatch", owners=["A", "B"],
            severity=7.0, trend="new", quantitative=True, values={1: 100, 2: 95},
        )
        assert c.id == "c1"
        assert c.severity == 7.0
        assert c.trend == "new"
        assert c.quantitative is True
        assert c.values == {1: 100, 2: 95}

    def test_decision_creation(self):
        d = Decision(
            round_num=2, domain="structures", decision="Use 150mm tube",
            reason="strength sufficient", made_by="Structures Chief",
        )
        assert d.round_num == 2
        assert d.domain == "structures"

    def test_alarm_flag_creation(self):
        a = AlarmFlag(
            flag_type="tool_failure", message="FreeCAD crashed",
            round_num=1, resolved=False,
        )
        assert a.flag_type == "tool_failure"
        assert not a.resolved

    def test_consensus_estimate_creation(self):
        ce = ConsensusEstimate(
            parameter="mass_kg", min_val=4.5, max_val=6.0, recommended=5.2,
            confidence="MEDIUM", unit="kg", basis="3 agents estimated",
            individual={
                "Agent A": {"value": 5.0, "reasoning": "calc", "confidence": "HIGH"},
                "Agent B": {"value": 5.5, "reasoning": "exp", "confidence": "MEDIUM"},
            },
        )
        assert ce.recommended == 5.2
        assert ce.confidence == "MEDIUM"
        assert len(ce.individual) == 2

    def test_round_record_creation(self):
        rr = RoundRecord(
            round_num=1, timestamp="2026-05-19T10:00:00",
            agent_analyses={"Agent A": "analysis text"},
            tool_results={"tool1": {"success": True}},
        )
        assert rr.round_num == 1
        assert rr.token_count == 0
        assert rr.conflicts == []

    def test_complexity_assessment_creation(self):
        ca = ComplexityAssessment(
            tier=ComplexityTier.RIGOROUS, score=8.5,
            reason="multi-domain with safety constraints",
            agent_names=["Director", "Propulsion Chief", "Structures Chief"],
            max_rounds=4, cross_review=True, devil_advocate=True,
            consensus_estimation=True, skill_crafter=False,
        )
        assert ca.tier == ComplexityTier.RIGOROUS
        assert ca.score == 8.5
        assert ca.cross_review is True


class TestStateManagerInit:
    """StateManager.init()."""

    def test_init_creates_state(self):
        mgr = StateManager()
        state = mgr.init("design a rocket")
        assert state.task_instruction == "design a rocket"
        assert state.round_num == 0
        assert state.history == []

    def test_init_independent_states(self):
        mgr = StateManager()
        s1 = mgr.init("task A")
        s2 = mgr.init("task B")
        assert s1.task_instruction == "task A"
        assert s2.task_instruction == "task B"
        assert s1 is not s2


class TestStateManagerStartRound:
    """StateManager.start_round()."""

    def test_increments_round(self):
        mgr = StateManager()
        s = mgr.init("test")
        s2 = mgr.start_round(s)
        assert s2.round_num == 1
        assert s.round_num == 0  # original unchanged

    def test_appends_round_record(self):
        mgr = StateManager()
        s = mgr.init("test")
        s2 = mgr.start_round(s)
        assert len(s2.history) == 1
        assert s2.history[0].round_num == 1

    def test_multiple_rounds(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s = mgr.start_round(s)
        s = mgr.start_round(s)
        assert s.round_num == 3
        assert len(s.history) == 3


class TestStateManagerUpdatePlan:
    """StateManager.update_after_plan()."""

    def test_updates_conflicts(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        conflicts = [make_conflict("c1", 5.0)]
        s = mgr.update_after_plan(
            s, analyses={"A": "ok"}, conflicts=conflicts,
            dependency_graph={"A": ["B"]}, tool_requests=["tool1"],
        )
        assert s.active_conflicts == conflicts
        assert s.dependency_graph == {"A": ["B"]}
        assert s.history[-1].conflicts == conflicts
        assert s.history[-1].phase_outputs["plan"]["tool_requests"] == ["tool1"]

    def test_convergence_log_appended(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        conflicts = [make_conflict("c1", 5.0)]
        s = mgr.update_after_plan(
            s, analyses={}, conflicts=conflicts,
            dependency_graph={}, tool_requests=[],
        )
        assert len(s.convergence_log) == 1
        assert s.convergence_log[0]["conflict_id"] == "c1"
        assert s.convergence_log[0]["entries"][0]["severity"] == 5.0

    def test_convergence_log_existing_conflict(self):
        mgr = StateManager()
        s = mgr.init("test")
        s.convergence_log = [{
            "conflict_id": "c1", "description": "conflict c1",
            "entries": [{"round": 1, "severity": 7.0}], "trend": "new",
        }]
        s = mgr.start_round(s)
        conflicts = [make_conflict("c1", 5.0)]
        s = mgr.update_after_plan(
            s, analyses={}, conflicts=conflicts,
            dependency_graph={}, tool_requests=[],
        )
        assert len(s.convergence_log[0]["entries"]) == 2
        assert s.convergence_log[0]["entries"][1]["severity"] == 5.0


class TestStateManagerUpdateDo:
    """StateManager.update_after_do()."""

    def test_stores_tool_results(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        results = {"tool1": {"success": True, "performance": {"mass_kg": 5.0}}}
        s = mgr.update_after_do(s, results)
        assert s.history[-1].tool_results == results

    def test_writes_performance_to_task_params(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        results = {"sim": {"success": True, "performance": {"Isp_s": 200.0}}}
        s = mgr.update_after_do(s, results)
        assert s.task_params["sim_Isp_s"] == 200.0

    def test_skips_failed_results(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        results = {"sim": {"success": False, "error": "crashed"}}
        s = mgr.update_after_do(s, results)
        assert "sim_" not in str(s.task_params)


class TestStateManagerUpdateCheck:
    """StateManager.update_after_check()."""

    def test_updates_conflicts_and_reviews(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        conflicts = [make_conflict("c1", 3.0)]
        s = mgr.update_after_check(
            s, reviews={"A": "review"}, updated_conflicts=conflicts,
            devil_advocate="no issues",
        )
        assert s.active_conflicts == conflicts
        assert s.history[-1].cross_review == {"A": "review"}
        assert s.history[-1].phase_outputs["check"]["devil_advocate"] == "no issues"


class TestStateManagerUpdateAct:
    """StateManager.update_after_act()."""

    def test_extends_decisions_and_alarms(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        decisions = [Decision(round_num=1, domain="d", decision="dec",
                              reason="r", made_by="Agent")]
        alarms = [AlarmFlag(flag_type="oscillation", message="osc", round_num=1)]
        s = mgr.update_after_act(s, decisions, alarms, token_count=1000,
                                 estimated_cost=0.002)
        assert s.decisions == decisions
        assert s.alarm_flags == alarms

    def test_accumulates_cost(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s = mgr.update_after_act(s, [], [], token_count=500, estimated_cost=0.001)
        assert s.cost_summary["total_tokens"] == 500
        assert s.cost_summary["estimated_cost"] == 0.001
        assert s.cost_summary["calls"] == 1

        s = mgr.start_round(s)
        s = mgr.update_after_act(s, [], [], token_count=300, estimated_cost=0.0006)
        assert s.cost_summary["total_tokens"] == 800
        assert s.cost_summary["estimated_cost"] == pytest.approx(0.0016)
        assert s.cost_summary["calls"] == 2

    def test_stores_in_round_record(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        decisions = [Decision(round_num=1, domain="d", decision="dec",
                              reason="r", made_by="A")]
        s = mgr.update_after_act(s, decisions, [], token_count=100,
                                 estimated_cost=0.0001)
        record = s.history[-1]
        assert record.decisions == decisions
        assert record.token_count == 100
        assert record.phase_outputs["act"]["decisions"] == ["dec"]


class TestStateManagerSummary:
    """StateManager.summary()."""

    def test_empty_state(self):
        mgr = StateManager()
        s = mgr.init("test")
        assert mgr.summary(s) == ""

    def test_summary_after_round(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s.history[-1].agent_analyses = {"Agent A": "found issue with thrust"}
        s.history[-1].tool_results = {
            "sim": {"performance": {"Isp_s": 200.0}},
        }
        summary = mgr.summary(s)
        assert "Round 1" in summary
        assert "Agent A" in summary
        assert "Isp_s=200.0" in summary

    def test_summary_with_conflicts(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s.history[-1].conflicts = [make_conflict("c1", 8.0)]
        summary = mgr.summary(s)
        assert "冲突" in summary
        assert "8.0" in summary

    def test_summary_with_decisions(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s.history[-1].decisions = [
            Decision(round_num=1, domain="d", decision="pick aluminum",
                     reason="r", made_by="A"),
        ]
        summary = mgr.summary(s)
        assert "pick aluminum" in summary

    def test_summary_respects_max_chars(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s.history[-1].agent_analyses = {"Agent A": "x" * 200}
        summary = mgr.summary(s, max_chars=50)
        assert len(summary) <= 50


class TestStateManagerDelta:
    """StateManager.delta()."""

    def test_no_changes(self):
        mgr = StateManager()
        s1 = mgr.init("test")
        s2 = mgr.init("test")
        assert mgr.delta(s1, s2) == "无变化"

    def test_param_change(self):
        mgr = StateManager()
        s1 = SharedState(task_instruction="test", task_params={"x": 10.0})
        s2 = SharedState(task_instruction="test", task_params={"x": 11.0})
        delta = mgr.delta(s1, s2)
        assert "x: 10.0 → 11.0" in delta
        assert "Δ10.0%" in delta

    def test_new_param(self):
        mgr = StateManager()
        s1 = SharedState(task_instruction="test", task_params={})
        s2 = SharedState(task_instruction="test", task_params={"y": 5.0})
        delta = mgr.delta(s1, s2)
        assert "y: None → 5.0" in delta

    def test_conflict_resolved(self):
        mgr = StateManager()
        s1 = SharedState(task_instruction="test",
                         active_conflicts=[make_conflict("c1", 5.0)])
        s2 = SharedState(task_instruction="test", active_conflicts=[])
        delta = mgr.delta(s1, s2)
        assert "已解决" in delta

    def test_conflict_new(self):
        mgr = StateManager()
        s1 = SharedState(task_instruction="test", active_conflicts=[])
        s2 = SharedState(task_instruction="test",
                         active_conflicts=[make_conflict("c1", 5.0)])
        delta = mgr.delta(s1, s2)
        assert "新冲突" in delta

    def test_conflict_severity_change(self):
        mgr = StateManager()
        s1 = SharedState(task_instruction="test",
                         active_conflicts=[make_conflict("c1", 5.0)])
        s2 = SharedState(task_instruction="test",
                         active_conflicts=[make_conflict("c1", 3.0)])
        delta = mgr.delta(s1, s2)
        assert "↓" in delta


class TestStateManagerSaveRound:
    """StateManager.save_round()."""

    def test_saves_round_files(self):
        mgr = StateManager()
        s = mgr.init("test")
        s = mgr.start_round(s)
        s.history[-1].agent_analyses = {"Agent A": "analysis"}
        s.history[-1].tool_results = {"tool1": {"success": True}}
        s.history[-1].conflicts = [make_conflict("c1", 3.0)]
        s.history[-1].decisions = [
            Decision(round_num=1, domain="d", decision="dec",
                     reason="r", made_by="A"),
        ]
        s.history[-1].alarm_flags = [
            AlarmFlag(flag_type="test", message="msg", round_num=1),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round_1"
            mgr.save_round(s, round_dir)

            assert (round_dir / "record.json").exists()
            assert (round_dir / "state.json").exists()

            record = json.loads((round_dir / "record.json").read_text(encoding="utf-8"))
            assert record["round_num"] == 1
            assert len(record["conflicts"]) == 1
            assert len(record["decisions"]) == 1

            snap = json.loads((round_dir / "state.json").read_text(encoding="utf-8"))
            assert snap["task_instruction"] == "test"


class TestComplexityTier:
    """ComplexityTier enum."""

    def test_tier_values(self):
        assert ComplexityTier.LITE.value == "lite"
        assert ComplexityTier.STANDARD.value == "standard"
        assert ComplexityTier.RIGOROUS.value == "rigorous"

    def test_tier_from_value(self):
        assert ComplexityTier("lite") == ComplexityTier.LITE
        assert ComplexityTier("rigorous") == ComplexityTier.RIGOROUS


class TestCheckpointRestore:
    """Checkpoint and restore — full state roundtrip."""

    @pytest.fixture
    def populated_state(self):
        from sigma.provenance import AuditTrail, ProvenanceEntry
        from datetime import datetime
        state = SharedState(task_instruction="测试任务")
        state.round_num = 2
        state.task_params = {"thrust_n": 1500, "mass_kg": 5.0}
        state.convergence_log = [{"conflict_id": "c1", "description": "推力估算不一致", "entries": []}]
        state.decisions = [Decision(round_num=1, domain="推进", decision="使用KNSB", reason="成熟方案", made_by="Director")]
        state.alarm_flags = [AlarmFlag(flag_type="tool_failure", message="mass_calc 模拟结果", round_num=1, resolved=True)]
        state.active_conflicts = [Conflict(id="c1", description="推力估算不一致", owners=["Propulsion", "Sim"],
                                           severity=3.0, trend="shrinking", quantitative=True, values={1: 5.0, 2: 3.0})]
        state.cost_summary = {"total_tokens": 5000, "estimated_cost": 0.03, "calls": 2}
        state.history = [
            RoundRecord(round_num=1, timestamp=datetime.now().isoformat(),
                       agent_analyses={"A": "analysis1"}, tool_results={"t": {"v": 1}},
                       conflicts=[], decisions=[], alarm_flags=[]),
            RoundRecord(round_num=2, timestamp=datetime.now().isoformat(),
                       agent_analyses={"B": "analysis2"}, tool_results={},
                       conflicts=[], decisions=[], alarm_flags=[]),
        ]
        state.audit_trail.add("thrust_n", 1500, "tool", "thrust_calc", 1, "ok", "HIGH")
        return state

    def test_checkpoint_writes_file(self, populated_state, tmp_path):
        sm = StateManager()
        path = tmp_path / "checkpoint.json"
        sm.checkpoint(populated_state, path)
        assert path.exists()

    def test_restore_recovers_all_fields(self, populated_state, tmp_path):
        sm = StateManager()
        path = tmp_path / "checkpoint.json"
        sm.checkpoint(populated_state, path)
        restored = sm.restore(path)
        assert restored.task_instruction == populated_state.task_instruction
        assert restored.round_num == populated_state.round_num
        assert restored.task_params == populated_state.task_params
        assert len(restored.decisions) == len(populated_state.decisions)
        assert restored.decisions[0].domain == "推进"
        assert len(restored.alarm_flags) == len(populated_state.alarm_flags)
        assert len(restored.active_conflicts) == 1
        assert restored.active_conflicts[0].id == "c1"
        assert len(restored.history) == 2
        assert len(restored.audit_trail) == 1

    def test_restore_preserves_audit_trail(self, populated_state, tmp_path):
        sm = StateManager()
        path = tmp_path / "checkpoint.json"
        sm.checkpoint(populated_state, path)
        restored = sm.restore(path)
        entries = restored.audit_trail.trace("thrust_n")
        assert len(entries) == 1
        assert entries[0].value == 1500
        assert entries[0].source_type == "tool"

    def test_restore_empty_state(self, tmp_path):
        sm = StateManager()
        path = tmp_path / "empty.json"
        sm.checkpoint(SharedState(task_instruction="empty"), path)
        restored = sm.restore(path)
        assert restored.task_instruction == "empty"
        assert restored.round_num == 0
        assert len(restored.audit_trail) == 0
