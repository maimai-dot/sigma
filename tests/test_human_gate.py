"""Tests for HumanGate — Founder-in-the-loop gate at AERC phase transitions."""

import pytest
from unittest import mock

from sigma.human_gate import HumanGate, GateDecision, GateAction


# ── GateDecision ─────────────────────────────────────────────────────

class TestGateDecision:
    def test_approve_continues(self):
        d = GateDecision(action=GateAction.APPROVE)
        assert d.should_continue is True
        assert d.should_retry is False

    def test_revise_retries(self):
        d = GateDecision(action=GateAction.REVISE, feedback="try again")
        assert d.should_continue is True
        assert d.should_retry is True
        assert d.feedback == "try again"

    def test_override_continues_no_retry(self):
        d = GateDecision(action=GateAction.OVERRIDE, overrides={"mass_kg": 10.0})
        assert d.should_continue is True
        assert d.should_retry is False
        assert d.overrides["mass_kg"] == 10.0

    def test_reject_stops(self):
        d = GateDecision(action=GateAction.REJECT, feedback="unsafe")
        assert d.should_continue is False
        assert d.should_retry is False

    def test_defaults(self):
        d = GateDecision(action=GateAction.APPROVE)
        assert d.feedback == ""
        assert d.overrides == {}


# ── HumanGate ─────────────────────────────────────────────────────────

class TestHumanGateInit:
    def test_defaults(self):
        gate = HumanGate()
        assert gate.interactive is True
        assert gate.callback is None
        assert gate.verbose is True

    def test_non_interactive(self):
        gate = HumanGate(interactive=False)
        assert gate.interactive is False

    def test_with_callback(self):
        def cb(phase, ctx):
            return GateDecision(action=GateAction.APPROVE)
        gate = HumanGate(callback=cb)
        assert gate.callback is cb


class TestHumanGateCheck:
    def test_unknown_phase_returns_approve(self):
        gate = HumanGate(interactive=False)
        d = gate.check("nonexistent_phase")
        assert d.action == GateAction.APPROVE

    def test_non_interactive_returns_approve(self):
        gate = HumanGate(interactive=False)
        for phase in HumanGate.PHASES:
            d = gate.check(phase)
            assert d.action == GateAction.APPROVE

    def test_callback_receives_phase_and_context(self):
        calls = []
        def cb(phase, ctx):
            calls.append((phase, ctx.get("key")))
            return GateDecision(action=GateAction.APPROVE)
        gate = HumanGate(interactive=False, callback=cb)
        d = gate.check("after_plan", key="value1")
        assert d.action == GateAction.APPROVE
        assert calls == [("after_plan", "value1")]

    def test_callback_returns_revise(self):
        def cb(phase, ctx):
            return GateDecision(action=GateAction.REVISE, feedback="redo plan")
        gate = HumanGate(interactive=False, callback=cb)
        d = gate.check("after_plan", plan=None)
        assert d.action == GateAction.REVISE
        assert d.feedback == "redo plan"

    def test_callback_returns_reject(self):
        def cb(phase, ctx):
            return GateDecision(action=GateAction.REJECT, feedback="stop everything")
        gate = HumanGate(interactive=False, callback=cb)
        d = gate.check("after_check", check=None)
        assert d.action == GateAction.REJECT

    def test_callback_with_overrides(self):
        def cb(phase, ctx):
            return GateDecision(
                action=GateAction.OVERRIDE,
                overrides={"Isp": 180, "mass_kg": 5.0},
            )
        gate = HumanGate(interactive=False, callback=cb)
        d = gate.check("after_do")
        assert d.action == GateAction.OVERRIDE
        assert d.overrides["Isp"] == 180
        assert d.overrides["mass_kg"] == 5.0


class TestHumanGateInteractive:
    def test_approve_on_empty_input(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", return_value=""):
            d = gate._interactive_check("after_plan", "P→D", {})
            assert d.action == GateAction.APPROVE

    def test_approve_on_a(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", return_value="a"):
            d = gate._interactive_check("after_do", "D→C", {})
            assert d.action == GateAction.APPROVE

    def test_approve_full_word(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", return_value="approve"):
            d = gate._interactive_check("after_check", "C→A", {})
            assert d.action == GateAction.APPROVE

    def test_reject_on_x(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", side_effect=["x", "danger"]):
            d = gate._interactive_check("after_act", "A→Next", {})
            assert d.action == GateAction.REJECT
            assert "danger" in d.feedback

    def test_revise_with_feedback(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", side_effect=["r", "fix thrust calc", ""]):
            d = gate._interactive_check("after_plan", "P→D", {})
            assert d.action == GateAction.REVISE
            assert "fix thrust calc" in d.feedback

    def test_override_with_params(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", side_effect=[
            "o", "adjusted mass", "", "mass_kg=8.0", "Isp=150", ""
        ]):
            d = gate._interactive_check("after_act", "A→Next", {})
            assert d.action == GateAction.OVERRIDE
            assert d.feedback == "adjusted mass"
            assert d.overrides["mass_kg"] == 8.0
            assert d.overrides["Isp"] == 150.0

    def test_invalid_then_valid(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", side_effect=["invalid", "q", "a"]):
            d = gate._interactive_check("after_plan", "P→D", {})
            assert d.action == GateAction.APPROVE

    def test_eof_returns_reject(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", side_effect=EOFError):
            d = gate._interactive_check("after_check", "C→A", {})
            assert d.action == GateAction.REJECT

    def test_keyboard_interrupt_returns_reject(self):
        gate = HumanGate(interactive=True)
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            d = gate._interactive_check("after_act", "A→Next", {})
            assert d.action == GateAction.REJECT


class TestHumanGatePhaseLabels:
    def test_all_phases_have_labels(self):
        for phase in HumanGate.PHASES:
            assert phase in HumanGate.PHASE_LABELS

    def test_valid_phases(self):
        valid = {"after_plan", "after_do", "after_check", "after_act"}
        assert set(HumanGate.PHASES) == valid


class TestHumanGatePrintContext:
    def test_after_plan_context(self, capsys):
        from sigma.protocol import PlanOutput
        gate = HumanGate(interactive=False)
        plan = PlanOutput(
            agent_analyses={"Agent A": "analysis", "Agent B": "analysis"},
            conflicts=[],
            dependency_graph={},
            tool_requests=["tool_1"],
            knowledge_gaps=[],
        )
        gate._print_context("after_plan", "P→D", {"plan": plan})
        captured = capsys.readouterr().out
        assert "HumanGate" in captured
        assert "P→D" in captured

    def test_after_do_context(self, capsys):
        from sigma.protocol import DoOutput
        gate = HumanGate(interactive=False)
        do_out = DoOutput(
            tool_results={"tool_a": {"success": True}},
            abnormal_results=[],
        )
        gate._print_context("after_do", "D→C", {"do_output": do_out})
        captured = capsys.readouterr().out
        assert "tool_a" in captured

    def test_after_check_context(self, capsys):
        from sigma.protocol import CheckOutput
        gate = HumanGate(interactive=False)
        check = CheckOutput(
            reviews={},
            updated_conflicts=[],
            devil_advocate="this might be dangerous",
            data_deviations=[],
        )
        gate._print_context("after_check", "C→A", {"check": check})
        captured = capsys.readouterr().out
        assert "dangerous" in captured

    def test_after_act_context(self, capsys):
        from sigma.protocol import ActOutput
        from sigma.state import SharedState, ConsensusEstimate
        gate = HumanGate(interactive=False)
        state = SharedState(task_instruction="test")
        state.round_num = 2
        state.max_rounds = 4
        act = ActOutput(
            verdict="converging", decisions=[], alarm_flags=[],
            should_continue=True, reason="more needed",
            consensus=[ConsensusEstimate(
                parameter="mass_kg", min_val=4.0, max_val=6.0,
                recommended=5.0, confidence="MEDIUM", unit="kg",
                basis="expert", individual={},
            )],
        )
        gate._print_context("after_act", "A→Next", {"act": act, "state": state})
        captured = capsys.readouterr().out
        assert "CONVERGING" in captured
        assert "mass_kg" in captured

    def test_empty_context(self, capsys):
        gate = HumanGate(interactive=False)
        gate._print_context("after_plan", "P→D", {})
        captured = capsys.readouterr().out
        assert "HumanGate" in captured


class TestHumanGateMultilineInput:
    def test_read_multiline_single_line(self):
        gate = HumanGate(interactive=False)
        with mock.patch("builtins.input", side_effect=["single line", ""]):
            result = gate._read_multiline("test")
            assert result == "single line"

    def test_read_multiline_multiple(self):
        gate = HumanGate(interactive=False)
        with mock.patch("builtins.input", side_effect=["line1", "line2", ""]):
            result = gate._read_multiline("test")
            assert result == "line1\nline2"

    def test_read_multiline_empty(self):
        gate = HumanGate(interactive=False)
        with mock.patch("builtins.input", side_effect=[""]):
            result = gate._read_multiline("test")
            assert result == ""

    def test_read_overrides(self):
        gate = HumanGate(interactive=False)
        with mock.patch("builtins.input", side_effect=["mass_kg=10.0", "Isp=150", ""]):
            result = gate._read_overrides()
            assert result["mass_kg"] == 10.0
            assert result["Isp"] == 150.0

    def test_read_overrides_non_numeric(self):
        gate = HumanGate(interactive=False)
        with mock.patch("builtins.input", side_effect=["name=test_rocket", ""]):
            result = gate._read_overrides()
            assert result["name"] == "test_rocket"


# ── GateAction Enum ──────────────────────────────────────────────────

class TestGateAction:
    def test_all_four_actions(self):
        actions = {a.value for a in GateAction}
        assert actions == {"approve", "revise", "override", "reject"}
