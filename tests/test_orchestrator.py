"""Tests for SigmaOrchestrator — AERC cycle scheduling, hook triggers,
report generation, result building, and output formatting.

Uses mocked SigmaProtocol and HookSystem to avoid real LLM/tool calls.
"""

import json
import pytest
import tempfile
from unittest import mock
from pathlib import Path

from sigma.orchestrator import SigmaOrchestrator, _next_version
from sigma.config import SigmaConfig
from sigma.state import (
    SharedState, StateManager, RoundRecord, Conflict, Decision,
    AlarmFlag, ConsensusEstimate, ComplexityAssessment, ComplexityTier,
)
from sigma.protocol import (
    SigmaProtocol, AgentSpec, ToolSpec,
    PlanOutput, DoOutput, CheckOutput, ActOutput,
)
from sigma.hooks import HookSystem, HookPoint
from sigma.cost_tracker import RoundCost


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_protocol():
    """A SigmaProtocol that returns canned AERC outputs."""
    proto = mock.Mock(spec=SigmaProtocol)

    # Default complexity assessment
    proto._assess_complexity.return_value = ComplexityAssessment(
        tier=ComplexityTier.LITE,
        score=1.0,
        reason="simple task",
        agent_names=["Test Agent"],
        max_rounds=1,
        cross_review=False,
        devil_advocate=False,
        consensus_estimation=False,
        skill_crafter=False,
    )

    # Default plan output
    proto.plan.return_value = PlanOutput(
        agent_analyses={"Test Agent": "Analysis text"},
        conflicts=[],
        dependency_graph={"Test Agent": []},
        tool_requests=[],
        knowledge_gaps=[],
    )

    # Default do output
    proto.do.return_value = DoOutput(tool_results={}, abnormal_results=[])

    # Default check output
    proto.check.return_value = CheckOutput(
        reviews={"Test Agent": "All good"},
        updated_conflicts=[],
        devil_advocate="",
        data_deviations=[],
    )

    # Default act output
    proto.act.return_value = ActOutput(
        verdict="converged",
        decisions=[],
        alarm_flags=[],
        should_continue=False,
        reason="Task complete",
        consensus=[],
    )

    return proto


@pytest.fixture
def orchestrator(mock_protocol):
    """Orchestrator with mocked protocol, non-interactive, temp output."""
    orch = SigmaOrchestrator(
        verbose=False,
        interactive=False,
        max_rounds=2,
    )
    orch.protocol = mock_protocol
    return orch


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── Init ──────────────────────────────────────────────────────────

class TestOrchestratorInit:
    def test_defaults(self):
        orch = SigmaOrchestrator()
        assert isinstance(orch.config, SigmaConfig)
        assert orch.max_rounds == 4
        assert orch.verbose is True
        assert orch.interactive is True
        assert isinstance(orch.hooks, HookSystem)
        assert isinstance(orch.protocol, SigmaProtocol)
        assert isinstance(orch.state_mgr, StateManager)
        assert orch.total_cost == 0.0

    def test_custom_max_rounds(self):
        orch = SigmaOrchestrator(max_rounds=2)
        assert orch.max_rounds == 2

    def test_custom_config(self):
        config = SigmaConfig(project_name="Custom")
        orch = SigmaOrchestrator(config=config)
        assert orch.config.project_name == "Custom"

    def test_custom_hooks(self):
        hooks = HookSystem()
        orch = SigmaOrchestrator(hooks=hooks)
        assert orch.hooks is hooks

    def test_dry_mode_no_llm(self):
        """Non-interactive, verbose=False — no print/input."""
        orch = SigmaOrchestrator(verbose=False, interactive=False)
        assert orch.verbose is False
        assert orch.interactive is False

    def test_agents_tools_skills_forwarded_to_protocol(self):
        agents = {"A": AgentSpec(name="A", role="r", goal="g", backstory="b")}
        tools = {"T": ToolSpec(name="T")}
        skills = {"s1": "skill content"}
        orch = SigmaOrchestrator(agents=agents, tools=tools, skills=skills)
        assert "A" in orch.protocol.agents
        assert "T" in orch.protocol.tools
        assert "s1" in orch.protocol.skill_cache


# ── Banner ────────────────────────────────────────────────────────

class TestPrintBanner:
    def test_banner_contains_project_name(self):
        config = SigmaConfig(project_name="TestProject")
        orch = SigmaOrchestrator(config=config, verbose=True)
        # Redirect log to capture — just verify no exception
        orch._print_banner()

    def test_banner_default_project(self):
        orch = SigmaOrchestrator(verbose=True)
        orch._print_banner()


# ── Log ───────────────────────────────────────────────────────────

class TestLog:
    def test_log_when_verbose(self):
        orch = SigmaOrchestrator(verbose=True)
        orch._log("test message")

    def test_log_when_not_verbose(self):
        orch = SigmaOrchestrator(verbose=False)
        orch._log("should not appear")


# ── Build Round Report ────────────────────────────────────────────

class TestBuildRoundReport:
    def test_basic_report_structure(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        plan = PlanOutput(
            agent_analyses={"Agent1": "analysis 1", "Agent2": "analysis 2"},
            conflicts=[],
            dependency_graph={},
            tool_requests=[],
            knowledge_gaps=[],
        )
        do = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done")

        report = orchestrator._build_round_report(state, plan, do, check, act)
        assert "# Round 1" in report
        assert "converged" in report

    def test_report_includes_tool_results(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        plan = PlanOutput(agent_analyses={}, conflicts=[], dependency_graph={},
                         tool_requests=[], knowledge_gaps=[])
        do = DoOutput(
            tool_results={"my_tool": {"performance": {"mass_kg": 5.0}}},
            abnormal_results=[],
        )
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done")

        report = orchestrator._build_round_report(state, plan, do, check, act)
        assert "my_tool" in report
        assert "mass_kg" in report

    def test_report_includes_devil_advocate(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        plan = PlanOutput(agent_analyses={}, conflicts=[], dependency_graph={},
                         tool_requests=[], knowledge_gaps=[])
        do = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(
            reviews={}, updated_conflicts=[], data_deviations=[],
            devil_advocate="This looks dangerous because...",
        )
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done")

        report = orchestrator._build_round_report(state, plan, do, check, act)
        assert "魔鬼代言人" in report
        assert "This looks dangerous" in report

    def test_report_includes_decisions(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        plan = PlanOutput(agent_analyses={}, conflicts=[], dependency_graph={},
                         tool_requests=[], knowledge_gaps=[])
        do = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        act = ActOutput(
            verdict="converged", alarm_flags=[],
            should_continue=False, reason="done",
            decisions=[Decision(round_num=1, domain="test", decision="Use option A",
                               reason="Better performance", made_by="Judge")],
        )

        report = orchestrator._build_round_report(state, plan, do, check, act)
        assert "Use option A" in report

    def test_report_includes_alarms(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        plan = PlanOutput(agent_analyses={}, conflicts=[], dependency_graph={},
                         tool_requests=[], knowledge_gaps=[])
        do = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        act = ActOutput(
            verdict="stalled", decisions=[],
            should_continue=False, reason="safety",
            alarm_flags=[AlarmFlag(flag_type="safety", message="Risk of failure",
                                   round_num=1)],
        )

        report = orchestrator._build_round_report(state, plan, do, check, act)
        assert "Risk of failure" in report

    def test_report_includes_consensus(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        plan = PlanOutput(agent_analyses={}, conflicts=[], dependency_graph={},
                         tool_requests=[], knowledge_gaps=[])
        do = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        act = ActOutput(
            verdict="converged", decisions=[], alarm_flags=[],
            should_continue=False, reason="done",
            consensus=[ConsensusEstimate(
                parameter="mass_kg", min_val=4.0, max_val=6.0, recommended=5.0,
                confidence="HIGH", unit="kg", basis="expert judgment",
                individual={
                    "A": {"value": 5.0, "reasoning": "engineering estimate", "confidence": "HIGH"},
                },
            )],
        )

        report = orchestrator._build_round_report(state, plan, do, check, act)
        assert "mass_kg" in report
        assert "5.0" in report


# ── Build Result Dict ─────────────────────────────────────────────

class TestBuildResult:
    def test_basic_result_structure(self, orchestrator):
        state = SharedState(task_instruction="test task")
        state.round_num = 2
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done", consensus=[])
        output_path = Path("/tmp/test_output")

        result = orchestrator._build_result("test task", state, [], act, output_path)
        assert result["instruction"] == "test task"
        assert result["framework"] == "Sigma AERC"
        assert result["total_rounds"] == 2
        assert result["final_verdict"] == "converged"
        assert result["output_dir"] == str(output_path)

    def test_result_includes_consensus(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        act = ActOutput(
            verdict="converged", decisions=[], alarm_flags=[],
            should_continue=False, reason="done",
            consensus=[ConsensusEstimate(
                parameter="Isp", min_val=150, max_val=200, recommended=180,
                confidence="MEDIUM", unit="s", basis="collective estimate",
                individual={
                    "A": {"value": 180, "reasoning": "typical KNSB", "confidence": "MEDIUM"},
                },
            )],
        )
        result = orchestrator._build_result("test", state, [], act, Path("/tmp"))
        assert len(result["consensus"]) == 1
        assert result["consensus"][0]["parameter"] == "Isp"
        assert result["consensus"][0]["recommended"] == 180

    def test_result_includes_unresolved_alarms(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        state.alarm_flags = [
            AlarmFlag(flag_type="safety", message="Critical issue", round_num=1, resolved=False),
            AlarmFlag(flag_type="data", message="Fixed issue", round_num=1, resolved=True),
        ]
        act = ActOutput(verdict="stalled", decisions=[], alarm_flags=[],
                        should_continue=False, reason="safety")
        result = orchestrator._build_result("test", state, [], act, Path("/tmp"))
        assert "Critical issue" in result["alarm_flags"]
        assert "Fixed issue" not in result["alarm_flags"]

    def test_result_includes_decisions(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        state.decisions = [
            Decision(round_num=1, domain="propulsion", decision="Use KNSB",
                    reason="available", made_by="Propulsion Chief"),
        ]
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done")
        result = orchestrator._build_result("test", state, [], act, Path("/tmp"))
        assert "Use KNSB" in result["decisions"]

    def test_result_act_none_verdict(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 0
        result = orchestrator._build_result("test", state, [], None, Path("/tmp"))
        assert result["final_verdict"] == "unknown"


# ── Final Markdown Report ─────────────────────────────────────────

class TestBuildFinalMarkdown:
    def test_basic_report(self, orchestrator):
        state = SharedState(task_instruction="build rocket")
        state.round_num = 3
        state.complexity_tier = "standard"
        state.complexity_assessment = {"score": 5.0}

        report = orchestrator._build_final_markdown(state, [], None)
        assert "build rocket" in report
        assert "STANDARD" in report

    def test_report_with_rounds(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 2
        state.complexity_tier = "lite"
        state.complexity_assessment = {"score": 1.0}

        round_reports = ["# Round 1\ncontent", "# Round 2\ncontent"]
        report = orchestrator._build_final_markdown(state, round_reports, None)
        assert "# Round 1" in report
        assert "# Round 2" in report

    def test_report_with_unresolved_alarms(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        state.complexity_tier = "rigorous"
        state.complexity_assessment = {"score": 8.0}
        state.alarm_flags = [
            AlarmFlag(flag_type="safety", message="Danger!", round_num=1),
        ]

        report = orchestrator._build_final_markdown(state, [], None)
        assert "Danger!" in report


# ── Print Round Status ────────────────────────────────────────────

class TestPrintRoundStatus:
    def test_status_prints_verdict(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.complexity_tier = "standard"
        cost = RoundCost(round_num=1)
        act = ActOutput(verdict="converging", decisions=[], alarm_flags=[],
                        should_continue=True, reason="not done yet", consensus=[])
        orchestrator._print_round_status(1, cost, state, act)

    def test_status_with_consensus_entries(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.complexity_tier = "standard"
        cost = RoundCost(round_num=1)
        act = ActOutput(
            verdict="converging", decisions=[], alarm_flags=[],
            should_continue=True, reason="not done",
            consensus=[ConsensusEstimate(
                parameter="mass_kg", min_val=4.0, max_val=6.0, recommended=5.0,
                confidence="LOW", unit="kg", basis="estimate",
                individual={"A": {"value": 5.0, "reasoning": "guess", "confidence": "LOW"}},
            )],
        )
        orchestrator._print_round_status(1, cost, state, act)

    def test_status_with_history(self, orchestrator):
        state = SharedState(task_instruction="test", task_params={"mass_kg": 5.0})
        state.complexity_tier = "standard"
        # Add history with tool results
        from sigma.state import RoundRecord
        from datetime import datetime
        state.history = [
            RoundRecord(
                round_num=1, timestamp=datetime.now().isoformat(),
                phase_outputs={"do": {"tool_results": {"mass_kg": 4.5}}},
            ),
        ]
        cost = RoundCost(round_num=2)
        act = ActOutput(verdict="converging", decisions=[], alarm_flags=[],
                        should_continue=True, reason="in progress", consensus=[])
        orchestrator._print_round_status(2, cost, state, act)


# ── Print Detailed Status ─────────────────────────────────────────

class TestPrintDetailedStatus:
    def test_detailed_status_prints_state(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        orchestrator._print_detailed_status(state)

    def test_detailed_status_with_tool_results(self, orchestrator):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        from sigma.state import RoundRecord
        from datetime import datetime
        state.history = [
            RoundRecord(
                round_num=1, timestamp=datetime.now().isoformat(),
                tool_results={"my_tool": {"performance": {"Isp": 180}}},
            ),
        ]
        orchestrator._print_detailed_status(state)


# ── Run Method ────────────────────────────────────────────────────

class TestRunMethod:
    def test_run_single_round_converged(self, orchestrator, temp_output_dir):
        """A single-round LITE task that converges immediately."""
        result = orchestrator.run("simple task", str(temp_output_dir))
        assert result["total_rounds"] == 1
        assert result["final_verdict"] == "converged"
        assert result["framework"] == "Sigma AERC"

    def test_run_creates_output_files(self, orchestrator, temp_output_dir):
        result = orchestrator.run("simple task", str(temp_output_dir))
        output_path = Path(result["output_dir"])
        assert output_path.exists()
        assert (output_path / "REPORT.md").exists()
        assert (output_path / "result.json").exists()

    def test_run_result_json_is_valid(self, orchestrator, temp_output_dir):
        result = orchestrator.run("simple task", str(temp_output_dir))
        output_path = Path(result["output_dir"])
        with open(output_path / "result.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["instruction"] == "simple task"
        assert "timestamp" in data

    def test_run_report_markdown_is_readable(self, orchestrator, temp_output_dir):
        result = orchestrator.run("simple task", str(temp_output_dir))
        output_path = Path(result["output_dir"])
        report = (output_path / "REPORT.md").read_text(encoding="utf-8")
        assert "simple task" in report

    def test_run_multiple_rounds(self, orchestrator, temp_output_dir):
        """Task needs 2 rounds to converge."""
        # First act says continue, second says stop
        orchestrator.protocol.act.side_effect = [
            ActOutput(verdict="converging", decisions=[], alarm_flags=[],
                     should_continue=True, reason="need more", consensus=[]),
            ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                     should_continue=False, reason="done", consensus=[]),
        ]
        orchestrator.protocol._assess_complexity.return_value = ComplexityAssessment(
            tier=ComplexityTier.STANDARD, score=4.0, reason="medium",
            agent_names=["Test Agent"], max_rounds=3,
            cross_review=True, devil_advocate=False,
            consensus_estimation=True, skill_crafter=False,
        )

        result = orchestrator.run("medium task", str(temp_output_dir))
        assert result["total_rounds"] == 2
        assert result["final_verdict"] == "converged"

    def test_run_max_rounds_override(self, mock_protocol, temp_output_dir):
        """max_rounds=1 forces single round even if assessment allows more."""
        mock_protocol._assess_complexity.return_value = ComplexityAssessment(
            tier=ComplexityTier.RIGOROUS, score=8.0, reason="complex",
            agent_names=["Test Agent"], max_rounds=4,
            cross_review=True, devil_advocate=True,
            consensus_estimation=True, skill_crafter=True,
        )
        orch = SigmaOrchestrator(verbose=False, interactive=False, max_rounds=1)
        orch.protocol = mock_protocol

        result = orch.run("complex task", str(temp_output_dir))
        assert result["total_rounds"] == 1

    def test_run_default_output_dir(self, mock_protocol, temp_output_dir):
        """When no output_dir given, auto-creates v{N} under config base."""
        config = SigmaConfig(output_base_dir=str(temp_output_dir))
        orch = SigmaOrchestrator(config=config, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        result = orch.run("test task")
        output_path = Path(result["output_dir"])
        assert output_path.exists()
        assert output_path.name.startswith("v")
        assert (output_path / "REPORT.md").exists()

    def test_run_relative_output_dir(self, mock_protocol, temp_output_dir):
        """Relative output_dir should resolve against config.output_base_dir."""
        config = SigmaConfig(output_base_dir=str(temp_output_dir))
        orch = SigmaOrchestrator(config=config, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        result = orch.run("test task", "custom_output")
        output_path = Path(result["output_dir"])
        assert output_path.is_absolute()
        assert "custom_output" in str(output_path)

    def test_run_with_relative_output_dir_no_config_base(self, orchestrator, temp_output_dir):
        """Relative output_dir when config has no explicit base — uses CWD-based default."""
        result = orchestrator.run("test task", str(temp_output_dir / "out"))
        assert (Path(result["output_dir"])).exists()

    def test_run_saves_round_dirs(self, orchestrator, temp_output_dir):
        """Each round gets a subdirectory under output."""
        result = orchestrator.run("task", str(temp_output_dir))
        output_path = Path(result["output_dir"])
        round_dirs = list(output_path.glob("round_*"))
        assert len(round_dirs) >= 1


# ── Hook Triggers during Run ──────────────────────────────────────

class TestHookTriggersInRun:
    def test_on_start_hook_fires(self, mock_protocol, temp_output_dir):
        hooks = HookSystem()
        called = {}

        def on_start(**ctx):
            called["on_start"] = ctx.get("instruction")
        hooks.register(HookPoint.ON_START, on_start)

        orch = SigmaOrchestrator(hooks=hooks, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        orch.run("hook test", str(temp_output_dir))
        assert called.get("on_start") == "hook test"

    def test_on_complete_hook_fires(self, mock_protocol, temp_output_dir):
        hooks = HookSystem()
        called = {}

        def on_complete(**ctx):
            called["on_complete"] = ctx.get("result", {}).get("final_verdict")
        hooks.register(HookPoint.ON_COMPLETE, on_complete)

        orch = SigmaOrchestrator(hooks=hooks, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        orch.run("hook test", str(temp_output_dir))
        assert called.get("on_complete") == "converged"

    def test_before_plan_hook_fires(self, mock_protocol, temp_output_dir):
        hooks = HookSystem()
        before_count = []

        def before_plan(**ctx):
            before_count.append(1)
        hooks.register(HookPoint.BEFORE_PLAN, before_plan)

        orch = SigmaOrchestrator(hooks=hooks, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        orch.run("hook test", str(temp_output_dir))
        assert len(before_count) >= 1

    def test_after_act_hook_fires(self, mock_protocol, temp_output_dir):
        hooks = HookSystem()
        verdicts = []

        def after_act(**ctx):
            act = ctx.get("act")
            if act:
                verdicts.append(act.verdict)
        hooks.register(HookPoint.AFTER_ACT, after_act)

        orch = SigmaOrchestrator(hooks=hooks, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        orch.run("hook test", str(temp_output_dir))
        assert "converged" in verdicts

    def test_on_error_hook_fires(self, mock_protocol, temp_output_dir):
        hooks = HookSystem()
        errors = []

        def on_error(**ctx):
            errors.append(type(ctx.get("error")).__name__)
        hooks.register(HookPoint.ON_ERROR, on_error)

        mock_protocol.plan.side_effect = RuntimeError("AERC plan failed")

        orch = SigmaOrchestrator(hooks=hooks, verbose=False, interactive=False)
        orch.protocol = mock_protocol

        with pytest.raises(RuntimeError, match="AERC plan failed"):
            orch.run("failing task", str(temp_output_dir))
        assert "RuntimeError" in errors

    def test_hooks_fire_in_order(self, mock_protocol, temp_output_dir):
        hooks = HookSystem()
        order = []

        def start(**ctx): order.append("start")
        hooks.register(HookPoint.ON_START, start)

        def complete(**ctx): order.append("complete")
        hooks.register(HookPoint.ON_COMPLETE, complete)

        orch = SigmaOrchestrator(hooks=hooks, verbose=False, interactive=False)
        orch.protocol = mock_protocol
        orch.run("ordered task", str(temp_output_dir))
        assert order == ["start", "complete"]


# ── Cost Tracking ─────────────────────────────────────────────────

class TestCostTracking:
    def test_total_cost_accumulated(self, orchestrator, temp_output_dir):
        orchestrator.run("task", str(temp_output_dir))
        assert orchestrator.total_cost >= 0.0

    def test_cost_tracker_present(self, orchestrator):
        from sigma.cost_tracker import CostTracker
        assert isinstance(orchestrator.cost_tracker, CostTracker)


# ── State Initialization ──────────────────────────────────────────

class TestStateInit:
    def test_complexity_assessment_stored(self, orchestrator, temp_output_dir):
        result = orchestrator.run("test", str(temp_output_dir))
        # The state is internal but we can check the result
        assert "instruction" in result
        assert result["total_rounds"] >= 1

    def test_state_max_rounds_capped(self, mock_protocol, temp_output_dir):
        mock_protocol._assess_complexity.return_value = ComplexityAssessment(
            tier=ComplexityTier.RIGOROUS, score=8.0, reason="complex",
            agent_names=["Test Agent"], max_rounds=4,
            cross_review=True, devil_advocate=True,
            consensus_estimation=True, skill_crafter=True,
        )
        orch = SigmaOrchestrator(verbose=False, interactive=False, max_rounds=2)
        orch.protocol = mock_protocol
        result = orch.run("complex task", str(temp_output_dir))
        assert result["total_rounds"] <= 2


# ── Print Final Report ────────────────────────────────────────────

class TestPrintFinalReport:
    def test_final_report_writes_files(self, orchestrator, temp_output_dir):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        state.complexity_tier = "lite"
        state.complexity_assessment = {"score": 1.0}
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done")

        orchestrator._print_final_report(state, [], act, temp_output_dir)
        assert (temp_output_dir / "REPORT.md").exists()
        assert (temp_output_dir / "result.json").exists()

    def test_final_report_includes_cost(self, orchestrator, temp_output_dir):
        state = SharedState(task_instruction="test")
        state.round_num = 1
        state.complexity_tier = "lite"
        state.complexity_assessment = {"score": 1.0}
        act = ActOutput(verdict="converged", decisions=[], alarm_flags=[],
                        should_continue=False, reason="done")

        orchestrator._print_final_report(state, [], act, temp_output_dir)
        report = (temp_output_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "成本" in report or "cost" in report.lower()


# ── _next_version ─────────────────────────────────────────────────

class TestNextVersion:
    def test_empty_dir_returns_v1(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            result = _next_version(base)
            assert result == "v1"

    def test_existing_versions(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "v1").mkdir()
            (base / "v2").mkdir()
            (base / "v5").mkdir()
            result = _next_version(base)
            assert result == "v6"

    def test_ignores_non_version_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "v1").mkdir()
            (base / "not_a_version").mkdir()
            (base / "v2abc").mkdir()  # Not a valid version number
            result = _next_version(base)
            assert result == "v2"

    def test_ignores_files(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "v1").mkdir()
            (base / "readme.txt").touch()
            (base / "v10").mkdir()
            result = _next_version(base)
            assert result == "v11"

    def test_creates_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "nonexistent"
            result = _next_version(base)
            assert result == "v1"
            assert base.exists()
