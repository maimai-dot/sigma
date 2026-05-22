"""Tests for async protocol methods — plan_async, check_async, do_async."""

import pytest
from unittest import mock
from datetime import datetime

from sigma.llm import AsyncUniversalBackend, LLMResponse
from sigma.config import SigmaConfig
from sigma.state import SharedState, RoundRecord, ComplexityAssessment, ComplexityTier
from sigma.protocol import SigmaProtocol, AgentSpec, PlanOutput, DoOutput, CheckOutput
from sigma.cost_tracker import RoundCost


@pytest.fixture
def async_llm():
    """Fresh mock AsyncOpenAIBackend for each test."""
    be = mock.AsyncMock(spec=AsyncUniversalBackend)
    be.chat.return_value = LLMResponse(
        content="Async analysis: thrust=1500N, mass=5.2kg",
        input_tokens=80, output_tokens=40,
    )
    be.retry = mock.Mock()
    return be


@pytest.fixture
def agent_specs():
    return {
        "Propulsion": AgentSpec(name="Propulsion", role="推进工程师", goal="设计推进系统",
                                backstory="推进专家", tool_names=["thrust_calc"]),
        "Structures": AgentSpec(name="Structures", role="结构工程师", goal="设计箭体结构",
                                backstory="结构专家", tool_names=["mass_calc"]),
    }


@pytest.fixture
def sample_state():
    state = SharedState(task_instruction="验证箭设计方案")
    state.round_num = 1
    state.history = [RoundRecord(round_num=0, timestamp=datetime.now().isoformat())]
    return state


# ── Plan Async ────────────────────────────────────────────────────

class TestPlanAsync:
    @pytest.mark.asyncio
    async def test_basic_plan_async(self, async_llm, agent_specs, sample_state):
        config = SigmaConfig(project_name="Test", creed="Test creed")
        proto = SigmaProtocol(config=config, agents=agent_specs, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        plan = await proto.plan_async(sample_state, rc)
        assert isinstance(plan, PlanOutput)
        assert len(plan.agent_analyses) == 2
        assert "Propulsion" in plan.agent_analyses

    @pytest.mark.asyncio
    async def test_plan_async_with_complexity(self, async_llm, agent_specs, sample_state):
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents=agent_specs, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        ca = ComplexityAssessment(
            tier=ComplexityTier.LITE, score=1.5, reason="simple",
            agent_names=["Propulsion"], max_rounds=1,
            cross_review=False, devil_advocate=False,
            consensus_estimation=False, skill_crafter=False,
        )
        plan = await proto.plan_async(sample_state, rc, ca)
        assert len(plan.agent_analyses) == 1
        assert "Propulsion" in plan.agent_analyses

    @pytest.mark.asyncio
    async def test_plan_async_with_cross_review(self, async_llm, agent_specs, sample_state):
        config = SigmaConfig(project_name="Test", creed="Test creed")
        proto = SigmaProtocol(config=config, agents=agent_specs, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        ca = ComplexityAssessment(
            tier=ComplexityTier.RIGOROUS, score=7.0, reason="complex",
            agent_names=["Propulsion", "Structures"], max_rounds=4,
            cross_review=True, devil_advocate=True,
            consensus_estimation=True, skill_crafter=True,
        )
        plan = await proto.plan_async(sample_state, rc, ca)
        assert "交叉审查" in plan.agent_analyses["Propulsion"]

    @pytest.mark.asyncio
    async def test_plan_async_error_handling(self, agent_specs, sample_state):
        """Agent that raises should get [ERROR: ...] not crash."""
        bad_llm = mock.AsyncMock(spec=AsyncUniversalBackend)
        bad_llm.chat.side_effect = [LLMResponse(content="ok"), RuntimeError("boom")]
        bad_llm.retry = mock.Mock()
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents=agent_specs, llm_backend=bad_llm)
        rc = RoundCost(round_num=1)
        plan = await proto.plan_async(sample_state, rc)
        assert "Propulsion" in plan.agent_analyses
        assert "[ERROR:" in plan.agent_analyses["Structures"]


# ── Check Async ───────────────────────────────────────────────────

class TestCheckAsync:
    @pytest.fixture
    def mock_plan(self):
        return PlanOutput(
            agent_analyses={"Propulsion": "thrust analysis", "Structures": "mass analysis"},
            conflicts=[],
            dependency_graph={},
            tool_requests=[],
            knowledge_gaps=[],
        )

    @pytest.fixture
    def mock_do(self):
        return DoOutput(
            tool_results={"thrust_calc": {"thrust_n": 1500, "success": True}},
            abnormal_results=[],
        )

    @pytest.mark.asyncio
    async def test_basic_check_async(self, async_llm, agent_specs, sample_state, mock_plan, mock_do):
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents=agent_specs, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        check = await proto.check_async(sample_state, mock_plan, mock_do, rc)
        assert isinstance(check, CheckOutput)
        assert len(check.reviews) == 2

    @pytest.mark.asyncio
    async def test_check_async_with_devil_advocate(self, async_llm, agent_specs,
                                                    sample_state, mock_plan, mock_do):
        safety = AgentSpec(name="Safety Officer", role="安全官", goal="确保安全", backstory="安全专家")
        all_agents = {**agent_specs, "Safety Officer": safety}
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents=all_agents, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        ca = ComplexityAssessment(
            tier=ComplexityTier.RIGOROUS, score=7.0, reason="complex",
            agent_names=["Propulsion", "Structures"], max_rounds=4,
            cross_review=True, devil_advocate=True,
            consensus_estimation=True, skill_crafter=True,
        )
        check = await proto.check_async(sample_state, mock_plan, mock_do, rc, ca)
        assert check.devil_advocate != ""


# ── Do Async ──────────────────────────────────────────────────────

class TestDoAsync:
    @pytest.mark.asyncio
    async def test_do_async_delegates_to_sync(self, async_llm, sample_state):
        """do_async simply delegates to do() since tools are sync."""
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents={}, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        plan = PlanOutput(
            agent_analyses={}, conflicts=[], dependency_graph={},
            tool_requests=[], knowledge_gaps=[],
        )
        do = await proto.do_async(plan, sample_state, rc)
        assert isinstance(do, DoOutput)
        assert do.tool_results == {}


# ── Async LLM Call ────────────────────────────────────────────────

class TestAsyncCallLLM:
    @pytest.mark.asyncio
    async def test_a_call_llm_basic(self, async_llm):
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents={}, llm_backend=async_llm)
        rc = RoundCost(round_num=1)
        resp = await proto._a_call_llm("system", "user", rc)
        assert isinstance(resp, str)
        assert len(resp) > 0
        assert rc.calls >= 1

    @pytest.mark.asyncio
    async def test_a_call_llm_no_cost_tracker(self, async_llm):
        config = SigmaConfig(project_name="Test")
        proto = SigmaProtocol(config=config, agents={}, llm_backend=async_llm)
        resp = await proto._a_call_llm("system", "user")
        assert len(resp) > 0
