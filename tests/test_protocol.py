"""Tests for SigmaProtocol — AERC engine core.

Tests cover: AgentSpec/ToolSpec, complexity assessment (rule engine),
tool dispatch, JSON parsing, unit inference, parameter classification,
dependency graph, and full phase methods (plan/do/check/act) with mocked LLM.
"""

import json
import pytest
from unittest import mock
from pathlib import Path

from sigma.protocol import (
    SigmaProtocol, AgentSpec, ToolSpec,
    PlanOutput, DoOutput, CheckOutput, ActOutput,
)
from sigma.state import (
    SharedState, StateManager, Conflict, Decision, AlarmFlag,
    ComplexityAssessment, ComplexityTier, ConsensusEstimate,
)
from sigma.config import SigmaConfig
from sigma.llm import LLMResponse, UniversalBackend
from sigma.cost_tracker import RoundCost
from sigma.triggers import Trigger


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def basic_config():
    return SigmaConfig()


@pytest.fixture
def rocket_config():
    """Domain config similar to RocketFactory's SigmaConfig."""
    return SigmaConfig(
        project_name="RocketTest",
        creed="We are building rockets.",
        domain_keywords={
            "propulsion": ["推进", "发动机", "比冲", "KNSB", "N2O", "LOX", "propellant", "Isp"],
            "structure": ["结构", "管", "质量", "mass", "强度", "壁厚"],
            "aerodynamics": ["气动", "减阻", "drag", "stability"],
            "gnc": ["飞控", "传感器", "航电", "GNC"],
            "thermal": ["热防护", "烧蚀", "ablative", "温度", "temperature"],
            "supply": ["采购", "BOM", "供应链", "supply"],
        },
        role_map={
            "director": "Director",
            "propulsion_chief": "Propulsion Chief",
            "structures_chief": "Structures Chief",
            "safety_officer": "Safety Officer",
        },
        domain_agent_map={
            "propulsion": "Propulsion Chief",
            "structure": "Structures Chief",
        },
        standard_exclude_agents=[],
    )


@pytest.fixture
def sample_agents():
    return {
        "Director": AgentSpec(
            name="Director", role="任务总监",
            goal="拆解任务，协调各角色",
            backstory="资深系统工程师",
            tool_names=[],
        ),
        "Propulsion Chief": AgentSpec(
            name="Propulsion Chief", role="推进总工",
            goal="设计推进系统",
            backstory="推进专家",
            tool_names=["rocketcea_analyzer", "propellant_comparator"],
        ),
        "Structures Chief": AgentSpec(
            name="Structures Chief", role="结构总工",
            goal="设计箭体结构",
            backstory="结构力学专家",
            tool_names=["freecad_mass_extractor"],
        ),
        "Safety Officer": AgentSpec(
            name="Safety Officer", role="安全官",
            goal="保障安全",
            backstory="安全审查专家",
            tool_names=[],
        ),
    }


@pytest.fixture
def sample_tools():
    return {
        "freecad_mass_extractor": ToolSpec(
            name="freecad_mass_extractor",
            instance=None,  # Will be mocked
            aliases=["质量提取", "freecad", "mass"],
            default_params={},
            expected_outputs=["mass_kg", "volume_mm3"],
        ),
        "rocketcea_analyzer": ToolSpec(
            name="rocketcea_analyzer",
            instance=None,
            aliases=["CEA", "推进剂分析", "rocketcea"],
            default_params={"fuel": "KNSB", "ox": "N2O", "Pc": 20},
            expected_outputs=["Isp", "T_comb", "C_star"],
        ),
    }


@pytest.fixture
def mock_llm():
    """Creates a mock LLM backend that returns preset responses."""
    llm = mock.Mock(spec=UniversalBackend)
    llm.chat.return_value = LLMResponse(
        content="Mock analysis result.",
        input_tokens=100,
        output_tokens=50,
    )
    return llm


@pytest.fixture
def sample_state():
    state = SharedState(task_instruction="计算火箭质量")
    state.round_num = 1
    from sigma.state import RoundRecord
    from datetime import datetime
    state.history = [RoundRecord(round_num=0, timestamp=datetime.now().isoformat())]
    return state


# ── AgentSpec ─────────────────────────────────────────────────────

class TestAgentSpec:
    def test_minimal_spec(self):
        s = AgentSpec(name="test", role="tester", goal="test things", backstory="was born to test")
        assert s.name == "test"
        assert s.role == "tester"
        assert s.tool_names == []
        assert s.skill_files == []

    def test_full_spec(self):
        s = AgentSpec(
            name="Engineer", role="engineer", goal="design",
            backstory="experienced", skill_files=["a.md", "b.md"],
            tool_names=["tool1", "tool2"],
            tool_instances=[mock.Mock(), mock.Mock()],
        )
        assert len(s.skill_files) == 2
        assert len(s.tool_names) == 2
        assert len(s.tool_instances) == 2

    def test_system_prompt_basic(self):
        s = AgentSpec(name="E", role="engineer", goal="design it", backstory="I design things")
        prompt = s.system_prompt()
        assert "I design things" in prompt
        assert "design it" in prompt

    def test_system_prompt_with_creed(self):
        s = AgentSpec(name="E", role="engineer", goal="design it", backstory="I design things")
        prompt = s.system_prompt(creed="WE BUILD ROCKETS")
        assert "WE BUILD ROCKETS" in prompt
        assert prompt.startswith("WE BUILD ROCKETS")

    def test_system_prompt_with_extra_context(self):
        s = AgentSpec(name="E", role="engineer", goal="design it", backstory="I design things")
        prompt = s.system_prompt(extra_context="More info here")
        assert "More info here" in prompt
        assert prompt.endswith("More info here")

    def test_system_prompt_with_both(self):
        s = AgentSpec(name="E", role="engineer", goal="design it", backstory="I design things")
        prompt = s.system_prompt(extra_context="Extra", creed="CREED")
        assert prompt.startswith("CREED")
        assert prompt.endswith("Extra")

    def test_hash_based_on_name(self):
        s1 = AgentSpec(name="A", role="r1", goal="g1", backstory="b1")
        s2 = AgentSpec(name="A", role="r2", goal="g2", backstory="b2")
        s3 = AgentSpec(name="B", role="r1", goal="g1", backstory="b1")
        assert hash(s1) == hash(s2)
        assert hash(s1) != hash(s3)

    def test_tool_instances_default(self):
        s = AgentSpec(name="T", role="r", goal="g", backstory="b")
        assert s.tool_instances == []


# ── ToolSpec ──────────────────────────────────────────────────────

class TestToolSpec:
    def test_minimal_spec(self):
        t = ToolSpec(name="my_tool")
        assert t.name == "my_tool"
        assert t.instance is None
        assert t.aliases == []
        assert t.default_params == {}
        assert t.expected_outputs == []
        assert t.description == ""

    def test_full_spec(self):
        instance = mock.Mock()
        t = ToolSpec(
            name="analyzer",
            instance=instance,
            aliases=["分析器", "analyze"],
            default_params={"mode": "fast"},
            expected_outputs=["result_1", "result_2"],
            description="An analysis tool",
        )
        assert t.instance is instance
        assert "分析器" in t.aliases
        assert t.default_params["mode"] == "fast"
        assert len(t.expected_outputs) == 2


# ── Complexity Assessment (Rule Engine) ──────────────────────────

class TestComplexityAssessment:
    """_assess_complexity is pure rule engine — no LLM calls."""

    def test_lite_tier_short_query(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._assess_complexity("hello")
        assert result.tier == ComplexityTier.LITE
        assert result.max_rounds == 1
        assert not result.cross_review
        assert not result.devil_advocate

    def test_lite_tier_simple_calculation(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._assess_complexity("计算150mm管子的质量")
        assert result.tier == ComplexityTier.LITE
        assert result.max_rounds == 1

    def test_standard_tier_with_domain_overlap(self, rocket_config):
        proto = SigmaProtocol(config=rocket_config)
        result = proto._assess_complexity(
            "设计KNSB推进剂的发动机，要求比冲大于180s，推力500N，用于150mm箭体结构"
        )
        # Should hit propulsion + structure domains + multiple params
        assert result.tier in (ComplexityTier.STANDARD, ComplexityTier.RIGOROUS)
        assert result.max_rounds >= 3

    def test_rigorous_tier_complex_design(self, rocket_config, sample_agents):
        proto = SigmaProtocol(config=rocket_config, agents=sample_agents)
        result = proto._assess_complexity(
            "设计1公里级验证箭完整方案：包含KNSB推进系统选型、箭体结构设计、"
            "气动减阻优化、飞控传感器布局、热防护方案、供应链BOM清单、"
            "安全评审报告。要求总重<15kg，高度<3m，推力>500N，比冲>180s，"
            "在民用材料限制下6个月内完工。"
        )
        assert result.tier == ComplexityTier.RIGOROUS
        assert result.max_rounds == 4
        assert result.cross_review is True
        assert result.devil_advocate is True
        assert result.consensus_estimation is True
        assert result.skill_crafter is True

    def test_score_never_exceeds_10(self, rocket_config):
        proto = SigmaProtocol(config=rocket_config)
        # Craft an instruction that hits every scoring dimension
        result = proto._assess_complexity(
            "设计优化方案选择：推进系统KNSB N2O LOX发动机设计结构减阻飞控传感器"
            "热防护烧蚀供应链采购BOM 150mm 500N 180s 15kg 3m 20bar 300K 85% 60度"
            "方案一或者方案二选择最佳方案 " * 10
        )
        assert result.score <= 10.0

    def test_length_scoring(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        short = proto._assess_complexity("hi")
        medium = proto._assess_complexity("a" * 100)
        long_text = proto._assess_complexity("a" * 300)
        very_long = proto._assess_complexity("a" * 500)
        assert short.score <= medium.score <= very_long.score
        assert long_text.score > short.score

    def test_parameter_count_scoring(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        no_params = proto._assess_complexity("hello world")
        some_params = proto._assess_complexity("150mm 500N 180s 15kg 3m")
        assert some_params.score > no_params.score

    def test_decision_keyword_adds_score(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        no_choice = proto._assess_complexity("计算质量")
        with_choice = proto._assess_complexity("计算质量或选择方案")
        assert with_choice.score >= no_choice.score

    def test_action_verbs_weight(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        query = proto._assess_complexity("搜索资料")
        calc = proto._assess_complexity("计算火箭质量")
        design = proto._assess_complexity("设计发动机方案")
        assert design.score >= calc.score
        assert calc.score >= query.score

    def test_returns_complexity_assessment_with_agent_names(self, rocket_config, sample_agents):
        proto = SigmaProtocol(config=rocket_config, agents=sample_agents)
        result = proto._assess_complexity("设计火箭推进系统")
        assert isinstance(result, ComplexityAssessment)
        assert isinstance(result.agent_names, list)
        assert len(result.agent_names) > 0


# ── Agent Selection ──────────────────────────────────────────────

class TestAgentSelection:
    def test_lite_selects_director_plus_domain_agents(self, rocket_config, sample_agents):
        proto = SigmaProtocol(config=rocket_config, agents=sample_agents)
        domain_scores = {"propulsion": 3.0, "structure": 1.5}
        selected = proto._select_agents_for_tier(
            ComplexityTier.LITE, domain_scores, "test",
        )
        assert "Director" in selected
        assert "Propulsion Chief" in selected
        assert len(selected) <= 4

    def test_lite_deduplicates(self, rocket_config, sample_agents):
        proto = SigmaProtocol(config=rocket_config, agents=sample_agents)
        domain_scores = {"propulsion": 3.0, "structure": 1.5}
        selected = proto._select_agents_for_tier(
            ComplexityTier.LITE, domain_scores, "test",
        )
        assert len(selected) == len(set(selected))

    def test_standard_returns_all_except_excluded(self, rocket_config, sample_agents):
        config = SigmaConfig(
            domain_keywords=rocket_config.domain_keywords,
            standard_exclude_agents=["Safety Officer"],
        )
        proto = SigmaProtocol(config=config, agents=sample_agents)
        selected = proto._select_agents_for_tier(
            ComplexityTier.STANDARD, {}, "test",
        )
        assert "Safety Officer" not in selected
        assert "Director" in selected

    def test_standard_exclude_empty_returns_all(self, rocket_config, sample_agents):
        proto = SigmaProtocol(config=rocket_config, agents=sample_agents)
        selected = proto._select_agents_for_tier(
            ComplexityTier.STANDARD, {}, "test",
        )
        assert len(selected) == len(sample_agents)

    def test_rigorous_returns_all_agents(self, rocket_config, sample_agents):
        proto = SigmaProtocol(config=rocket_config, agents=sample_agents)
        selected = proto._select_agents_for_tier(
            ComplexityTier.RIGOROUS, {}, "test",
        )
        assert set(selected) == set(sample_agents.keys())


# ── SigmaProtocol Init ───────────────────────────────────────────

class TestSigmaProtocolInit:
    def test_defaults(self):
        proto = SigmaProtocol()
        assert isinstance(proto.config, SigmaConfig)
        assert isinstance(proto.agents, dict)
        assert isinstance(proto.tools, dict)
        assert isinstance(proto.skill_cache, dict)
        assert isinstance(proto.llm, UniversalBackend)

    def test_custom_config(self, rocket_config):
        proto = SigmaProtocol(config=rocket_config)
        assert proto.config.project_name == "RocketTest"

    def test_custom_llm(self, mock_llm):
        proto = SigmaProtocol(llm_backend=mock_llm)
        assert proto.llm is mock_llm

    def test_custom_model(self):
        proto = SigmaProtocol(model="gpt-4")
        assert proto.model == "gpt-4"

    def test_verbose_false_suppresses_log(self):
        proto = SigmaProtocol(verbose=False)
        assert proto.verbose is False

    def test_agents_and_tools_injected(self, sample_agents, sample_tools):
        proto = SigmaProtocol(agents=sample_agents, tools=sample_tools)
        assert "Director" in proto.agents
        assert "freecad_mass_extractor" in proto.tools

    def test_reasonable_ranges_injected_to_triggers(self):
        config = SigmaConfig(reasonable_ranges={"mass_kg": (0.1, 100.0)})
        proto = SigmaProtocol(config=config)
        assert "mass_kg" in proto.triggers.REASONABLE_RANGES
        assert proto.triggers.REASONABLE_RANGES["mass_kg"] == (0.1, 100.0)


# ── LLM Call ─────────────────────────────────────────────────────

class TestCallLLM:
    def test_returns_content(self, mock_llm):
        proto = SigmaProtocol(llm_backend=mock_llm)
        result = proto._call_llm("system", "user")
        assert result == "Mock analysis result."

    def test_tracks_cost(self, mock_llm):
        proto = SigmaProtocol(llm_backend=mock_llm)
        cost = RoundCost(round_num=1)
        proto._call_llm("system prompt text here", "user", cost)
        assert cost.calls >= 1

    def test_error_response_without_cost_tracking(self, mock_llm):
        mock_llm.chat.return_value = LLMResponse(content="[LLM_ERROR: timeout]", retries=3)
        proto = SigmaProtocol(llm_backend=mock_llm)
        result = proto._call_llm("sys", "user")
        assert "LLM_ERROR" in result


# ── Tool Request Extraction ──────────────────────────────────────

class TestExtractToolRequests:
    def test_explicit_marker_chinese(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {"A": "需要调用 [需要工具: freecad_mass_extractor] 来计算质量"}
        requests = proto._extract_tool_requests(analyses)
        assert "freecad_mass_extractor" in requests

    def test_explicit_marker_english(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {"A": "[NEED_TOOL: rocketcea_analyzer] for combustion analysis"}
        requests = proto._extract_tool_requests(analyses)
        assert "rocketcea_analyzer" in requests

    def test_explicit_marker_multiple_tools(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {"A": "[需要工具: freecad_mass_extractor, rocketcea_analyzer]"}
        requests = proto._extract_tool_requests(analyses)
        assert "freecad_mass_extractor" in requests
        assert "rocketcea_analyzer" in requests

    def test_tool_name_mentioned_in_text(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {"A": "We should use freecad_mass_extractor for this"}
        requests = proto._extract_tool_requests(analyses)
        assert "freecad_mass_extractor" in requests

    def test_alias_match(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {"A": "我们需要做推进剂分析来确认比冲"}
        requests = proto._extract_tool_requests(analyses)
        assert "rocketcea_analyzer" in requests

    def test_short_alias_not_matched(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {"A": "CEA is a standard tool"}  # "CEA" is only 3 chars, < 4
        requests = proto._extract_tool_requests(analyses)
        # "CEA" alias len=3 < 4 so it's not matched by alias; but "rocketcea" name IS in tool name
        # Actually "CEA" would match by tool name if "cea" is in tool_name.lower()
        # rocketcea_analyzer has "cea" in it — no, "cea" is NOT in "rocketcea_analyzer" literally
        # Wait: "rocketcea_analyzer".lower() = "rocketcea_analyzer", "cea" is not a substring of that
        # So CEA should not match
        assert "rocketcea_analyzer" not in requests

    def test_no_tools_returns_empty(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        analyses = {"A": "No tool references here"}
        requests = proto._extract_tool_requests(analyses)
        assert requests == []

    def test_duplicate_requests_deduplicated(self, basic_config, sample_tools):
        proto = SigmaProtocol(config=basic_config, tools=sample_tools)
        analyses = {
            "A": "[需要工具: freecad_mass_extractor]",
            "B": "[需要工具: freecad_mass_extractor]",
        }
        requests = proto._extract_tool_requests(analyses)
        assert len(requests) == 1


# ── Dependency Graph ─────────────────────────────────────────────

class TestDependencyGraph:
    def test_detects_name_mentions(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        analyses = {
            "Director": "I depend on Propulsion Chief for thrust data",
        }
        graph = proto._build_dependency_graph(analyses)
        assert "Propulsion Chief" in graph.get("Director", [])

    def test_self_not_in_deps(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        analyses = {
            "Director": "As Director I need to coordinate with Director",
        }
        graph = proto._build_dependency_graph(analyses)
        assert "Director" not in graph.get("Director", [])

    def test_max_five_deps(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        analyses = {
            "Director": "Depends on " + " ".join(sample_agents.keys()),
        }
        graph = proto._build_dependency_graph(analyses)
        assert len(graph.get("Director", [])) <= 5

    def test_no_deps_returns_empty_list(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        analyses = {"Director": "Just my own analysis"}
        graph = proto._build_dependency_graph(analyses)
        assert graph.get("Director") == []


# ── JSON Parsing ─────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_direct_json(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response('{"value": 42, "confidence": "HIGH"}')
        assert result == {"value": 42, "confidence": "HIGH"}

    def test_json_in_code_fence(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response('```json\n{"value": 99}\n```')
        assert result == {"value": 99}

    def test_json_in_code_fence_no_lang(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response('```\n{"value": 99}\n```')
        assert result == {"value": 99}

    def test_json_braces_extraction(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response('The answer is {"value": 7.5} as shown')
        assert result == {"value": 7.5}

    def test_invalid_json_returns_none(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response("Not JSON at all")
        assert result is None

    def test_empty_string_returns_none(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response("")
        assert result is None

    def test_nested_json(self, basic_config):
        proto = SigmaProtocol(config=basic_config)
        result = proto._parse_json_response(
            '{"value": 100, "individual": {"A": {"value": 95}}}'
        )
        assert result["value"] == 100
        assert result["individual"]["A"]["value"] == 95


# ── Unit Inference ───────────────────────────────────────────────

class TestInferUnit:
    def test_known_units(self):
        proto = SigmaProtocol()
        assert proto._infer_unit("Isp") == "s"
        assert proto._infer_unit("T_comb") == "K"
        assert proto._infer_unit("mass_kg") == "kg"
        assert proto._infer_unit("thrust_N") == "N"
        assert proto._infer_unit("velocity") == "m/s"
        assert proto._infer_unit("pressure_bar") == "bar"
        assert proto._infer_unit("diameter_mm") == "mm"

    def test_unknown_param_returns_empty(self):
        proto = SigmaProtocol()
        assert proto._infer_unit("unknown_thing") == ""


# ── Numeric Parameter Detection ──────────────────────────────────

class TestIsNumericParam:
    def test_suffix_params(self):
        proto = SigmaProtocol()
        assert proto._is_numeric_param("thrust_n")
        assert proto._is_numeric_param("mass_kg")
        assert proto._is_numeric_param("velocity_ms")
        assert proto._is_numeric_param("pressure_bar")
        assert proto._is_numeric_param("aspect_ratio")
        assert proto._is_numeric_param("density_kg")
        assert proto._is_numeric_param("temperature_K")

    def test_keyword_params(self):
        proto = SigmaProtocol()
        assert proto._is_numeric_param("Isp")
        assert proto._is_numeric_param("chamber_temperature")
        assert proto._is_numeric_param("exhaust_velocity")
        assert proto._is_numeric_param("mass_estimate")
        assert proto._is_numeric_param("chamber_pressure")

    def test_non_numeric_params(self):
        proto = SigmaProtocol()
        assert not proto._is_numeric_param("material_choice")
        assert not proto._is_numeric_param("design_type")
        assert not proto._is_numeric_param("supplier_name")


# ── Plan Phase (with mocked LLM) ─────────────────────────────────

class TestPlanPhase:
    def test_plan_with_single_agent(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        cost = RoundCost(round_num=1)
        assessment = ComplexityAssessment(
            tier=ComplexityTier.LITE,
            score=1.0,
            reason="simple",
            agent_names=["Director"],
            max_rounds=1,
            cross_review=False,
            devil_advocate=False,
            consensus_estimation=False,
            skill_crafter=False,
        )
        result = proto.plan(sample_state, cost, assessment)
        assert isinstance(result, PlanOutput)
        assert "Director" in result.agent_analyses
        assert len(result.agent_analyses) == 1

    def test_plan_with_cross_review(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        cost = RoundCost(round_num=1)
        assessment = ComplexityAssessment(
            tier=ComplexityTier.STANDARD,
            score=4.0,
            reason="medium",
            agent_names=["Director", "Propulsion Chief", "Structures Chief"],
            max_rounds=3,
            cross_review=True,
            devil_advocate=False,
            consensus_estimation=True,
            skill_crafter=False,
        )
        result = proto.plan(sample_state, cost, assessment)
        assert len(result.agent_analyses) == 3

    def test_plan_without_complexity_uses_all_agents(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        cost = RoundCost(round_num=1)
        result = proto.plan(sample_state, cost, None)
        assert len(result.agent_analyses) == len(sample_agents)

    def test_plan_output_structure(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        cost = RoundCost(round_num=1)
        result = proto.plan(sample_state, cost)
        assert isinstance(result.dependency_graph, dict)
        assert isinstance(result.tool_requests, list)
        assert isinstance(result.knowledge_gaps, list)

    def test_plan_handles_agent_error(self, sample_state, sample_agents):
        llm = mock.Mock(spec=UniversalBackend)
        llm.chat.side_effect = RuntimeError("boom")
        proto = SigmaProtocol(llm_backend=llm, agents=sample_agents)
        cost = RoundCost(round_num=1)
        assessment = ComplexityAssessment(
            tier=ComplexityTier.LITE, score=1.0, reason="test",
            agent_names=["Director"], max_rounds=1,
            cross_review=False, devil_advocate=False,
            consensus_estimation=False, skill_crafter=False,
        )
        result = proto.plan(sample_state, cost, assessment)
        assert "[ERROR" in result.agent_analyses.get("Director", "")


# ── Do Phase ─────────────────────────────────────────────────────

class TestDoPhase:
    def test_do_no_tool_requests(self, sample_state):
        proto = SigmaProtocol()
        plan = PlanOutput(
            agent_analyses={}, conflicts=[], dependency_graph={},
            tool_requests=[], knowledge_gaps=[],
        )
        cost = RoundCost(round_num=1)
        result = proto.do(plan, sample_state, cost)
        assert isinstance(result, DoOutput)
        assert result.tool_results == {}
        assert result.abnormal_results == []

    def test_do_unknown_tool_skipped(self, sample_state):
        proto = SigmaProtocol()
        plan = PlanOutput(
            agent_analyses={}, conflicts=[], dependency_graph={},
            tool_requests=["nonexistent_tool"], knowledge_gaps=[],
        )
        cost = RoundCost(round_num=1)
        result = proto.do(plan, sample_state, cost)
        assert result.tool_results == {}

    def test_do_with_mocked_tool(self, sample_state, sample_tools):
        mock_tool = mock.Mock()
        mock_tool._run.return_value = {"success": True, "performance": {"mass_kg": 5.0}}
        sample_tools["freecad_mass_extractor"].instance = mock_tool

        proto = SigmaProtocol(tools=sample_tools)
        plan = PlanOutput(
            agent_analyses={}, conflicts=[], dependency_graph={},
            tool_requests=["freecad_mass_extractor"], knowledge_gaps=[],
        )
        cost = RoundCost(round_num=1)
        result = proto.do(plan, sample_state, cost)
        assert "freecad_mass_extractor" in result.tool_results
        assert result.tool_results["freecad_mass_extractor"]["performance"]["mass_kg"] == 5.0

    def test_do_retry_on_abnormal_result(self, sample_state, sample_tools):
        mock_tool = mock.Mock()
        # First call returns abnormal, second returns normal
        mock_tool._run.side_effect = [
            {"success": True, "performance": {"mass_kg": 9999}},  # Abnormal — out of range
            {"success": True, "performance": {"mass_kg": 5.0}},
        ]
        sample_tools["freecad_mass_extractor"].instance = mock_tool

        config = SigmaConfig(reasonable_ranges={"mass_kg": (0.1, 100.0)})
        proto = SigmaProtocol(config=config, tools=sample_tools)
        plan = PlanOutput(
            agent_analyses={}, conflicts=[], dependency_graph={},
            tool_requests=["freecad_mass_extractor"], knowledge_gaps=[],
        )
        cost = RoundCost(round_num=1)
        result = proto.do(plan, sample_state, cost)
        # After retry, should get the correct value
        final = result.tool_results.get("freecad_mass_extractor", {})
        perf = final.get("performance", {})
        assert perf.get("mass_kg") == 5.0

    def test_do_tool_exception_caught(self, sample_state, sample_tools):
        mock_tool = mock.Mock()
        mock_tool._run.side_effect = RuntimeError("tool crash")
        sample_tools["freecad_mass_extractor"].instance = mock_tool

        proto = SigmaProtocol(tools=sample_tools)
        plan = PlanOutput(
            agent_analyses={}, conflicts=[], dependency_graph={},
            tool_requests=["freecad_mass_extractor"], knowledge_gaps=[],
        )
        cost = RoundCost(round_num=1)
        result = proto.do(plan, sample_state, cost)
        tool_result = result.tool_results.get("freecad_mass_extractor", {})
        assert tool_result.get("success") is False
        assert "tool crash" in tool_result.get("error", "")


# ── Check Phase ──────────────────────────────────────────────────

class TestCheckPhase:
    def test_check_basic(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "analysis", "Propulsion Chief": "analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        do_output = DoOutput(tool_results={}, abnormal_results=[])
        cost = RoundCost(round_num=1)
        result = proto.check(sample_state, plan, do_output, cost, None)
        assert isinstance(result, CheckOutput)
        assert isinstance(result.reviews, dict)
        assert isinstance(result.updated_conflicts, list)

    def test_check_lite_skips_devil_advocate(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        do_output = DoOutput(tool_results={}, abnormal_results=[])
        cost = RoundCost(round_num=1)
        assessment = ComplexityAssessment(
            tier=ComplexityTier.LITE, score=1.0, reason="test",
            agent_names=["Director"], max_rounds=1,
            cross_review=False, devil_advocate=False,
            consensus_estimation=False, skill_crafter=False,
        )
        result = proto.check(sample_state, plan, do_output, cost, assessment)
        assert result.devil_advocate == ""

    def test_check_rigorous_includes_devil_advocate(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        do_output = DoOutput(tool_results={}, abnormal_results=[])
        cost = RoundCost(round_num=1)
        assessment = ComplexityAssessment(
            tier=ComplexityTier.RIGOROUS, score=8.0, reason="complex",
            agent_names=list(sample_agents.keys()), max_rounds=4,
            cross_review=True, devil_advocate=True,
            consensus_estimation=True, skill_crafter=True,
        )
        result = proto.check(sample_state, plan, do_output, cost, assessment)
        assert len(result.devil_advocate) > 0

    def test_check_detects_data_deviations(self, mock_llm, sample_state, sample_agents, sample_tools):
        config = SigmaConfig(reasonable_ranges={"mass_kg": (0.1, 100.0)})
        proto = SigmaProtocol(config=config, llm_backend=mock_llm, agents=sample_agents, tools=sample_tools)
        # Seed task_params so deviation check has a previous value to compare
        sample_state.task_params["mass_kg"] = 5.0
        plan = PlanOutput(
            agent_analyses={"Director": "analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        do_output = DoOutput(tool_results={
            "freecad_mass_extractor": {
                "success": True,
                "performance": {"mass_kg": 9999},
            },
        }, abnormal_results=[])
        cost = RoundCost(round_num=1)
        result = proto.check(sample_state, plan, do_output, cost, None)
        assert len(result.data_deviations) >= 1


# ── Act Phase ────────────────────────────────────────────────────

class TestActPhase:
    def test_act_first_round_continues(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        check = CheckOutput(
            reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[],
        )
        cost = RoundCost(round_num=1)
        result = proto.act(sample_state, plan, check, cost, prev_state=None)
        assert result.verdict == "converging"
        assert result.should_continue is True

    def test_act_with_physical_impossible(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={
                "Director": "The Isp is 500s and this is physically impossible for this propellant",
            },
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        check = CheckOutput(
            reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[],
        )
        cost = RoundCost(round_num=1)
        prev_state = SharedState(task_instruction="test")
        prev_state.round_num = 1
        sample_state.round_num = 2
        sample_state.history = [
            type("R", (), {"phase_outputs": {"do": {"tool_results": {}}}})(),
        ]
        result = proto.act(sample_state, plan, check, cost, prev_state=prev_state)
        # "不可能" / "impossible" keyword should trigger alarm
        assert result.verdict == "STALLED"
        assert result.should_continue is False
        assert len(result.alarm_flags) > 0

    def test_act_second_round_uses_judge(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "reasonable analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        check = CheckOutput(
            reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[],
        )
        cost = RoundCost(round_num=2)
        prev_state = SharedState(task_instruction="test")
        prev_state.round_num = 1
        sample_state.round_num = 2
        sample_state.history = [
            type("R", (), {"phase_outputs": {"do": {"tool_results": {}}}})(),
        ]
        result = proto.act(sample_state, plan, check, cost, prev_state=prev_state)
        assert result.verdict in ("converged", "converging")

    def test_act_creates_decisions_for_resolved_conflicts(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "reasonable analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        check = CheckOutput(
            reviews={},
            updated_conflicts=[
                Conflict(id="c1", description="resolved issue", owners=["Director"],
                         severity=2.0, trend="improving", quantitative=False, values={}),
            ],
            devil_advocate="", data_deviations=[],
        )
        cost = RoundCost(round_num=2)
        prev_state = SharedState(task_instruction="test")
        prev_state.round_num = 1
        prev_state.task_params["mass_kg"] = 5.0
        sample_state.round_num = 2
        sample_state.history = [
            type("R", (), {"phase_outputs": {"do": {"tool_results": {}}}})(),
        ]
        sample_state.task_params["mass_kg"] = 5.1
        result = proto.act(sample_state, plan, check, cost, prev_state=prev_state)
        assert len(result.decisions) >= 1
        assert any("resolved" in d.decision for d in result.decisions)

    def test_act_skips_consensus_estimation_in_lite(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={"Director": "analysis"},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        check = CheckOutput(
            reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[],
        )
        do_output = DoOutput(tool_results={
            "some_tool": {"success": "simulated", "note": "no data"},
        }, abnormal_results=[])
        cost = RoundCost(round_num=2)
        prev_state = SharedState(task_instruction="test")
        prev_state.round_num = 1
        prev_state.task_params["mass_kg"] = 5.0
        sample_state.round_num = 2
        sample_state.history = [
            type("R", (), {"phase_outputs": {"do": {"tool_results": {}}}})(),
        ]
        sample_state.task_params["mass_kg"] = 5.0
        assessment = ComplexityAssessment(
            tier=ComplexityTier.LITE, score=1.0, reason="test",
            agent_names=["Director"], max_rounds=1,
            cross_review=False, devil_advocate=False,
            consensus_estimation=False, skill_crafter=False,
        )
        result = proto.act(sample_state, plan, check, cost, prev_state=prev_state,
                          do_output=do_output, complexity=assessment)
        assert result.consensus == []


# ── Skill Loading ────────────────────────────────────────────────

class TestSkillLoading:
    def test_get_skills_for_agent(self, sample_agents):
        proto = SigmaProtocol(
            agents=sample_agents,
            skills={"propulsion_basics": "KNSB is a solid propellant..."},
        )
        agent = AgentSpec(
            name="Test", role="r", goal="g", backstory="b",
            skill_files=["skills/propulsion_basics.md"],
        )
        result = proto._get_skills_for(agent)
        assert "KNSB" in result

    def test_get_skills_empty_when_no_match(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents, skills={})
        agent = AgentSpec(
            name="Test", role="r", goal="g", backstory="b",
            skill_files=["skills/nonexistent.md"],
        )
        result = proto._get_skills_for(agent)
        assert result == ""

    def test_get_skills_no_skill_files(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        agent = AgentSpec(name="Test", role="r", goal="g", backstory="b")
        result = proto._get_skills_for(agent)
        assert result == ""


# ── Tool Defaults ────────────────────────────────────────────────

class TestToolDefaults:
    def test_try_tool_defaults_from_spec(self, sample_tools):
        mock_tool = mock.Mock()
        mock_tool._run.return_value = {"success": True, "performance": {"Isp": 180}}
        sample_tools["rocketcea_analyzer"].instance = mock_tool
        sample_tools["rocketcea_analyzer"].default_params = {
            "fuel": "KNSB", "ox": "N2O", "Pc": 20,
        }

        proto = SigmaProtocol(tools=sample_tools)
        result = proto._try_tool_defaults(sample_tools["rocketcea_analyzer"], "rocketcea_analyzer")
        assert result["success"] is True
        mock_tool._run.assert_called_once_with(fuel="KNSB", ox="N2O", Pc=20)

    def test_try_tool_defaults_from_config(self):
        mock_tool = mock.Mock()
        mock_tool._run.return_value = {"success": True, "performance": {}}
        spec = ToolSpec(
            name="my_tool", instance=mock_tool,
            expected_outputs=["result"],
        )
        config = SigmaConfig(default_tool_params={"my_tool": {"mode": "default"}})
        proto = SigmaProtocol(config=config, tools={"my_tool": spec})
        result = proto._try_tool_defaults(spec, "my_tool")
        assert result["success"] is True
        mock_tool._run.assert_called_once_with(mode="default")

    def test_try_tool_defaults_no_defaults_returns_simulated(self, sample_tools):
        mock_tool = mock.Mock()
        spec = ToolSpec(name="bare_tool", instance=mock_tool)
        proto = SigmaProtocol(tools={"bare_tool": spec})
        result = proto._try_tool_defaults(spec, "bare_tool")
        assert result["success"] == "simulated"


# ── Params for Tool ──────────────────────────────────────────────

class TestParamsForTool:
    def test_returns_expected_outputs(self, sample_tools):
        proto = SigmaProtocol(tools=sample_tools)
        params = proto._params_for_tool("freecad_mass_extractor")
        assert "mass_kg" in params
        assert "volume_mm3" in params

    def test_fuzzy_name_match(self, sample_tools):
        proto = SigmaProtocol(tools=sample_tools)
        params = proto._params_for_tool("freecad")
        assert "mass_kg" in params

    def test_unknown_tool_returns_empty(self):
        proto = SigmaProtocol()
        params = proto._params_for_tool("nonexistent")
        assert params == []


# ── Identify Unreliable Params ───────────────────────────────────

class TestIdentifyUnreliableParams:
    def test_simulated_results_flagged(self, sample_tools, sample_state):
        proto = SigmaProtocol(tools=sample_tools)
        do_output = DoOutput(
            tool_results={
                "freecad_mass_extractor": {
                    "success": "simulated",
                    "note": "no data",
                },
            },
            abnormal_results=[],
        )
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        result = proto._identify_unreliable_params(do_output, check, sample_state)
        assert "mass_kg" in result or "volume_mm3" in result

    def test_failed_results_flagged(self, sample_tools, sample_state):
        proto = SigmaProtocol(tools=sample_tools)
        do_output = DoOutput(
            tool_results={
                "freecad_mass_extractor": {
                    "success": False,
                    "error": "tool failed",
                },
            },
            abnormal_results=[],
        )
        check = CheckOutput(reviews={}, updated_conflicts=[], devil_advocate="", data_deviations=[])
        result = proto._identify_unreliable_params(do_output, check, sample_state)
        assert "mass_kg" in result or "volume_mm3" in result

    def test_review_anomalies_flagged(self, sample_state):
        proto = SigmaProtocol()
        sample_state.task_params["Isp"] = 180
        do_output = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(
            reviews={"A": "Isp数据不可靠，需要重新计算"},
            updated_conflicts=[], devil_advocate="", data_deviations=[],
        )
        result = proto._identify_unreliable_params(do_output, check, sample_state)
        assert "Isp" in result

    def test_empty_when_all_reliable(self, sample_state):
        proto = SigmaProtocol()
        do_output = DoOutput(tool_results={}, abnormal_results=[])
        check = CheckOutput(reviews={"A": "Everything looks good"}, updated_conflicts=[],
                          devil_advocate="", data_deviations=[])
        result = proto._identify_unreliable_params(do_output, check, sample_state)
        assert result == []


# ── Relevant Agents for Param ────────────────────────────────────

class TestRelevantAgentsForParam:
    def test_finds_agents_mentioning_param(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={
                "Director": "Need to check Isp value",
                "Propulsion Chief": "Isp is critical for performance",
                "Structures Chief": "Structure mass only",
            },
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        state = SharedState(task_instruction="test")
        result = proto._relevant_agents_for_param("Isp", plan, state)
        assert "Director" in result
        assert "Propulsion Chief" in result

    def test_always_includes_director(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={},
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        state = SharedState(task_instruction="test")
        result = proto._relevant_agents_for_param("unknown_param", plan, state)
        assert "Director" in result

    def test_returns_at_most_five(self, sample_agents):
        proto = SigmaProtocol(agents=sample_agents)
        plan = PlanOutput(
            agent_analyses={
                name: f"mass is important for the {name}" for name in sample_agents
            },
            conflicts=[], dependency_graph={}, tool_requests=[], knowledge_gaps=[],
        )
        state = SharedState(task_instruction="test")
        result = proto._relevant_agents_for_param("mass", plan, state)
        assert len(result) <= 5


# ── Devil Advocate ───────────────────────────────────────────────

class TestDevilAdvocate:
    def test_returns_empty_without_safety_officer(self, mock_llm, sample_state):
        proto = SigmaProtocol(llm_backend=mock_llm, agents={
            "Director": AgentSpec(name="Director", role="d", goal="g", backstory="b"),
        })
        result = proto._devil_advocate(sample_state, {}, RoundCost(round_num=1))
        assert result == ""

    def test_calls_llm_when_safety_officer_exists(self, mock_llm, sample_state, sample_agents):
        proto = SigmaProtocol(llm_backend=mock_llm, agents=sample_agents)
        result = proto._devil_advocate(sample_state, {}, RoundCost(round_num=1))
        assert len(result) > 0
        assert mock_llm.chat.called
