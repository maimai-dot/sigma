"""
AERC 协议引擎 — Σ 框架核心
P (Plan): 独立分析 + 交叉审查
D (Do): 并行工具调用
C (Check): 数据驱动的跨角色审查
A (Act): 收敛判断 + 知识沉淀
"""

import concurrent.futures
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, get_type_hints

from sigma.state import (
    SharedState, StateManager, Conflict, Decision, AlarmFlag, RoundRecord,
    ConsensusEstimate, ComplexityTier, ComplexityAssessment,
)
from sigma.convergence import ConvergenceJudge
from sigma.triggers import TriggerSystem, Trigger
from sigma.cost_tracker import CostTracker, RoundCost, DEEPSEEK_PRICING
from sigma.config import SigmaConfig
from sigma.llm import LLMBackend, LLMResponse, UniversalBackend, AgentLLMConfig
from sigma.cache import CachedLLMBackend, CacheConfig
from sigma.schema_validator import validate_against_schema

# ── Agent Registry ──────────────────────────────────────────────

@dataclass
class AgentSpec:
    """智能体规格定义（从 agents/*.py 提取）."""
    name: str
    role: str
    goal: str
    backstory: str
    skill_files: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    tool_instances: list = field(default_factory=list)
    output_schema: dict | None = None
    llm_config: AgentLLMConfig | None = None
    """Per-agent LLM overrides: model, temperature, max_tokens, max_rpm."""

    def system_prompt(self, extra_context: str = "", creed: str = "") -> str:
        if creed:
            prompt = f"{creed}\n\n{self.backstory}\n\n你的职责: {self.goal}"
        else:
            prompt = f"{self.backstory}\n\n你的职责: {self.goal}"
        if extra_context:
            prompt += f"\n\n{extra_context}"
        return prompt

    def get_llm_params(self, defaults: dict) -> dict:
        """Resolve LLM params, merging per-agent overrides with defaults.

        Args:
            defaults: dict with keys model, max_tokens, temperature.

        Returns:
            Merged dict with per-agent overrides taking precedence.
        """
        if self.llm_config is None:
            return defaults
        result = dict(defaults)
        if self.llm_config.model:
            result["model"] = self.llm_config.model
        if self.llm_config.max_tokens:
            result["max_tokens"] = self.llm_config.max_tokens
        if self.llm_config.temperature is not None:
            result["temperature"] = self.llm_config.temperature
        return result

    def __hash__(self):
        return hash(self.name)


@dataclass
class ToolSpec:
    """工具元数据 — 取代旧的 _TOOL_REGISTRY + 硬编码的别名/默认参数."""
    name: str
    instance: Any = None
    aliases: list[str] = field(default_factory=list)
    default_params: dict = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    description: str = ""

    _TYPE_MAP = {
        str: "string", int: "integer", float: "number",
        bool: "boolean", list: "array", dict: "object",
    }

    def to_openai_schema(self) -> dict:
        """生成 OpenAI function calling 格式的工具定义."""
        properties = {}
        required: list[str] = []

        if self.instance and hasattr(self.instance, "_run"):
            try:
                hints = get_type_hints(self.instance._run)
                sig = inspect.signature(self.instance._run)
            except Exception:
                hints = {}
                sig = None
        else:
            hints = {}
            sig = None

        if sig:
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                py_type = hints.get(param_name, str)
                json_type = self._TYPE_MAP.get(py_type, "string")

                # Handle Optional[X]
                origin = getattr(py_type, "__origin__", None)
                if origin is type(None):  # noqa: E721
                    continue
                if origin is list:
                    json_type = "array"
                elif origin is dict:
                    json_type = "object"

                properties[param_name] = {"type": json_type}
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

        # 如果无法从签名获取参数，从 default_params 反推
        if not properties and self.default_params:
            for key, val in self.default_params.items():
                json_type = self._TYPE_MAP.get(type(val), "string")
                properties[key] = {"type": json_type}

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or f"调用 {self.name} 工具",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ── Plan Output ─────────────────────────────────────────────────

@dataclass
class PlanOutput:
    agent_analyses: dict[str, str]       # agent_name → analysis text
    conflicts: list[Conflict]
    dependency_graph: dict[str, list[str]]
    tool_requests: list[str]              # 需要调用的工具名列表
    knowledge_gaps: list[Trigger]


@dataclass
class DoOutput:
    tool_results: dict[str, dict]         # tool_name → result
    abnormal_results: list[Trigger]


@dataclass
class CheckOutput:
    reviews: dict[str, str]               # agent_name → review text
    updated_conflicts: list[Conflict]
    devil_advocate: str
    data_deviations: list[Trigger]


@dataclass
class ActOutput:
    verdict: str                          # CONVERGED | CONVERGING | OSCILLATING | SLOW | STALLED
    decisions: list[Decision]
    alarm_flags: list[AlarmFlag]
    should_continue: bool
    reason: str
    consensus: list[ConsensusEstimate] = field(default_factory=list)


# ── Sigma Protocol ─────────────────────────────────────────────

class SigmaProtocol:
    """AERC 协议引擎."""

    def __init__(
        self,
        config: SigmaConfig | None = None,
        agents: dict[str, AgentSpec] | None = None,
        tools: dict[str, "ToolSpec"] | None = None,
        skills: dict[str, str] | None = None,
        llm_backend: LLMBackend | None = None,
        model: str | None = None,
        verbose: bool = True,
        stream_callback: callable | None = None,
    ):
        """
        Args:
            stream_callback: Optional callback(str) called for each token
                             during streaming LLM responses.
        """
        self.verbose = verbose
        self.config = config or SigmaConfig()
        self.state_mgr = StateManager()
        self.judge = ConvergenceJudge()
        self.triggers = TriggerSystem()
        self.cost_tracker = CostTracker(DEEPSEEK_PRICING)
        self._stream_callback = stream_callback

        # LLM 后端（可注入，默认 UniversalBackend，兼容任何 OpenAI 格式 API）
        llm = llm_backend or UniversalBackend()
        cache_cfg = self.config.cache_config
        if cache_cfg and cache_cfg.enabled:
            llm = CachedLLMBackend(llm, cache_cfg)
        self.llm = llm
        self.model = model or self.config.default_model

        # 智能体 + 工具 + 技能（由应用层注入）
        self.agents = agents or {}
        self.tools = tools or {}
        self.skill_cache = skills or {}

        # 若有应用配置的合理范围，注入触发器
        if self.config.reasonable_ranges:
            self.triggers.REASONABLE_RANGES.update(self.config.reasonable_ranges)

    def _log(self, msg: str) -> None:
        if self.verbose:
            from sigma.log import get_logger
            get_logger("sigma.protocol").info(msg)

    # ── Complexity Assessment ───────────────────────────────────

    def _assess_complexity(self, instruction: str) -> ComplexityAssessment:
        """评估任务复杂度，返回分层配置。

        纯规则驱动，零 LLM 调用 — 为框架的元决策保持快速/免费。
        """
        text = instruction.lower()
        score = 0.0
        reasons: list[str] = []

        # 1. 任务长度 (0-1 分)
        length = len(instruction)
        if length < 60:
            score += 0.0
        elif length < 150:
            score += 0.3
            reasons.append(f"中等长度 ({length} 字符)")
        elif length < 400:
            score += 0.7
            reasons.append(f"较长 ({length} 字符)")
        else:
            score += 1.0
            reasons.append(f"很长 ({length} 字符)")

        # 2. 工程领域覆盖 (0-3 分)
        domain_scores: dict[str, float] = {}
        for domain, keywords in self.config.domain_keywords.items():
            hits = sum(1 for kw in keywords if kw.lower() in text)
            if hits > 0:
                domain_scores[domain] = min(hits, 3) * 0.5
        score += sum(domain_scores.values())
        domain_count = len(domain_scores)
        if domain_count >= 5:
            reasons.append(f"跨 {domain_count} 个工程领域")
        elif domain_count >= 3:
            reasons.append(f"涉及 {domain_count} 个领域")
        elif domain_count > 0:
            reasons.append(f"聚焦 {domain_count} 个领域")
        else:
            reasons.append("无明确工程领域")

        # 3. 数值参数数量 (0-2 分)
        num_pattern = re.compile(
            r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|kg|g|s|ms|bar|atm|K|N|kN|%|度|°)'
        )
        param_count = len(num_pattern.findall(instruction))
        if param_count <= 1:
            score += 0.0
        elif param_count <= 4:
            score += 0.5
            reasons.append(f"{param_count} 个数值参数")
        elif param_count <= 10:
            score += 1.0
            reasons.append(f"{param_count} 个数值参数")
        else:
            score += 2.0
            reasons.append(f"{param_count} 个数值参数")

        # 4. 约束关键词 (0-2 分)
        constraint_score = 0.0
        for kw, weight in self.config.constraint_keywords.items():
            if kw.lower() in text:
                constraint_score += weight
        score += min(constraint_score, 2.0)

        # 5. 行动动词 (0-2 分)
        action_score = 0.0
        for verb, weight in self.config.action_weights.items():
            if verb.lower() in text:
                action_score = max(action_score, weight)
        score += action_score * 0.4  # 缩放到 0-2 范围
        if action_score >= 4.0:
            reasons.append("设计/优化级任务")
        elif action_score >= 2.0:
            reasons.append("计算/分析级任务")
        else:
            reasons.append("查询/搜索级任务")

        # 6. 是否包含"或/或者/选择/方案" → 决策复杂度 (0-0.5 分)
        decision_kw = ["或", "或者", "选择", "方案", "or", "choice", "option", "alternative"]
        if any(kw in text for kw in decision_kw):
            score += 0.5
            reasons.append("包含决策选择")

        # 归一化到 0-10
        score = min(score, 10.0)

        # 确定分层
        if score <= 2.5:
            tier = ComplexityTier.LITE
        elif score <= 6.0:
            tier = ComplexityTier.STANDARD
        else:
            tier = ComplexityTier.RIGOROUS

        # 确定激活角色列表
        agent_names = self._select_agents_for_tier(tier, domain_scores, instruction)
        max_rounds = {"lite": 1, "standard": 3, "rigorous": 4}[tier.value]

        return ComplexityAssessment(
            tier=tier,
            score=score,
            reason="; ".join(reasons) if reasons else "默认评估",
            agent_names=agent_names,
            max_rounds=max_rounds,
            cross_review=tier != ComplexityTier.LITE,
            devil_advocate=tier == ComplexityTier.RIGOROUS,
            consensus_estimation=tier != ComplexityTier.LITE,
            skill_crafter=tier == ComplexityTier.RIGOROUS,
        )

    def _select_agents_for_tier(
        self, tier: ComplexityTier, domain_scores: dict[str, float], instruction: str,
    ) -> list[str]:
        """根据分层和任务内容选择激活角色."""
        all_agents = list(self.agents.keys())

        if tier == ComplexityTier.LITE:
            # Director + 2-3 个最相关领域角色
            selected = []
            # Director 总是包含
            for name in all_agents:
                if "director" in name.lower():
                    selected.append(name)
                    break
            # 根据领域得分选择 top 2-3
            domain_agent_map = self.config.domain_agent_map
            ranked_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
            for domain, _ in ranked_domains[:3]:
                agent_name = domain_agent_map.get(domain)
                if agent_name and agent_name in all_agents:
                    selected.append(agent_name)
            # 去重
            seen = set()
            result = []
            for a in selected:
                if a not in seen:
                    result.append(a)
                    seen.add(a)
            return result[:4]  # LITE 最多 4 个

        elif tier == ComplexityTier.STANDARD:
            # 根据配置排除部分角色（默认排除空集，不排除任何角色）
            exclude = self.config.standard_exclude_agents
            return [n for n in all_agents if n not in exclude]

        else:  # RIGOROUS
            return all_agents

    def _call_llm(
        self, system_prompt: str, user_prompt: str,
        cost: Optional[RoundCost] = None,
        agent_name: str = "",
    ) -> str:
        """调用 LLM，返回响应文本。自动使用流式输出（带进度显示）。

        Args:
            agent_name: If set, per-agent LLM config (model/temperature/max_tokens)
                        overrides global defaults from SigmaConfig.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Resolve per-agent LLM params
        model = self.model
        max_tokens = self.config.default_max_tokens
        temperature = self.config.default_temperature
        if agent_name:
            agent_spec = self.agents.get(agent_name)
            if agent_spec and agent_spec.llm_config:
                cfg = agent_spec.llm_config
                if cfg.model:
                    model = cfg.model
                if cfg.max_tokens:
                    max_tokens = cfg.max_tokens
                if cfg.temperature is not None:
                    temperature = cfg.temperature

        # Use streaming for real backends (skip Mock objects whose spec enables hasattr)
        is_mock = "mock" in type(self.llm).__module__
        use_stream = not is_mock and hasattr(self.llm, "chat_stream")
        if use_stream:
            resp = self.llm.chat_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                on_chunk=self._stream_chunk_callback,
            )
        else:
            resp = self.llm.chat(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if cost:
            cost.calls += 1
            if resp.input_tokens or resp.output_tokens:
                self.cost_tracker.record_call(cost, resp.input_tokens, resp.output_tokens)
            else:
                self.cost_tracker.record_call(
                    cost, len(system_prompt) // 4, 500,
                )
        return resp.content

    def _call_with_tools(
        self, system_prompt: str, user_prompt: str,
        tool_schemas: list[dict],
        cost: Optional[RoundCost] = None,
        agent_name: str = "",
    ) -> LLMResponse:
        """调用 LLM 并传入工具定义，返回完整 LLMResponse（含 tool_calls）."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        model = self.model
        max_tokens = self.config.default_max_tokens
        temperature = self.config.default_temperature
        if agent_name:
            agent_spec = self.agents.get(agent_name)
            if agent_spec and agent_spec.llm_config:
                cfg = agent_spec.llm_config
                if cfg.model:
                    model = cfg.model
                if cfg.max_tokens:
                    max_tokens = cfg.max_tokens
                if cfg.temperature is not None:
                    temperature = cfg.temperature

        is_mock = "mock" in type(self.llm).__module__
        use_stream = not is_mock and hasattr(self.llm, "chat_stream")
        if use_stream:
            resp = self.llm.chat_stream(
                messages=messages, model=model,
                max_tokens=max_tokens, temperature=temperature,
                tools=tool_schemas,
                on_chunk=self._stream_chunk_callback,
            )
        else:
            resp = self.llm.chat(
                messages=messages, model=model,
                max_tokens=max_tokens, temperature=temperature,
                tools=tool_schemas,
            )

        if cost:
            cost.calls += 1
            if resp.input_tokens or resp.output_tokens:
                self.cost_tracker.record_call(cost, resp.input_tokens, resp.output_tokens)
            else:
                self.cost_tracker.record_call(cost, len(system_prompt) // 4, 500)
        return resp

    def _stream_chunk_callback(self, chunk: str) -> None:
        """Called for each token during streaming.

        Delegates to user-provided callback if set; otherwise shows progress
        milestones when verbose is enabled.
        """
        if self._stream_callback:
            self._stream_callback(chunk)
            return
        if self.verbose:
            self._on_stream_chunk(chunk)

    def _on_stream_chunk(self, chunk: str) -> None:
        """Show progress milestones during streaming."""
        if not hasattr(self, "_stream_buf"):
            self._stream_buf = ""
        self._stream_buf += chunk
        if "\n" in self._stream_buf or len(self._stream_buf) >= 300:
            self._log(f"    ...")
            self._stream_buf = ""

    def _call_llm_validated(
        self, system_prompt: str, user_prompt: str,
        cost: Optional[RoundCost] = None,
        schema: dict | None = None,
        max_retries: int = 2,
    ) -> tuple[str, dict | None]:
        """Call LLM, parse JSON, validate against schema, retry on failure."""
        prompt = user_prompt
        for _ in range(max_retries + 1):
            resp = self._call_llm(system_prompt, prompt, cost)

            # Check for LLM error
            if resp.startswith("[LLM_ERROR"):
                return resp, None

            if schema is None:
                return resp, None

            data = self._parse_json_response(resp)
            if data is None:
                prompt = user_prompt + "\n\n[ERROR] 你的回复不是有效的JSON。请只输出JSON对象。重试。"
                continue

            errors = validate_against_schema(data, schema)
            if not errors:
                return resp, data

            prompt = user_prompt + f"\n\n[ERROR] 输出格式错误: {'; '.join(errors)}。请严格按照schema输出JSON。重试。"

        return resp, None

    # ── P: Plan ────────────────────────────────────────────────

    def plan(self, state: SharedState, cost: RoundCost,
             complexity: Optional[ComplexityAssessment] = None) -> PlanOutput:
        """P 阶段：独立分析 + 交叉审查（复杂度自适应）."""
        # 根据复杂度选择激活角色
        if complexity is not None and complexity.agent_names:
            agent_names = complexity.agent_names
        else:
            agent_names = list(self.agents.keys())

        n_agents = len(agent_names)
        tier_label = complexity.tier.value if complexity else "rigorous"
        self._log(f"\n  [P] 独立分析 — {n_agents} 个智能体 [{tier_label}]...")

        # Phase 1: 并行独立分析
        analyses = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_agents)) as pool:
            futures = {
                pool.submit(self._agent_analyze, name, state, cost): name
                for name in agent_names
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    analyses[name] = fut.result()
                except Exception as e:
                    analyses[name] = f"[ERROR: {e}]"

        # 提取冲突
        conflicts = self._extract_conflicts(analyses, state.round_num)

        # 检测知识缺口
        knowledge_gaps = self.triggers.check_knowledge_gap(analyses)

        # 构建依赖图
        dep_graph = self._build_dependency_graph(analyses)

        # Phase 2: 交叉审查（LITE 跳过）
        do_cross_review = complexity is None or complexity.cross_review
        if do_cross_review and len(agent_names) >= 2:
            self._log(f"  [P] 交叉审查 — {n_agents} 角色并行互审...")
            cross_reviews = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_agents)) as pool:
                futures = {
                    pool.submit(
                        self._agent_cross_review, name, analyses,
                        conflicts, state, cost,
                    ): name
                    for name in agent_names
                }
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    try:
                        cross_reviews[name] = fut.result()
                    except Exception as e:
                        cross_reviews[name] = f"[CROSS_REVIEW_ERROR: {e}]"

            # 合并交叉审查到分析
            for name, review in cross_reviews.items():
                if name in analyses:
                    analyses[name] = analyses[name] + "\n\n--- 交叉审查 ---\n" + review

        # 提取工具需求
        tool_requests = self._extract_tool_requests(analyses)

        # 更新冲突
        updated_conflicts = self._extract_conflicts(analyses, state.round_num)

        self.state_mgr.update_after_plan(
            state, analyses, updated_conflicts, dep_graph, tool_requests,
        )

        return PlanOutput(
            agent_analyses=analyses,
            conflicts=updated_conflicts,
            dependency_graph=dep_graph,
            tool_requests=tool_requests,
            knowledge_gaps=knowledge_gaps,
        )

    def _agent_analyze(
        self, name: str, state: SharedState, cost: RoundCost,
    ) -> str:
        """单个智能体的独立分析."""
        agent = self.agents.get(name)
        if not agent:
            return f"Agent '{name}' not found."

        # 构建 prompt
        shared_context = state.to_context()
        skills_text = self._get_skills_for(agent)

        system = agent.system_prompt(
            f"\n\n当前工程共享状态:\n{shared_context}"
            f"\n\n你的可用工具: {', '.join(agent.tool_names) if agent.tool_names else '无'}"
            f"\n\n{skills_text}",
            creed=self.config.creed,
        )

        user = (
            f"任务: {state.task_instruction}\n\n"
            "请从你的专业视角独立分析这个任务:\n"
            "1. 你的专业判断是什么？\n"
            "2. 哪些关键参数需要计算或验证？\n"
            "   如需调用工具，请用 [需要工具: 工具名] 格式明确标注\n"
            "   可用工具: " + ", ".join(agent.tool_names) + "\n"
            "3. 你依赖哪些其他专业角色的判断？（标注 [依赖: 角色名]）\n"
            "4. 你的初步结论和建议\n"
            "5. 是否有不确定或需要搜索的知识点？\n\n"
            "请给出具体数值和推理过程，不要泛泛而谈。"
        )
        return self._call_llm(system, user, cost, agent_name=name)

    def _agent_cross_review(
        self, name: str, analyses: dict[str, str],
        conflicts: list[Conflict], state: SharedState, cost: RoundCost,
    ) -> str:
        """单个智能体交叉审查其他智能体的相关结论."""
        agent = self.agents.get(name)
        if not agent:
            return ""

        # 找出与本角色相关的冲突
        relevant_conflicts = [
            c for c in conflicts if name in c.owners
            or any(name in state.dependency_graph.get(o, []) for o in c.owners)
        ]

        # 找出本角色依赖的其他角色的结论
        deps = state.dependency_graph.get(name, [])
        relevant_analyses = {}
        for dep_name in deps:
            if dep_name in analyses:
                text = analyses[dep_name]
                # 只取前 800 字
                relevant_analyses[dep_name] = text[:800]

        if not relevant_conflicts and not relevant_analyses:
            return ""

        conflict_text = "\n".join(
            f"- {c.description} (严重度: {c.severity}/10)"
            for c in relevant_conflicts
        ) or "无直接冲突"

        other_text = "\n\n".join(
            f"**{n}**: {t}" for n, t in relevant_analyses.items()
        ) or "无相关角色输出"

        system = agent.system_prompt()
        user = (
            f"请审查以下与你相关的信息和冲突:\n\n"
            f"## 与你相关的其他角色结论\n{other_text}\n\n"
            f"## 与你相关的冲突\n{conflict_text}\n\n"
            "请回答:\n"
            "1. 其他角色的结论是否有你以为错误或遗漏的？\n"
            "2. 与你相关的冲突，你的立场是什么？\n"
            "3. 你是否需要调整自己之前的分析？"
        )
        return self._call_llm(system, user, cost, agent_name=name)

    def _extract_conflicts(
        self, analyses: dict[str, str], round_num: int,
    ) -> list[Conflict]:
        """从智能体分析中提取冲突（简化版：基于关键词 + 值差异）."""
        # 使用 LLM 做一次汇总冲突提取
        summary_prompt = (
            "以下是各工程角色的分析。请提取角色之间的分歧和冲突:\n\n"
        )
        for name, text in analyses.items():
            summary_prompt += f"### {name}\n{text[:400]}\n\n"
        summary_prompt += (
            "请列出所有冲突，格式: JSON array [{id, description, owners, severity(0-10)}]"
            "只输出 JSON，不要其他文字。"
        )

        try:
            resp = self._call_llm(
                "你是系统工程评审官，负责发现跨角色冲突。",
                summary_prompt,
            )
            # 尝试解析 JSON
            json_start = resp.find("[")
            json_end = resp.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(resp[json_start:json_end])
                conflicts = []
                for item in data:
                    conflicts.append(Conflict(
                        id=item.get("id", f"c{len(conflicts)}"),
                        description=item.get("description", ""),
                        owners=item.get("owners", []),
                        severity=float(item.get("severity", 5)),
                        trend="new",
                        quantitative=False,
                        values={round_num: float(item.get("severity", 5))},
                    ))
                return conflicts
        except Exception:
            pass
        return []

    def _build_dependency_graph(
        self, analyses: dict[str, str],
    ) -> dict[str, list[str]]:
        """从分析文本中提取依赖关系."""
        graph = {}
        for name, text in analyses.items():
            deps = []
            for other_name in self.agents:
                if other_name != name and other_name.lower() in text.lower():
                    deps.append(other_name)
            graph[name] = deps[:5]  # 最多 5 个依赖
        return graph

    def _extract_tool_requests(self, analyses: dict[str, str]) -> list[str]:
        """提取需要调用的工具列表（匹配 [需要工具: xxx] 或 ToolSpec.aliases）."""
        requested = set()
        for text in analyses.values():
            text_str = str(text)
            text_lower = text_str.lower()
            # 匹配明确的工具请求标记
            for match in re.finditer(
                r'\[(?:需要工具|工具需求|调用工具|NEED_TOOL)\s*[:：]\s*([^\]]+)\]',
                text_str, re.IGNORECASE,
            ):
                for name in match.group(1).split(","):
                    requested.add(name.strip())
            # 匹配注册的工具名
            for tool_name, spec in self.tools.items():
                if tool_name.lower() in text_lower:
                    requested.add(tool_name)
                # 别名匹配（从 ToolSpec.aliases）
                for alias in spec.aliases:
                    if len(alias) >= 4 and alias.lower() in text_lower:
                        requested.add(tool_name)
                        break
        return list(requested)

    def _get_skills_for(self, agent: AgentSpec) -> str:
        """获取智能体关联的技能文件内容."""
        texts = []
        for skill_file in agent.skill_files:
            key = Path(skill_file).stem
            if key in self.skill_cache:
                texts.append(self.skill_cache[key][:1500])
        if texts:
            return "\n\n参考技能:\n" + "\n---\n".join(texts)
        return ""

    # ── D: Do ──────────────────────────────────────────────────

    def do(
        self, plan: PlanOutput, state: SharedState, cost: RoundCost,
    ) -> DoOutput:
        """D 阶段：并行工具调用 + 即时合理性检查.

        优先使用 OpenAI function calling（如果注册了工具），
        fallback 到文本标记提取 + 默认参数执行。
        """
        abnormal = []
        results = {}

        # ── 尝试 function calling 路径 ──
        if self.tools:
            tool_schemas = [spec.to_openai_schema() for spec in self.tools.values()]
            self._log(f"  [D] Function calling — {len(tool_schemas)} 个工具可用...")

            fc_prompt = (
                f"任务: {state.task_instruction}\n\n"
                "你需要调用哪些工具来完成上述任务？请用具体参数调用。"
            )
            try:
                resp = self._call_with_tools(
                    "你是一个工程执行系统。根据任务需求，选择合适的工具并给出具体参数。",
                    fc_prompt, tool_schemas, cost,
                )
            except Exception:
                resp = LLMResponse(content="")

            if resp.tool_calls:
                self._log(f"  [D] LLM 请求 {len(resp.tool_calls)} 个工具调用")
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(5, len(resp.tool_calls)),
                ) as pool:
                    futures = {}
                    for tc in resp.tool_calls:
                        func_name = tc["function"]["name"]
                        try:
                            func_args = json.loads(tc["function"]["arguments"])
                        except (json.JSONDecodeError, KeyError):
                            func_args = {}
                        spec = self.tools.get(func_name)
                        if spec and spec.instance:
                            futures[pool.submit(
                                self._execute_tool_with_args, spec, func_name, func_args,
                            )] = func_name

                    for fut in concurrent.futures.as_completed(futures):
                        tool_name = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as e:
                            result = {"success": False, "error": str(e)}

                        triggers = self.triggers.check_tool_abnormal(tool_name, result)
                        if triggers and any(t.severity >= 6 for t in triggers):
                            self._log(f"    ⚠ {tool_name} 异常")
                            for t in triggers:
                                abnormal.append(t)

                        if result.get("success", True):
                            self._log(f"    ✓ {tool_name}")
                        else:
                            self._log(f"    ✗ {tool_name}: {result.get('error', 'Unknown')}")
                        results[tool_name] = result

                self.state_mgr.update_after_do(state, results)
                return DoOutput(tool_results=results, abnormal_results=abnormal)

        # ── Fallback: 文本标记提取 ──
        if not plan.tool_requests:
            if not self.tools:
                self._log("    无工具请求，跳过")
            return DoOutput(tool_results={}, abnormal_results=[])

        self._log(f"  [D] 文本标记模式 — {len(plan.tool_requests)} 个工具...")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(5, len(plan.tool_requests)),
        ) as pool:
            futures = {}
            for tool_name in plan.tool_requests:
                spec = self.tools.get(tool_name)
                if spec and spec.instance:
                    futures[pool.submit(self._execute_tool, spec, tool_name)] = tool_name

            for fut in concurrent.futures.as_completed(futures):
                tool_name = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"success": False, "error": str(e)}

                triggers = self.triggers.check_tool_abnormal(tool_name, result)
                if triggers and any(t.severity >= 6 for t in triggers):
                    self._log(f"    ⚠ {tool_name} 异常，重试中...")
                    spec = self.tools.get(tool_name)
                    if spec and spec.instance:
                        try:
                            result = self._execute_tool(spec, tool_name)
                            triggers = self.triggers.check_tool_abnormal(tool_name, result)
                        except Exception as e:
                            result = {"success": False, "error": str(e)}

                if result.get("success", True):
                    self._log(f"    ✓ {tool_name}")
                else:
                    self._log(f"    ✗ {tool_name}: {result.get('error', 'Unknown')}")
                    for t in triggers:
                        abnormal.append(t)

                results[tool_name] = result

        self.state_mgr.update_after_do(state, results)
        return DoOutput(tool_results=results, abnormal_results=abnormal)

    def _execute_tool(self, spec: "ToolSpec", tool_name: str, max_retries: int = 1) -> dict:
        """执行单个工具，优先使用 ToolSpec 默认参数."""
        tool = spec.instance
        result = self._try_tool_defaults(spec, tool_name)
        if result.get("success") and result.get("success") != "simulated":
            return result

        for attempt in range(max_retries + 1):
            try:
                if hasattr(tool, "_run"):
                    result = tool._run()
                    if isinstance(result, dict) and result.get("success"):
                        return result
                elif hasattr(tool, "run"):
                    result = tool.run()
                    if isinstance(result, dict):
                        return result
            except TypeError:
                pass
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    return {"success": False, "error": str(e)}
        return {
            "success": "simulated",
            "note": f"Tool {tool_name} — no defaults or parameterless call failed",
        }

    def _execute_tool_with_args(
        self, spec: "ToolSpec", tool_name: str, args: dict,
    ) -> dict:
        """使用 LLM 提供的参数执行工具（function calling 路径）."""
        tool = spec.instance
        try:
            if hasattr(tool, "_run"):
                result = tool._run(**args)
            elif hasattr(tool, "run"):
                result = tool.run(**args)
            else:
                return {"success": False, "error": f"Tool {tool_name} has no callable method"}
            if isinstance(result, dict):
                return result
            return {"success": True, "result": result}
        except TypeError as e:
            # 参数不匹配 → fallback 到默认参数
            self._log(f"    ⚠ {tool_name} 参数不匹配 ({e})，使用默认参数...")
            return self._execute_tool(spec, tool_name)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _try_tool_defaults(self, spec: "ToolSpec", tool_name: str) -> dict:
        """尝试用默认参数调用工具（优先 ToolSpec.default_params，其次 config.default_tool_params）."""
        # 1. 从 ToolSpec 读取默认参数
        if spec.default_params:
            try:
                return spec.instance._run(**spec.default_params)
            except Exception as e:
                return {"success": False, "error": str(e)}
        # 2. 从 config 读取默认参数
        for key, params in self.config.default_tool_params.items():
            if key in tool_name:
                try:
                    return spec.instance._run(**params)
                except Exception as e:
                    return {"success": False, "error": str(e)}
        # 3. 无可用默认参数
        return {"success": "simulated", "note": f"Tool {tool_name} has no defaults"}

    # ── C: Check ───────────────────────────────────────────────

    def check(
        self, state: SharedState, plan: PlanOutput, do_output: DoOutput, cost: RoundCost,
        complexity: Optional[ComplexityAssessment] = None,
    ) -> CheckOutput:
        """C 阶段：数据驱动的跨角色审查（复杂度自适应）."""
        n_reviewers = min(6, len(plan.agent_analyses))
        self._log(f"  [C] 数据审查 — {n_reviewers} 个角色并行...")

        reviews = {}
        relevant_agents = list(plan.agent_analyses.keys())[:6]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_reviewers)) as pool:
            futures = {
                pool.submit(
                    self._agent_review_data, name, state, do_output.tool_results, cost,
                ): name
                for name in relevant_agents
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    reviews[name] = fut.result()
                except Exception as e:
                    reviews[name] = f"[REVIEW_ERROR: {e}]"

        # 魔鬼代言人（仅 RIGOROUS 模式执行）
        do_devil = complexity is None or complexity.devil_advocate
        devil = ""
        if do_devil:
            devil = self._devil_advocate(state, do_output.tool_results, cost)

        # 更新冲突
        updated_conflicts = self._extract_conflicts(reviews, state.round_num)

        # 数据偏差检测
        deviations = []
        for tool_name, result in do_output.tool_results.items():
            perf = result.get("performance", {})
            if perf:
                devs = self.triggers.check_data_deviation(
                    state.task_params, perf,
                )
                deviations.extend(devs)

        self.state_mgr.update_after_check(state, reviews, updated_conflicts, devil)
        return CheckOutput(
            reviews=reviews,
            updated_conflicts=updated_conflicts,
            devil_advocate=devil,
            data_deviations=deviations,
        )

    def _agent_review_data(
        self, name: str, state: SharedState, tool_results: dict, cost: RoundCost,
    ) -> str:
        """单个智能体审查计算结果."""
        agent = self.agents.get(name)
        if not agent:
            return ""

        results_text = ""
        for tname, res in tool_results.items():
            if isinstance(res, dict):
                perf = res.get("performance", {})
                if perf:
                    results_text += f"\n{tname}: " + ", ".join(
                        f"{k}={v}" for k, v in perf.items()
                    )
                elif res.get("success") == "simulated":
                    results_text += f"\n{tname}: [模拟结果] {res.get('note', '')}"
                else:
                    err = res.get("error", "")
                    if err:
                        results_text += f"\n{tname}: [失败] {err}"

        system = agent.system_prompt()
        user = (
            f"以下是工具计算结果，请从你的专业角度审查:\n{results_text}\n\n"
            f"参考共享状态:\n{state.to_context()}\n\n"
            "请回答:\n"
            "1. 这些数据是否符合你的专业预期？\n"
            "2. 是否有异常或矛盾之处？\n"
            "3. 这些数据是否要求你调整之前的分析？\n"
            "4. 如果数据不够，还需要什么额外计算？"
        )
        return self._call_llm(system, user, cost, agent_name=name)

    def _devil_advocate(
        self, state: SharedState, tool_results: dict, cost: RoundCost,
    ) -> str:
        """魔鬼代言人审查（由安全官执行）."""
        safety = self.agents.get("Safety Officer")
        if not safety:
            return ""

        results_text = ""
        for tname, res in tool_results.items():
            perf = res.get("performance", {})
            if perf:
                results_text += f"\n{tname}: {json.dumps(perf)}"

        system = safety.system_prompt()
        user = (
            "请刻意从最坏情况出发审视当前方案:\n\n"
            f"当前状态: {state.to_context()}\n"
            f"计算结果: {results_text}\n\n"
            "请回答:\n"
            "1. 如果这个方案失败了，最可能的三个原因是什么？\n"
            "2. 哪个假设最脆弱？\n"
            "3. 有没有数值在'刚刚好够'的边缘而没留安全裕度？\n"
            "4. 有没有人被其他角色的结论说服了但没有真正验证？"
        )
        return self._call_llm(system, user, cost, agent_name="Safety Officer")

    # ── Consensus Estimation ───────────────────────────────────

    def _estimate_single_param(
        self, param: str, plan: PlanOutput, state: SharedState,
        do_output: DoOutput, cost: RoundCost,
    ) -> ConsensusEstimate | None:
        """Estimate one parameter via multi-agent consensus (runs in parallel per-param)."""
        relevant = self._relevant_agents_for_param(param, plan, state)
        if len(relevant) < 2:
            return None

        # Parallel independent estimates across agents
        estimates = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(relevant)) as pool:
            futures = {
                pool.submit(
                    self._agent_estimate_param, name, param, state, do_output, cost,
                ): name
                for name in relevant
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    est = fut.result()
                    if est:
                        estimates[name] = est
                except Exception:
                    pass

        if len(estimates) < 2:
            return None

        estimates = self._cross_review_estimates(estimates, param, state, cost)

        values = [e["value"] for e in estimates.values()]
        mins = [e.get("min", e["value"]) for e in estimates.values()]
        maxs = [e.get("max", e["value"]) for e in estimates.values()]
        confidences = [e.get("confidence", "MEDIUM") for e in estimates.values()]

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        recommended = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2 if n % 2 == 0 else sorted_vals[n // 2]

        high_count = sum(1 for c in confidences if c.upper() == "HIGH")
        low_count = sum(1 for c in confidences if c.upper() == "LOW")
        if high_count >= len(confidences) / 2:
            agg_confidence = "HIGH"
        elif low_count > len(confidences) / 2:
            agg_confidence = "LOW"
        else:
            agg_confidence = "MEDIUM"

        if recommended > 0:
            span = (max(values) - min(values)) / recommended
            if span > 0.5:
                agg_confidence = "LOW"
            elif span > 0.25 and agg_confidence == "HIGH":
                agg_confidence = "MEDIUM"

        basis_parts = [f"{name}: {e.get('reasoning', '')[:100]}" for name, e in estimates.items()]
        basis = "; ".join(basis_parts)
        unit = self._infer_unit(param)

        return ConsensusEstimate(
            parameter=param,
            min_val=min(mins),
            max_val=max(maxs),
            recommended=recommended,
            confidence=agg_confidence,
            unit=unit,
            basis=basis,
            individual={
                name: {
                    "value": e["value"],
                    "reasoning": e.get("reasoning", ""),
                    "confidence": e.get("confidence", "MEDIUM"),
                }
                for name, e in estimates.items()
            },
        )

    def _run_consensus_estimation(
        self, state: SharedState, plan: PlanOutput,
        do_output: DoOutput, check: CheckOutput, cost: RoundCost,
    ) -> list[ConsensusEstimate]:
        """当工具数据不可靠时，启动多角色共识估算。

        所有参数并行估算，大幅降低延迟。
        """
        unreliable_params = self._identify_unreliable_params(do_output, check, state)
        if not unreliable_params:
            return []

        self._log(f"  [A] 共识估算 — {len(unreliable_params)} 个参数并行...")
        results = []

        # 并行估算所有参数（每个参数内部也并行估算各角色）
        if len(unreliable_params) == 1:
            ce = self._estimate_single_param(
                unreliable_params[0], plan, state, do_output, cost,
            )
            if ce:
                results.append(ce)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(unreliable_params), 5),
            ) as pool:
                futures = {
                    pool.submit(
                        self._estimate_single_param, p, plan, state, do_output, cost,
                    ): p
                    for p in unreliable_params
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        ce = fut.result()
                        if ce:
                            results.append(ce)
                    except Exception:
                        pass

        # 写入 state
        for ce in results:
            state.task_params[f"{ce.parameter}"] = ce.recommended
            state.task_params[f"{ce.parameter}_confidence"] = ce.confidence
            state.task_params[f"{ce.parameter}_source"] = "consensus_estimated"

            state.audit_trail.add(
                parameter=ce.parameter, value=ce.recommended, source_type="consensus",
                source_name="consensus", round_num=state.round_num,
                evidence=ce.basis,
                confidence=ce.confidence,
            )

            self._log(
                f"    ✓ {param}: {recommended:.1f} {unit} "
                f"[{min(mins):.1f}–{max(maxs):.1f}, {agg_confidence}]"
            )

        return results

    def _identify_unreliable_params(
        self, do_output: DoOutput, check: CheckOutput, state: SharedState,
    ) -> list[str]:
        """识别需要共识估算的不可靠参数."""
        params = set()

        # 1. 工具返回 simulated 或失败
        for tname, res in do_output.tool_results.items():
            if isinstance(res, dict):
                if res.get("success") == "simulated":
                    # 提取该工具预期产出的参数名
                    params.update(self._params_for_tool(tname))
                elif res.get("success") is False:
                    params.update(self._params_for_tool(tname))

        # 2. C 阶段检测到数据偏差
        for dev in check.data_deviations:
            params.add(dev.message)

        # 3. 各角色审查中明确指出的异常
        for review in check.reviews.values():
            review_str = str(review).lower()
            for keyword in ["不可靠", "unreliable", "异常", "anomalous",
                          "不可能", "impossible", "错误", "wrong", "invalid"]:
                if keyword in review_str:
                    # 尝试从审查中提取参数名
                    for param_key in state.task_params:
                        if param_key.lower() in review_str:
                            params.add(param_key)

        # 去重 + 过滤（只保留可数值估算的参数）
        return [p for p in params if self._is_numeric_param(p)]

    def _params_for_tool(self, tool_name: str) -> list[str]:
        """根据 ToolSpec 获取工具产出的参数列表."""
        for name, spec in self.tools.items():
            if name.lower() in tool_name.lower() or tool_name.lower() in name.lower():
                return spec.expected_outputs
        return []

    def _is_numeric_param(self, param: str) -> bool:
        """判断参数是否可以数值估算."""
        numeric_suffixes = ["_s", "_K", "_kg", "_m", "_mm", "_mm3", "_ms", "_ms2",
                          "_gmol", "_bar", "_n", "_ratio", "_percent"]
        return any(param.endswith(s) for s in numeric_suffixes) or any(
            s in param for s in ["Isp", "mass", "velocity", "temperature", "pressure"]
        )

    def _relevant_agents_for_param(
        self, param: str, plan: PlanOutput, state: SharedState,
    ) -> list[str]:
        """确定与某个参数相关的角色列表."""
        scores = {}

        # 从 P 阶段分析中找：谁提到了这个参数
        for name, analysis in plan.agent_analyses.items():
            analysis_str = str(analysis).lower()
            param_lower = param.lower()
            score = 0
            if param_lower in analysis_str:
                score += 3
            # 偏科角色加权
            if "propulsion" in name.lower() and any(
                k in param_lower for k in ["isp", "c_star", "t_comb", "thrust"]
            ):
                score += 2
            if "structure" in name.lower() and any(
                k in param_lower for k in ["mass", "volume", "weight", "density"]
            ):
                score += 2
            if "sim" in name.lower() and any(
                k in param_lower for k in ["isp", "velocity", "apogee"]
            ):
                score += 2
            if "gnc" in name.lower() and any(
                k in param_lower for k in ["mass", "isp", "thrust", "velocity"]
            ):
                score += 2
            if score > 0:
                scores[name] = score

        # Director 和 Safety Officer 总是相关（提供全局视角）
        for name in self.agents:
            if "director" in name.lower():
                scores[name] = scores.get(name, 0) + 2
            if "safety" in name.lower():
                scores[name] = scores.get(name, 0) + 1

        # 按分数排序，取前 5
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [name for name, _ in ranked[:5]]

    # Schema for consensus estimation output
    ESTIMATE_SCHEMA = {
        "type": "object",
        "required": ["value"],
        "properties": {
            "value": {"type": "number"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "confidence": {"type": "string"},
            "reasoning": {"type": "string"},
        },
    }

    def _agent_estimate_param(
        self, name: str, param: str, state: SharedState,
        do_output: DoOutput, cost: RoundCost,
    ) -> Optional[dict]:
        """单个角色对某个参数进行独立估算."""
        agent = self.agents.get(name)
        if not agent:
            return None

        # 收集工具结果作为参考
        tool_context = ""
        for tname, res in do_output.tool_results.items():
            if isinstance(res, dict):
                perf = res.get("performance", {})
                if perf:
                    tool_context += f"\n{tname}: {json.dumps(perf)}"
                elif res.get("success") == "simulated":
                    tool_context += f"\n{tname}: [模拟/不可用]"

        system = agent.system_prompt()
        user = (
            f"参数 '{param}' 的工具计算结果不可靠或缺失，需要你用专业知识独立估算。\n\n"
            f"共享状态: {state.to_context()}\n"
            f"工具输出: {tool_context}\n\n"
            "请给出你的最佳估算，格式如下（只输出 JSON）:\n"
            "{\n"
            '  "value": <你的最佳估算值>,\n'
            '  "min": <合理下限>,\n'
            '  "max": <合理上限>,\n'
            '  "confidence": "HIGH"|"MEDIUM"|"LOW",\n'
            '  "reasoning": "<估算依据，不超过100字>"\n'
            "}"
        )

        schema = agent.output_schema or self.ESTIMATE_SCHEMA
        _, data = self._call_llm_validated(system, user, cost, schema)
        if data and "value" in data:
            try:
                data["value"] = float(data["value"])
                data["min"] = float(data.get("min", data["value"]))
                data["max"] = float(data.get("max", data["value"]))
                state.audit_trail.add(
                    parameter=param, value=data["value"], source_type="agent_estimate",
                    source_name=name, round_num=state.round_num,
                    evidence=data.get("reasoning", ""),
                    confidence=data.get("confidence", "MEDIUM"),
                )
                return data
            except (ValueError, TypeError):
                return None
        return None

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """从 LLM 响应中提取 JSON."""
# 尝试直接解析
        try:
            return json.loads(text)
        except Exception:
            pass
        # 尝试提取 ```json ... ``` 块
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        # 尝试提取 { ... } 块
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

    def _cross_review_estimates(
        self, estimates: dict, param: str, state: SharedState, cost: RoundCost,
    ) -> dict:
        """将各角色的估算互相展示，允许调整后收敛."""
        # 构建展示文本
        estimates_text = ""
        for name, e in estimates.items():
            estimates_text += (
                f"\n{name}: 估算值={e['value']}, 范围=[{e.get('min', '?')}–{e.get('max', '?')}], "
                f"置信度={e.get('confidence', '?')}, 依据={e.get('reasoning', '')}"
            )

        updated = {}
        for name in estimates:
            agent = self.agents.get(name)
            if not agent:
                continue

            system = agent.system_prompt()
            user = (
                f"关于参数 '{param}'，以下是所有同事的独立估算:\n"
                f"{estimates_text}\n\n"
                "看完其他同事的估算后，你可以选择:\n"
                "1. 如果认同某个同事的估算，保持不变\n"
                "2. 如果发现自己的估算有误，可以调整\n"
                "3. 如果有新的依据，可以更新\n\n"
                "请输出调整后的估算（JSON）:\n"
                '{"value": <数值>, "min": <下限>, "max": <上限>, '
                '"confidence": "HIGH"|"MEDIUM"|"LOW", '
                '"reasoning": "<最终依据>"}'
            )

            try:
                resp = self._call_llm(system, user, cost, agent_name=name)
                data = self._parse_json_response(resp)
                if data and "value" in data:
                    data["value"] = float(data["value"])
                    data["min"] = float(data.get("min", data["value"]))
                    data["max"] = float(data.get("max", data["value"]))
                    updated[name] = data
                else:
                    updated[name] = estimates[name]  # 保持原估算
            except Exception:
                updated[name] = estimates[name]

        return updated

    def _infer_unit(self, param: str) -> str:
        """从参数名推测单位."""
        unit_map = {
            "Isp": "s", "T_comb": "K", "C_star": "m/s",
            "mass": "kg", "volume": "mm³", "velocity": "m/s",
            "acceleration": "m/s²", "thrust": "N", "pressure": "bar",
            "density": "kg/m³", "length": "m", "diameter": "mm",
        }
        for key, unit in unit_map.items():
            if key.lower() in param.lower():
                return unit
        return ""

    # ── A: Act ─────────────────────────────────────────────────

    def act(
        self, state: SharedState, plan: PlanOutput, check: CheckOutput,
        cost: RoundCost, prev_state: Optional[SharedState] = None,
        do_output: Optional[DoOutput] = None,
        complexity: Optional[ComplexityAssessment] = None,
    ) -> ActOutput:
        """A 阶段：收敛判断 + 共识估算 + 知识沉淀 + 决策（复杂度自适应）."""
        self._log("  [A] 收敛判断 + 知识沉淀...")

        # 物理不可能检查
        physical_alarms = self.judge.check_physical_impossible(plan.agent_analyses)
        if physical_alarms:
            alarms = [AlarmFlag(
                flag_type=a["flag_type"], message=a["message"],
                round_num=state.round_num,
            ) for a in physical_alarms]
            return ActOutput(
                verdict="STALLED",
                decisions=[],
                alarm_flags=alarms,
                should_continue=False,
                reason="物理不可能被检测到",
            )

        # 收敛判断
        if prev_state and prev_state.round_num > 0:
            result = self.judge.judge(prev_state, state)
        else:
            # 第一轮，默认继续
            from sigma.convergence import JudgeResult, Verdict
            result = JudgeResult(
                verdict=Verdict.CONVERGING,
                reason="第一轮，启动迭代",
                quantitative_converged=[],
                quantitative_remaining=list(state.task_params.keys()),
                qualitative_resolved=[],
                qualitative_remaining=[c.id for c in state.active_conflicts],
                estimated_rounds_left=3,
                should_stop=False,
            )

        # 提取决策
        decisions = []
        for conflict in check.updated_conflicts:
            if conflict.severity < 3.0:
                decisions.append(Decision(
                    round_num=state.round_num,
                    domain=conflict.id,
                    decision=f"冲突已缓解: {conflict.description}",
                    reason=f"严重度降至 {conflict.severity}/10",
                    made_by="ConvergenceJudge",
                ))

        # Alarm flags — 来自 C 阶段数据偏差检测
        alarm_flags = []
        for deviation in check.data_deviations:
            alarm_flags.append(AlarmFlag(
                flag_type="data_deviation",
                message=deviation.message,
                round_num=state.round_num,
            ))

        # ── 共识估算：当工具数据不可靠时（LITE 跳过）──
        do_consensus = complexity is None or complexity.consensus_estimation
        consensus = []
        if do_consensus and do_output is not None:
            consensus = self._run_consensus_estimation(
                state, plan, do_output, check, cost,
            )

        # Skill Crafter: 仅 RIGOROUS 模式触发
        do_skill = complexity is None or complexity.skill_crafter
        if do_skill and plan.knowledge_gaps:
            self._run_skill_crafter(plan.knowledge_gaps, state, cost)

        should_continue = not result.should_stop

        self.state_mgr.update_after_act(
            state, decisions, alarm_flags,
            cost.total_tokens, cost.estimated_cost_rmb,
        )

        return ActOutput(
            verdict=result.verdict.value,
            decisions=decisions,
            alarm_flags=alarm_flags,
            should_continue=should_continue,
            reason=result.reason,
            consensus=consensus,
        )

    def _run_skill_crafter(
        self, gaps: list[Trigger], state: SharedState, cost: RoundCost,
    ) -> None:
        """触发 Skill Crafter 补全知识缺口."""
        self._log("  [Skill Crafter] 检测到知识缺口，启动搜索...")
        crafter = self.agents.get("Skill Crafter")
        if not crafter:
            return

        for gap in gaps[:2]:  # 每轮最多处理 2 个缺口
            system = crafter.system_prompt()
            user = (
                f"知识缺口: {gap.message}\n\n"
                "请:\n"
                "1. 判断这个知识点是否对你的技能库有补充价值\n"
                "2. 如果有，请搜索并生成一个技能文件片段\n"
                "3. 格式: 标题 + 知识要点 + 数据来源 + 适用场景"
            )
            resp = self._call_llm(system, user, cost)
            # 保存到 skills/generated/
            output_base = Path(self.config.output_base_dir) if self.config.output_base_dir else Path.cwd()
            generated_dir = output_base / "skills" / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            # 安全文件名：仅保留 ASCII，限长，防碰撞加 hash
            import hashlib
            ascii_slug = "".join(c for c in gap.message if c.isascii() and c.isalnum() or c in " -_")
            ascii_slug = ascii_slug.strip()[:40].replace(" ", "_").replace("-", "_")
            ascii_slug = ascii_slug.replace("__","_").strip("_")
            if not ascii_slug:
                ascii_slug = "skill"
            short_hash = hashlib.md5(gap.message.encode()).hexdigest()[:6]
            filename = f"{ascii_slug}_{short_hash}.md"
            (generated_dir / filename).write_text(resp, encoding="utf-8")
            self._log(f"    ✓ 技能文件已生成: skills/generated/{filename}")

    # ── Async variants ────────────────────────────────────────────

    async def _a_call_llm(
        self, system_prompt: str, user_prompt: str,
        cost: Optional[RoundCost] = None,
        agent_name: str = "",
    ) -> str:
        """Async LLM call.

        Args:
            agent_name: If set, per-agent LLM config overrides global defaults.
        """
        model = self.model
        max_tokens = self.config.default_max_tokens
        temperature = self.config.default_temperature
        if agent_name:
            agent_spec = self.agents.get(agent_name)
            if agent_spec and agent_spec.llm_config:
                cfg = agent_spec.llm_config
                if cfg.model:
                    model = cfg.model
                if cfg.max_tokens:
                    max_tokens = cfg.max_tokens
                if cfg.temperature is not None:
                    temperature = cfg.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = await self.llm.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if cost:
            cost.calls += 1
            if resp.input_tokens or resp.output_tokens:
                self.cost_tracker.record_call(cost, resp.input_tokens, resp.output_tokens)
            else:
                self.cost_tracker.record_call(cost, len(system_prompt) // 4, 500)
        return resp.content

    async def _agent_analyze_async(
        self, name: str, state: SharedState, cost: RoundCost,
    ) -> str:
        agent = self.agents.get(name)
        if not agent:
            return f"Agent '{name}' not found."
        shared_context = state.to_context()
        skills_text = self._get_skills_for(agent)
        system = agent.system_prompt(
            f"\n\n当前工程共享状态:\n{shared_context}"
            f"\n\n你的可用工具: {', '.join(agent.tool_names) if agent.tool_names else '无'}"
            f"\n\n{skills_text}",
            creed=self.config.creed,
        )
        user = (
            f"任务: {state.task_instruction}\n\n"
            "请从你的专业视角独立分析这个任务:\n"
            "1. 你的专业判断是什么？\n"
            "2. 哪些关键参数需要计算或验证？\n"
            "   如需调用工具，请用 [需要工具: 工具名] 格式明确标注\n"
            "   可用工具: " + ", ".join(agent.tool_names) + "\n"
            "3. 你依赖哪些其他专业角色的判断？（标注 [依赖: 角色名]）\n"
            "4. 你的初步结论和建议\n"
            "5. 是否有不确定或需要搜索的知识点？\n\n"
            "请给出具体数值和推理过程，不要泛泛而谈。"
        )
        return await self._a_call_llm(system, user, cost, agent_name=name)

    async def _agent_cross_review_async(
        self, name: str, analyses: dict[str, str],
        conflicts: list[Conflict], state: SharedState, cost: RoundCost,
    ) -> str:
        agent = self.agents.get(name)
        if not agent:
            return ""
        related = [n for n in analyses if n != name]
        context = "\n\n".join(f"### {n}\n{a}" for n, a in analyses.items())
        system = agent.system_prompt()
        user = (
            f"请审查以下 {', '.join(related)} 角色的分析结论。\n\n"
            "要点：\n"
            "1. 数据/计算逻辑是否有错误？请指正\n"
            "2. 结论是否与你的专业判断一致？何处不一致？\n"
            "3. 遗漏了哪些关键方面？\n\n"
            f"--- 分析全文 ---\n{context}"
        )
        return await self._a_call_llm(system, user, cost, agent_name=name)

    async def _agent_review_data_async(
        self, name: str, state: SharedState, tool_results: dict, cost: RoundCost,
    ) -> str:
        agent = self.agents.get(name)
        if not agent:
            return ""
        results_text = ""
        for tname, res in tool_results.items():
            if isinstance(res, dict):
                perf = res.get("performance", {})
                if perf:
                    results_text += f"\n{tname}: " + ", ".join(f"{k}={v}" for k, v in perf.items())
                elif res.get("success") == "simulated":
                    results_text += f"\n{tname}: [模拟结果] {res.get('note', '')}"
                else:
                    results_text += f"\n{tname}: {res.get('error', res)}"
        system = agent.system_prompt()
        user = (
            f"审查以下工具运行结果。\n\n"
            f"工具输出:\n{results_text}\n\n"
            "请评估:\n"
            "1. 数值是否合理？\n"
            "2. 是否有异常需要关注？\n"
            "3. 建议的修正或补充分析。"
        )
        return await self._a_call_llm(system, user, cost, agent_name="Safety Officer")

    async def plan_async(self, state: SharedState, cost: RoundCost,
                         complexity: Optional[ComplexityAssessment] = None) -> PlanOutput:
        """Async P phase using asyncio.gather."""
        if complexity is not None and complexity.agent_names:
            agent_names = complexity.agent_names
        else:
            agent_names = list(self.agents.keys())
        n_agents = len(agent_names)

        tasks = [self._agent_analyze_async(name, state, cost) for name in agent_names]
        results = await __import__("asyncio").gather(*tasks, return_exceptions=True)
        analyses = {}
        for name, result in zip(agent_names, results):
            analyses[name] = str(result) if not isinstance(result, Exception) else f"[ERROR: {result}]"

        conflicts = self._extract_conflicts(analyses, state.round_num)
        knowledge_gaps = self.triggers.check_knowledge_gap(analyses)
        dep_graph = self._build_dependency_graph(analyses)

        do_cross_review = complexity is None or complexity.cross_review
        if do_cross_review and n_agents >= 2:
            review_tasks = [
                self._agent_cross_review_async(name, analyses, conflicts, state, cost)
                for name in agent_names
            ]
            review_results = await __import__("asyncio").gather(*review_tasks, return_exceptions=True)
            for name, review in zip(agent_names, review_results):
                rtext = str(review) if not isinstance(review, Exception) else f"[CROSS_REVIEW_ERROR: {review}]"
                if name in analyses:
                    analyses[name] = analyses[name] + "\n\n--- 交叉审查 ---\n" + rtext

        tool_requests = self._extract_tool_requests(analyses)
        updated_conflicts = self._extract_conflicts(analyses, state.round_num)
        self.state_mgr.update_after_plan(state, analyses, updated_conflicts, dep_graph, tool_requests)

        return PlanOutput(
            agent_analyses=analyses, conflicts=updated_conflicts,
            dependency_graph=dep_graph, tool_requests=tool_requests,
            knowledge_gaps=knowledge_gaps,
        )

    async def do_async(self, plan: PlanOutput, state: SharedState,
                       cost: RoundCost) -> DoOutput:
        """Async D phase. Tool execution is already parallel with ThreadPoolExecutor."""
        return self.do(plan, state, cost)

    async def check_async(self, state: SharedState, plan: PlanOutput,
                          do_output: DoOutput, cost: RoundCost,
                          complexity: Optional[ComplexityAssessment] = None) -> CheckOutput:
        """Async C phase using asyncio.gather."""
        n_reviewers = min(6, len(plan.agent_analyses))
        relevant_agents = list(plan.agent_analyses.keys())[:6]

        if n_reviewers > 0:
            tasks = [self._agent_review_data_async(name, state, do_output.tool_results, cost)
                     for name in relevant_agents]
            results = await __import__("asyncio").gather(*tasks, return_exceptions=True)
            reviews = {}
            for name, result in zip(relevant_agents, results):
                reviews[name] = str(result) if not isinstance(result, Exception) else f"[REVIEW_ERROR: {result}]"
        else:
            reviews = {}

        do_devil = complexity is None or complexity.devil_advocate
        devil = ""
        if do_devil:
            devil = await self._devil_advocate_async(state, do_output.tool_results, cost)

        updated_conflicts = self._extract_conflicts(reviews, state.round_num)

        deviations = []
        for tool_name, result in do_output.tool_results.items():
            perf = result.get("performance", {})
            if perf:
                devs = self.triggers.check_data_deviation(state.task_params, perf)
                deviations.extend(devs)

        self.state_mgr.update_after_check(state, reviews, updated_conflicts, devil)
        return CheckOutput(
            reviews=reviews, updated_conflicts=updated_conflicts,
            devil_advocate=devil, data_deviations=deviations,
        )

    async def _devil_advocate_async(
        self, state: SharedState, tool_results: dict, cost: RoundCost,
    ) -> str:
        """Async devil's advocate review."""
        safety = self.agents.get("Safety Officer")
        if not safety:
            return ""
        results_text = ""
        for tname, res in tool_results.items():
            perf = res.get("performance", {})
            if perf:
                results_text += f"\n{tname}: {json.dumps(perf)}"
        system = safety.system_prompt()
        user = (
            "请刻意从最坏情况出发审视当前方案:\n\n"
            f"当前状态: {state.to_context()}\n"
            f"计算结果: {results_text}\n\n"
            "请回答:\n"
            "1. 如果这个方案失败了，最可能的三个原因是什么？\n"
            "2. 哪个假设最脆弱？\n"
            "3. 有没有数值在'刚刚好够'的边缘而没留安全裕度？\n"
            "4. 有没有人被其他角色的结论说服了但没有真正验证？"
        )
        return await self._a_call_llm(system, user, cost)
