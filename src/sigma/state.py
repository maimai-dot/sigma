"""
三层共享状态管理
Layer 1: 结构化共享状态（贯穿全轮，~2K tokens）
Layer 2: 轮次摘要（上一轮，~3K tokens）
Layer 3: 全量存档（每轮完整输出存文件）
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sigma.provenance import AuditTrail


class ComplexityTier(Enum):
    """任务复杂度分层."""
    LITE = "lite"            # 3 核角色, 1 轮, 无交叉审查, ~5 LLM 调用
    STANDARD = "standard"    # 6 核角色, 2-3 轮, 简化交叉审查, ~30 调用
    RIGOROUS = "rigorous"    # 全部 8 角色, 最多 4 轮, 完整审查, ~80 调用


@dataclass
class ComplexityAssessment:
    """复杂度评估结果."""
    tier: ComplexityTier
    score: float                # 0-10 复杂度评分
    reason: str                 # 评估理由
    agent_names: list[str]      # 本轮应激活的角色
    max_rounds: int
    cross_review: bool          # 是否执行交叉审查
    devil_advocate: bool        # 是否执行魔鬼代言人
    consensus_estimation: bool  # 是否执行共识估算
    skill_crafter: bool         # 是否启用技能工匠


@dataclass
class Conflict:
    """跨角色冲突记录."""
    id: str
    description: str
    owners: list[str]  # 涉及的角色名
    severity: float    # 0.0-10.0，严重程度
    trend: str         # "new" | "shrinking" | "growing" | "stalled"
    quantitative: bool # 是否可以数值化衡量
    values: dict       # {round: value} 数值化跟踪


@dataclass
class Decision:
    """本轮做出的决策."""
    round_num: int
    domain: str
    decision: str
    reason: str
    made_by: str


@dataclass
class AlarmFlag:
    """告警标记."""
    flag_type: str    # "physical_limit" | "tool_failure" | "missing_expertise" | "oscillation"
    message: str
    round_num: int
    resolved: bool = False


@dataclass
class ConsensusEstimate:
    """多角色共识估算结果."""
    parameter: str       # 参数名，如 "engine_Isp_sl_s"
    min_val: float
    max_val: float
    recommended: float   # 共识推荐值
    confidence: str      # HIGH / MEDIUM / LOW
    unit: str
    basis: str           # 估算依据摘要
    individual: dict     # {agent_name: {"value": float, "reasoning": str, "confidence": str}}


@dataclass
class RoundRecord:
    """单轮完整记录."""
    round_num: int
    timestamp: str
    phase_outputs: dict = field(default_factory=dict)
    agent_analyses: dict = field(default_factory=dict)
    tool_results: dict = field(default_factory=dict)
    cross_review: dict = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    alarm_flags: list[AlarmFlag] = field(default_factory=list)
    token_count: int = 0


@dataclass
class SharedState:
    """Layer 1: 贯穿所有轮次的结构化共享状态."""
    task_instruction: str
    task_params: dict = field(default_factory=dict)
    round_num: int = 0
    max_rounds: int = 4
    complexity_tier: str = "standard"       # ComplexityTier.value
    complexity_assessment: dict = field(default_factory=dict)
    convergence_log: list[dict] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    alarm_flags: list[AlarmFlag] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    active_conflicts: list[Conflict] = field(default_factory=list)
    cost_summary: dict = field(default_factory=lambda: {
        "total_tokens": 0,
        "estimated_cost": 0.0,
        "calls": 0,
    })
    history: list[RoundRecord] = field(default_factory=list)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def to_context(self) -> str:
        """将共享状态序列化为 LLM 上下文."""
        lines = [
            "## 共享工程状态",
            f"任务: {self.task_instruction}",
            f"当前轮次: {self.round_num}/{self.max_rounds}",
        ]
        if self.task_params:
            lines.append("\n### 设计参数")
            for k, v in self.task_params.items():
                lines.append(f"  {k}: {v}")
        if self.decisions:
            lines.append("\n### 已做决策")
            for d in self.decisions[-5:]:
                lines.append(f"  [R{d.round_num}] {d.domain}: {d.decision}")
        if self.active_conflicts:
            lines.append("\n### 当前冲突")
            for c in self.active_conflicts:
                lines.append(f"  ⚠ {c.description} (严重度: {c.severity}/10, 趋势: {c.trend})")
        if self.alarm_flags:
            lines.append("\n### 告警")
            for a in self.alarm_flags:
                if not a.resolved:
                    lines.append(f"  [{a.flag_type}] {a.message}")
        return "\n".join(lines)


class StateManager:
    """共享状态管理器."""

    def init(self, instruction: str) -> SharedState:
        """初始化 Round 0 状态."""
        return SharedState(task_instruction=instruction)

    def start_round(self, state: SharedState) -> SharedState:
        """开始新的一轮."""
        from copy import deepcopy
        new_state = deepcopy(state)
        new_state.round_num += 1
        new_state.history.append(RoundRecord(
            round_num=new_state.round_num,
            timestamp=datetime.now().isoformat(),
        ))
        return new_state

    def update_after_plan(
        self, state: SharedState, analyses: dict, conflicts: list[Conflict],
        dependency_graph: dict, tool_requests: list[str],
    ) -> SharedState:
        """P 阶段后更新状态."""
        state.active_conflicts = conflicts
        state.dependency_graph = dependency_graph
        record = state.history[-1]
        record.agent_analyses = analyses
        record.conflicts = conflicts
        record.phase_outputs["plan"] = {
            "tool_requests": tool_requests,
            "dependency_graph": dependency_graph,
        }
        # 记录冲突日志
        for c in conflicts:
            existing = [e for e in state.convergence_log
                       if e.get("conflict_id") == c.id]
            if not existing:
                state.convergence_log.append({
                    "conflict_id": c.id,
                    "description": c.description,
                    "entries": [{"round": state.round_num, "severity": c.severity}],
                    "trend": "new",
                })
            else:
                existing[0]["entries"].append({
                    "round": state.round_num, "severity": c.severity,
                })
        return state

    def update_after_do(
        self, state: SharedState, tool_results: dict,
    ) -> SharedState:
        """D 阶段后更新状态."""
        record = state.history[-1]
        record.tool_results = tool_results
        record.phase_outputs["do"] = {"tool_results": tool_results}
        # 将工具结果写入 task_params
        for key, result in tool_results.items():
            if isinstance(result, dict) and result.get("success"):
                perf = result.get("performance", {})
                for k, v in perf.items():
                    param_key = f"{key}_{k}"
                    state.task_params[param_key] = v
                    state.audit_trail.add(
                        parameter=param_key, value=v, source_type="tool",
                        source_name=key, round_num=state.round_num,
                        evidence=str(result.get("raw", result.get("note", ""))),
                        confidence="HIGH" if result.get("success") is True else "MEDIUM",
                    )
        return state

    def update_after_check(
        self, state: SharedState, reviews: dict, updated_conflicts: list[Conflict],
        devil_advocate: str,
    ) -> SharedState:
        """C 阶段后更新状态."""
        state.active_conflicts = updated_conflicts
        record = state.history[-1]
        record.cross_review = reviews
        record.phase_outputs["check"] = {
            "reviews": reviews,
            "devil_advocate": devil_advocate,
        }
        return state

    def update_after_act(
        self, state: SharedState, decisions: list[Decision],
        alarm_flags: list[AlarmFlag], token_count: int, estimated_cost: float,
    ) -> SharedState:
        """A 阶段后更新状态."""
        state.decisions.extend(decisions)
        state.alarm_flags.extend(alarm_flags)
        record = state.history[-1]
        record.decisions = decisions
        record.alarm_flags = alarm_flags
        record.token_count = token_count
        record.phase_outputs["act"] = {
            "decisions": [d.decision for d in decisions],
            "alarm_flags": [a.message for a in alarm_flags],
        }
        state.cost_summary["total_tokens"] += token_count
        state.cost_summary["estimated_cost"] += estimated_cost
        state.cost_summary["calls"] += 1
        return state

    def summary(self, state: SharedState, max_chars: int = 500) -> str:
        """Layer 2: 生成上一轮的结构化摘要."""
        if not state.history:
            return ""
        record = state.history[-1]
        lines = [f"## Round {record.round_num} 摘要"]
        # 各角色关键发现
        if record.agent_analyses:
            for agent, analysis in record.agent_analyses.items():
                snippet = str(analysis)[:120]
                lines.append(f"- **{agent}**: {snippet}...")
        # 工具结果
        if record.tool_results:
            lines.append("\n### 计算结果")
            for tool, result in record.tool_results.items():
                if isinstance(result, dict):
                    perf = result.get("performance", {})
                    if perf:
                        vals = ", ".join(f"{k}={v}" for k, v in perf.items())
                        lines.append(f"- {tool}: {vals}")
        # 冲突
        if record.conflicts:
            lines.append("\n### 冲突")
            for c in record.conflicts:
                lines.append(f"- {c.description} (严重度: {c.severity}/10)")
        # 决策
        if record.decisions:
            lines.append("\n### 决策")
            for d in record.decisions:
                lines.append(f"- {d.decision}")
        return "\n".join(lines)[:max_chars]

    def delta(self, prev_state: SharedState, curr_state: SharedState) -> str:
        """计算两轮之间的增量变化."""
        lines = []
        # 参数变化
        prev_params = prev_state.task_params
        curr_params = curr_state.task_params
        for key in set(prev_params) | set(curr_params):
            pv = prev_params.get(key)
            cv = curr_params.get(key)
            if pv != cv:
                if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
                    delta_pct = abs(cv - pv) / max(abs(pv), 0.001) * 100
                    lines.append(f"  {key}: {pv} → {cv} (Δ{delta_pct:.1f}%)")
                else:
                    lines.append(f"  {key}: {pv} → {cv}")
        # 冲突变化
        curr_conflicts = {c.id: c.severity for c in curr_state.active_conflicts}
        prev_conflicts = {c.id: c.severity for c in prev_state.active_conflicts}
        for cid in set(curr_conflicts) | set(prev_conflicts):
            ps = prev_conflicts.get(cid)
            cs = curr_conflicts.get(cid)
            if cs is None:
                lines.append(f"  冲突已解决: {cid}")
            elif ps is None:
                lines.append(f"  新冲突: {cid} (严重度: {cs})")
            else:
                delta = cs - ps
                arrow = "↓" if delta < 0 else "↑" if delta > 0 else "→"
                lines.append(f"  {cid}: {ps} → {cs} {arrow}")
        return "\n".join(lines) if lines else "无变化"

    def save_round(self, state: SharedState, round_dir: Path) -> None:
        """Layer 3: 保存完整轮次存档."""
        import json
        round_dir.mkdir(parents=True, exist_ok=True)
        record = state.history[-1]
        data = {
            "round_num": record.round_num,
            "timestamp": record.timestamp,
            "agent_analyses": {k: str(v)[:2000] for k, v in record.agent_analyses.items()},
            "tool_results": record.tool_results,
            "conflicts": [{
                "id": c.id, "description": c.description,
                "severity": c.severity, "trend": c.trend,
            } for c in record.conflicts],
            "decisions": [{
                "domain": d.domain, "decision": d.decision, "reason": d.reason,
            } for d in record.decisions],
            "alarm_flags": [{
                "type": a.flag_type, "message": a.message,
            } for a in record.alarm_flags],
            "token_count": record.token_count,
        }
        (round_dir / "record.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        # 保存共享状态快照
        snapshot = {
            "task_instruction": state.task_instruction,
            "task_params": state.task_params,
            "round_num": state.round_num,
            "cost_summary": state.cost_summary,
        }
        (round_dir / "state.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    def checkpoint(self, state: SharedState, path: Path) -> None:
        """完整序列化 SharedState 到单个文件，支持中断恢复."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize_state(state)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def restore(self, path: Path) -> SharedState:
        """从 checkpoint 文件完整恢复 SharedState."""
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize_state(data)


def _serialize_state(state: SharedState) -> dict:
    """Serialize SharedState to a JSON-safe dict."""
    return {
        "task_instruction": state.task_instruction,
        "task_params": state.task_params,
        "round_num": state.round_num,
        "max_rounds": state.max_rounds,
        "complexity_tier": state.complexity_tier,
        "complexity_assessment": state.complexity_assessment,
        "convergence_log": state.convergence_log,
        "decisions": [_serialize_decision(d) for d in state.decisions],
        "alarm_flags": [_serialize_alarm(a) for a in state.alarm_flags],
        "dependency_graph": state.dependency_graph,
        "active_conflicts": [_serialize_conflict(c) for c in state.active_conflicts],
        "cost_summary": state.cost_summary,
        "history": [_serialize_round(r) for r in state.history],
        "audit_trail": state.audit_trail.to_dict(),
    }


def _deserialize_state(data: dict) -> SharedState:
    """Deserialize SharedState from a JSON-safe dict."""
    return SharedState(
        task_instruction=data["task_instruction"],
        task_params=data.get("task_params", {}),
        round_num=data.get("round_num", 0),
        max_rounds=data.get("max_rounds", 4),
        complexity_tier=data.get("complexity_tier", "standard"),
        complexity_assessment=data.get("complexity_assessment", {}),
        convergence_log=data.get("convergence_log", []),
        decisions=[_deserialize_decision(d) for d in data.get("decisions", [])],
        alarm_flags=[_deserialize_alarm(a) for a in data.get("alarm_flags", [])],
        dependency_graph=data.get("dependency_graph", {}),
        active_conflicts=[_deserialize_conflict(c) for c in data.get("active_conflicts", [])],
        cost_summary=data.get("cost_summary", {}),
        history=[_deserialize_round(r) for r in data.get("history", [])],
        audit_trail=AuditTrail.from_dict(data.get("audit_trail", {"entries": []})),
    )


def _serialize_decision(d: Decision) -> dict:
    return {"round_num": d.round_num, "domain": d.domain, "decision": d.decision,
            "reason": d.reason, "made_by": d.made_by}


def _deserialize_decision(d: dict) -> Decision:
    return Decision(round_num=d["round_num"], domain=d["domain"], decision=d["decision"],
                    reason=d.get("reason", ""), made_by=d.get("made_by", ""))


def _serialize_alarm(a: AlarmFlag) -> dict:
    return {"flag_type": a.flag_type, "message": a.message, "round_num": a.round_num,
            "resolved": a.resolved}


def _deserialize_alarm(a: dict) -> AlarmFlag:
    return AlarmFlag(flag_type=a["flag_type"], message=a["message"],
                     round_num=a["round_num"], resolved=a.get("resolved", False))


def _serialize_conflict(c: Conflict) -> dict:
    return {"id": c.id, "description": c.description, "owners": c.owners,
            "severity": c.severity, "trend": c.trend, "quantitative": c.quantitative,
            "values": c.values}


def _deserialize_conflict(c: dict) -> Conflict:
    return Conflict(id=c["id"], description=c["description"], owners=c.get("owners", []),
                    severity=c["severity"], trend=c.get("trend", "new"),
                    quantitative=c.get("quantitative", False), values=c.get("values", {}))


def _serialize_round(r: RoundRecord) -> dict:
    return {"round_num": r.round_num, "timestamp": r.timestamp,
            "phase_outputs": r.phase_outputs, "agent_analyses": r.agent_analyses,
            "tool_results": r.tool_results, "cross_review": r.cross_review,
            "conflicts": [_serialize_conflict(c) for c in r.conflicts],
            "decisions": [_serialize_decision(d) for d in r.decisions],
            "alarm_flags": [_serialize_alarm(a) for a in r.alarm_flags],
            "token_count": r.token_count}


def _deserialize_round(r: dict) -> RoundRecord:
    return RoundRecord(round_num=r["round_num"], timestamp=r.get("timestamp", ""),
                       phase_outputs=r.get("phase_outputs", {}),
                       agent_analyses=r.get("agent_analyses", {}),
                       tool_results=r.get("tool_results", {}),
                       cross_review=r.get("cross_review", {}),
                       conflicts=[_deserialize_conflict(c) for c in r.get("conflicts", [])],
                       decisions=[_deserialize_decision(d) for d in r.get("decisions", [])],
                       alarm_flags=[_deserialize_alarm(a) for a in r.get("alarm_flags", [])],
                       token_count=r.get("token_count", 0))
