"""TauConfig — all configurable parameters for the Tau hierarchical framework.

Applications override fields to adapt Tau to different domains.
Prompts use {param_key} as placeholder — resolved via .replace(), not .format().
This avoids JSON brace escaping hell.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigma.guardrails import GuardrailSet


@dataclass
class TauConfig:
    """Configuration for TauOrchestrator, TauDecomposer, TauResolver, and Detector.

    Every field has a reasonable default. Override for domain-specific needs.
    """

    # ── Decomposition ───────────────────────────────────────────────

    decompose_system_prompt: str = (
        "你是一位工程总监（Director），负责将复杂任务拆解为子任务并分配给专业部门。\n\n"
        "拆解规则：\n"
        "1. 每个子任务只分配给 1-2 个最相关的部门/角色\n"
        "2. 识别接口参数（interface_params）：一个部门产出、另一个部门需要的共享参数\n"
        "3. 明确依赖关系（dependencies）：哪些子任务必须先完成\n"
        "4. 每个子任务有清晰的预期产出（expected_outputs）\n\n"
        "输出格式（JSON）：\n"
        "{\n"
        '  "subtasks": [\n'
        "    {\n"
        '      "id": "st_1",\n'
        '      "description": "子任务描述",\n'
        '      "assigned_agents": ["角色名"],\n'
        '      "interface_params": ["参数名"],\n'
        '      "dependencies": [],\n'
        '      "expected_outputs": ["产出描述"]\n'
        "    }\n"
        "  ],\n"
        '  "interface_map": {"参数名": ["st_1", "st_2"]}\n'
        "}\n\n"
        "只输出 JSON，不要其他文字。"
    )

    # ── Conflict Detection ──────────────────────────────────────────

    detection_threshold: float = 0.10
    """Relative difference above which two values for the same param are flagged."""

    severity_max: float = 10.0
    """Maximum severity score."""

    # ── Resolution ──────────────────────────────────────────────────

    light_threshold: float = 2.0
    """Severity below this: try DIRECT discussion first."""

    sigma_threshold: float = 5.0
    """Severity below this: SIGMA AERC; above: DIRECTOR decision."""

    max_light_iter: int = 2
    """After this many iterations, skip DIRECT and go to SIGMA/DIRECTOR."""

    direct_discussion_rounds: int = 3
    """Max back-and-forth rounds in DIRECT free discussion before escalating."""

    direct_convergence_ratio: float = 0.20
    """Relative difference below which DIRECT discussion is considered converged."""

    direct_system_prompt: str = (
        "你是一位中立的工程协调员。两个部门对同一个参数产生了分歧。"
        "请阅读双方的分析，找出共识点，给出一个合理的折中值。"
        '只输出 JSON：{"value": <数值>, "rationale": "<理由>"}'
    )

    sigma_blind_system_prompt: str = (
        "你是一位专业工程师。请独立估算参数 {param_key} 的数值。"
        "不要与其他人讨论，只基于你的专业知识给出最佳估算。"
        '只输出 JSON：{"value": <数值>, "reasoning": "<推理过程>"}'
    )

    sigma_review_prompt: str = (
        "以下是其他工程师对 {param_key} 的估算：\n{estimates}\n\n"
        "请审查这些估算，给出你的评估。"
        '只输出 JSON：{"agreed_value": <数值>, "confidence": "HIGH|MEDIUM|LOW"}'
    )

    direct_agent_discussion_prompt: str = (
        "你正在与另一位工程师讨论参数 {param_key} 的取值。\n"
        "你的专业领域：{my_role}\n"
        "对方来自：{other_role}\n\n"
        "当前状态：\n"
        "  你的估算：{my_value}\n"
        "  对方估算：{other_value}\n"
        "  对方最新意见：{other_message}\n\n"
        "请阅读对方的意见，然后：\n"
        "1. 如果对方的分析有道理，可以调整你的估算\n"
        "2. 如果坚持你的估算，请用数据说明理由\n"
        "3. 如果双方差距已经很小（<10%），可以建议接受平均值\n\n"
        '只输出 JSON：{{"value": <你的最新估算>, "message": "<给对方的反馈>", "accept_avg": true|false}}'
    )

    director_decision_system_prompt: str = (
        "你是一位工程总监（Director）。两个部门对一个关键参数无法达成一致，"
        "盲审也无法收敛。请基于所有证据做出方向性决策。"
        '只输出 JSON：{"value": <数值>, "rationale": "<决策理由>"}'
    )

    # ── Consensus Convergence ───────────────────────────────────────

    consensus_convergence_ratio: float = 0.30
    """Maximum range/median ratio for consensus to be considered converged."""

    # ── Executor ─────────────────────────────────────────────────────

    executor_max_workers: int = 6
    """Max parallel workers for subtask execution."""

    executor_timeout_per_subtask: float = 300.0
    """Seconds before a subtask times out."""

    # ── Guardrails ────────────────────────────────────────────────────

    guardrails: "GuardrailSet | None" = None
    """Optional GuardrailSet for output validation. Set to a GuardrailSet instance
    to validate all subtask outputs against physical/domain constraints."""

    # ── Helpers ─────────────────────────────────────────────────────

    def format_prompt(self, prompt: str, **kwargs) -> str:
        """Replace {key} placeholders with values."""
        result = prompt
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result
