"""
事件驱动触发器
知识缺口检测: P 阶段输出含不确定表述 → 触发 Skill Crafter
工具异常检测: D 阶段返回值异常 → 标记重试
数据偏差检测: C 阶段发现数据和预期偏差 > 30% → 触发重新评估
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trigger:
    trigger_type: str       # "knowledge_gap" | "tool_abnormal" | "data_deviation"
    source: str             # 触发源：agent 名或 tool 名
    message: str
    severity: float         # 0.0-10.0
    context: dict           # 额外上下文
    handled: bool = False


class TriggerSystem:
    """事件驱动触发器系统."""

    # 知识缺口关键词
    GAP_KEYWORDS = [
        "不确定", "需要查", "建议搜索", "我不清楚", "需要确认",
        "需要验证", "待核实", "需要实验", "缺乏数据", "unknown",
        "uncertain", "need to verify", "need to check", "not sure",
        "需要标定", "需要测量", "经验值", "估计", "假设",
    ]

    # 标准工具返回值合理性范围
    REASONABLE_RANGES = {
        "Isp": (50, 500),          # s
        "T_comb": (500, 5000),     # K
        "mass_kg": (0.01, 50000),  # kg
        "C_star": (300, 8000),     # m/s
        "Gamma": (1.0, 2.0),       # unitless
        "Molar_mass": (1, 200),    # g/mol
        "Cf": (0.5, 3.0),          # unitless
    }

    def check_knowledge_gap(self, analyses: dict[str, str]) -> list[Trigger]:
        """检测分析中的知识缺口."""
        triggers = []
        for agent, text in analyses.items():
            text_str = str(text)
            for kw in self.GAP_KEYWORDS:
                if kw.lower() in text_str.lower():
                    # 找关键词附近的上下文
                    idx = text_str.lower().find(kw.lower())
                    snippet = text_str[max(0, idx - 80):idx + len(kw) + 80]
                    triggers.append(Trigger(
                        trigger_type="knowledge_gap",
                        source=agent,
                        message=f"知识缺口: '{kw}' — ...{snippet}...",
                        severity=5.0,
                        context={"agent": agent, "keyword": kw, "snippet": snippet},
                    ))
        return triggers

    def check_tool_abnormal(self, tool_name: str, result: dict) -> list[Trigger]:
        """检测工具返回值是否异常."""
        triggers = []
        if not isinstance(result, dict):
            return triggers
        if not result.get("success", False) and result.get("success") != "simulated":
            triggers.append(Trigger(
                trigger_type="tool_abnormal",
                source=tool_name,
                message=f"工具调用失败: {result.get('error', 'Unknown')}",
                severity=8.0,
                context={"tool": tool_name, "error": result.get("error")},
            ))
            return triggers
        # 检查性能数值是否在合理范围内
        perf = result.get("performance", {})
        for key, value in perf.items():
            if not isinstance(value, (int, float)):
                continue
            range_info = None
            for prefix, (lo, hi) in self.REASONABLE_RANGES.items():
                if prefix in key:
                    range_info = (lo, hi)
                    break
            if range_info and not (range_info[0] <= value <= range_info[1]):
                triggers.append(Trigger(
                    trigger_type="tool_abnormal",
                    source=tool_name,
                    message=f"异常值: {key}={value}, 期望范围 [{range_info[0]}, {range_info[1]}]",
                    severity=7.0,
                    context={"tool": tool_name, "key": key, "value": value, "range": range_info},
                ))
        return triggers

    def check_data_deviation(
        self, expected: dict, actual: dict,
    ) -> list[Trigger]:
        """检测预期值与实际工具结果的偏差."""
        triggers = []
        for key, exp_val in expected.items():
            if key not in actual:
                continue
            act_val = actual[key]
            if not isinstance(exp_val, (int, float)) or not isinstance(act_val, (int, float)):
                continue
            denominator = max(abs(exp_val), 0.001)
            deviation = abs(act_val - exp_val) / denominator
            if deviation > 0.30:  # 30%
                triggers.append(Trigger(
                    trigger_type="data_deviation",
                    source=key,
                    message=f"数据偏差 {deviation:.0%}: 预期 {exp_val}, 实际 {act_val}",
                    severity=min(10.0, deviation * 10),
                    context={"expected": exp_val, "actual": act_val, "deviation": deviation},
                ))
        return triggers

    def needs_skill_crafter(self, triggers: list[Trigger]) -> bool:
        """判断是否需要启动 Skill Crafter."""
        return any(
            t.trigger_type == "knowledge_gap" and t.severity >= 3.0
            for t in triggers
        )

    def needs_tool_retry(self, triggers: list[Trigger]) -> list[str]:
        """返回需要重试的工具列表."""
        return [t.source for t in triggers if t.trigger_type == "tool_abnormal"]
