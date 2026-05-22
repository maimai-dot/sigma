"""
收敛判断系统
数值型: 变化率 < 5% → 收敛
定性冲突: 本轮不再提起 → 收敛
速率阈值: 变化率 < 10%/轮 → 过慢，提示 Founder
振荡检测: 连续 2 轮冲突不缩小 → 停车
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigma.state import SharedState, Conflict


class Verdict(Enum):
    CONVERGED = "converged"          # 完全收敛，输出最终方案
    CONVERGING = "converging"        # 正在收敛，继续下一轮
    SLOW = "slow"                    # 收敛过慢，提示 Founder 但仍可继续
    OSCILLATING = "oscillating"      # 振荡，立即停车
    STALLED = "stalled"              # 完全停滞，立即停车


@dataclass
class JudgeResult:
    verdict: Verdict
    reason: str
    quantitative_converged: list[str]   # 已收敛的数值参数
    quantitative_remaining: list[str]   # 未收敛的数值参数
    qualitative_resolved: list[str]     # 已解决的定性冲突
    qualitative_remaining: list[str]    # 未解决的定性冲突
    estimated_rounds_left: int          # 预计还需多少轮
    should_stop: bool


class ConvergenceJudge:
    """收敛判断器."""

    NUMERIC_THRESHOLD = 0.05       # 5% 变化率 → 已收敛
    SLOW_RATE_THRESHOLD = 0.10     # < 10%/轮 → 过慢
    OSCILLATION_WINDOW = 2         # 连续 N 轮不缩小 → 振荡

    def judge(self, prev_state: "SharedState", curr_state: "SharedState") -> JudgeResult:
        """综合收敛判断."""
        quant_converged, quant_remaining = self._check_quantitative(prev_state, curr_state)
        qual_resolved, qual_remaining = self._check_qualitative(prev_state, curr_state)
        is_oscillating = self.detect_oscillation(curr_state)
        rate = self.convergence_rate(curr_state)
        rounds_left = self._estimate_rounds_left(curr_state, rate)

        # 全部收敛
        if not quant_remaining and not qual_remaining:
            return JudgeResult(
                verdict=Verdict.CONVERGED,
                reason="所有数值参数和定性冲突均已收敛",
                quantitative_converged=quant_converged,
                quantitative_remaining=[],
                qualitative_resolved=qual_resolved,
                qualitative_remaining=[],
                estimated_rounds_left=0,
                should_stop=True,
            )

        # 振荡 → 硬刹车
        if is_oscillating:
            return JudgeResult(
                verdict=Verdict.OSCILLATING,
                reason=f"连续 {self.OSCILLATION_WINDOW} 轮冲突未缩小，系统进入振荡",
                quantitative_converged=quant_converged,
                quantitative_remaining=quant_remaining,
                qualitative_resolved=qual_resolved,
                qualitative_remaining=qual_remaining,
                estimated_rounds_left=-1,
                should_stop=True,
            )

        # 硬上限
        if curr_state.round_num >= curr_state.max_rounds:
            return JudgeResult(
                verdict=Verdict.STALLED,
                reason=f"已达最大轮次 {curr_state.max_rounds}，强制停车",
                quantitative_converged=quant_converged,
                quantitative_remaining=quant_remaining,
                qualitative_resolved=qual_resolved,
                qualitative_remaining=qual_remaining,
                estimated_rounds_left=0,
                should_stop=True,
            )

        # 收敛过慢
        if rate > 0 and rate < self.SLOW_RATE_THRESHOLD:
            return JudgeResult(
                verdict=Verdict.SLOW,
                reason=f"收敛速率 {rate:.1%}/轮，过慢，预计还需 {rounds_left} 轮",
                quantitative_converged=quant_converged,
                quantitative_remaining=quant_remaining,
                qualitative_resolved=qual_resolved,
                qualitative_remaining=qual_remaining,
                estimated_rounds_left=rounds_left,
                should_stop=False,
            )

        # 正常收敛中
        return JudgeResult(
            verdict=Verdict.CONVERGING,
            reason=f"收敛速率 {rate:.1%}/轮，预计还需 {rounds_left} 轮",
            quantitative_converged=quant_converged,
            quantitative_remaining=quant_remaining,
            qualitative_resolved=qual_resolved,
            qualitative_remaining=qual_remaining,
            estimated_rounds_left=rounds_left,
            should_stop=False,
        )

    def _check_quantitative(
        self, prev: "SharedState", curr: "SharedState",
    ) -> tuple[list[str], list[str]]:
        """检查数值型参数的收敛状态."""
        converged, remaining = [], []
        prev_params = prev.task_params
        curr_params = curr.task_params

        for key, cv in curr_params.items():
            pv = prev_params.get(key)
            if pv is None:
                continue
            if not isinstance(cv, (int, float)) or not isinstance(pv, (int, float)):
                continue
            denominator = max(abs(pv), 0.001)
            change_rate = abs(cv - pv) / denominator
            if change_rate < self.NUMERIC_THRESHOLD:
                converged.append(f"{key}: {pv}→{cv} (Δ{change_rate:.1%})")
            else:
                remaining.append(f"{key}: {pv}→{cv} (Δ{change_rate:.1%})")
        return converged, remaining

    def _check_qualitative(
        self, prev: "SharedState", curr: "SharedState",
    ) -> tuple[list[str], list[str]]:
        """检查定性冲突的收敛状态."""
        prev_ids = {c.id for c in prev.active_conflicts}
        curr_ids = {c.id for c in curr.active_conflicts}
        prev_map = {c.id: c for c in prev.active_conflicts}
        curr_map = {c.id: c for c in curr.active_conflicts}

        resolved = [cid for cid in prev_ids if cid not in curr_ids]
        remaining = []
        for cid in curr_ids & prev_ids:
            ps = prev_map[cid].severity
            cs = curr_map[cid].severity
            if cs >= ps:  # 没缩小
                remaining.append(cid)
            else:
                converged_entry = f"{cid} (严重度: {ps}→{cs})"
                # 严重度降到 < 2 可以认为已解决
                if cs < 2.0:
                    resolved.append(converged_entry)
                else:
                    remaining.append(converged_entry)
        return resolved, remaining

    def detect_oscillation(self, state: "SharedState") -> bool:
        """检测振荡：连续 OSCILLATION_WINDOW 轮冲突总量未缩小."""
        log = state.convergence_log
        if len(log) < self.OSCILLATION_WINDOW + 1:
            return False

        # 取最近 N+1 轮各冲突的严重度变化
        recent_severity_sums = []
        log_entries = list(log)  # 按顺序
        for entry in log_entries:
            sevs = [e["severity"] for e in entry.get("entries", [])]
            recent_severity_sums.append(sum(sevs) if sevs else 0)

        if len(recent_severity_sums) < self.OSCILLATION_WINDOW + 1:
            return False

        recent = recent_severity_sums[-(self.OSCILLATION_WINDOW + 1):]
        # 检查最近 OPTION_WINDOW 轮是否有持续下降趋势
        improvements = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        # 所有相邻差值都 >= 0（没缩小），且至少存在一个差值 > 0（有扩大）
        if all(imp >= 0 for imp in improvements) and any(imp > 0 for imp in improvements):
            return True
        # 如果全部差值约等于 0（停滞）
        if all(abs(imp) < 0.5 for imp in improvements):
            return True
        return False

    def convergence_rate(self, state: "SharedState") -> float:
        """计算当前收敛速率（基于冲突严重度总和的变化）."""
        log = state.convergence_log
        if len(log) < 2:
            return 1.0  # 第一轮，无历史
        entries = list(log)[-2:]  # 最近两轮
        prev_sum = sum(e.get("severity", 0) for e in entries[0].get("entries", []))
        curr_sum = sum(e.get("severity", 0) for e in entries[1].get("entries", []))
        if prev_sum == 0:
            return 1.0 if curr_sum == 0 else 0.0
        return max(0.0, (prev_sum - curr_sum) / prev_sum)

    def _estimate_rounds_left(self, state: "SharedState", rate: float) -> int:
        """估算剩余轮次."""
        if rate <= 0:
            return state.max_rounds - state.round_num
        total_severity = sum(c.severity for c in state.active_conflicts)
        if total_severity <= 0:
            return 0
        # 每轮减少 rate 的比例，计算剩余轮次
        rounds = 0
        remaining = total_severity
        while remaining > 2.0 and rounds < state.max_rounds:
            remaining *= (1 - rate)
            rounds += 1
        return rounds

    def check_physical_impossible(self, analyses: dict) -> list[dict]:
        """检测物理不可能声明."""
        alarms = []
        keywords = [
            "物理上不可能实现", "违反热力学定律", "physically impossible",
            "违反物理定律", "违反了能量守恒",
        ]
        for agent, text in analyses.items():
            text_str = str(text).lower()
            for kw in keywords:
                if kw.lower() in text_str:
                    alarms.append({
                        "agent": agent,
                        "flag_type": "physical_limit",
                        "message": f"{agent} 提出物理限制: ...{text_str[max(0, text_str.find(kw.lower())-50):text_str.find(kw.lower())+100]}",
                        "round_num": 0,
                    })
        return alarms
