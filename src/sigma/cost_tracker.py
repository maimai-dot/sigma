"""
Token 成本追踪
每轮记录 LLM 调用次数、token 消耗、预估成本
支持 DeepSeek 定价模型
"""

from dataclasses import dataclass, field
from typing import Optional


# DeepSeek V4 Pro 定价 (¥/1M tokens)
DEEPSEEK_PRICING = {
    "input": 1.0,    # ¥1/M input tokens
    "output": 4.0,   # ¥4/M output tokens (cached) — 实际 output 价格
    "cached_input": 0.25,  # ¥0.25/M cached input
}


@dataclass
class RoundCost:
    round_num: int
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    estimated_cost_rmb: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostTracker:
    """成本追踪器."""

    def __init__(self, pricing: Optional[dict] = None):
        self.pricing = pricing or DEEPSEEK_PRICING
        self.rounds: list[RoundCost] = []

    def start_round(self, round_num: int) -> RoundCost:
        cost = RoundCost(round_num=round_num)
        self.rounds.append(cost)
        return cost

    def record_call(
        self, cost: RoundCost, input_tokens: int, output_tokens: int,
    ) -> None:
        """记录一次 LLM 调用."""
        cost.input_tokens += input_tokens
        cost.output_tokens += output_tokens
        cost.calls += 1
        # 估算成本
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        cost.estimated_cost_rmb += input_cost + output_cost

    def record_estimated_call(
        self, cost: RoundCost, input_tokens: int, output_tokens: int,
    ) -> None:
        """记录估算调用（用 token 计数但非精确值时使用）."""
        cost.input_tokens += input_tokens
        cost.output_tokens += output_tokens
        cost.calls += 1
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        cost.estimated_cost_rmb += input_cost + output_cost

    def round_summary(self, cost: RoundCost) -> str:
        return (
            f"Token: {cost.total_tokens:,} | "
            f"预估 ¥{cost.estimated_cost_rmb:.2f} | "
            f"调用 {cost.calls} 次"
        )

    def total_summary(self) -> str:
        total_tokens = sum(r.total_tokens for r in self.rounds)
        total_cost = sum(r.estimated_cost_rmb for r in self.rounds)
        total_calls = sum(r.calls for r in self.rounds)
        return (
            f"总 Token: {total_tokens:,} | "
            f"总预估 ¥{total_cost:.2f} | "
            f"LLM 调用: {total_calls} 次 | "
            f"轮次: {len(self.rounds)}"
        )
