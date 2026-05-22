"""Benchmark metrics — quantify multi-agent framework performance.

Each metric produces a 0.0–1.0 score and a human-readable summary.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricResult:
    """Single metric evaluation result."""
    name: str
    score: float               # 0.0–1.0 (higher is better)
    detail: str                 # human-readable explanation
    raw: dict = field(default_factory=dict)


@dataclass
class TaskMetrics:
    """All metrics for one benchmark task run."""
    task_id: str
    tier: str
    rounds_completed: int
    verdict: str
    total_tokens: int
    estimated_cost_rmb: float

    schema_compliance: MetricResult
    convergence: MetricResult
    cost_efficiency: MetricResult
    parameter_accuracy: MetricResult
    key_concept_coverage: MetricResult

    composite_score: float      # weighted average

    # Latency (optional, for live runs)
    elapsed_seconds: float = 0.0
    time_to_first_output: float = 0.0


def score_schema_compliance(
    result: dict, expected_schema: dict | None,
) -> MetricResult:
    """Score whether the output conforms to expected structure."""
    if expected_schema is None:
        return MetricResult(
            name="schema_compliance", score=1.0,
            detail="无 schema 约束，默认通过",
        )

    from sigma.schema_validator import validate_against_schema
    errors = validate_against_schema(result, expected_schema)
    if not errors:
        return MetricResult(
            name="schema_compliance", score=1.0,
            detail="输出完全符合 schema",
        )

    penalty = min(len(errors) * 0.2, 1.0)
    return MetricResult(
        name="schema_compliance",
        score=max(0.0, 1.0 - penalty),
        detail=f"{len(errors)} 个 schema 错误: {'; '.join(errors[:3])}",
        raw={"errors": errors},
    )


def score_convergence(
    verdict: str, rounds: int, max_rounds: int,
    should_stop_reason: str = "",
) -> MetricResult:
    """Score convergence quality.

    - Converged within max_rounds: 1.0
    - Converging (stopped by user): 0.7
    - Oscillating/stalled: 0.0
    - Slow: 0.4
    """
    if verdict == "converged":
        if rounds <= 2:
            return MetricResult(
                name="convergence", score=1.0,
                detail=f"快速收敛 (第 {rounds} 轮)",
                raw={"verdict": verdict, "rounds": rounds},
            )
        return MetricResult(
            name="convergence", score=0.85,
            detail=f"收敛 (第 {rounds} 轮)",
            raw={"verdict": verdict, "rounds": rounds},
        )
    elif verdict == "converging":
        return MetricResult(
            name="convergence", score=0.7,
            detail=f"收敛中，停止于第 {rounds} 轮",
            raw={"verdict": verdict, "rounds": rounds, "reason": should_stop_reason},
        )
    elif verdict == "slow":
        return MetricResult(
            name="convergence", score=0.4,
            detail=f"收敛过慢 (第 {rounds} 轮): {should_stop_reason}",
            raw={"verdict": verdict, "rounds": rounds, "reason": should_stop_reason},
        )
    elif verdict == "oscillating":
        return MetricResult(
            name="convergence", score=0.0,
            detail=f"振荡停止: {should_stop_reason}",
            raw={"verdict": verdict, "rounds": rounds, "reason": should_stop_reason},
        )
    else:  # stalled
        return MetricResult(
            name="convergence", score=0.0,
            detail=f"停滞: {should_stop_reason}",
            raw={"verdict": verdict, "rounds": rounds, "reason": should_stop_reason},
        )


def score_cost_efficiency(
    total_tokens: int, estimated_cost: float,
    baseline_tokens: Optional[int] = None,
    baseline_cost: Optional[float] = None,
) -> MetricResult:
    """Score cost efficiency. If baseline provided, compute ratio."""
    if baseline_tokens is not None and baseline_tokens > 0:
        ratio = baseline_tokens / max(total_tokens, 1)
        if ratio >= 2.0:
            score = 1.0
        elif ratio >= 1.5:
            score = 0.9
        elif ratio >= 1.0:
            score = 0.7
        elif ratio >= 0.7:
            score = 0.5
        else:
            score = 0.3
        return MetricResult(
            name="cost_efficiency", score=score,
            detail=f"Token 使用: {total_tokens} (基线 {baseline_tokens}, 节省 {ratio:.1%})",
            raw={"tokens": total_tokens, "baseline": baseline_tokens, "ratio": ratio, "cost": estimated_cost},
        )

    # Without baseline, rate based on task tier expectations
    if total_tokens < 2000:
        return MetricResult(name="cost_efficiency", score=1.0, detail=f"极低 token 消耗 ({total_tokens})")
    elif total_tokens < 5000:
        return MetricResult(name="cost_efficiency", score=0.9, detail=f"低 token 消耗 ({total_tokens})")
    elif total_tokens < 15000:
        return MetricResult(name="cost_efficiency", score=0.7, detail=f"中等 token 消耗 ({total_tokens})")
    elif total_tokens < 40000:
        return MetricResult(name="cost_efficiency", score=0.5, detail=f"较高 token 消耗 ({total_tokens})")
    else:
        return MetricResult(name="cost_efficiency", score=0.3, detail=f"高 token 消耗 ({total_tokens})")


def score_parameter_accuracy(
    actual_params: dict, expected_params: dict, tolerance: float = 0.15,
) -> MetricResult:
    """Score how close computed parameters are to ground truth."""
    if not expected_params:
        return MetricResult(
            name="parameter_accuracy", score=1.0,
            detail="无 ground truth 参数，跳过",
        )

    matches = 0
    details = []
    for key, expected in expected_params.items():
        # Fuzzy match on parameter key
        actual_val = None
        matched_key = None
        for ak, av in actual_params.items():
            if key.replace("_", "") in ak.replace("_", "") or ak.replace("_", "") in key.replace("_", ""):
                if isinstance(av, (int, float)):
                    actual_val = av
                    matched_key = ak
                    break

        if actual_val is None:
            details.append(f"{key}: 未找到")
            continue

        denominator = max(abs(expected), 0.001)
        error = abs(actual_val - expected) / denominator
        if error <= tolerance:
            matches += 1
            details.append(f"{key}: {actual_val} (目标 {expected}, 误差 {error:.1%}) ✓")
        else:
            details.append(f"{key}: {actual_val} (目标 {expected}, 误差 {error:.1%}) ✗")

    if len(expected_params) == 0:
        score = 1.0
    else:
        score = matches / len(expected_params)

    return MetricResult(
        name="parameter_accuracy", score=score,
        detail=f"{matches}/{len(expected_params)} 参数在容差内",
        raw={"details": details, "tolerance": tolerance},
    )


def score_key_concept_coverage(
    agent_analyses: dict[str, str], key_concepts: list[str],
) -> MetricResult:
    """Score whether agent analyses cover expected key concepts."""
    if not key_concepts:
        return MetricResult(
            name="key_concept_coverage", score=1.0,
            detail="无关键概念要求",
        )

    all_text = " ".join(str(v) for v in agent_analyses.values()).lower()
    covered = sum(1 for kc in key_concepts if kc.lower() in all_text)
    score = covered / len(key_concepts) if key_concepts else 1.0

    return MetricResult(
        name="key_concept_coverage", score=score,
        detail=f"{covered}/{len(key_concepts)} 关键概念覆盖",
        raw={"covered": covered, "total": len(key_concepts),
             "missing": [kc for kc in key_concepts if kc.lower() not in all_text]},
    )


def score_latency(
    elapsed_seconds: float, task_tier: str,
) -> MetricResult:
    """Score latency based on tier expectations.

    Target per-tier latencies (wall clock):
      - LITE: < 5s
      - STANDARD: < 30s
      - RIGOROUS: < 120s
    """
    targets = {"lite": 5, "standard": 30, "rigorous": 120}
    target = targets.get(task_tier, 60)

    if elapsed_seconds <= 0:
        return MetricResult(name="latency", score=0.0, detail="无法测量延迟")

    ratio = target / elapsed_seconds
    if ratio >= 1.0:
        score = 1.0
        detail = f"{elapsed_seconds:.1f}s (目标 <{target}s) ✓"
    elif ratio >= 0.7:
        score = 0.8
        detail = f"{elapsed_seconds:.1f}s (目标 <{target}s, 略慢)"
    elif ratio >= 0.5:
        score = 0.5
        detail = f"{elapsed_seconds:.1f}s (目标 <{target}s, 偏慢)"
    else:
        score = 0.2
        detail = f"{elapsed_seconds:.1f}s (目标 <{target}s, 很慢)"

    return MetricResult(
        name="latency", score=score, detail=detail,
        raw={"elapsed": elapsed_seconds, "target": target},
    )


def compute_composite(metrics: list[MetricResult], weights: dict[str, float] | None = None) -> float:
    """Compute weighted composite score from individual metrics."""
    if weights is None:
        weights = {
            "schema_compliance": 0.15,
            "convergence": 0.30,
            "cost_efficiency": 0.15,
            "parameter_accuracy": 0.25,
            "key_concept_coverage": 0.15,
        }
    total = 0.0
    weight_sum = 0.0
    for m in metrics:
        w = weights.get(m.name, 0.2)
        total += m.score * w
        weight_sum += w
    return total / max(weight_sum, 0.001)


def compute_all_metrics(
    task_id: str, tier: str, result: dict, expected_params: dict,
    tolerance: float, key_concepts: list[str],
    agent_analyses: dict[str, str] | None = None,
    baseline_tokens: int | None = None,
    baseline_cost: float | None = None,
    elapsed_seconds: float = 0.0,
    time_to_first_output: float = 0.0,
) -> TaskMetrics:
    """Compute all metrics for one task run."""
    sc = score_schema_compliance(result, result.get("_schema"))
    conv = score_convergence(
        result.get("final_verdict", "unknown"),
        result.get("total_rounds", 0),
        result.get("max_rounds", 4),
    )
    cost = score_cost_efficiency(
        result.get("cost_summary", {}).get("total_tokens", 0),
        result.get("cost_summary", {}).get("estimated_cost", 0),
        baseline_tokens, baseline_cost,
    )
    acc = score_parameter_accuracy(
        result.get("parameters", {}), expected_params, tolerance,
    )
    kcc = score_key_concept_coverage(
        agent_analyses or {}, key_concepts,
    )
    lat = score_latency(elapsed_seconds, tier)

    composite = compute_composite([sc, conv, cost, acc, kcc, lat])

    return TaskMetrics(
        task_id=task_id,
        tier=tier,
        rounds_completed=result.get("total_rounds", 0),
        verdict=result.get("final_verdict", "unknown"),
        total_tokens=result.get("cost_summary", {}).get("total_tokens", 0),
        estimated_cost_rmb=result.get("cost_summary", {}).get("estimated_cost", 0),
        elapsed_seconds=elapsed_seconds,
        time_to_first_output=time_to_first_output,
        schema_compliance=sc,
        convergence=conv,
        cost_efficiency=cost,
        parameter_accuracy=acc,
        key_concept_coverage=kcc,
        composite_score=composite,
    )
