"""Benchmark report generator — produces comparison tables and Markdown reports.

Generates patent-ready comparison data: Σ vs bare-LLM, tier comparison,
and per-task breakdown with quantitative metrics.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sigma.benchmark.metrics import TaskMetrics
from sigma.benchmark.runner import BenchmarkSuiteResult, BenchmarkRun


@dataclass
class ComparisonPoint:
    """A single data point for side-by-side comparison."""
    label: str
    avg_composite: float
    convergence_rate: float
    total_tokens: int
    total_cost: float
    avg_rounds: float
    schema_compliance: float
    param_accuracy: float
    task_count: int


def compare_suites(
    suites: list[BenchmarkSuiteResult],
) -> list[ComparisonPoint]:
    """Generate comparison points from multiple benchmark suites."""
    points = []
    for suite in suites:
        if not suite.runs:
            continue
        n = len(suite.runs)
        points.append(ComparisonPoint(
            label=suite.config_label,
            avg_composite=suite.avg_composite,
            convergence_rate=suite.convergence_rate,
            total_tokens=suite.total_tokens,
            total_cost=suite.total_cost,
            avg_rounds=sum(r.metrics.rounds_completed for r in suite.runs) / n,
            schema_compliance=sum(r.metrics.schema_compliance.score for r in suite.runs) / n,
            param_accuracy=sum(r.metrics.parameter_accuracy.score for r in suite.runs) / n,
            task_count=n,
        ))
    return points


def _tier_label(tier: str) -> str:
    return {"lite": "⚡ LITE", "standard": "◆ STANDARD", "rigorous": "★ RIGOROUS"}.get(tier, tier)


def generate_markdown_report(
    suite: BenchmarkSuiteResult,
    baseline: BenchmarkSuiteResult | None = None,
    title: str = "Σ (Sigma) Benchmark Report",
) -> str:
    """Generate a full Markdown benchmark report.

    If baseline is provided, includes side-by-side comparison.
    """
    lines = [
        f"# {title}",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Suite**: {suite.suite_name} ({suite.config_label})",
        f"**Tasks**: {len(suite.runs)}",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Composite Score | **{suite.avg_composite:.2f}** / 1.00 |")
    lines.append(f"| Convergence Rate | **{suite.convergence_rate:.0%}** |")
    lines.append(f"| Total Tokens | {suite.total_tokens:,} |")
    lines.append(f"| Total Cost (RMB) | ¥{suite.total_cost:.4f} |")
    lines.append(f"| Avg Rounds | {sum(r.metrics.rounds_completed for r in suite.runs) / max(len(suite.runs), 1):.1f} |")
    lines.append("")

    # Comparison if baseline provided
    if baseline:
        lines.append("## Comparison: Σ vs Baseline")
        lines.append("")
        lines.append(f"| Metric | Baseline ({baseline.config_label}) | Σ ({suite.config_label}) | Improvement |")
        lines.append(f"|--------|----------------------------------|--------------------------|-------------|")
        lines.append(f"| Composite | {baseline.avg_composite:.2f} | {suite.avg_composite:.2f} | {_delta(baseline.avg_composite, suite.avg_composite)} |")
        lines.append(f"| Convergence | {baseline.convergence_rate:.0%} | {suite.convergence_rate:.0%} | {_delta(baseline.convergence_rate, suite.convergence_rate)} |")
        lines.append(f"| Tokens | {baseline.total_tokens:,} | {suite.total_tokens:,} | {_delta(baseline.total_tokens, suite.total_tokens, lower_better=True)} |")
        lines.append(f"| Cost | ¥{baseline.total_cost:.4f} | ¥{suite.total_cost:.4f} | {_delta(baseline.total_cost, suite.total_cost, lower_better=True)} |")
        lines.append("")

    # Per-tier breakdown
    lines.append("## Per-Tier Breakdown")
    lines.append("")
    tiers = {}
    for run in suite.runs:
        t = run.task.expected_tier
        if t not in tiers:
            tiers[t] = []
        tiers[t].append(run)

    for tier_name in ["lite", "standard", "rigorous"]:
        if tier_name not in tiers:
            continue
        tier_runs = tiers[tier_name]
        avg = sum(r.metrics.composite_score for r in tier_runs) / len(tier_runs)
        avg_cost = sum(r.metrics.estimated_cost_rmb for r in tier_runs) / len(tier_runs)
        lines.append(f"### {_tier_label(tier_name)} ({len(tier_runs)} tasks)")
        lines.append(f"- Avg Composite: **{avg:.2f}**")
        lines.append(f"- Avg Cost: ¥{avg_cost:.4f}")
        lines.append("")

    # Per-task detail
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append(f"| Task | Tier | Rounds | Verdict | Tokens | Cost | Latency | Score |")
    lines.append(f"|------|------|--------|---------|--------|------|---------|-------|")
    for run in suite.runs:
        m = run.metrics
        lines.append(
            f"| {run.task.id} | {_tier_label(run.task.expected_tier)} | "
            f"{m.rounds_completed} | {m.verdict} | {m.total_tokens:,} | "
            f"¥{m.estimated_cost_rmb:.4f} | {run.elapsed_seconds:.1f}s | **{m.composite_score:.2f}** |"
        )
    lines.append("")

    # Detailed metrics per task
    lines.append("## Detailed Metrics")
    lines.append("")
    for run in suite.runs:
        m = run.metrics
        lines.append(f"### {run.task.id} — {run.task.description}")
        lines.append(f"- **Composite**: {m.composite_score:.2f}")
        lines.append(f"- Schema Compliance: {m.schema_compliance.score:.2f} — {m.schema_compliance.detail}")
        lines.append(f"- Convergence: {m.convergence.score:.2f} — {m.convergence.detail}")
        lines.append(f"- Cost Efficiency: {m.cost_efficiency.score:.2f} — {m.cost_efficiency.detail}")
        lines.append(f"- Latency: {run.elapsed_seconds:.1f}s (TTFO: {m.time_to_first_output:.1f}s)")
        lines.append(f"- Parameter Accuracy: {m.parameter_accuracy.score:.2f} — {m.parameter_accuracy.detail}")
        lines.append(f"- Concept Coverage: {m.key_concept_coverage.score:.2f} — {m.key_concept_coverage.detail}")
        if m.parameter_accuracy.raw.get("details"):
            lines.append(f"  ```")
            for d in m.parameter_accuracy.raw["details"]:
                lines.append(f"  {d}")
            lines.append(f"  ```")
        lines.append("")

    return "\n".join(lines)


def generate_json_report(
    suite: BenchmarkSuiteResult,
    baseline: BenchmarkSuiteResult | None = None,
) -> dict:
    """Generate structured JSON benchmark report."""
    runs_data = []
    for run in suite.runs:
        m = run.metrics
        runs_data.append({
            "task_id": run.task.id,
            "tier": run.task.expected_tier,
            "category": run.task.category,
            "composite_score": m.composite_score,
            "rounds": m.rounds_completed,
            "verdict": m.verdict,
            "tokens": m.total_tokens,
            "cost_rmb": m.estimated_cost_rmb,
            "schema_compliance": m.schema_compliance.score,
            "convergence": m.convergence.score,
            "cost_efficiency": m.cost_efficiency.score,
            "parameter_accuracy": m.parameter_accuracy.score,
            "key_concept_coverage": m.key_concept_coverage.score,
            "elapsed_seconds": run.elapsed_seconds,
            "error": run.error,
        })

    report = {
        "title": suite.suite_name,
        "config": suite.config_label,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "avg_composite": suite.avg_composite,
            "convergence_rate": suite.convergence_rate,
            "total_tokens": suite.total_tokens,
            "total_cost_rmb": suite.total_cost,
            "task_count": len(suite.runs),
        },
        "runs": runs_data,
    }

    if baseline:
        points = compare_suites([baseline, suite])
        report["comparison"] = [
            {"label": p.label, "avg_composite": p.avg_composite,
             "convergence_rate": p.convergence_rate,
             "total_tokens": p.total_tokens, "total_cost_rmb": p.total_cost}
            for p in points
        ]

    return report


def _delta(baseline: float, current: float, lower_better: bool = False) -> str:
    """Format delta with direction arrow."""
    if baseline == 0:
        return "N/A"
    change = (current - baseline) / abs(baseline)
    pct = f"{abs(change):.0%}"
    if lower_better:
        if change < 0:
            return f"↓ {pct} better"
        elif change > 0:
            return f"↑ {pct} worse"
        return "— same"
    else:
        if change > 0:
            return f"↑ {pct} better"
        elif change < 0:
            return f"↓ {pct} worse"
        return "— same"


def save_report(suite: BenchmarkSuiteResult, output_dir: Path,
                baseline: BenchmarkSuiteResult | None = None,
                title: str = "Σ (Sigma) Benchmark Report") -> Path:
    """Save both Markdown and JSON reports to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md = generate_markdown_report(suite, baseline, title)
    md_path = output_dir / "BENCHMARK.md"
    md_path.write_text(md, encoding="utf-8")

    js = generate_json_report(suite, baseline)
    js_path = output_dir / "benchmark.json"
    js_path.write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path
