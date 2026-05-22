"""Sigma Benchmark — quantitative evaluation framework for multi-agent systems.

Provides standardized task suites, automated metrics, and comparison
reporting for patent-grade framework evaluation.
"""

from sigma.benchmark.tasks import BenchmarkTask, TASKS
from sigma.benchmark.metrics import (
    MetricResult, TaskMetrics,
    score_schema_compliance, score_convergence, score_cost_efficiency,
    score_parameter_accuracy, score_key_concept_coverage,
    compute_composite, compute_all_metrics,
)
from sigma.benchmark.runner import (
    BenchmarkRun, BenchmarkSuiteResult,
    run_task_replay, run_suite_replay,
    run_task_live, run_suite_live,
)
from sigma.benchmark.reporter import (
    ComparisonPoint, compare_suites,
    generate_markdown_report, generate_json_report, save_report,
)

__all__ = [
    # Tasks
    "BenchmarkTask", "TASKS",
    # Metrics
    "MetricResult", "TaskMetrics",
    "score_schema_compliance", "score_convergence", "score_cost_efficiency",
    "score_parameter_accuracy", "score_key_concept_coverage",
    "compute_composite", "compute_all_metrics",
    # Runner
    "BenchmarkRun", "BenchmarkSuiteResult",
    "run_task_replay", "run_suite_replay",
    "run_task_live", "run_suite_live",
    # Reporter
    "ComparisonPoint", "compare_suites",
    "generate_markdown_report", "generate_json_report", "save_report",
]
