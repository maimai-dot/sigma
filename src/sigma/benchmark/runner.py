"""Benchmark execution engine — runs benchmark suites through Sigma.

Supports replay mode (no real LLM calls) for testing and CI,
and live mode for actual benchmarking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sigma.benchmark.tasks import BenchmarkTask, TASKS
from sigma.benchmark.metrics import compute_all_metrics, TaskMetrics


@dataclass
class BenchmarkRun:
    """Result of a single benchmark task execution."""
    task: BenchmarkTask
    result: dict
    metrics: TaskMetrics
    elapsed_seconds: float
    error: str | None = None


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from a full benchmark suite."""
    runs: list[BenchmarkRun]
    suite_name: str
    config_label: str

    @property
    def avg_composite(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.metrics.composite_score for r in self.runs) / len(self.runs)

    @property
    def convergence_rate(self) -> float:
        """Fraction of tasks that reached converged/converging."""
        if not self.runs:
            return 0.0
        good = sum(1 for r in self.runs
                   if r.metrics.verdict in ("converged", "converging"))
        return good / len(self.runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.metrics.total_tokens for r in self.runs)

    @property
    def total_cost(self) -> float:
        return sum(r.metrics.estimated_cost_rmb for r in self.runs)


def run_task_replay(task: BenchmarkTask) -> BenchmarkRun:
    """Run a single benchmark task in replay mode (no real LLM).

    Simulates Sigma output structure for metric computation validation.
    """
    import time
    start = time.monotonic()

    # Simulate result structure based on tier expectations
    tier_rounds = {"lite": 1, "standard": 2, "rigorous": 3}
    tier_tokens = {"lite": 800, "standard": 5000, "rigorous": 18000}
    tier_cost = {"lite": 0.003, "standard": 0.02, "rigorous": 0.08}

    rounds = tier_rounds.get(task.expected_tier, 2)
    tokens = tier_tokens.get(task.expected_tier, 5000)
    cost = tier_cost.get(task.expected_tier, 0.02)

    # Simulate parameter accuracy based on whether task has expected_params
    params = {}
    if task.expected_params:
        import random
        rng = random.Random(hash(task.id) % (2**31))
        for key, val in task.expected_params.items():
            noise = 1.0 + rng.uniform(-task.tolerance * 0.8, task.tolerance * 0.8)
            params[key] = round(val * noise, 4)

    # Simulate agent analyses coverage
    analyses = {}
    if task.key_concepts:
        analyses["agent_0"] = " ".join(task.key_concepts)

    result = {
        "instruction": task.instruction,
        "framework": "Sigma AERC",
        "timestamp": "2026-01-01T00:00:00",
        "total_rounds": rounds,
        "max_rounds": 4,
        "final_verdict": "converged",
        "parameters": params,
        "decisions": [],
        "alarm_flags": [],
        "consensus": [],
        "cost_summary": {
            "total_tokens": tokens,
            "estimated_cost": cost,
            "calls": rounds * 4,
        },
    }

    elapsed = time.monotonic() - start
    time_to_first = elapsed * 0.15  # replay: first output ~15% of total

    metrics = compute_all_metrics(
        task_id=task.id, tier=task.expected_tier,
        result=result, expected_params=task.expected_params,
        tolerance=task.tolerance, key_concepts=task.key_concepts,
        agent_analyses=analyses,
        elapsed_seconds=elapsed,
        time_to_first_output=time_to_first,
    )

    return BenchmarkRun(task=task, result=result, metrics=metrics, elapsed_seconds=elapsed)


def run_suite_replay(
    tasks: list[BenchmarkTask] | None = None,
    suite_name: str = "Sigma Benchmark Suite",
    config_label: str = "replay",
) -> BenchmarkSuiteResult:
    """Run a full benchmark suite in replay mode."""
    tasks = tasks or TASKS
    runs = [run_task_replay(t) for t in tasks]
    return BenchmarkSuiteResult(runs=runs, suite_name=suite_name, config_label=config_label)


def run_task_live(
    task: BenchmarkTask,
    orchestrator,  # SigmaOrchestrator
) -> BenchmarkRun:
    """Run a single benchmark task with real LLM calls."""
    import time
    start = time.monotonic()

    try:
        result = orchestrator.run(instruction=task.instruction)
        error = None
    except Exception as e:
        result = {
            "instruction": task.instruction,
            "framework": "Sigma AERC",
            "timestamp": "",
            "total_rounds": 0,
            "max_rounds": 4,
            "final_verdict": "error",
            "parameters": {},
            "decisions": [],
            "alarm_flags": [],
            "consensus": [],
            "cost_summary": {"total_tokens": 0, "estimated_cost": 0, "calls": 0},
        }
        error = str(e)

    metrics = compute_all_metrics(
        task_id=task.id, tier=task.expected_tier,
        result=result, expected_params=task.expected_params,
        tolerance=task.tolerance, key_concepts=task.key_concepts,
    )

    elapsed = time.monotonic() - start
    return BenchmarkRun(task=task, result=result, metrics=metrics, elapsed_seconds=elapsed, error=error)


def run_suite_live(
    orchestrator,
    tasks: list[BenchmarkTask] | None = None,
    suite_name: str = "Sigma Benchmark Suite",
    config_label: str = "live",
) -> BenchmarkSuiteResult:
    """Run a full benchmark suite with real LLM calls."""
    tasks = tasks or TASKS
    runs = [run_task_live(t, orchestrator) for t in tasks]
    return BenchmarkSuiteResult(runs=runs, suite_name=suite_name, config_label=config_label)
