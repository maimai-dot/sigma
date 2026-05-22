"""Tests for the benchmark framework — tasks, metrics, runner, reporter."""

import json
import pytest
from pathlib import Path

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
)
from sigma.benchmark.reporter import (
    ComparisonPoint, compare_suites,
    generate_markdown_report, generate_json_report, save_report,
)


# ── Task Definitions ──────────────────────────────────────────────

class TestBenchmarkTasks:
    def test_all_tasks_have_required_fields(self):
        for t in TASKS:
            assert t.id
            assert t.instruction
            assert t.expected_tier in ("lite", "standard", "rigorous")
            assert t.category in ("lookup", "calculation", "analysis", "design")

    def test_tasks_cover_all_tiers(self):
        tiers = {t.expected_tier for t in TASKS}
        assert "lite" in tiers
        assert "standard" in tiers
        assert "rigorous" in tiers

    def test_tasks_have_min_agents(self):
        for t in TASKS:
            assert t.min_agents >= 2

    def test_task_count_is_10(self):
        assert len(TASKS) == 10

    def test_lite_tasks_exist(self):
        lite = [t for t in TASKS if t.expected_tier == "lite"]
        assert len(lite) >= 2
        for t in lite:
            assert t.min_agents <= 4


# ── Metrics ───────────────────────────────────────────────────────

class TestSchemaCompliance:
    def test_no_schema_returns_perfect(self):
        r = score_schema_compliance({"a": 1}, None)
        assert r.score == 1.0

    def test_valid_data_matches_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        r = score_schema_compliance({"x": 42}, schema)
        assert r.score == 1.0

    def test_invalid_data_penalized(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        r = score_schema_compliance({"x": 42}, schema)
        assert r.score < 1.0


class TestConvergence:
    def test_converged_fast_is_perfect(self):
        r = score_convergence("converged", 1, 4)
        assert r.score == 1.0

    def test_converged_late_is_high(self):
        r = score_convergence("converged", 3, 4)
        assert r.score >= 0.8

    def test_oscillating_is_zero(self):
        r = score_convergence("oscillating", 3, 4)
        assert r.score == 0.0

    def test_stalled_is_zero(self):
        r = score_convergence("stalled", 4, 4)
        assert r.score == 0.0

    def test_slow_is_moderate(self):
        r = score_convergence("slow", 3, 4)
        assert 0.3 <= r.score <= 0.5

    def test_converging_is_good(self):
        r = score_convergence("converging", 2, 4)
        assert r.score >= 0.6


class TestCostEfficiency:
    def test_very_low_tokens_is_perfect(self):
        r = score_cost_efficiency(500, 0.001)
        assert r.score == 1.0

    def test_low_tokens_is_high(self):
        r = score_cost_efficiency(3000, 0.01)
        assert r.score >= 0.8

    def test_high_tokens_is_low(self):
        r = score_cost_efficiency(50000, 0.2)
        assert r.score <= 0.5

    def test_with_baseline_saving(self):
        r = score_cost_efficiency(5000, 0.02, baseline_tokens=10000)
        assert r.score >= 0.8

    def test_with_baseline_worse(self):
        r = score_cost_efficiency(10000, 0.04, baseline_tokens=5000)
        assert r.score <= 0.6


class TestParameterAccuracy:
    def test_no_expected_returns_perfect(self):
        r = score_parameter_accuracy({}, {})
        assert r.score == 1.0

    def test_exact_match(self):
        r = score_parameter_accuracy({"mass_kg": 7.5}, {"mass_kg": 7.5})
        assert r.score == 1.0

    def test_within_tolerance(self):
        r = score_parameter_accuracy({"mass_kg": 7.8}, {"mass_kg": 7.5}, tolerance=0.10)
        assert r.score == 1.0  # 4% error < 10%

    def test_outside_tolerance(self):
        r = score_parameter_accuracy({"mass_kg": 10.0}, {"mass_kg": 7.5}, tolerance=0.10)
        assert r.score == 0.0  # 33% error > 10%

    def test_fuzzy_key_match(self):
        r = score_parameter_accuracy(
            {"tube_mass_kg": 7.5}, {"mass_kg": 7.5}, tolerance=0.05,
        )
        assert r.score >= 0.5  # Fuzzy match should find it

    def test_missing_param(self):
        r = score_parameter_accuracy({"other": 1.0}, {"mass_kg": 7.5})
        assert r.score == 0.0

    def test_multiple_params_mixed(self):
        r = score_parameter_accuracy(
            {"isp_s": 158, "c_star_m_s": 900},
            {"isp_s": 155, "c_star_m_s": 920},
            tolerance=0.15,
        )
        assert r.score == 1.0  # Both within 15%


class TestKeyConceptCoverage:
    def test_no_concepts_returns_perfect(self):
        r = score_key_concept_coverage({}, [])
        assert r.score == 1.0

    def test_all_covered(self):
        r = score_key_concept_coverage(
            {"a": "讨论固体火箭发动机推力"},
            ["固体", "推力"],
        )
        assert r.score == 1.0

    def test_partial_coverage(self):
        r = score_key_concept_coverage(
            {"a": "讨论发动机设计"},
            ["推力", "比冲"],
        )
        assert r.score == 0.0

    def test_case_insensitive(self):
        r = score_key_concept_coverage(
            {"a": "KNSB propellant analysis"},
            ["knsb"],
        )
        assert r.score == 1.0


class TestComposite:
    def test_all_perfect_is_1(self):
        metrics = [
            MetricResult("a", 1.0, "ok"),
            MetricResult("b", 1.0, "ok"),
        ]
        assert compute_composite(metrics, {"a": 0.5, "b": 0.5}) == 1.0

    def test_weighted_average(self):
        metrics = [
            MetricResult("a", 1.0, "ok"),
            MetricResult("b", 0.0, "bad"),
        ]
        c = compute_composite(metrics, {"a": 0.5, "b": 0.5})
        assert c == 0.5

    def test_default_weights_sum_to_1(self):
        metrics = [
            MetricResult("schema_compliance", 0.5, ""),
            MetricResult("convergence", 0.5, ""),
            MetricResult("cost_efficiency", 0.5, ""),
            MetricResult("parameter_accuracy", 0.5, ""),
            MetricResult("key_concept_coverage", 0.5, ""),
        ]
        c = compute_composite(metrics)
        assert 0.0 < c <= 1.0


class TestComputeAllMetrics:
    def test_returns_task_metrics(self):
        result = {
            "total_rounds": 2, "max_rounds": 4,
            "final_verdict": "converged",
            "parameters": {"mass_kg": 7.6},
            "cost_summary": {"total_tokens": 3000, "estimated_cost": 0.01},
        }
        tm = compute_all_metrics(
            task_id="test", tier="lite",
            result=result, expected_params={"mass_kg": 7.5},
            tolerance=0.15, key_concepts=["质量"],
            agent_analyses={"a": "计算质量"},
        )
        assert isinstance(tm, TaskMetrics)
        assert tm.composite_score >= 0.8
        assert tm.verdict == "converged"


# ── Runner (Replay) ───────────────────────────────────────────────

class TestRunnerReplay:
    def test_run_task_replay_succeeds(self):
        task = TASKS[0]
        run = run_task_replay(task)
        assert run.error is None
        assert run.metrics.composite_score > 0
        assert run.elapsed_seconds >= 0

    def test_run_suite_replay_all_tasks(self):
        suite = run_suite_replay()
        assert len(suite.runs) == len(TASKS)
        assert suite.avg_composite > 0
        assert suite.convergence_rate >= 0

    def test_run_suite_replay_subset(self):
        subset = TASKS[:3]
        suite = run_suite_replay(tasks=subset)
        assert len(suite.runs) == 3

    def test_run_task_replay_params_close_to_expected(self):
        task = [t for t in TASKS if t.id == "b001_lookup_density"][0]
        run = run_task_replay(task)
        density = run.result["parameters"].get("density_kg_m3")
        assert density is not None
        # Should be within tolerance
        expected = task.expected_params["density_kg_m3"]
        assert abs(density - expected) / expected <= task.tolerance

    def test_benchmark_suite_result_properties(self):
        suite = run_suite_replay(tasks=TASKS[:5])
        assert 0 <= suite.avg_composite <= 1
        assert 0 <= suite.convergence_rate <= 1
        assert suite.total_tokens > 0
        assert suite.total_cost > 0


# ── Reporter ──────────────────────────────────────────────────────

class TestReporter:
    @pytest.fixture
    def suite(self):
        return run_suite_replay(tasks=TASKS[:5], suite_name="Test Suite", config_label="test")

    def test_generate_markdown_report(self, suite):
        md = generate_markdown_report(suite, title="Test Report")
        assert "# Test Report" in md
        assert "Executive Summary" in md
        assert "Per-Task Results" in md
        # Check all tasks appear
        for run in suite.runs:
            assert run.task.id in md

    def test_generate_markdown_with_baseline(self, suite):
        baseline = run_suite_replay(tasks=TASKS[:5], config_label="baseline")
        md = generate_markdown_report(suite, baseline, title="Comparison")
        assert "Comparison:" in md
        assert "baseline" in md

    def test_generate_json_report(self, suite):
        js = generate_json_report(suite)
        assert js["title"] == "Test Suite"
        assert js["summary"]["task_count"] == 5
        assert len(js["runs"]) == 5
        assert "composite_score" in js["runs"][0]

    def test_generate_json_with_baseline(self, suite):
        baseline = run_suite_replay(tasks=TASKS[:5], config_label="baseline")
        js = generate_json_report(suite, baseline)
        assert "comparison" in js
        assert len(js["comparison"]) == 2

    def test_compare_suites(self, suite):
        baseline = run_suite_replay(tasks=TASKS[:5], config_label="baseline")
        points = compare_suites([baseline, suite])
        assert len(points) == 2
        assert points[0].label == "baseline"
        assert points[1].label == "test"

    def test_save_report(self, suite, tmp_path):
        report_path = save_report(suite, tmp_path)
        assert report_path.exists()
        assert (tmp_path / "BENCHMARK.md").exists()
        assert (tmp_path / "benchmark.json").exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Composite Score" in content


class TestComparisonPoint:
    def test_delta_helper(self):
        from sigma.benchmark.reporter import _delta
        assert "better" in _delta(0.5, 1.0)  # 100% improvement
        assert "worse" in _delta(1.0, 0.5)   # 50% degradation
        assert "same" in _delta(0.5, 0.5)    # Equal


# ── Integration ───────────────────────────────────────────────────

class TestBenchmarkIntegration:
    def test_full_pipeline(self, tmp_path):
        """End-to-end: run suite → generate report → save → verify."""
        suite = run_suite_replay(tasks=TASKS[:3], suite_name="Integration Test")
        assert len(suite.runs) == 3
        for run in suite.runs:
            assert run.metrics.composite_score > 0

        # Generate reports
        md = generate_markdown_report(suite)
        js = generate_json_report(suite)
        assert len(md) > 500
        assert js["summary"]["task_count"] == 3

        # Save
        report_path = save_report(suite, tmp_path)
        assert report_path.exists()
        assert json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))

    def test_tier_coverage_in_report(self):
        """Verify all three tiers are represented."""
        suite = run_suite_replay()
        md = generate_markdown_report(suite)
        assert "LITE" in md
        assert "STANDARD" in md
        assert "RIGOROUS" in md

    def test_replay_results_are_deterministic(self):
        """Replay mode should produce consistent results."""
        suite1 = run_suite_replay(tasks=TASKS[:3])
        suite2 = run_suite_replay(tasks=TASKS[:3])
        for r1, r2 in zip(suite1.runs, suite2.runs):
            assert r1.metrics.composite_score == r2.metrics.composite_score
            assert r1.metrics.verdict == r2.metrics.verdict
