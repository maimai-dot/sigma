"""Σ/Tau — Hierarchical task decomposition with graduated conflict resolution.

Tau (Τ) mirrors a real engineering organization:
  Tau decomposes → Departments execute independently
  → Interface conflicts detected → Graduated escalation:
    1. Direct discussion (lightweight)
    2. Sigma AERC blind review (significant conflicts)
    3. Tau decision (final escalation)
"""

from sigma.tau.types import (
    SubTask, TaskGraph, SubtaskResult,
    InterfaceConflict, ConflictReport,
    ResolutionResult, TauState,
)
from sigma.tau.decomposer import TauDecomposer
from sigma.tau.executor import IndependentExecutor, ExecutorConfig, AsyncIndependentExecutor
from sigma.tau.detector import InterfaceConflictDetector
from sigma.tau.resolver import TauResolver
from sigma.tau.orchestrator import TauOrchestrator
from sigma.tau.mode_selector import select_mode, ModeSelection
from sigma.tau.hooks import TauHookSystem, TauHookPoint
from sigma.tau.config import TauConfig
from sigma.tau.capability import AgentCapability, CapabilityRegistry
from sigma.tau.benchmark import (
    TauBenchmarkTask, TauBenchmarkMetrics, TAU_BENCHMARK_TASKS,
    score_decomposition_quality, score_resolution_effectiveness,
    score_convergence_efficiency, compute_tau_composite,
    run_tau_benchmark_replay, run_tau_suite_replay, TauSuiteResult,
)

__all__ = [
    "SubTask",
    "TaskGraph",
    "SubtaskResult",
    "InterfaceConflict",
    "ConflictReport",
    "ResolutionResult",
    "TauState",
    "TauDecomposer",
    "IndependentExecutor",
    "ExecutorConfig",
    "AsyncIndependentExecutor",
    "InterfaceConflictDetector",
    "TauResolver",
    "TauOrchestrator",
    "select_mode",
    "ModeSelection",
    "TauHookSystem",
    "TauHookPoint",
    "TauConfig",
    "AgentCapability",
    "CapabilityRegistry",
    # Tau Benchmark
    "TauBenchmarkTask",
    "TauBenchmarkMetrics",
    "TAU_BENCHMARK_TASKS",
    "score_decomposition_quality",
    "score_resolution_effectiveness",
    "score_convergence_efficiency",
    "compute_tau_composite",
    "run_tau_benchmark_replay",
    "run_tau_suite_replay",
    "TauSuiteResult",
]
