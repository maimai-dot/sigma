"""Tau Benchmark — evaluation suite for hierarchical decomposition + conflict resolution.

Metrics specific to Tau:
  - Decomposition quality: subtask count, interface param coverage, assignment accuracy
  - Resolution effectiveness: conflicts found, resolution rate by level, escalation count
  - Convergence efficiency: iterations to completion, cost per iteration
  - End-to-end success: task completion rate

Replay mode (no real LLM) for CI; live mode for real benchmarking.
"""

from dataclasses import dataclass, field
import time


@dataclass
class TauBenchmarkTask:
    """A benchmark task for evaluating the Tau hierarchical framework."""

    id: str
    instruction: str
    description: str = ""

    # Expected decomposition quality
    expected_min_subtasks: int = 2
    expected_max_subtasks: int = 8
    expected_interface_params: list[str] = field(default_factory=list)
    """Interface params the decomposer should identify."""

    # Expected agent assignments
    valid_agent_names: list[str] = field(default_factory=list)

    # Ground truth for parameter accuracy (same as Sigma benchmark tasks)
    expected_params: dict[str, float] = field(default_factory=dict)
    tolerance: float = 0.15


@dataclass
class TauBenchmarkMetrics:
    """Tau-specific metrics for one benchmark task."""

    task_id: str

    # Decomposition
    subtask_count: int
    interface_param_count: int
    interface_param_coverage: float         # fraction of expected params found
    agent_assignment_validity: float        # fraction of agents valid

    # Conflict detection
    conflicts_detected: int
    conflict_detection_precision: float     # of expected conflicts

    # Resolution
    rounds_to_complete: int
    max_rounds: int
    resolution_escalations: int
    director_decisions: int
    resolved: bool

    # Cost
    estimated_tokens: int
    estimated_cost_rmb: float

    # Composite
    composite_score: float


TAU_BENCHMARK_TASKS: list[TauBenchmarkTask] = [
    TauBenchmarkTask(
        id="tau001_simple_sequential",
        instruction="首先计算6061-T6铝管质量（外径150mm，壁厚3mm，长度2m），然后根据质量选择合适的降落伞尺寸",
        description="Simple sequential two-step — LITE complexity",
        expected_min_subtasks=2,
        expected_max_subtasks=4,
        expected_interface_params=["mass_kg"],
        valid_agent_names=["Structures Chief", "GNC Chief", "Sim Chief", "Engine A", "Engine B"],
    ),
    TauBenchmarkTask(
        id="tau002_propulsion_design",
        instruction="设计KNSB固体火箭发动机：首先选择推进剂配比并估算比冲，然后计算推力曲线，最后确定喷管尺寸",
        description="Three-step propulsion design — STANDARD complexity",
        expected_min_subtasks=3,
        expected_max_subtasks=6,
        expected_interface_params=["isp_s", "thrust_n", "chamber_pressure_bar"],
        valid_agent_names=["Propulsion Chief", "Sim Chief", "Structures Chief", "Supply Agent"],
    ),
    TauBenchmarkTask(
        id="tau003_full_rocket",
        instruction="设计一枚1km级验证火箭：拆解推进、结构、飞控、安全、供应链五个方面，各部门独立分析后整合",
        description="Multi-department full rocket — RIGOROUS complexity",
        expected_min_subtasks=5,
        expected_max_subtasks=10,
        expected_interface_params=["mass_kg", "thrust_n", "stability_margin_cal", "cost_rmb"],
        valid_agent_names=[
            "Propulsion Chief", "Structures Chief", "GNC Chief",
            "Sim Chief", "Supply Agent", "Safety Officer",
        ],
    ),
    TauBenchmarkTask(
        id="tau004_cross_department_integration",
        instruction="并行分析推进系统推力输出和箭体结构质量限制，汇总为总体设计参数",
        description="Cross-department with shared interface params",
        expected_min_subtasks=2,
        expected_max_subtasks=5,
        expected_interface_params=["thrust_n", "mass_kg"],
        valid_agent_names=["Propulsion Chief", "Structures Chief", "Sim Chief", "Engine A", "Engine B"],
    ),
    TauBenchmarkTask(
        id="tau005_material_supply_chain",
        instruction="首先分析三种火箭箭体材料的性能，然后评估供应链可获取性，最后给出综合推荐",
        description="Analysis with supply chain — STANDARD complexity",
        expected_min_subtasks=3,
        expected_max_subtasks=6,
        expected_interface_params=["cost_rmb", "density_kg_m3", "yield_strength_mpa"],
        valid_agent_names=["Structures Chief", "Supply Agent", "Safety Officer"],
    ),
]


def score_decomposition_quality(
    task: TauBenchmarkTask, task_graph,
) -> dict:
    """Score decomposition: subtask count, interface param coverage, assignment validity."""
    subtasks = task_graph.subtasks if hasattr(task_graph, 'subtasks') else []
    n = len(subtasks)

    # Subtask count in expected range
    if n < task.expected_min_subtasks:
        count_score = n / max(task.expected_min_subtasks, 1)
    elif n > task.expected_max_subtasks:
        count_score = max(0.0, task.expected_max_subtasks / max(n, 1))
    else:
        count_score = 1.0

    # Interface param coverage
    all_params: set[str] = set()
    for st in subtasks:
        for p in (st.interface_params if hasattr(st, 'interface_params') else []):
            all_params.add(p)
    expected_set = set(task.expected_interface_params)
    if expected_set:
        coverage = len(all_params & expected_set) / len(expected_set)
    else:
        coverage = 1.0  # No expected params → skip

    # Assignment validity
    valid = set(task.valid_agent_names)
    if valid:
        total_assignments = 0
        valid_assignments = 0
        for st in subtasks:
            agents = st.assigned_agents if hasattr(st, 'assigned_agents') else []
            total_assignments += len(agents)
            valid_assignments += sum(1 for a in agents if a in valid)
        assignment_validity = valid_assignments / max(total_assignments, 1)
    else:
        assignment_validity = 1.0

    composite = (count_score * 0.3 + coverage * 0.4 + assignment_validity * 0.3)
    return {
        "subtask_count": n,
        "count_score": count_score,
        "interface_params_found": sorted(all_params),
        "param_coverage": coverage,
        "assignment_validity": assignment_validity,
        "composite": composite,
    }


def score_resolution_effectiveness(
    state,  # TauState
) -> dict:
    """Score conflict resolution: detection rate, escalation count, resolution success."""
    conflicts = state.conflict_history if hasattr(state, 'conflict_history') else []
    resolutions = state.resolution_history if hasattr(state, 'resolution_history') else []

    total_conflicts = sum(len(c.conflicts) for c in conflicts)
    total_resolved = sum(len(r.resolved) for r in resolutions)
    director_decisions = sum(1 for r in resolutions if r.director_decision)

    if total_conflicts > 0:
        resolution_rate = total_resolved / total_conflicts
    else:
        resolution_rate = 1.0  # No conflicts = perfect

    escalation_penalty = min(director_decisions * 0.15, 0.45)
    composite = max(0.0, min(1.0, resolution_rate - escalation_penalty))

    return {
        "total_conflicts": total_conflicts,
        "total_resolved": total_resolved,
        "director_decisions": director_decisions,
        "resolution_rate": resolution_rate,
        "composite": composite,
    }


def score_convergence_efficiency(
    state,  # TauState
) -> dict:
    """Score how efficiently Tau converges."""
    iteration = state.iteration if hasattr(state, 'iteration') else 0
    completed = state.completed if hasattr(state, 'completed') else False
    max_iter = state.max_iterations if hasattr(state, 'max_iterations') else 5

    if completed:
        if iteration <= 1:
            score = 1.0
        elif iteration <= 2:
            score = 0.9
        elif iteration <= 3:
            score = 0.7
        else:
            score = 0.5
    else:
        score = 0.0

    return {
        "iterations": iteration,
        "max_iterations": max_iter,
        "completed": completed,
        "composite": score,
    }


def compute_tau_composite(
    decomp_score: float,
    resolution_score: float,
    convergence_score: float,
) -> float:
    """Weighted composite for Tau benchmarks."""
    return decomp_score * 0.40 + resolution_score * 0.30 + convergence_score * 0.30


def run_tau_benchmark_replay(
    task: TauBenchmarkTask,
    orchestrator,  # TauOrchestrator
    llm_call,
) -> TauBenchmarkMetrics:
    """Run a single Tau benchmark task in replay/offline mode."""
    start = time.monotonic()

    try:
        state = orchestrator.run(task.instruction)
    except Exception:
        # Minimal fallback metrics on failure
        return TauBenchmarkMetrics(
            task_id=task.id,
            subtask_count=0, interface_param_count=0,
            interface_param_coverage=0.0, agent_assignment_validity=0.0,
            conflicts_detected=0, conflict_detection_precision=0.0,
            rounds_to_complete=0, max_rounds=5,
            resolution_escalations=0, director_decisions=0,
            resolved=False, estimated_tokens=0, estimated_cost_rmb=0.0,
            composite_score=0.0,
        )

    elapsed = time.monotonic() - start

    task_graph = state.task_graph
    decomp = score_decomposition_quality(task, task_graph)
    resolution = score_resolution_effectiveness(state)
    convergence = score_convergence_efficiency(state)

    composite = compute_tau_composite(
        decomp["composite"], resolution["composite"], convergence["composite"],
    )

    cost = state.cost_summary if state.cost_summary else ""

    return TauBenchmarkMetrics(
        task_id=task.id,
        subtask_count=decomp["subtask_count"],
        interface_param_count=len(decomp["interface_params_found"]),
        interface_param_coverage=decomp["param_coverage"],
        agent_assignment_validity=decomp["assignment_validity"],
        conflicts_detected=resolution["total_conflicts"],
        conflict_detection_precision=1.0,  # Replay: assume perfect
        rounds_to_complete=convergence["iterations"],
        max_rounds=convergence["max_iterations"],
        resolution_escalations=resolution["director_decisions"],
        director_decisions=resolution["director_decisions"],
        resolved=convergence["completed"],
        estimated_tokens=_estimate_tokens_from_cost(cost),
        estimated_cost_rmb=_parse_cost(cost),
        composite_score=composite,
    )


def run_tau_suite_replay(
    tasks: list[TauBenchmarkTask] | None = None,
    orchestrator_constructor=None,
    agents: dict | None = None,
    llm_call=None,
) -> list[TauBenchmarkMetrics]:
    """Run all Tau benchmark tasks in replay mode."""
    tasks = tasks or TAU_BENCHMARK_TASKS
    results = []
    for task in tasks:
        if orchestrator_constructor and agents and llm_call:
            orch = orchestrator_constructor(agents=agents)
            metrics = run_tau_benchmark_replay(task, orch, llm_call)
        else:
            metrics = TauBenchmarkMetrics(
                task_id=task.id,
                subtask_count=0, interface_param_count=0,
                interface_param_coverage=0.0, agent_assignment_validity=0.0,
                conflicts_detected=0, conflict_detection_precision=0.0,
                rounds_to_complete=0, max_rounds=5,
                resolution_escalations=0, director_decisions=0,
                resolved=False, estimated_tokens=0, estimated_cost_rmb=0.0,
                composite_score=0.0,
            )
        results.append(metrics)
    return results


@dataclass
class TauSuiteResult:
    """Aggregated Tau benchmark suite results."""
    metrics: list[TauBenchmarkMetrics]
    suite_name: str = "Tau Benchmark Suite"

    @property
    def avg_composite(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.composite_score for m in self.metrics) / len(self.metrics)

    @property
    def resolution_rate(self) -> float:
        resolved = sum(1 for m in self.metrics if m.resolved)
        return resolved / max(len(self.metrics), 1)

    @property
    def avg_decomp_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.interface_param_coverage * 0.5 + m.agent_assignment_validity * 0.5
                   for m in self.metrics) / len(self.metrics)

    @property
    def avg_rounds(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.rounds_to_complete for m in self.metrics) / len(self.metrics)


def _estimate_tokens_from_cost(cost_str: str) -> int:
    """Crude reverse-engineering from cost summary."""
    if not cost_str:
        return 0
    return 500  # Placeholder


def _parse_cost(cost_str: str) -> float:
    """Parse cost from summary string."""
    if not cost_str:
        return 0.0
    import re
    m = re.search(r'¥(\d+\.?\d*)', cost_str)
    return float(m.group(1)) if m else 0.0
