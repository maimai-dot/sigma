"""Core data types for the Tau hierarchical framework.

Models a real engineering organization:
- Tau decomposes tasks → SubTask[]
- Departments execute independently → SubtaskResult[]
- Interface params are the "contracts" between departments
- Conflicts are detected at interfaces → resolved via graduated escalation
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SubTask:
    """A single decomposed subtask assigned to specific agents."""
    id: str
    description: str
    assigned_agents: list[str]       # 1-2 agents responsible
    interface_params: list[str] = field(default_factory=list)
    """Parameter keys this subtask produces that other subtasks depend on."""
    dependencies: list[str] = field(default_factory=list)
    """IDs of subtasks that must complete before this can start."""
    expected_outputs: list[str] = field(default_factory=list)
    """What this subtask should produce (natural language)."""
    output_model: type | None = None
    """Optional Pydantic BaseModel subclass for structured output parsing."""
    condition: callable | None = None
    """Optional callable(dict[str, SubtaskResult]) -> bool.
    If set and returns False, the subtask is skipped.
    Only evaluated when all dependencies are met."""
    human_input: str = ""
    """If non-empty, the executor pauses and calls the human_input_callback
    with this prompt before executing the subtask."""

    @classmethod
    def from_dict(cls, d: dict) -> "SubTask":
        return cls(**d)


@dataclass
class TaskGraph:
    """Decomposed task with subtasks and interface map."""
    instruction: str
    subtasks: list[SubTask]
    interface_map: dict[str, list[str]] = field(default_factory=dict)
    """param_key → [subtask_ids that produce/consume this param]"""

    @property
    def root_tasks(self) -> list[SubTask]:
        """Subtasks with no dependencies — can start immediately."""
        return [s for s in self.subtasks if not s.dependencies]

    def dependents_of(self, subtask_id: str) -> list[SubTask]:
        """Subtasks that depend on the given subtask."""
        return [s for s in self.subtasks if subtask_id in s.dependencies]

    @classmethod
    def from_dict(cls, d: dict) -> "TaskGraph":
        return cls(
            instruction=d["instruction"],
            subtasks=[SubTask.from_dict(s) for s in d["subtasks"]],
            interface_map=d.get("interface_map", {}),
        )


@dataclass
class SubtaskResult:
    """Output of a single subtask execution."""
    subtask_id: str
    success: bool
    agent_analyses: dict[str, str] = field(default_factory=dict)
    tool_results: dict[str, dict] = field(default_factory=dict)
    interface_params: dict[str, float] = field(default_factory=dict)
    """Computed values for interface parameters with confidence."""
    param_confidence: dict[str, str] = field(default_factory=dict)
    """Confidence per interface param: HIGH/MEDIUM/LOW."""
    error: str = ""
    guardrails_report: dict | None = None
    """Guardrail check results serialized as dict (GuardrailReport.to_dict())."""

    @classmethod
    def from_dict(cls, d: dict) -> "SubtaskResult":
        return cls(**d)


@dataclass
class InterfaceConflict:
    """A conflict between two subtasks on a shared interface parameter."""
    param_key: str
    subtask_a: str
    subtask_b: str
    value_a: float
    value_b: float
    severity: float              # 0-10, based on relative difference
    description: str = ""

    @property
    def relative_diff(self) -> float:
        denominator = max(abs(self.value_a), abs(self.value_b), 0.001)
        return abs(self.value_a - self.value_b) / denominator

    @classmethod
    def from_dict(cls, d: dict) -> "InterfaceConflict":
        return cls(**d)


@dataclass
class ConflictReport:
    """Result of interface conflict detection."""
    conflicts: list[InterfaceConflict] = field(default_factory=list)
    resolved_params: list[str] = field(default_factory=list)
    """Interface params that are consistent across subtasks."""

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def max_severity(self) -> float:
        return max((c.severity for c in self.conflicts), default=0.0)

    def affecting_subtask(self, subtask_id: str) -> list[InterfaceConflict]:
        return [c for c in self.conflicts
                if c.subtask_a == subtask_id or c.subtask_b == subtask_id]

    @classmethod
    def from_dict(cls, d: dict) -> "ConflictReport":
        return cls(
            conflicts=[InterfaceConflict.from_dict(c) for c in d.get("conflicts", [])],
            resolved_params=d.get("resolved_params", []),
        )


@dataclass
class ResolutionResult:
    """Result of conflict resolution attempt."""
    resolved: list[str]           # param keys that were resolved
    unresolved: list[str]         # param keys still in conflict
    consensus_values: dict[str, float] = field(default_factory=dict)
    director_decision: str = ""   # Director's directional decision if unresolved
    round_count: int = 0
    involved_agents: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ResolutionResult":
        return cls(**d)


@dataclass
class TauState:
    """Running state for a Tau-led execution."""
    instruction: str
    task_graph: TaskGraph | None = None
    subtask_results: dict[str, SubtaskResult] = field(default_factory=dict)
    conflict_history: list[ConflictReport] = field(default_factory=list)
    resolution_history: list[ResolutionResult] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5
    completed: bool = False
    final_verdict: str = ""
    cost_summary: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for checkpoint."""
        return {
            "instruction": self.instruction,
            "task_graph": asdict(self.task_graph) if self.task_graph else None,
            "subtask_results": {
                k: asdict(v) for k, v in self.subtask_results.items()
            },
            "conflict_history": [asdict(c) for c in self.conflict_history],
            "resolution_history": [asdict(r) for r in self.resolution_history],
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "completed": self.completed,
            "final_verdict": self.final_verdict,
            "cost_summary": self.cost_summary,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TauState":
        """Restore from JSON-compatible dict."""
        task_graph = None
        if d.get("task_graph"):
            task_graph = TaskGraph.from_dict(d["task_graph"])

        subtask_results = {}
        for k, v in d.get("subtask_results", {}).items():
            subtask_results[k] = SubtaskResult(**v)

        conflict_history = [
            ConflictReport.from_dict(c) for c in d.get("conflict_history", [])
        ]
        resolution_history = [
            ResolutionResult.from_dict(r) for r in d.get("resolution_history", [])
        ]

        return cls(
            instruction=d["instruction"],
            task_graph=task_graph,
            subtask_results=subtask_results,
            conflict_history=conflict_history,
            resolution_history=resolution_history,
            iteration=d.get("iteration", 0),
            max_iterations=d.get("max_iterations", 5),
            completed=d.get("completed", False),
            final_verdict=d.get("final_verdict", ""),
            cost_summary=d.get("cost_summary", ""),
            duration_ms=d.get("duration_ms", 0.0),
        )
