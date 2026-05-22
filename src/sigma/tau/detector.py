"""InterfaceConflictDetector — compare shared interface parameters across subtasks.

After all subtasks complete, check the "contract" parameters:
if two subtasks produce different values for the same parameter, it's a conflict.
"""

from sigma.tau.types import (
    SubtaskResult, InterfaceConflict, ConflictReport, TaskGraph,
)


class InterfaceConflictDetector:
    """Detects conflicts at the interfaces between subtasks."""

    # Relative difference threshold for flagging a conflict
    DEFAULT_THRESHOLD = 0.10  # 10%

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def detect(
        self,
        task_graph: TaskGraph,
        results: dict[str, SubtaskResult],
    ) -> ConflictReport:
        """Compare interface parameters across subtasks, generate conflict report."""
        conflicts: list[InterfaceConflict] = []
        resolved: list[str] = []

        for param_key, subtask_ids in task_graph.interface_map.items():
            if len(subtask_ids) < 2:
                resolved.append(param_key)
                continue

            # Collect values from each subtask
            values: dict[str, float] = {}
            for sid in subtask_ids:
                r = results.get(sid)
                if r and param_key in r.interface_params:
                    values[sid] = r.interface_params[param_key]

            if len(values) < 2:
                resolved.append(param_key)
                continue

            # Compare all pairs
            ids = list(values.keys())
            max_severity = 0.0
            worst_pair = (ids[0], ids[1])

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    diff = abs(values[a] - values[b]) / max(abs(values[a]), abs(values[b]), 0.001)
                    severity = min(10.0, diff * 20)  # 10% diff → severity 2.0, 50% diff → 10.0
                    if severity > max_severity:
                        max_severity = severity
                        worst_pair = (a, b)

            if max_severity > self.threshold * 10:
                conflicts.append(InterfaceConflict(
                    param_key=param_key,
                    subtask_a=worst_pair[0],
                    subtask_b=worst_pair[1],
                    value_a=values[worst_pair[0]],
                    value_b=values[worst_pair[1]],
                    severity=max_severity,
                    description=(
                        f"{param_key}: {worst_pair[0]}={values[worst_pair[0]]:.2f} vs "
                        f"{worst_pair[1]}={values[worst_pair[1]]:.2f} "
                        f"(差异 {max_severity / 20:.0%})"
                    ),
                ))
            else:
                resolved.append(param_key)

        return ConflictReport(conflicts=conflicts, resolved_params=resolved)
