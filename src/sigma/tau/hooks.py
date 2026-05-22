"""Tau Hook System — inject custom logic at Tau framework phase boundaries.

Mirrors sigma.hooks but reflects Tau's hierarchical flow:
  Decompose → Execute (per subtask) → Detect → Resolve → Iterate

Applications register callbacks that fire at each phase.
Callbacks receive a context dict and can return modifications.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TauHookPoint(Enum):
    """Named hook points in the Tau hierarchical execution cycle."""

    # Top-level lifecycle
    ON_START = "on_start"
    ON_COMPLETE = "on_complete"
    ON_ERROR = "on_error"

    # Iteration boundary
    ON_ITERATION_START = "on_iteration_start"
    ON_ITERATION_END = "on_iteration_end"

    # Decompose phase
    ON_DECOMPOSE_START = "on_decompose_start"
    ON_DECOMPOSE_END = "on_decompose_end"

    # Execute phase (per subtask granularity)
    ON_SUBTASK_START = "on_subtask_start"
    ON_SUBTASK_END = "on_subtask_end"
    ON_ALL_SUBTASKS_COMPLETE = "on_all_subtasks_complete"

    # Detect phase
    ON_DETECT_START = "on_detect_start"
    ON_DETECT_END = "on_detect_end"

    # Resolve phase
    ON_RESOLVE_START = "on_resolve_start"
    ON_RESOLVE_END = "on_resolve_end"

    # Escalation events (within resolve)
    ON_ESCALATE = "on_escalate"          # fired when DIRECT→SIGMA or SIGMA→DIRECTOR


@dataclass(order=True)
class _HookEntry:
    priority: int
    callback: Callable = field(compare=False)


class TauHookSystem:
    """Event-driven hook registry for the Tau hierarchical cycle.

    Usage:
        hooks = TauHookSystem()
        hooks.register(TauHookPoint.ON_DECOMPOSE_END, my_callback, priority=10)
        hooks.fire(TauHookPoint.ON_DECOMPOSE_END, instruction="...", task_graph=...)
    """

    def __init__(self):
        self._hooks: dict[TauHookPoint, list[_HookEntry]] = defaultdict(list)

    def register(self, point: TauHookPoint, callback: Callable, priority: int = 0):
        """Register a callback for a hook point.

        Args:
            point: Which Tau event to hook into.
            callback: Called as callback(**context). May return a dict to update context.
            priority: Lower runs first. Default 0.
        """
        from bisect import insort
        insort(self._hooks[point], _HookEntry(priority=priority, callback=callback))

    def fire(self, point: TauHookPoint, **context) -> dict[str, Any]:
        """Fire all callbacks registered for a hook point.

        Each callback receives the context as kwargs. If a callback returns a
        dict, those keys are merged into the context for subsequent callbacks.
        The final context dict is returned.
        """
        results = dict(context)
        for entry in self._hooks.get(point, []):
            try:
                ret = entry.callback(**results)
                if isinstance(ret, dict):
                    results.update(ret)
            except Exception as e:
                if point == TauHookPoint.ON_ERROR:
                    raise
                self.fire(TauHookPoint.ON_ERROR, error=e, **results)
                raise
        return results

    def clear(self):
        """Remove all registered hooks."""
        self._hooks.clear()
