"""Hook system — inject custom logic at AERC phase boundaries.

Applications register callbacks that fire before/after each AERC phase.
Callbacks receive a context dict and can return modifications or raise to halt.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class HookPoint(Enum):
    """Named hook points in the AERC cycle and task lifecycle."""
    # AERC phase hooks
    ON_START = "on_start"
    ON_ROUND_START = "on_round_start"
    BEFORE_PLAN = "before_plan"
    AFTER_PLAN = "after_plan"
    BEFORE_DO = "before_do"
    AFTER_DO = "after_do"
    BEFORE_CHECK = "before_check"
    AFTER_CHECK = "after_check"
    BEFORE_ACT = "before_act"
    AFTER_ACT = "after_act"
    ON_ROUND_END = "on_round_end"
    ON_ERROR = "on_error"
    ON_COMPLETE = "on_complete"

    # Task-level hooks (per subtask/task execution)
    BEFORE_TASK = "before_task"
    """Fired before a single task/subtask executes. Context: task_id, task_desc, agent_names."""
    AFTER_TASK = "after_task"
    """Fired after a task/subtask completes successfully. Context: task_id, result, duration_ms."""
    ON_TASK_ERROR = "on_task_error"
    """Fired when a task/subtask fails. Context: task_id, error, attempt."""


@dataclass(order=True)
class _HookEntry:
    priority: int
    callback: Callable = field(compare=False)


class HookSystem:
    """Event-driven hook registry for the AERC cycle.

    Usage:
        hooks = HookSystem()
        hooks.register(HookPoint.AFTER_PLAN, my_callback, priority=10)
        hooks.trigger(HookPoint.AFTER_PLAN, state=state, analyses=analyses)
    """

    def __init__(self):
        self._hooks: dict[HookPoint, list[_HookEntry]] = defaultdict(list)

    def register(self, point: HookPoint, callback: Callable, priority: int = 0):
        """Register a callback for a hook point.

        Args:
            point: Which AERC event to hook into.
            callback: Called as callback(**context). May return a dict to update context.
            priority: Lower runs first. Default 0.
        """
        from bisect import insort
        insort(self._hooks[point], _HookEntry(priority=priority, callback=callback))

    def trigger(self, point: HookPoint, **context) -> dict[str, Any]:
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
                if point == HookPoint.ON_ERROR:
                    raise  # Don't recurse into ON_ERROR from ON_ERROR
                self.trigger(HookPoint.ON_ERROR, error=e, **results)
                raise
        return results

    def clear(self):
        """Remove all registered hooks."""
        self._hooks.clear()
