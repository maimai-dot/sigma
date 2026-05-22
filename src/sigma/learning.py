"""Learning Store — capture execution history and feed lessons back into prompts.

Design:
  ExecutionRecord captures what happened in each run.
  LearningStore persists records and retrieves similar past tasks.
  FeedbackInjector enriches decomposer/agent prompts with relevant lessons.

This is NOT reinforcement learning — it's pattern-based experience retrieval.
The LLM reads past lessons and decides how to apply them.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any
from pathlib import Path


@dataclass
class ExecutionRecord:
    """A single execution run captured for learning."""

    # Identity
    instruction_hash: str = ""
    instruction: str = ""

    # Mode
    mode: str = ""                     # "sigma" | "tau"

    # Decomposition (Tau)
    subtask_count: int = 0
    assign_pattern: list[str] = field(default_factory=list)  # agent→subtask mapping

    # Execution
    params_produced: dict[str, float] = field(default_factory=dict)
    param_confidence: dict[str, str] = field(default_factory=dict)
    iterations: int = 0
    completed: bool = False

    # Resolution
    total_conflicts: int = 0
    total_resolved: int = 0
    resolution_levels: dict[str, int] = field(default_factory=dict)  # DIRECT/SIGMA/DIRECTOR → count
    director_decisions: int = 0

    # Quality
    guardrail_blocks: int = 0
    guardrail_warns: int = 0
    success: bool = True
    verdict: str = ""

    # Meta
    timestamp: float = 0.0
    duration_ms: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def hash_instruction(instruction: str) -> str:
        return hashlib.sha256(instruction.encode()).hexdigest()[:16]


class LearningStore:
    """Persistent storage for execution records with similarity retrieval.

    Uses simple keyword-based retrieval (TF-IDF style via KnowledgeBase
    if available) or falls back to substring matching.
    """

    def __init__(self, path: str | Path | None = None, max_records: int = 1000):
        self._records: list[ExecutionRecord] = []
        self._max = max_records
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._load()

    def record(self, rec: ExecutionRecord) -> None:
        """Store an execution record."""
        if rec.instruction_hash:
            # Replace existing record for same instruction
            self._records = [r for r in self._records if r.instruction_hash != rec.instruction_hash]
        else:
            rec.instruction_hash = ExecutionRecord.hash_instruction(rec.instruction)
        self._records.append(rec)
        if len(self._records) > self._max:
            self._records = self._records[-self._max:]
        if self._path:
            self._save()

    def find_similar(self, instruction: str, top_k: int = 3) -> list[ExecutionRecord]:
        """Find execution records with similar instructions.

        Uses multi-keyword overlap scoring (fast, no dependencies).
        """
        query_words = set(_tokenize(instruction))
        if not query_words:
            return []

        scored = []
        for rec in self._records:
            rec_words = set(_tokenize(rec.instruction))
            overlap = len(query_words & rec_words)
            if overlap > 0:
                # Bonus for same tags or mode
                bonus = 0
                scored.append((overlap + bonus, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def lessons_for(self, instruction: str, top_k: int = 3) -> str:
        """Generate a natural-language 'lessons learned' summary from similar runs."""
        similar = self.find_similar(instruction, top_k)
        if not similar:
            return ""

        lines = ["\n## 历史经验（来自类似任务的教训）\n"]
        for i, rec in enumerate(similar):
            lines.append(f"### 案例 {i+1}: {rec.instruction[:80]}...")
            lines.append(f"- 模式: {rec.mode}, 迭代: {rec.iterations}轮, 成功: {rec.completed}")
            if rec.subtask_count:
                lines.append(f"- 拆解: {rec.subtask_count}个子任务, 分配: {rec.assign_pattern}")
            if rec.total_conflicts:
                lines.append(f"- 冲突: {rec.total_conflicts}个, 解决: {rec.total_resolved}个, "
                           f"总监决策: {rec.director_decisions}次")
            if rec.params_produced:
                params_str = ", ".join(f"{k}={v}" for k, v in rec.params_produced.items())
                lines.append(f"- 关键参数: {params_str}")
            if rec.verdict:
                lines.append(f"- 结论: {rec.verdict[:200]}")
            if rec.guardrail_blocks or rec.guardrail_warns:
                lines.append(f"- 护栏: {rec.guardrail_blocks}阻止, {rec.guardrail_warns}警告")
        lines.append("请参考以上经验，避免重复已知错误。\n")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics about stored executions."""
        if not self._records:
            return {"total": 0}
        total = len(self._records)
        completed = sum(1 for r in self._records if r.completed)
        avg_iters = sum(r.iterations for r in self._records) / total if total else 0
        modes = {}
        for r in self._records:
            modes[r.mode] = modes.get(r.mode, 0) + 1
        return {
            "total": total,
            "completed": completed,
            "completion_rate": completed / total if total else 0,
            "avg_iterations": round(avg_iters, 1),
            "modes": modes,
        }

    def clear(self) -> None:
        self._records.clear()
        if self._path and self._path.exists():
            self._path.unlink()

    def __len__(self) -> int:
        return len(self._records)

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records]
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = [ExecutionRecord.from_dict(d) for d in data]
        except (json.JSONDecodeError, OSError):
            pass


def _tokenize(text: str) -> list[str]:
    """Multi-language tokenizer: word tokens for ASCII, bigrams for CJK."""
    import re
    text = text.lower().strip()
    result = []

    # Extract ASCII word tokens (2+ letters)
    words = re.findall(r'[a-z0-9]{2,}', text)
    result.extend(words)

    # Extract CJK characters for bigram generation
    cjk = re.findall(r'[一-鿿㐀-䶿]', text)
    for j in range(len(cjk) - 1):
        result.append(cjk[j] + cjk[j + 1])
    if len(cjk) == 1:
        result.append(cjk[0])

    return [t for t in result if t not in _STOP_WORDS]


_STOP_WORDS = {
    "the", "a", "an", "is", "of", "to", "in", "and", "for", "on", "with",
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "这", "他", "也", "与", "及", "或", "被",
}


# ═══════════════════════════════════════════════════════════════════════
# Integration helpers
# ═══════════════════════════════════════════════════════════════════════

def record_from_tau_state(state, instruction: str, duration_ms: float = 0) -> ExecutionRecord:
    """Build an ExecutionRecord from a completed TauState."""
    from sigma.tau.types import TauState
    rec = ExecutionRecord(
        instruction_hash=ExecutionRecord.hash_instruction(instruction),
        instruction=instruction,
        mode="tau",
        iterations=state.iteration,
        completed=state.completed,
        success=state.completed,
        verdict=state.final_verdict,
        timestamp=time.time(),
        duration_ms=duration_ms,
    )

    if state.task_graph:
        rec.subtask_count = len(state.task_graph.subtasks)
        rec.assign_pattern = [
            f"{st.id}→{','.join(st.assigned_agents)}"
            for st in state.task_graph.subtasks
        ]

    for r in state.subtask_results.values():
        rec.params_produced.update(r.interface_params)
        rec.param_confidence.update(r.param_confidence)

    if state.conflict_history:
        rec.total_conflicts = sum(len(cr.conflicts) for cr in state.conflict_history)
    if state.resolution_history:
        for res in state.resolution_history:
            rec.total_resolved += len(res.resolved)
            rec.director_decisions += 1 if res.director_decision else 0
        rec.total_conflicts = max(rec.total_conflicts, rec.total_resolved)

    return rec
