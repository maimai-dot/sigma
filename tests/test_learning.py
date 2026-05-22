"""Tests for sigma.learning — execution history and lesson retrieval."""

import json
import tempfile
from pathlib import Path

import pytest
from sigma.learning import (
    ExecutionRecord, LearningStore, record_from_tau_state,
    _tokenize,
)


# ═══════════════════════════════════════════════════════════════════════
# ExecutionRecord
# ═══════════════════════════════════════════════════════════════════════

class TestExecutionRecord:
    def test_hash_instruction(self):
        h = ExecutionRecord.hash_instruction("设计一枚KNSB固体火箭")
        assert len(h) == 16
        assert isinstance(h, str)

    def test_hash_deterministic(self):
        a = ExecutionRecord.hash_instruction("hello")
        b = ExecutionRecord.hash_instruction("hello")
        assert a == b

    def test_hash_different(self):
        a = ExecutionRecord.hash_instruction("hello")
        b = ExecutionRecord.hash_instruction("world")
        assert a != b

    def test_to_dict_and_back(self):
        rec = ExecutionRecord(
            instruction="设计火箭",
            instruction_hash="abc123",
            mode="tau",
            subtask_count=3,
            iterations=2,
            completed=True,
            success=True,
            verdict="任务完成",
            duration_ms=1500.0,
        )
        d = rec.to_dict()
        rec2 = ExecutionRecord.from_dict(d)
        assert rec2.instruction == rec.instruction
        assert rec2.mode == rec.mode
        assert rec2.subtask_count == rec.subtask_count

    def test_default_values(self):
        rec = ExecutionRecord()
        assert rec.instruction_hash == ""
        assert rec.assign_pattern == []
        assert rec.params_produced == {}


# ═══════════════════════════════════════════════════════════════════════
# LearningStore
# ═══════════════════════════════════════════════════════════════════════

class TestLearningStore:
    def test_empty_store(self):
        ls = LearningStore()
        assert len(ls) == 0
        assert ls.find_similar("anything") == []
        assert ls.lessons_for("anything") == ""
        assert ls.stats() == {"total": 0}

    def test_record_and_find(self):
        ls = LearningStore()
        rec = ExecutionRecord(
            instruction="设计KNSB固体火箭发动机",
            instruction_hash=ExecutionRecord.hash_instruction("设计KNSB固体火箭发动机"),
            mode="tau",
            iterations=3,
            completed=True,
            success=True,
            verdict="很好的结果",
        )
        ls.record(rec)
        assert len(ls) == 1

        similar = ls.find_similar("设计火箭发动机")
        assert len(similar) == 1
        assert similar[0].instruction == "设计KNSB固体火箭发动机"

    def test_find_no_match(self):
        ls = LearningStore()
        rec = ExecutionRecord(
            instruction="火箭设计",
            instruction_hash=ExecutionRecord.hash_instruction("火箭设计"),
        )
        ls.record(rec)
        # Completely unrelated query
        similar = ls.find_similar("zzz xxx yyy")
        assert similar == []

    def test_replace_existing(self):
        ls = LearningStore()
        rec1 = ExecutionRecord(
            instruction="设计火箭",
            instruction_hash="same_hash",
            iterations=1,
        )
        rec2 = ExecutionRecord(
            instruction="设计火箭",
            instruction_hash="same_hash",
            iterations=5,
        )
        ls.record(rec1)
        ls.record(rec2)
        assert len(ls) == 1
        assert ls._records[0].iterations == 5

    def test_lessons_format(self):
        ls = LearningStore()
        rec = ExecutionRecord(
            instruction="设计KNSB固体火箭发动机 优化喷嘴设计",
            instruction_hash="hash1",
            mode="tau",
            subtask_count=4,
            assign_pattern=["st_0→A", "st_1→B"],
            iterations=2,
            completed=True,
            total_conflicts=3,
            total_resolved=3,
            director_decisions=1,
            params_produced={"thrust_n": 1500, "isp_s": 155},
            verdict="推力1500N，比冲155s",
        )
        ls.record(rec)
        lessons = ls.lessons_for("固体火箭发动机 喷嘴")
        assert "历史经验" in lessons
        assert "KNSB" in lessons
        assert "thrust_n=1500" in lessons
        assert "isp_s=155" in lessons

    def test_max_records(self):
        ls = LearningStore(max_records=3)
        for i in range(5):
            ls.record(ExecutionRecord(
                instruction=f"任务{i}",
                instruction_hash=f"hash{i}",
            ))
        assert len(ls) <= 3

    def test_clear(self):
        ls = LearningStore()
        ls.record(ExecutionRecord(
            instruction="test",
            instruction_hash="hash",
        ))
        assert len(ls) == 1
        ls.clear()
        assert len(ls) == 0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learning.json"

            # Write
            ls1 = LearningStore(path=path)
            rec = ExecutionRecord(
                instruction="火箭设计任务",
                instruction_hash="abc",
                mode="tau",
                iterations=3,
                completed=True,
            )
            ls1.record(rec)

            # Read back
            ls2 = LearningStore(path=path)
            assert len(ls2) == 1
            assert ls2._records[0].instruction == "火箭设计任务"
            assert ls2._records[0].mode == "tau"

    def test_persistence_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            ls = LearningStore(path=path)
            assert len(ls) == 0

    def test_stats(self):
        ls = LearningStore()
        ls.record(ExecutionRecord(
            instruction="task1", instruction_hash="h1",
            mode="tau", iterations=2, completed=True,
        ))
        ls.record(ExecutionRecord(
            instruction="task2", instruction_hash="h2",
            mode="sigma", iterations=4, completed=False,
        ))
        stats = ls.stats()
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["completion_rate"] == 0.5
        assert stats["avg_iterations"] == 3.0
        assert stats["modes"] == {"tau": 1, "sigma": 1}


# ═══════════════════════════════════════════════════════════════════════
# _tokenize
# ═══════════════════════════════════════════════════════════════════════

class TestTokenize:
    def test_english_words(self):
        tokens = _tokenize("design a solid rocket motor")
        assert "design" in tokens
        assert "solid" in tokens
        assert "rocket" in tokens
        assert "motor" in tokens
        # Stop words
        assert "a" not in tokens

    def test_chinese_chars(self):
        tokens = _tokenize("设计一枚火箭发动机")
        # CJK becomes bigrams
        assert "设计" in tokens
        assert "火箭" in tokens
        assert "发动" in tokens

    def test_mixed_text(self):
        tokens = _tokenize("KNSB推进剂 isp比冲")
        assert "knsb" in tokens
        # CJK becomes bigrams
        assert "推进" in tokens
        assert "进剂" in tokens
        assert "isp" in tokens
        assert "比冲" in tokens

    def test_empty_text(self):
        assert _tokenize("") == []

    def test_stop_words_filtered(self):
        tokens = _tokenize("the rocket is 的 in 了 flight")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "的" not in tokens
        assert "了" not in tokens
        assert "rocket" in tokens
        assert "flight" in tokens


# ═══════════════════════════════════════════════════════════════════════
# record_from_tau_state
# ═══════════════════════════════════════════════════════════════════════

class TestRecordFromTauState:
    def test_minimal_state(self):
        from sigma.tau.types import TauState
        state = TauState(
            instruction="design a rocket",
            iteration=2,
            completed=True,
        )
        rec = record_from_tau_state(state, "design a rocket")
        assert rec.instruction == "design a rocket"
        assert rec.mode == "tau"
        assert rec.iterations == 2
        assert rec.completed

    def test_state_with_task_graph(self):
        from sigma.tau.types import TauState, TaskGraph, SubTask
        graph = TaskGraph(
            instruction="test",
            subtasks=[
                SubTask(id="st_0", description="分析推力",
                        assigned_agents=["A"], interface_params=["thrust_n"]),
                SubTask(id="st_1", description="分析质量",
                        assigned_agents=["B"], interface_params=["mass_kg"]),
            ],
        )
        state = TauState(
            instruction="test",
            task_graph=graph,
            iteration=1,
            completed=True,
        )
        rec = record_from_tau_state(state, "test")
        assert rec.subtask_count == 2
        assert len(rec.assign_pattern) == 2

    def test_state_with_results(self):
        from sigma.tau.types import TauState, SubtaskResult
        state = TauState(
            instruction="test",
            iteration=1,
            completed=True,
            subtask_results={
                "st_0": SubtaskResult(
                    subtask_id="st_0",
                    success=True,
                    interface_params={"thrust_n": 1500},
                    param_confidence={"thrust_n": "HIGH"},
                ),
            },
        )
        rec = record_from_tau_state(state, "test")
        assert rec.params_produced == {"thrust_n": 1500}
        assert rec.param_confidence == {"thrust_n": "HIGH"}

    def test_state_with_duration(self):
        from sigma.tau.types import TauState
        state = TauState(
            instruction="test",
            iteration=1,
            completed=True,
            duration_ms=2500.0,
        )
        rec = record_from_tau_state(state, "test", duration_ms=2500.0)
        assert rec.duration_ms == 2500.0
