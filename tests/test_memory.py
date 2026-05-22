"""Tests for MemoryStore — cross-session SQLite memory."""

import pytest
from sigma.memory import MemoryStore
from sigma.state import SharedState
from sigma.provenance import AuditTrail


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def populated_state():
    state = SharedState(task_instruction="计算150mm铝管质量")
    state.round_num = 2
    state.cost_summary = {"total_tokens": 3000, "estimated_cost": 0.05, "calls": 2}
    state.audit_trail.add("mass_kg", 5.2, "tool", "mass_calc", 1, "ok", "HIGH")
    state.audit_trail.add("mass_kg", 5.15, "consensus", "consensus", 2, "agreed", "MEDIUM")
    from sigma.state import Decision
    state.decisions = [
        Decision(round_num=1, domain="结构", decision="使用6061-T6铝管", reason="强度足够", made_by="Director"),
    ]
    return state


class TestMemoryStoreInit:
    def test_creates_database(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db)
        assert db.exists()
        store.close()

    def test_session_count_zero_initially(self, store):
        assert store.session_count() == 0


class TestSaveAndQuery:
    def test_save_session_returns_id(self, store, populated_state):
        result = {"final_verdict": "converged", "project_name": "Test"}
        sid = store.save_session(populated_state, result)
        assert sid is not None
        assert sid >= 1
        assert store.session_count() == 1

    def test_find_similar_tasks(self, store, populated_state):
        store.save_session(populated_state, {"final_verdict": "converged"})
        results = store.find_similar_tasks("150mm铝管")
        assert len(results) >= 1

    def test_find_similar_no_match(self, store, populated_state):
        store.save_session(populated_state, {"final_verdict": "converged"})
        results = store.find_similar_tasks("xyz123")
        assert len(results) == 0

    def test_get_parameter_history(self, store, populated_state):
        store.save_session(populated_state, {"final_verdict": "converged"})
        history = store.get_parameter_history("mass_kg")
        assert len(history) == 2

    def test_get_parameter_history_nonexistent(self, store):
        history = store.get_parameter_history("nonexistent")
        assert len(history) == 0

    def test_get_decisions_by_domain(self, store, populated_state):
        store.save_session(populated_state, {"final_verdict": "converged"})
        decisions = store.get_decisions_by_domain("结构")
        assert len(decisions) == 1
        assert "6061-T6" in decisions[0]["decision"]

    def test_decisions_empty_domain(self, store, populated_state):
        store.save_session(populated_state, {"final_verdict": "converged"})
        decisions = store.get_decisions_by_domain("推进")
        assert len(decisions) == 0


class TestLessons:
    def test_add_lesson(self, store, populated_state):
        result = {"final_verdict": "converged"}
        sid = store.save_session(populated_state, result)
        lid = store.add_lesson(sid, "KNSB比冲约140-155s", agent_name="Propulsion", round_num=1)
        assert lid >= 1

    def test_recent_lessons(self, store, populated_state):
        result = {"final_verdict": "converged"}
        sid = store.save_session(populated_state, result)
        store.add_lesson(sid, "lesson 1", agent_name="A", round_num=1)
        store.add_lesson(sid, "lesson 2", agent_name="B", round_num=2)
        lessons = store.recent_lessons()
        assert len(lessons) == 2
        assert lessons[0]["lesson"] == "lesson 2"  # most recent first

    def test_recent_lessons_empty(self, store):
        assert store.recent_lessons() == []


class TestMultipleSessions:
    def test_isolated_sessions(self, store):
        s1 = SharedState(task_instruction="任务A")
        s1.audit_trail.add("x", 1.0, "tool", "t1", 1)
        s2 = SharedState(task_instruction="任务B")
        s2.audit_trail.add("y", 2.0, "tool", "t2", 1)

        id1 = store.save_session(s1, {"final_verdict": "ok"})
        id2 = store.save_session(s2, {"final_verdict": "ok"})
        assert id1 != id2
        assert store.session_count() == 2

        # Parameter from session 1 should not appear in session 2 queries
        history_a = store.get_parameter_history("x")
        history_b = store.get_parameter_history("y")
        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["value"] == "1.0"
