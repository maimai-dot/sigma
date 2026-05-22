"""Cross-session memory storage — SQLite-based persistence.

Stores session summaries, parameters, decisions, and lessons
so future sessions can learn from past runs.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MemoryStore:
    """SQLite-backed cross-session memory.

    Usage:
        store = MemoryStore("sigma_memory.db")
        store.save_session(state, result)
        similar = store.find_similar_tasks("计算质量")
        history = store.get_parameter_history("mass_kg")
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._init_tables()
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_instruction TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                verdict TEXT,
                project_name TEXT,
                total_rounds INTEGER,
                total_tokens INTEGER,
                estimated_cost REAL,
                result_json TEXT
            );
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                param_key TEXT NOT NULL,
                param_value TEXT,
                source_type TEXT,
                source_name TEXT,
                round_num INTEGER,
                confidence TEXT,
                evidence TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                round_num INTEGER,
                domain TEXT,
                decision TEXT,
                reason TEXT,
                made_by TEXT
            );
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                lesson_text TEXT NOT NULL,
                agent_name TEXT,
                round_num INTEGER,
                tags TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_params_key ON parameters(param_key);
            CREATE INDEX IF NOT EXISTS idx_decisions_domain ON decisions(domain);
        """)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────

    def save_session(self, state, result: dict) -> int:
        """Archive a completed session. Returns the session ID."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO sessions (task_instruction, timestamp, verdict, project_name,
               total_rounds, total_tokens, estimated_cost, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.task_instruction, now,
                result.get("final_verdict", ""),
                result.get("project_name", ""),
                state.round_num,
                state.cost_summary.get("total_tokens", 0),
                state.cost_summary.get("estimated_cost", 0.0),
                json.dumps(result, indent=2, ensure_ascii=False),
            ),
        )
        session_id = cur.lastrowid

        # Save parameters with provenance
        for entry in state.audit_trail.entries:
            self._conn.execute(
                """INSERT INTO parameters (session_id, param_key, param_value,
                   source_type, source_name, round_num, confidence, evidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, entry.parameter, str(entry.value),
                    entry.source_type, entry.source_name, entry.round_num,
                    entry.confidence, entry.evidence[:500],
                ),
            )

        # Save decisions
        for d in state.decisions:
            self._conn.execute(
                """INSERT INTO decisions (session_id, round_num, domain, decision, reason, made_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, d.round_num, d.domain, d.decision, d.reason, d.made_by),
            )

        self._conn.commit()
        return session_id

    def add_lesson(self, session_id: int, lesson_text: str,
                   agent_name: str = "", round_num: int = 0, tags: str = "") -> int:
        """Store a learned lesson."""
        cur = self._conn.execute(
            "INSERT INTO lessons (session_id, lesson_text, agent_name, round_num, tags) VALUES (?, ?, ?, ?, ?)",
            (session_id, lesson_text, agent_name, round_num, tags),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Query ──────────────────────────────────────────────────

    def find_similar_tasks(self, instruction: str, limit: int = 5) -> list[dict]:
        """Find past sessions with keyword overlap in the instruction."""
        keywords = [w for w in instruction.split() if len(w) >= 2]
        if not keywords:
            return []
        clauses = " OR ".join(["task_instruction LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        rows = self._conn.execute(
            f"""SELECT id, task_instruction, timestamp, verdict, total_rounds,
                total_tokens, estimated_cost FROM sessions
                WHERE {clauses} ORDER BY timestamp DESC LIMIT ?""",
            params + [limit],
        ).fetchall()
        return [
            {"id": r[0], "instruction": r[1], "timestamp": r[2], "verdict": r[3],
             "total_rounds": r[4], "total_tokens": r[5], "estimated_cost": r[6]}
            for r in rows
        ]

    def get_parameter_history(self, param_key: str, limit: int = 20) -> list[dict]:
        """All recorded values for a parameter across sessions."""
        rows = self._conn.execute(
            """SELECT p.session_id, p.param_value, p.source_type, p.source_name,
               p.round_num, p.confidence, s.task_instruction, s.timestamp
               FROM parameters p JOIN sessions s ON p.session_id = s.id
               WHERE p.param_key = ? ORDER BY s.timestamp DESC LIMIT ?""",
            (param_key, limit),
        ).fetchall()
        return [
            {"session_id": r[0], "value": r[1], "source_type": r[2], "source_name": r[3],
             "round_num": r[4], "confidence": r[5], "task": r[6], "timestamp": r[7]}
            for r in rows
        ]

    def get_decisions_by_domain(self, domain: str, limit: int = 20) -> list[dict]:
        """Past decisions in a given domain."""
        rows = self._conn.execute(
            """SELECT d.decision, d.reason, d.made_by, d.round_num,
               s.task_instruction, s.timestamp
               FROM decisions d JOIN sessions s ON d.session_id = s.id
               WHERE d.domain LIKE ? ORDER BY s.timestamp DESC LIMIT ?""",
            (f"%{domain}%", limit),
        ).fetchall()
        return [
            {"decision": r[0], "reason": r[1], "made_by": r[2], "round_num": r[3],
             "task": r[4], "timestamp": r[5]}
            for r in rows
        ]

    def recent_lessons(self, limit: int = 10) -> list[dict]:
        """Most recent lessons."""
        rows = self._conn.execute(
            "SELECT lesson_text, agent_name, round_num, tags FROM lessons ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"lesson": r[0], "agent": r[1], "round": r[2], "tags": r[3]} for r in rows]

    def session_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()
