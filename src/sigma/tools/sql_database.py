"""
SQL Database Tool — execute SQL queries via sqlite3 (stdlib, zero-dependency).

Supports parameterized queries. Works with any SQLite database or :memory:.
For :memory: databases, keeps a persistent connection across _run() calls.
"""

import sqlite3
from dataclasses import dataclass, field

from sigma.agent import BaseTool

DEFAULT_DB = ":memory:"
MAX_ROWS = 1000
MAX_SQL_LENGTH = 8192
DANGEROUS_KEYWORDS = {"drop table", "drop index", "drop view", "alter table",
                       "attach", "detach", "vacuum", "reindex", "pragma"}


def _validate_sql(sql: str) -> str | None:
    """Basic safety check. Returns error message or None."""
    if not sql or not isinstance(sql, str):
        return "SQL query is required"
    if len(sql) > MAX_SQL_LENGTH:
        return f"SQL too long ({len(sql)} > {MAX_SQL_LENGTH} chars)"
    lower = sql.strip().lower()
    for kw in DANGEROUS_KEYWORDS:
        if lower.startswith(kw) or f" {kw} " in f" {lower} ":
            return f"Potentially destructive SQL blocked: '{kw}' in query"
    return None


@dataclass
class SQLDatabaseTool(BaseTool):
    """Execute SQL queries against a SQLite database.

    Uses parameterized queries (? placeholders with params list).
    Dangerous DDL statements (DROP TABLE, ALTER, etc.) are blocked.
    For :memory: databases, the connection persists across calls.
    """

    name: str = "sql_database"
    description: str = (
        "Execute SQL queries (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE) against "
        "a SQLite database. Uses parameterized ? placeholders. "
        "Dangerous operations (DROP, ALTER) are blocked. "
        "SELECT queries return columns + rows (max 1000). "
        "INSERT/UPDATE/DELETE return affected_rows count."
    )
    database: str = DEFAULT_DB
    _conn: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create connection. For :memory:, reuse across calls."""
        if self.database == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self.database)
        conn.row_factory = sqlite3.Row
        return conn

    def _run(self, sql: str = "", params: list | None = None, **kwargs) -> dict:
        """Execute a SQL query.

        Args:
            sql: SQL statement to execute. Use ? for parameters.
            params: Optional list of parameter values for ? placeholders.

        Returns:
            dict with: success, operation (query/execute), columns (for SELECT),
            rows (for SELECT), affected_rows (for INSERT/UPDATE/DELETE), error (on failure).
        """
        err = _validate_sql(sql)
        if err:
            return {"success": False, "error": err}

        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            lower_sql = sql.strip().lower()
            if lower_sql.startswith(("select", "pragma", "explain", "with")):
                rows = cursor.fetchmany(MAX_ROWS + 1)
                truncated = len(rows) > MAX_ROWS
                if truncated:
                    rows = rows[:MAX_ROWS]

                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [dict(zip(columns, row)) for row in rows]

                conn.commit()
                if self.database != ":memory:":
                    conn.close()
                return {
                    "success": True,
                    "operation": "query",
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data),
                    "truncated": truncated,
                }
            else:
                conn.commit()
                rowcount = cursor.rowcount
                if self.database != ":memory:":
                    conn.close()
                return {
                    "success": True,
                    "operation": "execute",
                    "affected_rows": rowcount if rowcount >= 0 else 0,
                }
        except sqlite3.Error as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close(self):
        """Close the persistent connection if any."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
