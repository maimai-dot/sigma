"""
JSON / CSV / TXT search tools — read, query, and search structured data files.

All zero-dependency (stdlib only). Agents can grep text, filter JSON arrays,
and query CSV rows by column value.
"""

import csv
import fnmatch
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from sigma.agent import BaseTool

MAX_FILE_BYTES = 16 * 1024 * 1024   # 16 MB
MAX_RESULTS = 500
FORBIDDEN_PATH_PATTERNS = {"..", "~", "$"}


def _safe_path(filepath: str, root: Path | None = None) -> Path:
    """Resolve and validate a file path, blocking traversal escapes.

    Returns the resolved absolute Path.
    Raises ValueError if the path contains forbidden patterns or escapes root.
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("filepath is required and must be a string")
    clean = filepath.strip()
    if not clean:
        raise ValueError("filepath cannot be empty")
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern in clean:
            raise ValueError(f"filepath contains forbidden pattern: {pattern}")

    p = Path(clean)
    if p.is_absolute():
        # Allow absolute paths — resolve() normalizes away traversal attempts
        return p.resolve()

    resolved = (root or Path.cwd()).resolve() / clean
    resolved = resolved.resolve()
    root_path = (root or Path.cwd()).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError:
        raise ValueError(f"Path traversal blocked: '{clean}' escapes root")
    return resolved


def _read_text_safe(path: Path) -> str | None:
    """Read a text file with size guard. Returns None on error."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ── JSON Tool ─────────────────────────────────────────────────────────

@dataclass
class JsonTool(BaseTool):
    """Read, query, and extract fields from JSON files.

    Supports reading entire JSON files, filtering arrays by key=value,
    and extracting specific fields via dotted paths like 'users[0].name'.
    """

    name: str = "json_tool"
    description: str = (
        "Read and query JSON files. Operations: read (load entire file), "
        "filter (filter array of objects by key=value), get (extract field "
        "by dotted path like 'data.items[0].name')."
    )

    def _run(
        self,
        operation: str = "",
        filepath: str = "",
        key: str = "",
        value: str = "",
        path: str = "",
        **kwargs,
    ) -> dict:
        """Query a JSON file.

        Args:
            operation: 'read', 'filter', or 'get'.
            filepath: Path to the JSON file.
            key: Object key to filter by (for 'filter' operation).
            value: Value to match (for 'filter' operation).
            path: Dotted path like 'users.0.name' (for 'get' operation).

        Returns:
            dict with: success, operation, data (parsed JSON), error (on failure).
        """
        if not filepath:
            return {"success": False, "error": "filepath is required"}
        if not operation:
            return {"success": False, "error": "operation is required"}

        try:
            p = _safe_path(filepath)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        text = _read_text_safe(p)
        if text is None:
            return {"success": False, "error": f"Cannot read: {filepath}"}

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}

        op = operation.strip().lower()

        if op == "read":
            return {"success": True, "operation": "read", "filepath": filepath, "data": data}

        if op == "filter":
            if not isinstance(data, list):
                return {"success": False, "error": "filter requires a JSON array at top level"}
            if not key:
                return {"success": False, "error": "key is required for filter"}
            results = [item for item in data if isinstance(item, dict) and str(item.get(key)) == value]
            if len(results) > MAX_RESULTS:
                results = results[:MAX_RESULTS]
            return {"success": True, "operation": "filter", "key": key, "value": value,
                    "match_count": len(results), "data": results}

        if op == "get":
            result = _resolve_json_path(data, path)
            return {"success": True, "operation": "get", "path": path, "data": result}

        return {"success": False, "error": f"Unknown operation: '{operation}'"}


def _resolve_json_path(data, dotted_path: str):
    """Walk a dotted path like 'users[0].name' through a JSON structure."""
    if not dotted_path:
        return data
    parts = dotted_path.replace("[", ".[").strip(".").split(".")
    current = data
    for part in parts:
        if not part:
            continue
        if part.endswith("]") and "[" in part:
            name, idx_str = part.split("[", 1)
            idx = int(idx_str.rstrip("]"))
            if name:
                current = current.get(name) if isinstance(current, dict) else current
            if isinstance(current, list) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


# ── CSV Tool ──────────────────────────────────────────────────────────

@dataclass
class CsvTool(BaseTool):
    """Read and query CSV files.

    Operations: read (all rows as dicts), filter (rows matching column value),
    columns (list column names).
    """

    name: str = "csv_tool"
    description: str = (
        "Read and query CSV files. Operations: read (all rows as list of dicts), "
        "filter (rows where column matches value), columns (list header names)."
    )

    def _run(
        self,
        operation: str = "",
        filepath: str = "",
        column: str = "",
        value: str = "",
        **kwargs,
    ) -> dict:
        """Query a CSV file.

        Args:
            operation: 'read', 'filter', or 'columns'.
            filepath: Path to the CSV file.
            column: Column name to filter by (for 'filter' operation).
            value: Value to match (for 'filter' operation).

        Returns:
            dict with: success, operation, headers, data (list of dicts), match_count.
        """
        if not filepath:
            return {"success": False, "error": "filepath is required"}
        if not operation:
            return {"success": False, "error": "operation is required"}

        try:
            p = _safe_path(filepath)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        text = _read_text_safe(p)
        if text is None:
            return {"success": False, "error": f"Cannot read: {filepath}"}

        try:
            reader = csv.DictReader(StringIO(text))
            headers = reader.fieldnames or []
            rows = list(reader)
        except csv.Error as e:
            return {"success": False, "error": f"CSV parse error: {e}"}

        op = operation.strip().lower()

        if op == "columns":
            return {"success": True, "operation": "columns", "filepath": filepath,
                    "columns": headers, "row_count": len(rows)}

        if op == "filter":
            if not column:
                return {"success": False, "error": "column is required for filter"}
            if column not in headers:
                return {"success": False, "error": f"Column '{column}' not found. Available: {headers}"}
            results = [r for r in rows if str(r.get(column, "")).strip() == value.strip()]
            if len(results) > MAX_RESULTS:
                results = results[:MAX_RESULTS]
            return {"success": True, "operation": "filter", "column": column, "value": value,
                    "match_count": len(results), "data": results}

        if op == "read":
            if len(rows) > MAX_RESULTS:
                rows = rows[:MAX_RESULTS]
            return {"success": True, "operation": "read", "filepath": filepath,
                    "columns": headers, "row_count": len(rows), "data": rows}

        return {"success": False, "error": f"Unknown operation: '{operation}'"}


# ── Text Grep Tool ────────────────────────────────────────────────────

@dataclass
class TxtGrepTool(BaseTool):
    """Search text files for lines matching a pattern.

    Supports plain-text substring and regex matching. Respects line limits.
    """

    name: str = "txt_grep"
    description: str = (
        "Search text files for lines matching a pattern. "
        "Use 'grep' operation for substring search, 'regex' for regex patterns. "
        "Returns matching lines with line numbers and context."
    )

    def _run(
        self,
        operation: str = "grep",
        filepath: str = "",
        pattern: str = "",
        context_lines: int = 0,
        max_matches: int = 200,
        **kwargs,
    ) -> dict:
        """Search a text file.

        Args:
            operation: 'grep' (substring) or 'regex'.
            filepath: Path to the text file.
            pattern: Search pattern (substring or regex).
            context_lines: Number of context lines before/after each match.
            max_matches: Maximum matches to return (default 200).

        Returns:
            dict with: success, operation, pattern, matches (list of {line_num, line,
            context_before, context_after}), match_count.
        """
        if not filepath:
            return {"success": False, "error": "filepath is required"}
        if not pattern:
            return {"success": False, "error": "pattern is required"}

        try:
            p = _safe_path(filepath)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        text = _read_text_safe(p)
        if text is None:
            return {"success": False, "error": f"Cannot read: {filepath}"}

        lines = text.splitlines()
        op = operation.strip().lower()

        if op == "grep":
            matcher = lambda ln: pattern in ln
        elif op == "regex":
            import re
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex: {e}"}
            matcher = lambda ln: regex.search(ln) is not None
        else:
            return {"success": False, "error": f"Unknown operation: '{operation}'"}

        matches = []
        for i, line in enumerate(lines):
            if matcher(line):
                ctx_before = lines[max(0, i - context_lines):i] if context_lines else []
                ctx_after = lines[i + 1:i + 1 + context_lines] if context_lines else []
                matches.append({
                    "line_num": i + 1,
                    "line": line,
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                })
                if len(matches) >= max_matches:
                    break

        return {
            "success": True,
            "operation": op,
            "pattern": pattern,
            "match_count": len(matches),
            "total_lines": len(lines),
            "matches": matches,
        }
