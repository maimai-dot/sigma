"""
Filesystem Tool — safe file read/write/list within a sandboxed root.

All paths are resolved relative to a configurable root directory.
Path traversal attempts are detected and blocked.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from sigma.agent import BaseTool

DEFAULT_ROOT = os.getcwd()
MAX_FILE_READ_BYTES = 1024 * 1024      # 1 MB
MAX_DIR_LIST = 500
FORBIDDEN_EXTENSIONS = {".exe", ".dll", ".so", ".dylib", ".bin"}
FORBIDDEN_PATTERNS = {"..", "~", "$"}


def _sanitize(root: Path, user_path: str) -> Path:
    """Resolve user_path relative to root, blocking traversal escapes.

    Returns the resolved absolute Path.
    Raises ValueError if the path escapes root, contains forbidden patterns,
    or resolves to a forbidden extension.
    """
    if not user_path or not isinstance(user_path, str):
        raise ValueError("Path is required and must be a string")

    clean = user_path.strip()
    if not clean:
        raise ValueError("Path cannot be empty")

    # Block patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in clean:
            raise ValueError(f"Path contains forbidden pattern: {pattern}")

    # Resolve relative to root
    candidate = (root / clean).resolve()

    # Must be within root
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"Path traversal blocked: '{clean}' escapes root")

    # Forbidden extension check for file operations
    suffix = candidate.suffix.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"Forbidden file extension: {suffix}")

    return candidate


@dataclass
class FileSystemTool(BaseTool):
    """Read, write, list, and check files within a sandboxed directory.

    All paths are confined to a configurable root. Path traversal attacks
    are detected and blocked.
    """

    name: str = "filesystem"
    description: str = (
        "Read, write, list, and check files within a sandboxed directory. "
        "Operations: read (returns file content as string), write (create/overwrite), "
        "list (directory listing), exists (check if path exists)."
    )
    root: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._root = Path(self.root or DEFAULT_ROOT).resolve()
        if not self._root.exists():
            self._root.mkdir(parents=True, exist_ok=True)

    def _run(
        self, operation: str = "", path: str = "", content: str = "", **kwargs
    ) -> dict:
        """Execute a filesystem operation.

        Args:
            operation: One of 'read', 'write', 'list', 'exists'.
            path: File or directory path relative to sandbox root.
            content: Text content to write (only for 'write' operation).

        Returns:
            dict with: success, operation, path, and operation-specific fields.
        """
        if not operation:
            return {"success": False, "error": "operation is required", "operation": ""}

        op = operation.strip().lower()

        if op == "exists":
            return self._op_exists(path)
        if op == "list":
            return self._op_list(path)
        if op == "read":
            return self._op_read(path)
        if op == "write":
            return self._op_write(path, content)

        return {"success": False, "error": f"Unknown operation: '{operation}'", "operation": op}

    def _op_exists(self, path: str) -> dict:
        try:
            target = _sanitize(self._root, path)
            return {"success": True, "operation": "exists", "path": str(target), "exists": target.exists()}
        except ValueError as e:
            return {"success": False, "error": str(e), "operation": "exists"}

    def _op_list(self, path: str) -> dict:
        if not path:
            path = "."
        try:
            target = _sanitize(self._root, path)
            if not target.exists():
                return {"success": False, "error": f"Directory not found: {path}", "operation": "list"}
            if not target.is_dir():
                return {"success": False, "error": f"Not a directory: {path}", "operation": "list"}

            entries = []
            for i, entry in enumerate(target.iterdir()):
                if i >= MAX_DIR_LIST:
                    entries.append("... [truncated]")
                    break
                suffix = "/" if entry.is_dir() else ""
                entries.append(entry.name + suffix)

            return {
                "success": True,
                "operation": "list",
                "path": str(target),
                "entries": sorted(entries),
                "count": len(entries),
            }
        except ValueError as e:
            return {"success": False, "error": str(e), "operation": "list"}

    def _op_read(self, path: str) -> dict:
        try:
            target = _sanitize(self._root, path)
            if not target.exists():
                return {"success": False, "error": f"File not found: {path}", "operation": "read"}
            if not target.is_file():
                return {"success": False, "error": f"Not a file: {path}", "operation": "read"}

            size = target.stat().st_size
            if size > MAX_FILE_READ_BYTES:
                return {"success": False, "error": f"File too large ({size} > {MAX_FILE_READ_BYTES} bytes)", "operation": "read"}

            text = target.read_text(encoding="utf-8", errors="replace")
            return {"success": True, "operation": "read", "path": str(target), "content": text, "size": len(text)}
        except ValueError as e:
            return {"success": False, "error": str(e), "operation": "read"}

    def _op_write(self, path: str, content: str) -> dict:
        try:
            target = _sanitize(self._root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"success": True, "operation": "write", "path": str(target), "size": len(content)}
        except ValueError as e:
            return {"success": False, "error": str(e), "operation": "write"}
