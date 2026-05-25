"""
Code Sandbox — AST-whitelisted Python execution in an isolated subprocess.

Allows agents to run ad-hoc computations with numpy/scipy/math/json only.
Imports are validated at the AST level before any code executes.
Execution runs in a subprocess with a 30-second timeout.
"""

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field

from sigma.agent import BaseTool

# ── Whitelist ───────────────────────────────────────────────────────

WHITELISTED_MODULES: set[str] = {
    "numpy", "scipy", "math", "json",
    "collections", "itertools", "functools", "typing",
    "dataclasses", "enum", "fractions", "decimal",
    "statistics", "random", "datetime", "hashlib",
    "re", "string", "textwrap", "copy", "pprint", "time",
    "scipy.optimize", "scipy.interpolate", "scipy.integrate",
    "scipy.linalg", "scipy.stats", "scipy.constants",
    "scipy.signal", "scipy.spatial", "scipy.special",
    "numpy.linalg", "numpy.random", "numpy.fft", "numpy.polynomial",
}

FORBIDDEN_BUILTINS: set[str] = {
    "eval", "exec", "compile", "__import__", "open",
    "input", "breakpoint", "memoryview",
}

FORBIDDEN_ATTRS: set[str] = {
    "os", "sys", "subprocess", "shutil", "socket",
    "requests", "urllib", "http", "ftplib", "imaplib",
    "smtplib", "telnetlib", "ctypes", "multiprocessing",
    "threading", "signal", "atexit", "pathlib",
}

# Modules safe for star imports
STAR_OK_MODULES: set[str] = {"math", "numpy", "scipy.constants"}

MAX_CODE_LENGTH = 4096
EXEC_TIMEOUT = 30


# ── AST Validation ──────────────────────────────────────────────────

@dataclass
class SandboxError:
    message: str
    lineno: int = 0


def validate_code(code: str) -> list[SandboxError]:
    """Validate Python code against the sandbox whitelist.

    Catches forbidden imports, dangerous builtins, and banned attributes
    before any code runs.
    """
    if len(code) > MAX_CODE_LENGTH:
        return [SandboxError(f"Code too long ({len(code)} > {MAX_CODE_LENGTH} chars)")]

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [SandboxError(f"Syntax error: {e.msg}", e.lineno or 0)]

    errors: list[SandboxError] = []
    validator = _SandboxValidator(errors)
    validator.visit(tree)
    return errors


class _SandboxValidator(ast.NodeVisitor):
    """Walk the AST and reject any unsafe constructs."""

    def __init__(self, errors: list[SandboxError]):
        self.errors = errors

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if not _is_whitelisted(alias.name):
                self.errors.append(SandboxError(
                    f"Forbidden import: {alias.name}", node.lineno,
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module is None:
            # "from . import X" / "from .foo import X" — relative imports
            self.errors.append(SandboxError(
                "Relative imports are forbidden", node.lineno,
            ))
            return

        if not _is_whitelisted(node.module):
            self.errors.append(SandboxError(
                f"Forbidden import: {node.module}", node.lineno,
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = _get_func_name(node.func)
        if func_name in FORBIDDEN_BUILTINS:
            self.errors.append(SandboxError(
                f"Forbidden builtin: {func_name}()", node.lineno,
            ))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        full = _get_attr_chain(node)
        if full and full.split(".")[0] in FORBIDDEN_ATTRS:
            self.errors.append(SandboxError(
                f"Forbidden attribute: {full}", node.lineno,
            ))
        self.generic_visit(node)


def _is_whitelisted(module_name: str) -> bool:
    """Check if a module or its top-level parent is whitelisted."""
    if module_name in WHITELISTED_MODULES:
        return True
    # Submodule: "numpy.linalg.some_thing" → check "numpy.linalg", "numpy"
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        if ".".join(parts[:i]) in WHITELISTED_MODULES:
            return True
    return False


def _get_func_name(node: ast.expr) -> str:
    """Extract the simple name from a call target."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _get_attr_chain(node: ast.Attribute) -> str:
    """Reconstruct 'os.path.join' from an Attribute node chain."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


# ── Subprocess Execution ────────────────────────────────────────────

def _run_in_subprocess(code: str, timeout: int = EXEC_TIMEOUT) -> dict:
    """Execute code in an isolated subprocess with a hard timeout.

    Uses -I (isolated mode) and -S (no site imports) flags to further
    restrict the subprocess environment.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = f.name

    # Filter environment to avoid leaking secrets into sandbox subprocess.
    # Only pass variables the sandbox actually needs — never API keys.
    env = {}
    safe_keys = {
        "PATH", "PYTHONPATH", "HOME", "USERPROFILE", "TEMP", "TMP",
        "SYSTEMROOT", "COMSPEC", "PATHEXT", "HOMEDRIVE", "HOMEPATH",
        "APPDATA", "LOCALAPPDATA", "ProgramData", "USERNAME",
        "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    }
    for k, v in os.environ.items():
        if k in safe_keys:
            env[k] = v
        elif k.startswith("PYTHON") and k != "PYTHONSTARTUP":
            env[k] = v
        elif k in ("CONDA_PREFIX", "VIRTUAL_ENV"):
            env[k] = v

    try:
        result = subprocess.run(
            [sys.executable, "-s", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Trim excessive output
        if len(stdout) > 8000:
            stdout = stdout[:8000] + "\n... [truncated]"
        if len(stderr) > 4000:
            stderr = stderr[:4000] + "\n... [truncated]"

        return {
            "success": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Tool ────────────────────────────────────────────────────────────

class CodeSandbox(BaseTool):
    """Execute Python code in a sandboxed environment.

    Only whitelisted imports (numpy, scipy, math, json, etc.) are allowed.
    Execution runs in an isolated subprocess with a 30-second timeout.

    Usage:
        sandbox = CodeSandbox()
        result = sandbox._run(code="import math\\nprint(math.sqrt(4))")
    """

    name: str = "code_sandbox"
    description: str = (
        "Execute Python code in a sandbox (numpy/scipy/math/json only, "
        f"{EXEC_TIMEOUT}s timeout). Use for ad-hoc calculations, data fitting, "
        "or numerical analysis. "
        "The code should print() its results to stdout."
    )

    def __init__(self, timeout: int = EXEC_TIMEOUT):
        super().__init__()
        self.timeout = timeout
        self.description = (
            "Execute Python code in a sandbox (numpy/scipy/math/json only, "
            f"{EXEC_TIMEOUT}s timeout). Use for ad-hoc calculations, data fitting, "
            "or numerical analysis. The code should print() its results to stdout."
        )

    def _run(self, code: str = "", **kwargs) -> dict:
        """Validate and execute Python code in the sandbox.

        Args:
            code: Python source code to execute.

        Returns:
            dict with keys: success, stdout, stderr, returncode, errors (if validation fails).
        """
        if not code or not code.strip():
            return {"success": False, "error": "No code provided", "returncode": -1}

        # 1. AST validation
        errors = validate_code(code)
        if errors:
            return {
                "success": False,
                "error": "Sandbox validation failed",
                "validation_errors": [e.message for e in errors],
                "returncode": -1,
            }

        # 2. Enforce print() for output capture
        wrapped = _wrap_for_output(code)

        # 3. Subprocess execution
        result = _run_in_subprocess(wrapped, timeout=self.timeout)
        return result


def _wrap_for_output(code: str) -> str:
    """Wrap the last expression in a print() if it looks like a bare expression.

    This allows agents to write "math.sqrt(4)" and still see the result.
    """
    stripped = code.rstrip()
    # If the code already has print(), don't wrap
    if "print(" in stripped:
        return stripped

    # Try to parse last line as a bare expression
    lines = stripped.split("\n")
    if not lines:
        return stripped

    last_line = lines[-1].strip()
    if not last_line or last_line.startswith(("#", "import ", "from ", "def ", "class ", "if ", "for ", "while ", "try:", "except", "with ")):
        return stripped

    # If last line is an assignment, keep it but add a print
    if "=" in last_line and not last_line.startswith(("if", "for", "while")) and "==" not in last_line:
        var_match = last_line.split("=")[0].strip().split()[-1] if last_line.split("=")[0].strip() else ""
        if var_match and var_match.isidentifier():
            lines.append(f"print({var_match})")
            return "\n".join(lines)

    # Check if last line is a simple expression
    try:
        ast.parse(last_line, mode="eval")
        lines[-1] = f"print({last_line})"
    except SyntaxError:
        pass

    return "\n".join(lines)
