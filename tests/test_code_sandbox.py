"""Tests for CodeSandbox — AST validation, subprocess execution, integration."""

import pytest
from sigma.code_sandbox import (
    validate_code, _run_in_subprocess, _wrap_for_output,
    CodeSandbox, SandboxError,
    WHITELISTED_MODULES, FORBIDDEN_BUILTINS, FORBIDDEN_ATTRS,
    MAX_CODE_LENGTH, EXEC_TIMEOUT,
)


# ── Whitelists ───────────────────────────────────────────────────────

class TestWhitelists:
    def test_numpy_in_whitelist(self):
        assert "numpy" in WHITELISTED_MODULES

    def test_scipy_in_whitelist(self):
        assert "scipy" in WHITELISTED_MODULES

    def test_math_in_whitelist(self):
        assert "math" in WHITELISTED_MODULES

    def test_eval_is_forbidden(self):
        assert "eval" in FORBIDDEN_BUILTINS

    def test_exec_is_forbidden(self):
        assert "exec" in FORBIDDEN_BUILTINS

    def test_open_is_forbidden(self):
        assert "open" in FORBIDDEN_BUILTINS

    def test_os_is_forbidden_attr(self):
        assert "os" in FORBIDDEN_ATTRS

    def test_subprocess_is_forbidden_attr(self):
        assert "subprocess" in FORBIDDEN_ATTRS


# ── AST Validation: Forbidden imports ────────────────────────────────

class TestASTImportValidation:
    def test_import_os_blocked(self):
        errors = validate_code("import os")
        assert len(errors) == 1
        assert "os" in errors[0].message

    def test_import_sys_blocked(self):
        errors = validate_code("import sys")
        assert len(errors) == 1

    def test_import_shutil_blocked(self):
        errors = validate_code("import shutil")
        assert len(errors) == 1

    def test_import_subprocess_blocked(self):
        errors = validate_code("import subprocess")
        assert len(errors) == 1

    def test_import_socket_blocked(self):
        errors = validate_code("import socket")
        assert len(errors) == 1

    def test_import_ctypes_blocked(self):
        errors = validate_code("import ctypes")
        assert len(errors) == 1

    def test_from_os_import_blocked(self):
        errors = validate_code("from os import path")
        assert len(errors) == 1

    def test_from_sys_import_blocked(self):
        errors = validate_code("from sys import argv")
        assert len(errors) == 1

    def test_relative_import_blocked(self):
        errors = validate_code("from . import foo")
        assert len(errors) >= 1


# ── AST Validation: Forbidden builtins ──────────────────────────────

class TestASTBuiltinValidation:
    def test_eval_blocked(self):
        errors = validate_code("eval('1+1')")
        assert len(errors) == 1
        assert "eval" in errors[0].message

    def test_exec_blocked(self):
        errors = validate_code("exec('x=1')")
        assert len(errors) == 1

    def test_compile_blocked(self):
        errors = validate_code("compile('1+1', '', 'eval')")
        assert len(errors) == 1

    def test_dunder_import_blocked(self):
        errors = validate_code('__import__("os")')
        assert len(errors) == 1

    def test_open_blocked(self):
        errors = validate_code("open('/etc/passwd')")
        assert len(errors) == 1


# ── AST Validation: Safe code ───────────────────────────────────────

class TestASTSafeCode:
    def test_import_math(self):
        assert validate_code("import math") == []

    def test_import_numpy(self):
        assert validate_code("import numpy") == []

    def test_import_numpy_as_np(self):
        assert validate_code("import numpy as np") == []

    def test_import_scipy(self):
        assert validate_code("import scipy") == []

    def test_import_scipy_submodules(self):
        assert validate_code("import scipy.optimize") == []
        assert validate_code("import scipy.interpolate") == []
        assert validate_code("import scipy.stats") == []

    def test_import_json(self):
        assert validate_code("import json") == []

    def test_from_math_import(self):
        assert validate_code("from math import sqrt, pi") == []

    def test_from_numpy_import(self):
        assert validate_code("from numpy import array, linspace") == []

    def test_import_collections(self):
        assert validate_code("import collections") == []

    def test_import_itertools(self):
        assert validate_code("import itertools") == []

    def test_import_re(self):
        assert validate_code("import re") == []

    def test_basic_arithmetic(self):
        assert validate_code("x = 1 + 2\nprint(x)") == []

    def test_function_definition(self):
        assert validate_code("def f(x):\n    return x*x\nprint(f(4))") == []

    def test_list_comprehension(self):
        assert validate_code("print([i*i for i in range(10)])") == []

    def test_numpy_computation(self):
        code = "import numpy as np\nx = np.array([1,2,3])\nprint(x.sum())"
        assert validate_code(code) == []


# ── AST Validation: Edge cases ──────────────────────────────────────

class TestASTEdgeCases:
    def test_syntax_error(self):
        errors = validate_code("for i in range(10)  # missing colon")
        assert len(errors) >= 1

    def test_code_too_long(self):
        long_code = "x = 1\n" * (MAX_CODE_LENGTH // 6 + 100)
        errors = validate_code(long_code)
        assert len(errors) >= 1
        assert "too long" in errors[0].message.lower()

    def test_empty_code(self):
        assert validate_code("") == []

    def test_comments_only(self):
        assert validate_code("# just a comment") == []

    def test_import_in_string_not_flagged(self):
        """String literals containing 'import os' are not AST nodes."""
        code = 'print("you can import os if you want")'
        assert validate_code(code) == []

    def test_multiple_errors(self):
        code = "import os\nimport sys"
        errors = validate_code(code)
        assert len(errors) == 2


# ── Subprocess Execution ────────────────────────────────────────────

class TestSubprocessExecution:
    def test_basic_execution(self):
        result = _run_in_subprocess("print(42)")
        assert result["success"] is True
        assert "42" in result["stdout"]

    def test_arithmetic(self):
        result = _run_in_subprocess("print(2 + 2 * 2)")
        assert "6" in result["stdout"]

    def test_math_module(self):
        result = _run_in_subprocess("import math\nprint(math.pi)")
        assert result["success"] is True

    def test_numpy_available(self):
        result = _run_in_subprocess("import numpy as np\nprint(np.__version__)")
        assert result["success"] is True

    def test_stderr_captured(self):
        result = _run_in_subprocess("import sys\nprint('out', file=sys.stderr)")
        assert "out" in result["stderr"]

    def test_runtime_error(self):
        result = _run_in_subprocess("1/0")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["stderr"]

    def test_timeout(self):
        result = _run_in_subprocess("import time\ntime.sleep(10)", timeout=1)
        assert result["success"] is False
        assert "timed out" in result["stderr"]
        assert result["returncode"] == -1

    def test_return_code_zero_on_success(self):
        result = _run_in_subprocess("print('ok')")
        assert result["returncode"] == 0

    def test_return_code_nonzero_on_error(self):
        result = _run_in_subprocess("raise ValueError('test')")
        assert result["returncode"] != 0


# ── CodeSandbox Tool ────────────────────────────────────────────────

class TestCodeSandboxTool:
    def test_numpy_calculation(self):
        sb = CodeSandbox()
        result = sb._run(code="import numpy as np\nprint(np.array([1,2,3]).sum())")
        assert result["success"] is True
        assert "6" in result["stdout"]

    def test_scipy_constants(self):
        sb = CodeSandbox()
        result = sb._run(code="import scipy.constants as C\nprint(C.c)")
        assert result["success"] is True

    def test_forbidden_import_blocked(self):
        sb = CodeSandbox()
        result = sb._run(code="import os\nprint(os.getcwd())")
        assert result["success"] is False
        assert "validation_errors" in result

    def test_eval_blocked_by_sandbox(self):
        sb = CodeSandbox()
        result = sb._run(code="eval('1+1')")
        assert result["success"] is False
        assert "validation_errors" in result

    def test_empty_code_rejected(self):
        sb = CodeSandbox()
        result = sb._run(code="")
        assert result["success"] is False
        assert "error" in result

    def test_safe_code_runs(self):
        sb = CodeSandbox()
        result = sb._run(code="import math\nprint(math.sqrt(16))")
        assert result["success"] is True
        assert "4.0" in result["stdout"]

    def test_custom_timeout(self):
        sb = CodeSandbox(timeout=1)
        result = sb._run(code="import time\ntime.sleep(5)\nprint('never')")
        assert result["success"] is False
        assert "timed out" in result["stderr"]

    def test_kwargs_code_param(self):
        sb = CodeSandbox()
        result = sb._run(code="print('hello')")
        assert result["stdout"] == "hello"

    def test_default_attributes(self):
        sb = CodeSandbox()
        assert sb.name == "code_sandbox"
        assert "sandbox" in sb.description.lower()

    def test_json_module(self):
        sb = CodeSandbox()
        result = sb._run(code="import json\nd = {'a': 1}\nprint(json.dumps(d))")
        assert result["success"] is True
        assert '{"a": 1}' in result["stdout"]

    def test_collections_module(self):
        sb = CodeSandbox()
        result = sb._run(code="from collections import Counter\nc = Counter('hello')\nprint(c['l'])")
        assert result["success"] is True
        assert "2" in result["stdout"]


# ── Output Wrap ─────────────────────────────────────────────────────

class TestWrapForOutput:
    def test_simple_expression(self):
        result = _wrap_for_output("math.sqrt(4)")
        assert result == "print(math.sqrt(4))"

    def test_already_has_print(self):
        code = "print(42)"
        assert _wrap_for_output(code) == code

    def test_assignment_wraps_variable(self):
        result = _wrap_for_output("x = 5 + 3")
        assert "print(x)" in result

    def test_import_only_no_wrap(self):
        code = "import math"
        result = _wrap_for_output(code)
        assert result == code

    def test_def_only_no_wrap(self):
        code = "def f(x):\n    return x*2"
        result = _wrap_for_output(code)
        assert result == code

    def test_comment_only_no_wrap(self):
        code = "# nothing here"
        assert _wrap_for_output(code) == code

    def test_multiline_with_expression_last(self):
        code = "import math\nx = math.sqrt(9)\nx"
        result = _wrap_for_output(code)
        assert result.endswith("print(x)")

    def test_print_with_parentheses_unaffected(self):
        code = "x = [1, 2, 3]\nprint(sum(x))"
        result = _wrap_for_output(code)
        assert result == code


# ── SandboxError ────────────────────────────────────────────────────

class TestSandboxError:
    def test_default_lineno(self):
        e = SandboxError(message="test")
        assert e.lineno == 0

    def test_with_lineno(self):
        e = SandboxError(message="test", lineno=5)
        assert e.lineno == 5


# ── Security edge cases ─────────────────────────────────────────────

class TestSecurityEdgeCases:
    def test_open_blocked(self):
        errors = validate_code("open('file.txt')")
        assert len(errors) >= 1

    def test_os_attribute_blocked(self):
        errors = validate_code("import os  # should be caught by import check")
        assert len(errors) == 1

    def test_underscore_underscore_import_blocked(self):
        errors = validate_code("__import__('os').system('ls')")
        assert len(errors) >= 1

    def test_attribute_access_on_os_blocked(self):
        """os.path.join should be caught as forbidden attribute."""
        # This needs an actual import first for the attribute chain to form
        code = "import os as _o\n_o.path.join('a', 'b')"
        errors = validate_code(code)
        # os is caught at import level
        assert len(errors) >= 1
