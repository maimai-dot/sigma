"""Tests for Sigma standard tools."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from sigma.tools import (
    CsvTool,
    DirectorySearchTool,
    FileSystemTool,
    HttpApiTool,
    JsonTool,
    SQLDatabaseTool,
    TxtGrepTool,
    WebScrapeTool,
)


# ── FileSystemTool ────────────────────────────────────────────────────

class TestFileSystemTool:
    def test_read_write(self):
        with tempfile.TemporaryDirectory() as d:
            tool = FileSystemTool(root=d)
            r = tool._run(operation="write", path="hello.txt", content="hello world")
            assert r["success"]
            assert r["size"] == 11

            r = tool._run(operation="read", path="hello.txt")
            assert r["success"]
            assert r["content"] == "hello world"

    def test_exists(self):
        with tempfile.TemporaryDirectory() as d:
            tool = FileSystemTool(root=d)
            assert tool._run(operation="exists", path="nope.txt")["exists"] is False

            Path(d, "yes.txt").write_text("ok")
            assert tool._run(operation="exists", path="yes.txt")["exists"] is True

    def test_list_directory(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.txt").write_text("a")
            Path(d, "b.txt").write_text("b")
            Path(d, "sub").mkdir()
            tool = FileSystemTool(root=d)
            r = tool._run(operation="list", path=".")
            assert r["success"]
            names = [e.rstrip("/") for e in r["entries"]]
            assert "a.txt" in names
            assert "b.txt" in names
            assert "sub" in names

    def test_path_traversal_blocked(self):
        tool = FileSystemTool(root="/tmp/sigma_test")
        r = tool._run(operation="read", path="../../../etc/passwd")
        assert not r["success"]
        assert "forbidden pattern" in r["error"].lower()

    def test_forbidden_extension(self):
        with tempfile.TemporaryDirectory() as d:
            tool = FileSystemTool(root=d)
            r = tool._run(operation="write", path="bad.exe", content="x")
            assert not r["success"]

    def test_empty_operation(self):
        tool = FileSystemTool()
        r = tool._run()
        assert not r["success"]


# ── HttpApiTool ───────────────────────────────────────────────────────

class TestHttpApiTool:
    def test_no_url(self):
        tool = HttpApiTool()
        r = tool._run(url="")
        assert not r["success"]

    def test_invalid_url(self):
        tool = HttpApiTool()
        r = tool._run(url="not-a-url")
        assert not r["success"]

    def test_unsupported_method(self):
        tool = HttpApiTool()
        r = tool._run(url="https://example.com", method="CONNECT")
        assert not r["success"]

    def test_real_get_request(self):
        """Real HTTP call to a stable test endpoint."""
        tool = HttpApiTool(timeout=15)
        r = tool._run(url="https://httpbin.org/get")
        if r["success"]:
            assert r["status_code"] == 200
            assert isinstance(r["body"], dict)


# ── SQLDatabaseTool ───────────────────────────────────────────────────

class TestSQLDatabaseTool:
    def test_create_and_select(self):
        tool = SQLDatabaseTool()
        r = tool._run(sql="CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
        assert r["success"]

        tool._run(sql="INSERT INTO items (name, price) VALUES (?, ?)", params=["widget", 9.99])
        tool._run(sql="INSERT INTO items (name, price) VALUES (?, ?)", params=["gadget", 19.50])

        r = tool._run(sql="SELECT * FROM items ORDER BY id")
        assert r["success"]
        assert r["operation"] == "query"
        assert r["row_count"] == 2
        assert r["columns"] == ["id", "name", "price"]
        assert r["rows"][0]["name"] == "widget"

    def test_select_with_parameter(self):
        tool = SQLDatabaseTool()
        tool._run(sql="CREATE TABLE x (k TEXT, v INTEGER)")
        tool._run(sql="INSERT INTO x VALUES (?, ?)", params=["a", 1])
        tool._run(sql="INSERT INTO x VALUES (?, ?)", params=["b", 2])

        r = tool._run(sql="SELECT * FROM x WHERE k = ?", params=["b"])
        assert r["row_count"] == 1
        assert r["rows"][0]["v"] == 2

    def test_update_affected_rows(self):
        tool = SQLDatabaseTool()
        tool._run(sql="CREATE TABLE x (k TEXT)")
        tool._run(sql="INSERT INTO x VALUES (?)", params=["old"])
        r = tool._run(sql="UPDATE x SET k = ?", params=["new"])
        assert r["success"]
        assert r["affected_rows"] >= 0

    def test_blocked_sql(self):
        tool = SQLDatabaseTool()
        r = tool._run(sql="DROP TABLE whatever")
        assert not r["success"]
        assert "blocked" in r["error"].lower()


# ── JsonTool ──────────────────────────────────────────────────────────

class TestJsonTool:
    def test_read(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "data.json")
            fp.write_text(json.dumps({"hello": "world", "count": 42}))
            tool = JsonTool()
            r = tool._run(operation="read", filepath=str(fp))
            assert r["success"]
            assert r["data"] == {"hello": "world", "count": 42}

    def test_filter(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "users.json")
            fp.write_text(json.dumps([
                {"name": "Alice", "role": "admin"},
                {"name": "Bob", "role": "user"},
                {"name": "Carol", "role": "admin"},
            ]))
            tool = JsonTool()
            r = tool._run(operation="filter", filepath=str(fp), key="role", value="admin")
            assert r["success"]
            assert r["match_count"] == 2

    def test_get_path(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "nested.json")
            fp.write_text(json.dumps({"users": [{"name": "A"}, {"name": "B"}]}))
            tool = JsonTool()
            r = tool._run(operation="get", filepath=str(fp), path="users[0].name")
            assert r["success"]
            assert r["data"] == "A"

    def test_no_filepath(self):
        tool = JsonTool()
        r = tool._run()
        assert not r["success"]


# ── CsvTool ───────────────────────────────────────────────────────────

class TestCsvTool:
    def test_read(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "data.csv")
            fp.write_text("name,score\nAlice,95\nBob,87\n")
            tool = CsvTool()
            r = tool._run(operation="read", filepath=str(fp))
            assert r["success"]
            assert r["row_count"] == 2
            assert r["data"][0]["name"] == "Alice"

    def test_columns(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "data.csv")
            fp.write_text("a,b,c\n1,2,3\n")
            tool = CsvTool()
            r = tool._run(operation="columns", filepath=str(fp))
            assert r["success"]
            assert r["columns"] == ["a", "b", "c"]

    def test_filter(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "data.csv")
            fp.write_text("city,country\nParis,FR\nBerlin,DE\nLyon,FR\n")
            tool = CsvTool()
            r = tool._run(operation="filter", filepath=str(fp), column="country", value="FR")
            assert r["success"]
            assert r["match_count"] == 2


# ── TxtGrepTool ───────────────────────────────────────────────────────

class TestTxtGrepTool:
    def test_grep(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "log.txt")
            fp.write_text("line one: ok\nline two: error\nline three: ok\n")
            tool = TxtGrepTool()
            r = tool._run(operation="grep", filepath=str(fp), pattern="error")
            assert r["success"]
            assert r["match_count"] == 1
            assert r["matches"][0]["line_num"] == 2

    def test_regex(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "data.txt")
            fp.write_text("abc123\nxyz\n456def\n")
            tool = TxtGrepTool()
            r = tool._run(operation="regex", filepath=str(fp), pattern=r"\d+")
            assert r["success"]
            assert r["match_count"] == 2

    def test_context(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d, "ctx.txt")
            fp.write_text("a\nb\nTARGET\nc\nd\n")
            tool = TxtGrepTool()
            r = tool._run(operation="grep", filepath=str(fp), pattern="TARGET", context_lines=1)
            assert r["success"]
            assert r["match_count"] == 1
            m = r["matches"][0]
            assert len(m["context_before"]) == 1
            assert len(m["context_after"]) == 1
            assert m["context_before"][0] == "b"

    def test_no_pattern(self):
        tool = TxtGrepTool()
        r = tool._run(filepath="/dev/null", pattern="")
        assert not r["success"]


# ── WebScrapeTool ─────────────────────────────────────────────────────

class TestWebScrapeTool:
    def test_no_url(self):
        tool = WebScrapeTool()
        r = tool._run(url="")
        assert not r["success"]

    def test_invalid_url(self):
        tool = WebScrapeTool()
        r = tool._run(url="ftp://example.com")
        assert not r["success"]

    def test_real_fetch(self):
        """Real fetch from a test endpoint."""
        tool = WebScrapeTool(timeout=15)
        r = tool._run(url="https://httpbin.org/json", operation="json")
        if r["success"]:
            assert r["status_code"] == 200
            assert "data" in r


# ── DirectorySearchTool ───────────────────────────────────────────────

class TestDirectorySearchTool:
    def test_find_py_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.py").write_text("pass")
            Path(d, "b.txt").write_text("hello")
            Path(d, "sub").mkdir()
            Path(d, "sub/c.py").write_text("pass")

            tool = DirectorySearchTool()
            r = tool._run(directory=d, pattern="*.py")
            assert r["success"]
            assert r["match_count"] == 2

    def test_directory_not_found(self):
        tool = DirectorySearchTool()
        r = tool._run(directory="/nonexistent/path")
        assert not r["success"]


# ── Document tools (graceful missing-dep handling) ────────────────────

class TestPdfTool:
    def test_missing_file(self):
        from sigma.tools.document_tools import PdfTool
        tool = PdfTool()
        r = tool._run(filepath="/nonexistent/file.pdf")
        assert not r["success"]


class TestExcelTool:
    def test_missing_file(self):
        from sigma.tools.document_tools import ExcelTool
        tool = ExcelTool()
        r = tool._run(filepath="/nonexistent/file.xlsx")
        assert not r["success"]
        r2 = tool._run(filepath="/nonexistent/file.xlsx", operation="sheets")
        assert not r2["success"]


class TestDocxTool:
    def test_missing_file(self):
        from sigma.tools.document_tools import DocxTool
        tool = DocxTool()
        r = tool._run(filepath="/nonexistent/file.docx")
        assert not r["success"]
