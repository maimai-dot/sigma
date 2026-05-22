"""Tests for discovery utilities — tool detection, AST parsing, skill loading."""

import sys
import tempfile
from pathlib import Path

import pytest
from sigma.discovery import (
    _is_tool_class,
    discover_agents_from_dir,
    discover_tools_from_dir,
    load_skills_from_dir,
    extract_agent_spec_from_source,
    _extract_from_module_constants,
)
from sigma.agent import BaseTool, Agent


class ToolA(BaseTool):
    name: str = "tool_a"
    def _run(self, **kwargs):
        return {"success": True}


class ToolB(BaseTool):
    name: str = "tool_b"
    def _run(self, **kwargs):
        return {"success": False}


class NotATool:
    def _run(self):
        pass  # _run exists but no BaseTool heritage (still detected by duck typing)


class TestIsToolClass:
    """Duck-typing tool detection."""

    def test_base_tool_subclass(self):
        assert _is_tool_class(ToolA)

    def test_duck_typing_with_run(self):
        assert _is_tool_class(NotATool)

    def test_rejects_non_class(self):
        assert not _is_tool_class("not a class")
        assert not _is_tool_class(42)

    def test_rejects_private_class(self):
        class _PrivateTool(BaseTool):
            def _run(self, **kwargs):
                return {}
        assert not _is_tool_class(_PrivateTool)

    def test_rejects_basetool(self):
        assert not _is_tool_class(BaseTool)

    def test_rejects_class_without_run(self):
        class NoRun:
            pass
        assert not _is_tool_class(NoRun)


class TestDiscoverToolsFromDir:
    """Tool discovery from a directory of Python modules."""

    def test_finds_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir()
            (tools_dir / "__init__.py").touch()

            (tools_dir / "my_tools.py").write_text("""
from sigma.agent import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    def _run(self, **kwargs):
        return {"success": True}
""", encoding="utf-8")

            # Need a parent to import from
            (Path(tmp) / "__init__.py").touch()

            import sys
            sys.path.insert(0, tmp)
            try:
                registry = discover_tools_from_dir(tools_dir)
                assert "my_tool" in registry
            finally:
                sys.path.remove(tmp)

    def test_skips_private_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir()
            (tools_dir / "__init__.py").touch()
            (tools_dir / "_private.py").write_text("""
from sigma.agent import BaseTool
class Hidden(BaseTool):
    name: str = "hidden"
    def _run(self, **kwargs):
        return {}
""", encoding="utf-8")
            (Path(tmp) / "__init__.py").touch()

            import sys
            sys.path.insert(0, tmp)
            try:
                registry = discover_tools_from_dir(tools_dir)
                assert "hidden" not in registry
            finally:
                sys.path.remove(tmp)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir(parents=True)
            registry = discover_tools_from_dir(tools_dir)
            assert registry == {}

    def test_nonexistent_dir(self):
        registry = discover_tools_from_dir(Path("/nonexistent/tools/dir"))
        assert registry == {}


class TestExtractAgentSpecFromSource:
    """AST-based agent spec extraction."""

    def test_extracts_basic_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test_agent.py"
            f.write_text("""
ROLE = "Test Engineer"
GOAL = "Validate the system"
BACKSTORY = "Experienced tester"
SKILL_FILES = ["testing.md", "validation.md"]
""", encoding="utf-8")

            spec = extract_agent_spec_from_source(f)
            assert spec is not None
            assert spec.role == "Test Engineer"
            assert spec.goal == "Validate the system"
            assert spec.backstory == "Experienced tester"
            assert "testing.md" in spec.skill_files

    def test_no_goal_or_backstory_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "empty_agent.py"
            f.write_text("ROLE = 'Nobody'\n", encoding="utf-8")
            spec = extract_agent_spec_from_source(f)
            assert spec is None

    def test_role_fallback_to_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "my_specialist.py"
            f.write_text("""
GOAL = "Do stuff"
BACKSTORY = "I do stuff"
""", encoding="utf-8")
            spec = extract_agent_spec_from_source(f)
            assert spec is not None
            # Falls back to title-cased stem when no ROLE constant
            # The fallback uses role_map or f.stem.replace("_", " ").title()
            assert spec.role == "My Specialist"

    def test_role_map_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "engineer.py"
            f.write_text("""
GOAL = "Design the system"
BACKSTORY = "Senior designer"
""", encoding="utf-8")
            spec = extract_agent_spec_from_source(
                f, role_map={"engineer": "Lead Engineer"},
            )
            assert spec is not None
            assert spec.role == "Lead Engineer"

    def test_unparseable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "bad.py"
            f.write_text("this is not valid python {{{", encoding="utf-8")
            spec = extract_agent_spec_from_source(f)
            assert spec is None


class TestLoadSkillsFromDir:
    """Skill file loading."""

    def test_loads_markdown_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skills_dir.mkdir()
            (skills_dir / "engineering.md").write_text(
                "# Engineering\nKnowledge content", encoding="utf-8")
            (skills_dir / "testing.md").write_text(
                "# Testing\nTest procedures", encoding="utf-8")

            cache = load_skills_from_dir(skills_dir)
            assert "engineering" in cache
            assert "testing" in cache
            assert "Knowledge content" in cache["engineering"]

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skills_dir.mkdir()
            cache = load_skills_from_dir(skills_dir)
            assert cache == {}

    def test_nonexistent_dir(self):
        cache = load_skills_from_dir(Path("/nonexistent/skills"))
        assert cache == {}

    def test_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            (skills_dir / "sub").mkdir(parents=True)
            (skills_dir / "sub" / "nested.md").write_text(
                "# Nested\nNested content", encoding="utf-8")
            cache = load_skills_from_dir(skills_dir)
            assert "nested" in cache


# ═══════════════════════════════════════════════════════════════════════
# discover_agents_from_dir
# ═══════════════════════════════════════════════════════════════════════

class TestDiscoverAgentsFromDir:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents_dir = Path(tmp) / "agents"
            agents_dir.mkdir()
            specs = discover_agents_from_dir(agents_dir)
            assert specs == {}

    def test_nonexistent_dir(self):
        specs = discover_agents_from_dir(Path("/no/such/agents"))
        assert specs == {}

    def test_discovers_with_ast_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents_dir = Path(tmp) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").touch()
            (agents_dir / "my_agent.py").write_text("""
ROLE = "Engineer"
GOAL = "Build rockets"
BACKSTORY = "Expert builder"
SKILL_FILES = ["propulsion.md"]
TOOLS = []
""", encoding="utf-8")
            (Path(tmp) / "__init__.py").touch()
            sys.path.insert(0, str(tmp))
            try:
                specs = discover_agents_from_dir(agents_dir)
                assert len(specs) >= 1
            finally:
                sys.path.remove(str(tmp))

    def test_skips_underscore_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents_dir = Path(tmp) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").touch()
            (agents_dir / "_private.py").write_text("""
ROLE = "Hidden"
GOAL = "Secret"
BACKSTORY = "Shh"
""", encoding="utf-8")
            (Path(tmp) / "__init__.py").touch()
            sys.path.insert(0, str(tmp))
            try:
                specs = discover_agents_from_dir(agents_dir)
                # _private.py should not be discovered
                assert len(specs) == 0
            finally:
                sys.path.remove(str(tmp))


# ═══════════════════════════════════════════════════════════════════════
# _extract_from_module_constants
# ═══════════════════════════════════════════════════════════════════════

class TestExtractFromModuleConstants:
    def test_extracts_valid_module(self):
        import types
        mod = types.ModuleType("test_mod")
        mod.ROLE = "Chief Engineer"
        mod.GOAL = "Design propulsion"
        mod.BACKSTORY = "Expert in rocket engines"
        mod.SKILL_FILES = ["propulsion.md"]
        mod.TOOLS = []

        spec = _extract_from_module_constants(mod, Path("chief.py"), {}, {})
        assert spec is not None
        assert spec.role == "Chief Engineer"
        assert spec.goal == "Design propulsion"

    def test_no_goal_or_backstory(self):
        import types
        mod = types.ModuleType("test_mod")
        mod.ROLE = "Nobody"
        # No GOAL or BACKSTORY
        spec = _extract_from_module_constants(mod, Path("nobody.py"), {}, {})
        assert spec is None

    def test_role_fallback_from_map(self):
        import types
        mod = types.ModuleType("test_mod")
        mod.GOAL = "Do things"
        mod.BACKSTORY = "I do things"
        # No ROLE constant, role should come from role_map
        spec = _extract_from_module_constants(
            mod, Path("my_agent.py"), {"my_agent": "Mapped Role"}, {},
        )
        assert spec is not None
        assert spec.role == "Mapped Role"

    def test_custom_role(self):
        import types
        mod = types.ModuleType("test_mod")
        mod.ROLE = "Custom Role"
        mod.GOAL = "Achieve"
        mod.BACKSTORY = "Experienced"
        spec = _extract_from_module_constants(mod, Path("custom.py"), {}, {})
        assert spec is not None
        assert spec.role == "Custom Role"
