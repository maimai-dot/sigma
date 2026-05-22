"""Tests for Sigma native Agent and BaseTool classes."""

from sigma.agent import Agent, BaseTool


class FakeTool(BaseTool):
    """A tool with class-level name."""
    name: str = "fake_tool"
    description: str = "A fake tool for testing"

    def _run(self, **kwargs):
        return {"success": True, "result": kwargs}


class NamelessTool(BaseTool):
    """A tool without class-level name — should fall back to class name."""

    def _run(self, **kwargs):
        return {"success": True}


class TestBaseTool:
    """BaseTool creation, name resolution, and _run dispatching."""

    def test_name_from_class_attribute(self):
        t = FakeTool()
        assert t.name == "fake_tool"

    def test_name_fallback_to_class_name(self):
        t = NamelessTool()
        assert t.name == "NamelessTool"

    def test_description(self):
        t = FakeTool()
        # description is set by dataclass default, not resolved in __post_init__
        assert t.description == ""

    def test_default_description(self):
        t = NamelessTool()
        assert t.description == ""

    def test_run_method(self):
        t = FakeTool()
        result = t._run(x=1, y=2)
        assert result == {"success": True, "result": {"x": 1, "y": 2}}


class TestAgent:
    """Agent creation and attribute access."""

    def test_minimal_agent(self):
        a = Agent()
        assert a.role == ""
        assert a.goal == ""
        assert a.backstory == ""
        assert a.tools == []
        assert a.skill_path == []
        assert not a.allow_delegation
        assert not a.verbose

    def test_full_agent(self):
        tool = FakeTool()
        a = Agent(
            role="Engineer",
            goal="Build things",
            backstory="Experienced builder",
            tools=[tool],
            skill_path=["engineering.md"],
            allow_delegation=True,
            verbose=True,
        )
        assert a.role == "Engineer"
        assert a.goal == "Build things"
        assert a.backstory == "Experienced builder"
        assert a.tools == [tool]
        assert a.skill_path == ["engineering.md"]
        assert a.allow_delegation
        assert a.verbose

    def test_absorbs_crewai_kwargs(self):
        """Agent should accept and ignore CrewAI-specific kwargs."""
        a = Agent(
            role="Engineer",
            goal="Test",
            backstory="Testing",
            system_message="CrewAI system message",
            llm="gpt-4",
            max_iter=15,
            some_unknown_kwarg=True,
        )
        assert a.role == "Engineer"
        assert a.goal == "Test"
        # CrewAI kwargs are absorbed, not stored

    def test_default_tools_none(self):
        a = Agent()
        assert a.tools == []

    def test_skill_path_default(self):
        a = Agent()
        assert a.skill_path == []
