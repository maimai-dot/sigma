"""Sigma native Agent and BaseTool — zero CrewAI dependency."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseTool:
    """Minimal tool base class for Sigma agents.

    Subclass this and implement _run() for your tools.
    The name field defaults to the class-level name attribute
    (or the class name itself), matching CrewAI BaseTool behavior.
    """

    name: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.name:
            cls = type(self)
            if hasattr(cls, "name") and isinstance(cls.name, str) and cls.name:
                self.name = cls.name
            else:
                self.name = cls.__name__

    def _run(self, **kwargs) -> dict:
        raise NotImplementedError


class Agent:
    """Sigma native agent definition.

    Drop-in replacement for CrewAI's Agent, holding role metadata.
    Accepts and ignores CrewAI-specific kwargs (system_message, llm, max_iter)
    so agent files can migrate with just an import change.
    """

    role: str
    goal: str
    backstory: str
    tools: list[Any]
    skill_path: list[str]
    allow_delegation: bool
    verbose: bool

    def __init__(
        self,
        role: str = "",
        goal: str = "",
        backstory: str = "",
        tools: list[Any] | None = None,
        skill_path: list[str] | None = None,
        allow_delegation: bool = False,
        verbose: bool = False,
        **kwargs,  # absorb CrewAI-specific: system_message, llm, max_iter, etc.
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.skill_path = skill_path or []
        self.allow_delegation = allow_delegation
        self.verbose = verbose
