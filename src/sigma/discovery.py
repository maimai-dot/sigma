"""Generic agent, tool, and skill discovery utilities.

Extracted from protocol.py -- parameterized so any application can use them
without hardcoded domain knowledge.
"""

import ast
import importlib
import sys
from pathlib import Path
from typing import Any, Optional

from sigma.agent import Agent, BaseTool


def _is_tool_class(obj: type) -> bool:
    """Check whether a class looks like a tool (duck typing).

    Accepts any class that defines its own _run method,
    regardless of which BaseTool it inherits from.
    """
    if not isinstance(obj, type):
        return False
    if obj.__name__.startswith("_") or obj.__name__ == "BaseTool":
        return False
    if "_run" not in obj.__dict__:
        return False
    return True


def discover_agents_from_dir(
    agents_dir: Path,
    role_map: dict[str, str] | None = None,
    tool_registry: dict[str, Any] | None = None,
) -> dict[str, "AgentSpec"]:
    """Discover agent definitions from a directory of Python files.

    Tries import first (CrewAI-style create_* functions), falls back to AST parsing.
    Returns dict of role_name -> AgentSpec.
    """
    from sigma.protocol import AgentSpec

    role_map = role_map or {}
    tool_registry = tool_registry or {}
    specs: dict[str, AgentSpec] = {}

    if not agents_dir.exists():
        return specs

    # Ensure project root is in path for imports
    project_root = str(agents_dir.parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for f in sorted(agents_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod_name = f"agents.{f.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            spec = extract_agent_spec_from_source(f, role_map, tool_registry)
            if spec:
                specs[spec.role] = spec
            continue

        # Look for Sigma Agent instances (ROLE, GOAL, etc. as module-level constants)
        spec = _extract_from_module_constants(mod, f, role_map, tool_registry)
        if spec:
            specs[spec.role] = spec
            continue

        # Fallback: look for create_* functions returning CrewAI Agent objects
        for attr_name in dir(mod):
            if not attr_name.startswith("create_"):
                continue
            fn = getattr(mod, attr_name)
            if not callable(fn):
                continue
            try:
                crewai_agent = fn()
                role = getattr(crewai_agent, "role", f.stem)
                goal = getattr(crewai_agent, "goal", "")
                backstory = getattr(crewai_agent, "backstory", "")
                tools = getattr(crewai_agent, "tools", []) or []
                skills = getattr(crewai_agent, "skill_path", []) or []

                spec = _build_spec_from_crewai_agent(
                    role, goal, backstory, tools, skills, tool_registry,
                )
                specs[role] = spec
            except Exception:
                continue

    return specs


def _extract_from_module_constants(
    mod: Any, f: Path, role_map: dict[str, str], tool_registry: dict[str, Any],
) -> Optional["AgentSpec"]:
    """Extract AgentSpec from module-level constants (Sigma native format).

    Looks for ROLE, GOAL, BACKSTORY, SKILL_FILES, TOOLS at module level.
    """
    from sigma.protocol import AgentSpec

    goal = getattr(mod, "GOAL", "")
    backstory = getattr(mod, "BACKSTORY", "")
    if not goal and not backstory:
        return None

    role = getattr(mod, "ROLE", "") or role_map.get(f.stem, f.stem.replace("_", " ").title())
    skill_files = getattr(mod, "SKILL_FILES", []) or []

    # Resolve tools
    tool_names = []
    tool_instances = []
    tools_attr = getattr(mod, "TOOLS", []) or []
    try:
        tools_iter = tools_attr() if callable(tools_attr) else tools_attr
    except Exception:
        tools_iter = tools_attr
    for t in tools_iter if hasattr(tools_iter, "__iter__") and not isinstance(tools_iter, str) else []:
        t_name = getattr(t, "name", type(t).__name__)
        tool_names.append(t_name)
        tool_instances.append(t)
        tool_registry[t_name] = t

    return AgentSpec(
        name=role, role=role, goal=goal, backstory=backstory,
        skill_files=list(skill_files),
        tool_names=tool_names, tool_instances=tool_instances,
    )


def _build_spec_from_crewai_agent(
    role: str, goal: str, backstory: str,
    tools: list, skills: list, tool_registry: dict[str, Any],
) -> "AgentSpec":
    """Build an AgentSpec from a CrewAI Agent object's attributes."""
    from sigma.protocol import AgentSpec

    tool_names = []
    tool_instances = []
    for t in tools:
        t_name = getattr(t, "name", type(t).__name__)
        tool_names.append(t_name)
        tool_instances.append(t)
        tool_registry[t_name] = t

    return AgentSpec(
        name=role, role=role, goal=goal, backstory=backstory,
        skill_files=list(skills) if skills else [],
        tool_names=tool_names, tool_instances=tool_instances,
    )


def extract_agent_spec_from_source(
    f: Path, role_map: dict[str, str] | None = None,
    tool_registry: dict[str, Any] | None = None,
) -> Optional["AgentSpec"]:
    """Parse a Python source file via AST to extract role/goal/backstory/tools."""
    from sigma.protocol import AgentSpec

    role_map = role_map or {}
    tool_registry = tool_registry or {}
    try:
        source = f.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    constants: dict[str, Any] = {}
    tool_class_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Constant):
                        constants[target.id] = node.value.value
                    elif isinstance(node.value, ast.List):
                        items = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                items.append(elt.value)
                        if items:
                            constants[target.id] = items
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOLS":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name):
                                tool_class_names.add(elt.func.id)

    goal = constants.get("GOAL", "")
    backstory = constants.get("BACKSTORY", "")
    skill_files = constants.get("SKILL_FILES", [])

    if not goal and not backstory:
        return None

    role = constants.get("ROLE", "") or role_map.get(
        f.stem, f.stem.replace("_", " ").title(),
    )

    # Match tool class names to registered tool instances
    tool_names = []
    tool_instances = []
    for cls_name in tool_class_names:
        if cls_name in tool_registry:
            tool_names.append(cls_name)
            tool_instances.append(tool_registry[cls_name])
        else:
            for reg_name, reg_instance in list(tool_registry.items()):
                if cls_name.lower() in reg_name.lower() or reg_name.lower() in cls_name.lower():
                    tool_names.append(reg_name)
                    tool_instances.append(reg_instance)
                    break

    return AgentSpec(
        name=role, role=role, goal=goal, backstory=backstory,
        skill_files=list(skill_files) if skill_files else [],
        tool_names=tool_names, tool_instances=tool_instances,
    )


def discover_tools_from_dir(tools_dir: Path) -> dict[str, Any]:
    """Auto-discover tool instances from a directory of Python modules."""
    registry: dict[str, Any] = {}
    if not tools_dir.exists():
        return registry

    project_root = str(tools_dir.parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for f in sorted(tools_dir.glob("*.py")):
        if f.name.startswith("_") or f.name in ("mission_control.py", "complexity_monitor.py"):
            continue
        mod_name = f"tools.{f.stem}"
        try:
            mod = importlib.import_module(mod_name)
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(mod, attr_name)
                if not isinstance(obj, type):
                    continue
                if _is_tool_class(obj):
                    try:
                        instance = obj()
                        name = getattr(instance, "name", attr_name)
                        registry[name] = instance
                    except Exception:
                        pass
        except Exception:
            pass
    return registry


def load_skills_from_dir(skills_dir: Path) -> dict[str, str]:
    """Pre-load all skill .md files into a name->content dict."""
    cache: dict[str, str] = {}
    if not skills_dir.exists():
        return cache
    for f in skills_dir.rglob("*.md"):
        try:
            cache[f.stem] = f.read_text(encoding="utf-8")
        except Exception:
            pass
    return cache
