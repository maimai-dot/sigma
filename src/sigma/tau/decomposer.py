"""TauDecomposer — LLM-driven task decomposition.

The Tau (总负责人) analyzes the instruction and breaks it into
subtasks, each assigned to specific agents, with interface parameters
that serve as contracts between departments.
"""

import json

from sigma.tau.types import SubTask, TaskGraph
from sigma.tau.config import TauConfig


class TauDecomposer:
    """Task decomposition engine — breaks high-level instruction into SubTask graph.

    Caches decomposition results by (instruction, agents) key to avoid
    redundant LLM calls for identical decomposition requests.
    """

    def __init__(self, config: TauConfig | None = None, cache_size: int = 128):
        self.config = config or TauConfig()
        self._cache: dict[str, TaskGraph] = {}
        self._cache_size = max(1, cache_size)

    def decompose(self, instruction: str, agent_names: list[str],
                  llm_call, skills: dict[str, str] | None = None,
                  capabilities=None,
                  lessons: str = "") -> TaskGraph:
        """Decompose instruction into a TaskGraph.

        Args:
            instruction: high-level task instruction
            agent_names: available agent/role names
            llm_call: callable(system_prompt, user_prompt) -> str
            skills: optional skill_name → skill_content for domain knowledge
            capabilities: optional CapabilityRegistry for agent capability context
        """
        cache_key = self._make_key(instruction, agent_names)
        if cache_key in self._cache:
            return self._cache[cache_key]

        system = self.config.decompose_system_prompt
        if skills:
            skill_summaries = []
            for name, content in skills.items():
                summary = content[:300] if len(content) > 300 else content
                skill_summaries.append(f"  [{name}]: {summary}")
            if skill_summaries:
                system += (
                    "\n\n可用的专业知识技能（拆解时可参考）：\n"
                    + "\n".join(skill_summaries)
                    + "\n在分配子任务时，将相关技能匹配到合适的部门。"
                )
        if capabilities is not None:
            ctx = capabilities.to_prompt_context(agent_names)
            if ctx:
                system += "\n\n" + ctx

        user = (
            f"任务：{instruction}\n\n"
            f"可用部门/角色：{', '.join(agent_names)}\n\n"
            "请拆解任务。"
        )
        if lessons:
            user += "\n" + lessons
        resp = llm_call(system, user)

        data = self._parse_json(resp)
        if data is None:
            fallback = self._fallback_graph(instruction, agent_names)
            self._store(cache_key, fallback)
            return fallback

        subtasks = []
        for item in data.get("subtasks", []):
            # Filter to valid agents only
            agents = [a for a in item.get("assigned_agents", []) if a in agent_names]
            if not agents:
                agents = [agent_names[0]] if agent_names else ["Tau"]

            subtasks.append(SubTask(
                id=item.get("id", f"st_{len(subtasks)}"),
                description=item.get("description", ""),
                assigned_agents=agents,
                interface_params=item.get("interface_params", []),
                dependencies=item.get("dependencies", []),
                expected_outputs=item.get("expected_outputs", []),
            ))

        interface_map = data.get("interface_map", {})
        if not interface_map:
            interface_map = self._infer_interface_map(subtasks)

        graph = TaskGraph(
            instruction=instruction,
            subtasks=subtasks,
            interface_map=interface_map,
        )
        self._store(cache_key, graph)
        return graph

    def _make_key(self, instruction: str, agent_names: list[str]) -> str:
        """Build cache key from instruction and sorted agent names."""
        return f"{instruction}|||{','.join(sorted(agent_names))}"

    def _store(self, key: str, graph: TaskGraph) -> None:
        """Store in cache, evicting oldest entry if at capacity."""
        if len(self._cache) >= self._cache_size:
            # Evict first key (simple FIFO eviction)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = graph

    def clear_cache(self) -> None:
        """Clear the decomposition cache."""
        self._cache.clear()

    def _fallback_graph(self, instruction: str, agent_names: list[str]) -> TaskGraph:
        """When LLM decomposition fails: one subtask per agent, sequential deps."""
        subtasks = []
        for i, name in enumerate(agent_names):
            subtasks.append(SubTask(
                id=f"st_{i}",
                description=f"从{name}视角分析：{instruction}",
                assigned_agents=[name],
            ))
        return TaskGraph(instruction=instruction, subtasks=subtasks)

    def _infer_interface_map(self, subtasks: list[SubTask]) -> dict[str, list[str]]:
        """Infer interface map from subtask interface_params."""
        imap: dict[str, list[str]] = {}
        for st in subtasks:
            for p in st.interface_params:
                if p not in imap:
                    imap[p] = []
                if st.id not in imap[p]:
                    imap[p].append(st.id)
        return imap

    def _parse_json(self, text: str) -> dict | None:
        """Extract JSON from LLM response."""
        if not text or text.startswith("[LLM_ERROR"):
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return None
