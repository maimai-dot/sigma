"""Agent CapabilityRegistry — informs the decomposer which agent can do what.

Without this, the decomposer only sees agent names and guesses domain fit.
With a registry, assignment accuracy improves significantly.
"""

from dataclasses import dataclass, field


@dataclass
class AgentCapability:
    """What an agent/role can do — domains, tools, expertise."""
    name: str
    domains: list[str] = field(default_factory=list)
    """Domain keywords: e.g., ["推进", "燃烧", "propulsion", "thermodynamics"]."""
    tools: list[str] = field(default_factory=list)
    """Tool names available to this agent."""
    expertise: str = ""
    """Free-text summary of what this agent excels at."""

    def summary(self) -> str:
        parts = [f"{self.name}:"]
        if self.domains:
            parts.append(f"领域={', '.join(self.domains)}")
        if self.tools:
            parts.append(f"工具={', '.join(self.tools)}")
        if self.expertise:
            parts.append(f"专长={self.expertise}")
        return " ".join(parts)


class CapabilityRegistry:
    """Registry mapping agent names to their capabilities.

    Used by TauDecomposer to inject capability context into the decompose prompt,
    resulting in more accurate task assignments.

    Usage:
        reg = CapabilityRegistry({
            "Propulsion Chief": AgentCapability(
                name="Propulsion Chief",
                domains=["推进", "燃烧", "发动机"],
                tools=["rocketcea"],
                expertise="固体/液体火箭发动机设计，推力与比冲估算",
            ),
        })
    """

    def __init__(self, entries: dict[str, AgentCapability] | None = None):
        self._entries: dict[str, AgentCapability] = entries or {}

    def register(self, cap: AgentCapability) -> None:
        self._entries[cap.name] = cap

    def get(self, name: str) -> AgentCapability | None:
        return self._entries.get(name)

    def to_prompt_context(self, agent_names: list[str] | None = None) -> str:
        """Generate capability context for injection into the decompose prompt.

        Args:
            agent_names: if provided, only include these agents.
                         if None, include all registered.
        """
        names = set(agent_names) if agent_names else set(self._entries.keys())
        entries = [self._entries[n] for n in names if n in self._entries]
        if not entries:
            return ""

        lines = ["各角色能力说明（拆解任务时请据此分配）："]
        for cap in entries:
            lines.append(f"  - {cap.summary()}")
        return "\n".join(lines)
