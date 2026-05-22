"""SigmaConfig — all application-specific configuration for the Sigma framework."""

from dataclasses import dataclass, field
from typing import Optional

from sigma.cache import CacheConfig


@dataclass
class SigmaConfig:
    """Application-level configuration injected into SigmaProtocol and SigmaOrchestrator.

    Every field has a reasonable default so sigma works out of the box.
    Applications override fields to add domain knowledge.
    """

    # ── Identity ──
    project_name: str = "Sigma Project"
    """Display name used in report titles and banners."""

    # ── Creed ──
    creed: str = ""
    """Optional creed injected into every agent's system prompt."""

    # ── Domain keywords for complexity assessment ──
    domain_keywords: dict[str, list[str]] = field(default_factory=dict)
    """Maps domain names to keyword lists for _assess_complexity().
    Example: {"propulsion": ["engine", "thrust"], "structures": ["mass", "strength"]}."""

    # ── Action verb weights for complexity assessment ──
    action_weights: dict[str, float] = field(default_factory=lambda: {
        "查": 1.0, "查找": 1.0, "搜索": 1.0, "lookup": 1.0, "find": 1.0,
        "计算": 2.0, "估算": 2.0, "calculate": 2.0, "compute": 2.0,
        "比较": 2.5, "对比": 2.5, "compare": 2.5,
        "分析": 3.0, "评估": 3.0, "analyze": 3.0, "evaluate": 3.0,
        "设计": 4.0, "优化": 4.0, "design": 4.0, "optimize": 4.0,
        "方案": 4.0, "全箭": 5.0, "总体": 5.0, "完整": 4.0,
    })

    constraint_keywords: dict[str, float] = field(default_factory=lambda: {
        "必须": 0.5, "must": 0.5, "require": 0.5, "要求": 0.5,
        "安全": 0.5, "safety": 0.5, "冗余": 0.5,
        "权衡": 1.0, "trade": 1.0, "tradeoff": 1.0,
        "同时": 0.5, "both": 0.5, "兼顾": 0.5,
        "对比": 0.5, "方案对比": 1.0,
        "验证": 0.5, "verify": 0.5, "validate": 0.5,
    })

    # ── Role mapping ──
    role_map: dict[str, str] = field(default_factory=dict)
    """Maps agent source file stems to display role names.
    Example: {"propulsion_chief": "Propulsion Chief"}."""

    # ── Domain-to-agent mapping for tier-based agent selection ──
    domain_agent_map: dict[str, str] = field(default_factory=dict)
    """Maps domain keys to agent names for _select_agents_for_tier()."""

    # ── Agent composition rules ──
    lite_max_agents: int = 4
    standard_exclude_agents: set[str] = field(default_factory=set)
    """Agent names to exclude from STANDARD tier."""

    # ── Tool defaults ──
    default_tool_params: dict[str, dict] = field(default_factory=dict)
    """Fallback default parameters for tools, keyed by tool name substring.

    Example: {"my_tool": {"param_a": 1.0, "param_b": "value"}}."""

    # ── Reasonable value ranges ──
    reasonable_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Expected output ranges for tool validation, keyed by parameter prefix.
    Overrides TriggerSystem defaults."""

    # ── Cache ──
    cache_config: CacheConfig | None = None

    # ── Human-in-the-loop ──
    enable_human_gate: bool = False
    """Insert Founder approval gates at all 4 AERC phase transitions."""
    human_gate_phases: list[str] = field(default_factory=lambda: [
        "after_plan", "after_do", "after_check", "after_act",
    ])
    """Which phases trigger HumanGate. Default: all 4."""
    human_gate_callback: object | None = None
    """Optional callable(str, dict) -> GateDecision for non-interactive gate use."""

    # ── Memory ──
    memory_db_path: str | None = None

    # ── Output ──
    output_base_dir: Optional[str] = None
    """Base directory for output artifacts."""

    # ── LLM settings ──
    default_model: str = "deepseek-v4-pro"
    default_max_tokens: int = 2048
    default_temperature: float = 0.2
