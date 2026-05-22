# Σ (Sigma) — Generic AERC Multi-Agent Collaboration Framework

**Analyze → Execute → Review → Converge loop with cross-review, convergence detection, and complexity-adaptive routing.**

Zero domain knowledge baked in. Inject everything via SigmaConfig + ToolSpec. Works with any OpenAI-compatible LLM.

## Installation

```bash
pip install -e .
# or from PyPI (future):
# pip install sigma-framework
```

Requires Python ≥ 3.12.

## Quick Start

```python
from sigma import (
    SigmaOrchestrator, SigmaConfig,
    AgentSpec, ToolSpec, BaseTool,
    UniversalBackend,
)

# 1. Define your domain
config = SigmaConfig(
    project_name="My Project",
    creed="We build things that work.",
    domain_keywords={
        "engineering": ["design", "calculate", "optimize"],
        "review": ["check", "verify", "audit"],
    },
    role_map={"engineer": "Lead Engineer", "reviewer": "Reviewer"},
    domain_agent_map={"engineering": "engineer", "review": "reviewer"},
)

# 2. Define your tools
class Calculator(BaseTool):
    name: str = "calculator"
    description: str = "Performs calculations"

    def _run(self, **kwargs):
        return {"success": True, "performance": {"result": 42}}

tools = {"calculator": ToolSpec(name="calculator", instance=Calculator())}

# 3. Define your agents
agents = {
    "engineer": AgentSpec(
        name="Engineer", role="Engineer",
        goal="Design solutions", backstory="Senior engineer",
        skill_files=[], tool_names=["calculator"],
        tool_instances=[Calculator()],
    ),
}

# 4. Run
backend = UniversalBackend(api_key="sk-...")
orchestrator = SigmaOrchestrator(
    config=config, agents=agents, tools=tools,
    llm_backend=backend, max_rounds=3,
)
result = orchestrator.run("Design a simple widget")
```

## Architecture

```
Founder (you)
  ▼
┌─────────── AERC Loop ───────────┐
│  A (Analyze) — Agents analyze independently
│  E (Execute) — Parallel tool execution
│  R (Review) — Cross-review + devil's advocate
│  C (Converge) — Convergence judge + consensus estimation
└──────────────────────────────────┘
  ▼
  Output (REPORT.md + result.json)
```

### Key Mechanisms

- **Cross-review**: Agents analyze independently (cannot see each other), then review each other's work
- **Convergence judge**: Numeric params Δ < 5% + no new qualitative conflicts → converged. Oscillation detection halts early.
- **Complexity-adaptive**: Pure rule-engine scores tasks 0-10, selects LITE/STANDARD/RIGOROUS tier
- **Consensus estimation**: When tools are unreliable, multiple agents estimate independently → converge to consensus range

### Three Complexity Tiers

| Tier | Score | Agents | Rounds | LLM Calls |
|------|-------|--------|--------|-----------|
| LITE | ≤ 2.5 | 2-4 | 1 | ~8 |
| STANDARD | ≤ 6.0 | 4-6 | 2-3 | ~30 |
| RIGOROUS | > 6.0 | 8 | ≤ 4 | ~80 |

## Configuration

All domain knowledge is injected via `SigmaConfig`:

| Field | Purpose |
|-------|---------|
| `project_name` | Display name in reports |
| `creed` | Injected into every agent's system prompt |
| `domain_keywords` | Used by complexity assessor to detect domains |
| `action_weights` | Weights for action verbs in complexity scoring |
| `role_map` | Maps agent file stems to display names |
| `domain_agent_map` | Maps domain keys to agents for tier selection |
| `default_tool_params` | Fallback params for tools |
| `reasonable_ranges` | Expected output ranges for tool validation |

## Running Tests

```bash
pytest tests/ -v
pytest --cov=sigma --cov-report=term
```

985 tests, 80% coverage on all pure-logic modules.

## Concepts

- **No domain knowledge in the framework**: Rocket propellants, medical terms, financial models — all injected via `SigmaConfig`
- **Tool duck-typing**: Any class with `_run()` is a tool — no specific base class required
- **Universal LLM backend**: Works with any OpenAI-compatible API (DeepSeek, GLM, Qwen, etc.)
- **Immutable state**: Every state update returns a new copy

## License

MIT
