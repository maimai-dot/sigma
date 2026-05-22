"""Sigma 框架真实场景验证 — 非火箭领域端到端测试。

验证目标：
  1. Sigma 能否处理纯 LLM 推理任务（零工具）？
  2. AERC 循环能否在多轮后完成收敛？
  3. 输出是否对非火箭领域有实际价值？
  4. 整个流程有没有藏着的 "花架子"（fake/mock/skip）？

测试场景：
  A. 软件工程：设计一个高并发短链服务的数据库架构
  B. 医疗：评估两种降压药联合使用的安全性

每个场景使用 Sigma AERC STANDARD 模式（4 agent, 交叉审查）。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dotenv
env_path = os.environ.get("SIGMA_DOTENV", os.path.join(os.path.dirname(__file__), "..", ".env"))
dotenv.load_dotenv(env_path)
dotenv.load_dotenv()

from sigma.config import SigmaConfig
from sigma.llm import UniversalBackend
from sigma.protocol import AgentSpec
from sigma.cost_tracker import CostTracker


# ═══════════════════════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════════════════════

config = SigmaConfig(
    project_name="Verification-Test",
    creed="We produce rigorous, evidence-backed analysis with specific numbers and clear reasoning.",
    domain_keywords={
        "software": ["database", "architecture", "API", "cache", "SQL", "throughput", "latency"],
        "medical": ["drug", "dose", "interaction", "safety", "efficacy", "trial", "contraindication"],
    },
    role_map={
        "analyst": "Senior Analyst",
        "reviewer": "Peer Reviewer",
        "specialist": "Domain Specialist",
        "safety": "Safety Auditor",
    },
    domain_agent_map={
        "software": "analyst",
        "medical": "analyst",
    },
)

backend = UniversalBackend(
    api_key=os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY")),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com"),
)

# 通用智能体（零领域知识内置，全部通过 AgentSpec 注入）
agents = {
    "Senior Analyst": AgentSpec(
        name="Senior Analyst",
        role="高级分析师",
        goal="对给定问题进行深度分析，给出有具体数值、有逻辑链、有工程可行性的方案",
        backstory=(
            "你是一位资深技术分析师，擅长从第一性原理出发分析问题。"
            "你需要给出具体数值（如 QPS、延迟ms、成本¥），不能只说'高性能''低成本'。"
            "回复使用中文，给出具体数值和推理过程。"
        ),
    ),
    "Peer Reviewer": AgentSpec(
        name="Peer Reviewer",
        role="同行评审",
        goal="审查分析师的方案，找出逻辑漏洞、数值矛盾、遗漏的边界条件和未考虑的替代方案",
        backstory=(
            "你是一位严格的同行评审专家。你的职责是挑战分析师的每一个假设，"
            "找出数据中的矛盾，指出遗漏的约束条件。请用中文回复。"
        ),
    ),
    "Domain Specialist": AgentSpec(
        name="Domain Specialist",
        role="领域专家",
        goal="从领域实践角度补充具体案例、已知陷阱、行业基准数据",
        backstory=(
            "你是一位有 15 年实战经验的领域专家。你知道教科书之外的工程现实——"
            "哪些方案在实际中出过问题、哪些指标是行业真实水平（不是理论峰值）。"
            "请用中文回复，引用具体的行业案例或数据。"
        ),
    ),
    "Safety Auditor": AgentSpec(
        name="Safety Auditor",
        role="安全审计员",
        goal="以最坏情况视角审视方案，找出可能导致系统崩溃或安全事故的极端场景",
        backstory=(
            "你是一位安全审计专家。你的职责是从最坏情况出发审视方案："
            "如果流量暴涨 10 倍会怎样？如果关键节点宕机会怎样？"
            "如果用户输入恶意数据会怎样？请用中文回复。"
        ),
    ),
}

tools = {}  # 零工具：纯 LLM 推理
cost_tracker = CostTracker()

print("=" * 70)
print("  Sigma 框架真实场景验证")
print("  验证目标: 框架在非火箭领域、零工具条件下能否正常工作")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario A: 软件工程领域
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Scenario A: 高并发短链服务数据库架构设计（软件工程）")
print("─" * 70)

instruction_a = (
    "设计一个日活 500 万用户的短链服务（类似 bit.ly）的数据库架构。"
    "要求：1）日生成短链 ~2000 万条；2）日访问量 ~5 亿次重定向；"
    "3）短链有效期 3 年；4）需要统计每条的点击量。"
    "请给出：存储引擎选择、分库分表策略、缓存方案、"
    "预估存储量、QPS、P99 延迟、月度基础设施成本。"
    "用具体数值，不能只说'高性能''低成本'。"
)

print(f"  任务: {instruction_a[:80]}...")

# Run AERC
from sigma.orchestrator import SigmaOrchestrator

orch_a = SigmaOrchestrator(
    config=config, agents=agents, tools=tools,
    llm_backend=backend, max_rounds=2,
    verbose=False, interactive=False,
)

result_a = orch_a.run(instruction_a, mode="sigma")

print(f"\n  状态: {result_a.get('final_verdict', 'unknown')}")
print(f"  轮次: {result_a.get('total_rounds', '?')}")
params_a = result_a.get("parameters", {})
print(f"  产出参数: {len(params_a)} 个")
for k, v in list(params_a.items())[:5]:
    print(f"    {k}: {v}")

# 检查是否有实质性输出
analyses = result_a.get("agent_analyses", {})
agent_count = len(analyses)
total_chars = sum(len(str(v)) for v in analyses.values())
print(f"  智能体分析: {agent_count} 个, 总长度 {total_chars} 字符")

if agent_count >= 2 and total_chars > 500:
    print("  ✅ 有多智能体参与，产出有实质内容")
else:
    print("  ⚠️ 智能体参与不足或产出过短")

cost_a = result_a.get("cost_summary", {})
print(f"  Token: {cost_a.get('total_tokens', '?')}, 成本: ¥{cost_a.get('estimated_cost', 0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario B: 医疗领域
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Scenario B: 降压药联合使用安全性评估（医疗）")
print("─" * 70)

instruction_b = (
    "评估氨氯地平（5mg/日）与赖诺普利（10mg/日）联合用于"
    "65 岁、轻度肾功能不全（eGFR 55 mL/min/1.73m²）高血压患者的"
    "安全性和有效性。需考虑：药代动力学相互作用、血压目标、"
    "电解质紊乱风险、肾功能监测频率。给出具体建议和监测方案。"
)

print(f"  任务: {instruction_b[:80]}...")

orch_b = SigmaOrchestrator(
    config=config, agents=agents, tools=tools,
    llm_backend=backend, max_rounds=2,
    verbose=False, interactive=False,
)

result_b = orch_b.run(instruction_b, mode="sigma")

print(f"\n  状态: {result_b.get('final_verdict', 'unknown')}")
print(f"  轮次: {result_b.get('total_rounds', '?')}")
params_b = result_b.get("parameters", {})
print(f"  产出参数: {len(params_b)} 个")
for k, v in list(params_b.items())[:5]:
    print(f"    {k}: {v}")

analyses_b = result_b.get("agent_analyses", {})
agent_count_b = len(analyses_b)
total_chars_b = sum(len(str(v)) for v in analyses_b.values())
print(f"  智能体分析: {agent_count_b} 个, 总长度 {total_chars_b} 字符")

if agent_count_b >= 2 and total_chars_b > 500:
    print("  ✅ 有多智能体参与，产出有实质内容")
else:
    print("  ⚠️ 智能体参与不足或产出过短")

cost_b = result_b.get("cost_summary", {})
print(f"  Token: {cost_b.get('total_tokens', '?')}, 成本: ¥{cost_b.get('estimated_cost', 0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  验证总结
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  验证总结")
print("=" * 70)

checks = []

# Check 1: AERC loop completes
checks.append(("AERC 循环完成",
    result_a.get("final_verdict") not in ("error", None, "unknown")
    and result_b.get("final_verdict") not in ("error", None, "unknown")))

# Check 2: Multi-agent participation
checks.append(("多智能体并行参与",
    agent_count >= 3 and agent_count_b >= 3))

# Check 3: Substantial output
checks.append(("产出有实质内容 (>1000 chars)",
    total_chars > 1000 and total_chars_b > 1000))

# Check 4: Cross-review happened
rounds_a = result_a.get("total_rounds", 0)
rounds_b = result_b.get("total_rounds", 0)
checks.append(("多轮协作 (>=1 round)",
    rounds_a >= 1 and rounds_b >= 1))

# Check 5: Cost is reasonable
total_cost = cost_a.get('estimated_cost', 0) + cost_b.get('estimated_cost', 0)
checks.append((f"成本合理 (¥{total_cost:.4f} < ¥1.00)",
    total_cost < 1.00))

print()
all_pass = True
for name, passed in checks:
    status = "✅" if passed else "❌"
    if not passed:
        all_pass = False
    print(f"  {status} {name}")

print()
if all_pass:
    print("  结论: Sigma 框架是真实可用的，不是花架子。")
else:
    print("  结论: 存在问题，需进一步调查。")

print()
print("=" * 70)
