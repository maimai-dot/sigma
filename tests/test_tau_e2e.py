"""Real LLM end-to-end test for Tau hierarchical framework.

Tests: Decomposition → Independent Execution → Conflict Detection → Resolution
Uses live DeepSeek V4 Pro API.

Two test scenarios:
  1. Sequential task (should have interface contracts)
  2. Structured output analysis
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
from sigma.llm import OpenAIBackend
from sigma.protocol import AgentSpec, ToolSpec
from sigma.cost_tracker import CostTracker
from sigma.tau import TauOrchestrator, CapabilityRegistry, AgentCapability

# ── Setup ──
config = SigmaConfig(project_name="Tau-E2E-Test")
backend = OpenAIBackend(
    api_key=os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY")),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com"),
)
cost_tracker = CostTracker()

def llm_call(system: str, user: str) -> str:
    resp = backend.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model="deepseek-v4-pro",
        max_tokens=2048,
        temperature=0.2,
    )
    return resp.content


# ── Agents ──
agents = {
    "Propulsion Chief": AgentSpec(
        name="Propulsion Chief",
        role="推进总工",
        goal="设计KNSB固体推进系统，给出推力、比冲、燃烧室压力、喷管喉径等具体数值",
        backstory="你是一位火箭推进专家，擅长KNSB固体火箭发动机设计。请用中文回复，给出具体数值和推理。",
    ),
    "Structures Chief": AgentSpec(
        name="Structures Chief",
        role="结构总工",
        goal="基于推进系统参数设计箭体结构，给出质量、材料、壁厚、尺寸",
        backstory="你是一位航天结构工程师。你接收推进系统给出的推力和压力参数作为设计约束。请用中文回复，给出具体数值。",
    ),
    "GNC Chief": AgentSpec(
        name="GNC Chief",
        role="飞控总工",
        goal="基于推进和结构参数设计飞控系统，评估稳定性和控制方案",
        backstory="你是一位飞控专家。你需要从推进系统获取推力、从结构获取质量和尺寸作为输入。请用中文回复，给出具体数值。",
    ),
}

# ── Capabilities ──
capabilities = CapabilityRegistry({
    "Propulsion Chief": AgentCapability(
        name="Propulsion Chief",
        domains=["推进", "燃烧", "发动机设计"],
        tools=[],
        expertise="KNSB固体推进剂(65%KNO3+35%山梨醇)发动机设计，推力/比冲/室压/喉径计算",
    ),
    "Structures Chief": AgentCapability(
        name="Structures Chief",
        domains=["结构", "强度", "材料"],
        tools=[],
        expertise="箭体结构设计，铝合金/碳纤维选型，质量估算，壁厚计算",
    ),
    "GNC Chief": AgentCapability(
        name="GNC Chief",
        domains=["飞控", "稳定性", "传感器"],
        tools=[],
        expertise="火箭姿态控制，稳定裕度分析，传感器选型，弹道仿真",
    ),
})

tools = {}

# ── Tau Orchestrator ──
tau = TauOrchestrator(
    agents=agents,
    tools=tools,
    llm_call=llm_call,
    max_iterations=3,
    verbose=True,
    cost_tracker=cost_tracker,
    capabilities=capabilities,
)


# ═══════════════════════════════════════════════════════════════════════
#  Scenario 1: Sequential Rocket Design
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Scenario 1: Sequential Rocket Engine Design")
print("=" * 70)

instruction_1 = (
    "设计KNSB固体火箭发动机用于1公里级验证箭。按顺序进行："
    "步骤1：推进系统设计——确定燃烧室压力、喷管喉径、推力、比冲。"
    "步骤2：基于步骤1的推力和压力，进行箭体结构设计——选择材料、计算壁厚、估算总质量。"
    "步骤3：基于步骤1的推力和步骤2的质量，进行飞控系统评估——计算稳定裕度、选择控制方案。"
    "注意：后续步骤必须以上一步的输出作为输入约束。"
)

print(f"\n指令: {instruction_1[:100]}...\n")
state1 = tau.run(instruction_1)

print("\n" + "-" * 50)
print("  Scenario 1 Results")
print("-" * 50)
print(f"  完成: {state1.completed}")
print(f"  迭代: {state1.iteration}/{state1.max_iterations}")
print(f"  子任务数: {len(state1.task_graph.subtasks)}")

for st in state1.task_graph.subtasks:
    r = state1.subtask_results.get(st.id)
    success = r.success if r else False
    params = r.interface_params if r else {}
    deps = f" (依赖: {', '.join(st.dependencies)})" if st.dependencies else ""
    print(f"\n  [{st.id}] {st.description[:80]}...{deps}")
    print(f"    部门: {st.assigned_agents}")
    print(f"    接口参数: {st.interface_params}")
    print(f"    成功: {success}")
    if params:
        for k, v in params.items():
            conf = r.param_confidence.get(k, "?") if r else "?"
            print(f"    {k}: {v:.2f} [{conf}]")
    if r and r.agent_analyses:
        for agent_name, analysis in r.agent_analyses.items():
            preview = analysis[:200].replace("\n", " ")
            print(f"    分析({agent_name}): {preview}...")

if state1.conflict_history:
    conflicts_total = sum(len(cr.conflicts) for cr in state1.conflict_history)
    print(f"\n  总冲突: {conflicts_total}")
    for i, cr in enumerate(state1.conflict_history):
        if cr.conflicts:
            for c in cr.conflicts:
                print(f"    Round {i+1}: {c.param_key} severity={c.severity:.1f} "
                      f"({c.subtask_a}={c.value_a:.2f} vs {c.subtask_b}={c.value_b:.2f})")

if state1.resolution_history:
    for i, r in enumerate(state1.resolution_history):
        print(f"  决议 Round {i+1}: resolved={r.resolved} consensus={r.consensus_values}")
        if r.director_decision:
            print(f"    总监决策: {r.director_decision}")

print(f"\n  {cost_tracker.total_summary()}")


# ═══════════════════════════════════════════════════════════════════════
#  Scenario 2: Cross-department Integration Task
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  Scenario 2: Cross-department Integration")
print("=" * 70)

# Reset cost tracker
cost_tracker2 = CostTracker()
tau2 = TauOrchestrator(
    agents=agents,
    tools=tools,
    llm_call=llm_call,
    max_iterations=3,
    verbose=False,
    cost_tracker=cost_tracker2,
    capabilities=capabilities,
)

instruction_2 = (
    "对一枚KNSB固体火箭进行集成参数确认："
    "1）推进总工给出燃烧室压力(bar)、推力(N)、比冲(s)的工程估算值。"
    "2）结构总工基于推进参数给出箭体质量(kg)、外径(mm)、壁厚(mm)。"
    "3）飞控总工基于推力和质量参数评估稳定裕度，确保静稳定裕度>1.5倍口径。"
    "如参数不一致，请通过接口协调达成一致。"
)

print(f"\n指令: {instruction_2}...\n")
state2 = tau2.run(instruction_2)

print("\n" + "-" * 50)
print("  Scenario 2 Results")
print("-" * 50)
print(f"  完成: {state2.completed}")
print(f"  迭代: {state2.iteration}/{state2.max_iterations}")
print(f"  子任务数: {len(state2.task_graph.subtasks)}")

for st in state2.task_graph.subtasks:
    r = state2.subtask_results.get(st.id)
    success = r.success if r else False
    params = r.interface_params if r else {}
    deps = f" (依赖: {', '.join(st.dependencies)})" if st.dependencies else ""
    print(f"\n  [{st.id}] {st.description[:80]}...{deps}")
    print(f"    部门: {st.assigned_agents}")
    print(f"    接口参数: {st.interface_params}")
    print(f"    成功: {success}")
    if params:
        for k, v in params.items():
            conf = r.param_confidence.get(k, "?") if r else "?"
            print(f"    {k}: {v:.2f} [{conf}]")

if state2.conflict_history:
    conflicts_total = sum(len(cr.conflicts) for cr in state2.conflict_history)
    print(f"\n  总冲突: {conflicts_total}")
    for i, cr in enumerate(state2.conflict_history):
        if cr.conflicts:
            for c in cr.conflicts:
                print(f"    Round {i+1}: {c.param_key} severity={c.severity:.1f} "
                      f"({c.subtask_a}={c.value_a:.2f} vs {c.subtask_b}={c.value_b:.2f})")

print(f"\n  {cost_tracker2.total_summary()}")

print("\n" + "=" * 70)
print("  Tau E2E 测试完成")
print("=" * 70)
