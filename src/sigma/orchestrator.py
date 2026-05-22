"""
Σ 协奏器 — AERC 循环调度 + Founder 交互 + 输出生成
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sigma.config import SigmaConfig
from sigma.llm import LLMBackend
from sigma.state import SharedState, StateManager, ComplexityAssessment, ComplexityTier
from sigma.convergence import ConvergenceJudge, Verdict
from sigma.triggers import TriggerSystem
from sigma.cost_tracker import CostTracker, RoundCost
from sigma.hooks import HookSystem, HookPoint
from sigma.memory import MemoryStore
from sigma.protocol import (
    SigmaProtocol, AgentSpec, ToolSpec,
    PlanOutput, DoOutput, CheckOutput, ActOutput,
)
from sigma.tau.mode_selector import select_mode, ModeSelection
from sigma.tau.orchestrator import TauOrchestrator
from sigma.human_gate import HumanGate, GateDecision, GateAction

# ── ANSI Terminal ──────────────────────────────────────────────
_RESET = "\033[0m"; _BOLD = "\033[1m"; _DIM = "\033[2m"
_RED = "\033[91m"; _GREEN = "\033[92m"; _YELLOW = "\033[93m"
_BLUE = "\033[94m"; _MAGENTA = "\033[95m"; _CYAN = "\033[96m"; _WHITE = "\033[97m"

if sys.platform == "win32":
    import os; os.system("")


class SigmaOrchestrator:
    """Σ 协奏器 — 管理 AERC 循环直到收敛或刹车."""

    def __init__(
        self,
        config: SigmaConfig | None = None,
        agents: dict[str, AgentSpec] | None = None,
        tools: dict[str, ToolSpec] | None = None,
        skills: dict[str, str] | None = None,
        llm_backend: LLMBackend | None = None,
        max_rounds: int = 4,
        verbose: bool = True,
        interactive: bool = True,
        hooks: HookSystem | None = None,
    ):
        self.config = config or SigmaConfig()
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.interactive = interactive
        self.hooks = hooks or HookSystem()
        self.protocol = SigmaProtocol(
            config=self.config,
            agents=agents or {},
            tools=tools or {},
            skills=skills or {},
            llm_backend=llm_backend,
            verbose=verbose,
        )
        self.state_mgr = StateManager()
        self.judge = ConvergenceJudge()
        self.cost_tracker = CostTracker()
        self.total_cost = 0.0

        # Human-in-the-loop gate (optional)
        self.gate: HumanGate | None = None
        if self.config.enable_human_gate:
            cb = self.config.human_gate_callback
            self.gate = HumanGate(
                interactive=self.interactive,
                callback=cb if callable(cb) else None,
                verbose=verbose,
            )

    def _check_gate(self, phase: str, **context) -> GateDecision:
        """Run a HumanGate check if enabled for this phase.

        Returns GateDecision. Caller should check:
        - decision.action == GateAction.REJECT → stop
        - decision.should_retry → re-run the phase
        """
        if not self.gate:
            return GateDecision(action=GateAction.APPROVE)
        if phase not in self.config.human_gate_phases:
            return GateDecision(action=GateAction.APPROVE)
        return self.gate.check(phase, **context)

    @classmethod
    def resume(cls, checkpoint_path: str | Path, **kwargs) -> dict:
        """从 checkpoint 恢复 AERC 循环.

        Usage:
            result = SigmaOrchestrator.resume("output/v1/checkpoint.json", config=cfg)
        """
        orchestrator = cls(**kwargs)
        return orchestrator.run(resume_from=Path(checkpoint_path))

    def run(self, instruction: str = "", output_dir: str | Path | None = None,
            resume_from: Path | None = None, mode: str = "auto") -> dict:
        """执行完整 AERC 循环（或 Tau 层次化执行）.

        output_dir: 产出目录（绝对路径）。若未提供，使用 config.output_base_dir / "output" / v{N}
        resume_from: 从 checkpoint 文件恢复继续执行。
        mode: "auto" | "sigma" | "tau"
              auto: 基于指令内容自动选择
              sigma: 强制 Sigma 协同模式
              tau: 强制 Tau 层次化拆解模式
        """
        # ── 模式选择 ──
        if mode not in ("auto", "sigma", "tau"):
            raise ValueError(f"mode must be 'auto', 'sigma', or 'tau', got '{mode}'")

        if mode == "auto":
            selection = select_mode(instruction)
            if self.verbose:
                self._log(f"  {_DIM}模式选择: {_CYAN}{selection.mode.upper()}{_RESET} "
                          f"{_DIM}({selection.reason}) [{selection.confidence}]{_RESET}")
            effective_mode = selection.mode
        else:
            effective_mode = mode

        # ── Tau 模式路由 ──
        if effective_mode == "tau":
            return self._run_tau(instruction, output_dir, resume_from)
        base = Path(self.config.output_base_dir) if self.config.output_base_dir else Path(".")

        # ── 恢复 or 初始化 ──
        if resume_from:
            resume_path = Path(resume_from)
            state = self.state_mgr.restore(resume_path)
            instruction = instruction or state.task_instruction
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    output_path = base / output_dir
            else:
                output_path = resume_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            # 从恢复的状态重建 ComplexityAssessment
            ca = state.complexity_assessment
            assessment = ComplexityAssessment(
                tier=ComplexityTier(ca.get("tier", "standard")),
                score=ca.get("score", 5.0),
                reason=ca.get("reason", ""),
                agent_names=ca.get("agent_names", []),
                max_rounds=ca.get("max_rounds", state.max_rounds),
                cross_review=ca.get("cross_review", False),
                devil_advocate=ca.get("devil_advocate", False),
                consensus_estimation=ca.get("consensus_estimation", False),
                skill_crafter=ca.get("skill_crafter", False),
            )
            self._log(f"  {_CYAN}从 checkpoint 恢复: Round {state.round_num}/{state.max_rounds}{_RESET}")
        else:
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    output_path = base / output_dir
            else:
                output_path = base / "output" / _next_version(base / "output")
            output_path.mkdir(parents=True, exist_ok=True)

            state = self.state_mgr.init(instruction)
            state.max_rounds = self.max_rounds

            # ── 复杂度评估 ──
            assessment = self.protocol._assess_complexity(instruction)
            state.complexity_tier = assessment.tier.value
            state.complexity_assessment = {
                "tier": assessment.tier.value,
                "score": assessment.score,
                "reason": assessment.reason,
                "agent_names": assessment.agent_names,
                "cross_review": assessment.cross_review,
                "devil_advocate": assessment.devil_advocate,
                "consensus_estimation": assessment.consensus_estimation,
                "skill_crafter": assessment.skill_crafter,
            }
            state.max_rounds = min(self.max_rounds, assessment.max_rounds)

        self._print_banner()
        self._log(f"{_BOLD}{_WHITE}📋 Founder 指令{_RESET}")
        self._log(f"  {_YELLOW}{instruction}{_RESET}")
        tiers = {"lite": _GREEN, "standard": _CYAN, "rigorous": _MAGENTA}
        tc = tiers.get(assessment.tier.value, _DIM)
        self._log(f"  {_DIM}复杂度: {tc}{_BOLD}{assessment.tier.value.upper()}{_RESET} "
                  f"{_DIM}(评分 {assessment.score:.1f}/10, {len(assessment.agent_names)} 角色, "
                  f"最多 {assessment.max_rounds} 轮){_RESET}")
        self._log(f"  {_DIM}产出: {output_path}{_RESET}")

        # AERC 循环
        prev_state = None
        final_act = None
        round_reports = []

        self.hooks.trigger(HookPoint.ON_START, instruction=instruction, state=state)

        try:
            while state.round_num < state.max_rounds:
                state = self.state_mgr.start_round(state)
                round_num = state.round_num
                round_dir = output_path / f"round_{round_num}"
                round_cost = self.cost_tracker.start_round(round_num)

                self.hooks.trigger(HookPoint.ON_ROUND_START, state=state, round_num=round_num)

                self._log(f"\n{_CYAN}{_BOLD}═══ Round {round_num}/{state.max_rounds} ═══{_RESET}")

                # ── P ──
                rejected = False
                self.hooks.trigger(HookPoint.BEFORE_PLAN, state=state, assessment=assessment)
                while True:
                    plan = self.protocol.plan(state, round_cost, assessment)
                    self.hooks.trigger(HookPoint.AFTER_PLAN, state=state, plan=plan)
                    gd = self._check_gate("after_plan", plan=plan, state=state)
                    if gd.action == GateAction.REJECT:
                        rejected = True; break
                    if gd.should_retry:
                        state.task_instruction += f"\n\n[修订意见]: {gd.feedback}"
                        continue
                    if gd.action == GateAction.OVERRIDE and gd.overrides:
                        for k, v in gd.overrides.items():
                            state.task_params[k] = v
                    break
                if rejected:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — P 阶段{_RESET}")
                    break
                self._log(f"  {_GREEN}✓ P: {len(plan.agent_analyses)} 个角色, {len(plan.conflicts)} 个冲突{_RESET}")

                if self._detect_mass_llm_failure(plan.agent_analyses):
                    error_count = sum(1 for v in plan.agent_analyses.values() if str(v).startswith("[LLM_ERROR"))
                    self._log(f"  {_RED}⏹ 检测到大规模 LLM 调用失败（{error_count}/{len(plan.agent_analyses)} 个智能体返回错误），中止{_RESET}")
                    final_act = ActOutput(
                        verdict="error", decisions=[], alarm_flags=[],
                        should_continue=False, reason="大规模 LLM 调用失败 — 超过半数智能体返回错误",
                    )
                    round_reports.append(self._build_round_report(state, plan, DoOutput(tool_results={}, abnormal_results=[]),
                                           CheckOutput(reviews={}, devil_advocate="", alarm_flags=[]), final_act))
                    break

                # ── D ──
                self.hooks.trigger(HookPoint.BEFORE_DO, state=state, plan=plan)
                while True:
                    do_output = self.protocol.do(plan, state, round_cost)
                    self.hooks.trigger(HookPoint.AFTER_DO, state=state, do_output=do_output)
                    gd = self._check_gate("after_do", do_output=do_output, plan=plan, state=state)
                    if gd.action == GateAction.REJECT:
                        rejected = True; break
                    if gd.should_retry:
                        continue
                    if gd.action == GateAction.OVERRIDE and gd.overrides:
                        for k, v in gd.overrides.items():
                            state.task_params[k] = v
                    break
                if rejected:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — D 阶段{_RESET}")
                    break

                # ── C ──
                self.hooks.trigger(HookPoint.BEFORE_CHECK, state=state, plan=plan, do_output=do_output)
                while True:
                    check = self.protocol.check(state, plan, do_output, round_cost, assessment)
                    self.hooks.trigger(HookPoint.AFTER_CHECK, state=state, check=check)
                    gd = self._check_gate("after_check", check=check, state=state)
                    if gd.action == GateAction.REJECT:
                        rejected = True; break
                    if gd.should_retry:
                        continue
                    if gd.action == GateAction.OVERRIDE and gd.overrides:
                        for k, v in gd.overrides.items():
                            state.task_params[k] = v
                    break
                if rejected:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — C 阶段{_RESET}")
                    break
                self._log(f"  {_GREEN}✓ C: 数据审查完成{_RESET}")

                # ── A ──
                self.hooks.trigger(HookPoint.BEFORE_ACT, state=state, check=check)
                act = self.protocol.act(state, plan, check, round_cost, prev_state, do_output, assessment)
                final_act = act
                self.hooks.trigger(HookPoint.AFTER_ACT, state=state, act=act)
                gd = self._check_gate("after_act", act=act, state=state)
                if gd.action == GateAction.REJECT:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — A 阶段{_RESET}")
                    break
                if gd.should_retry and gd.feedback:
                    state.task_instruction += f"\n\n[修订意见]: {gd.feedback}"
                self._log(f"  {_YELLOW}✓ A: {act.verdict} — {act.reason}{_RESET}")

                # 保存本轮
                self.state_mgr.save_round(state, round_dir)
                round_reports.append(self._build_round_report(state, plan, do_output, check, act))

                # 自动 checkpoint
                self.state_mgr.checkpoint(state, output_path / "checkpoint.json")

                self.hooks.trigger(HookPoint.ON_ROUND_END, state=state, round_num=round_num, act=act)

                # 成本
                self.total_cost += round_cost.estimated_cost_rmb
                self._print_round_status(round_num, round_cost, state, act)

                # 判断
                if not act.should_continue:
                    self._log(f"\n  {_GREEN}{_BOLD}⬢ 停止: {act.reason}{_RESET}")
                    break

                # Founder 交互
                if self.interactive:
                    cmd = input(f"\n  [{_BOLD}继续{_RESET}/{_DIM}停车{_RESET}/{_DIM}查看{_RESET}]: ").strip().lower()
                    if cmd in ("停车", "stop", "s"):
                        self._log(f"  {_YELLOW}Founder 决定停车{_RESET}")
                        break
                    elif cmd in ("查看", "view", "v"):
                        self._print_detailed_status(state)
                        cmd2 = input(f"\n  [{_BOLD}继续{_RESET}/{_DIM}停车{_RESET}]: ").strip().lower()
                        if cmd2 in ("停车", "stop", "s"):
                            break

                prev_state = state
        except Exception as e:
            # 异常时保存 checkpoint 以便恢复
            self.state_mgr.checkpoint(state, output_path / "checkpoint.json")
            self.hooks.trigger(HookPoint.ON_ERROR, error=e, instruction=instruction, state=state)
            raise

        # 最终报告
        self._print_final_report(state, round_reports, final_act, output_path)
        result = self._build_result(instruction, state, round_reports, final_act, output_path)

        # 跨会话 Memory
        if self.config.memory_db_path:
            try:
                memory = MemoryStore(self.config.memory_db_path)
                memory.save_session(state, result)
                memory.close()
            except Exception:
                pass

        self.hooks.trigger(HookPoint.ON_COMPLETE, instruction=instruction, state=state, result=result)
        return result

    async def arun(self, instruction: str = "", output_dir: str | Path | None = None,
                   resume_from: Path | None = None, mode: str = "auto") -> dict:
        """异步执行完整 AERC 循环 — 使用 asyncio.gather 替代 ThreadPoolExecutor.

        Ref: Tau 模式在当前实现中委托同步 run()（TauOrchestrator 尚未异步化）。
        """
        # ── 模式选择 ──
        if mode not in ("auto", "sigma", "tau"):
            raise ValueError(f"mode must be 'auto', 'sigma', or 'tau', got '{mode}'")

        if mode == "auto":
            selection = select_mode(instruction)
            if self.verbose:
                self._log(f"  {_DIM}模式选择: {_CYAN}{selection.mode.upper()}{_RESET} "
                          f"{_DIM}({selection.reason}) [{selection.confidence}]{_RESET}")
            effective_mode = selection.mode
        else:
            effective_mode = mode

        # ── Tau 模式路由 ──
        if effective_mode == "tau":
            return self._run_tau(instruction, output_dir, resume_from)

        base = Path(self.config.output_base_dir) if self.config.output_base_dir else Path(".")

        if resume_from:
            resume_path = Path(resume_from)
            state = self.state_mgr.restore(resume_path)
            instruction = instruction or state.task_instruction
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    output_path = base / output_dir
            else:
                output_path = resume_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            ca = state.complexity_assessment
            assessment = ComplexityAssessment(
                tier=ComplexityTier(ca.get("tier", "standard")),
                score=ca.get("score", 5.0),
                reason=ca.get("reason", ""),
                agent_names=ca.get("agent_names", []),
                max_rounds=ca.get("max_rounds", state.max_rounds),
                cross_review=ca.get("cross_review", False),
                devil_advocate=ca.get("devil_advocate", False),
                consensus_estimation=ca.get("consensus_estimation", False),
                skill_crafter=ca.get("skill_crafter", False),
            )
            self._log(f"  {_CYAN}从 checkpoint 恢复: Round {state.round_num}/{state.max_rounds}{_RESET}")
        else:
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    output_path = base / output_dir
            else:
                output_path = base / "output" / _next_version(base / "output")
            output_path.mkdir(parents=True, exist_ok=True)

            state = self.state_mgr.init(instruction)
            state.max_rounds = self.max_rounds

            assessment = self.protocol._assess_complexity(instruction)
            state.complexity_tier = assessment.tier.value
            state.complexity_assessment = {
                "tier": assessment.tier.value, "score": assessment.score,
                "reason": assessment.reason, "agent_names": assessment.agent_names,
                "cross_review": assessment.cross_review, "devil_advocate": assessment.devil_advocate,
                "consensus_estimation": assessment.consensus_estimation,
                "skill_crafter": assessment.skill_crafter,
            }
            state.max_rounds = min(self.max_rounds, assessment.max_rounds)

        self._print_banner()
        self._log(f"{_BOLD}{_WHITE}📋 Founder 指令{_RESET}")
        self._log(f"  {_YELLOW}{instruction}{_RESET}")
        tiers = {"lite": _GREEN, "standard": _CYAN, "rigorous": _MAGENTA}
        tc = tiers.get(assessment.tier.value, _DIM)
        self._log(f"  {_DIM}复杂度: {tc}{_BOLD}{assessment.tier.value.upper()}{_RESET} "
                  f"{_DIM}(评分 {assessment.score:.1f}/10, {len(assessment.agent_names)} 角色, "
                  f"最多 {assessment.max_rounds} 轮){_RESET}")
        self._log(f"  {_DIM}产出: {output_path}{_RESET}")

        prev_state = None
        final_act = None
        round_reports = []

        self.hooks.trigger(HookPoint.ON_START, instruction=instruction, state=state)

        try:
            while state.round_num < state.max_rounds:
                state = self.state_mgr.start_round(state)
                round_num = state.round_num
                round_dir = output_path / f"round_{round_num}"
                round_cost = self.cost_tracker.start_round(round_num)

                self.hooks.trigger(HookPoint.ON_ROUND_START, state=state, round_num=round_num)
                self._log(f"\n{_CYAN}{_BOLD}═══ Round {round_num}/{state.max_rounds} ═══{_RESET}")

                # ── P ──
                rejected = False
                self.hooks.trigger(HookPoint.BEFORE_PLAN, state=state, assessment=assessment)
                while True:
                    plan = await self.protocol.plan_async(state, round_cost, assessment)
                    self.hooks.trigger(HookPoint.AFTER_PLAN, state=state, plan=plan)
                    gd = self._check_gate("after_plan", plan=plan, state=state)
                    if gd.action == GateAction.REJECT:
                        rejected = True; break
                    if gd.should_retry:
                        state.task_instruction += f"\n\n[修订意见]: {gd.feedback}"
                        continue
                    if gd.action == GateAction.OVERRIDE and gd.overrides:
                        for k, v in gd.overrides.items():
                            state.task_params[k] = v
                    break
                if rejected:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — P 阶段{_RESET}")
                    break
                self._log(f"  {_GREEN}✓ P: {len(plan.agent_analyses)} 个角色, {len(plan.conflicts)} 个冲突{_RESET}")

                if self._detect_mass_llm_failure(plan.agent_analyses):
                    error_count = sum(1 for v in plan.agent_analyses.values() if str(v).startswith("[LLM_ERROR"))
                    self._log(f"  {_RED}⏹ 检测到大规模 LLM 调用失败（{error_count}/{len(plan.agent_analyses)} 个智能体返回错误），中止{_RESET}")
                    final_act = ActOutput(
                        verdict="error", decisions=[], alarm_flags=[],
                        should_continue=False, reason="大规模 LLM 调用失败 — 超过半数智能体返回错误",
                    )
                    round_reports.append(self._build_round_report(state, plan, DoOutput(tool_results={}, abnormal_results=[]),
                                           CheckOutput(reviews={}, devil_advocate="", alarm_flags=[]), final_act))
                    break

                # ── D ──
                self.hooks.trigger(HookPoint.BEFORE_DO, state=state, plan=plan)
                while True:
                    do_output = await self.protocol.do_async(plan, state, round_cost)
                    self.hooks.trigger(HookPoint.AFTER_DO, state=state, do_output=do_output)
                    gd = self._check_gate("after_do", do_output=do_output, plan=plan, state=state)
                    if gd.action == GateAction.REJECT:
                        rejected = True; break
                    if gd.should_retry:
                        continue
                    if gd.action == GateAction.OVERRIDE and gd.overrides:
                        for k, v in gd.overrides.items():
                            state.task_params[k] = v
                    break
                if rejected:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — D 阶段{_RESET}")
                    break

                # ── C ──
                self.hooks.trigger(HookPoint.BEFORE_CHECK, state=state, plan=plan, do_output=do_output)
                while True:
                    check = await self.protocol.check_async(state, plan, do_output, round_cost, assessment)
                    self.hooks.trigger(HookPoint.AFTER_CHECK, state=state, check=check)
                    gd = self._check_gate("after_check", check=check, state=state)
                    if gd.action == GateAction.REJECT:
                        rejected = True; break
                    if gd.should_retry:
                        continue
                    if gd.action == GateAction.OVERRIDE and gd.overrides:
                        for k, v in gd.overrides.items():
                            state.task_params[k] = v
                    break
                if rejected:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — C 阶段{_RESET}")
                    break
                self._log(f"  {_GREEN}✓ C: 数据审查完成{_RESET}")

                # ── A ──
                self.hooks.trigger(HookPoint.BEFORE_ACT, state=state, check=check)
                act = self.protocol.act(state, plan, check, round_cost, prev_state, do_output, assessment)
                final_act = act
                self.hooks.trigger(HookPoint.AFTER_ACT, state=state, act=act)
                gd = self._check_gate("after_act", act=act, state=state)
                if gd.action == GateAction.REJECT:
                    self._log(f"  {_RED}⏹ Gate 拒绝 — A 阶段{_RESET}")
                    break
                if gd.should_retry and gd.feedback:
                    state.task_instruction += f"\n\n[修订意见]: {gd.feedback}"
                self._log(f"  {_YELLOW}✓ A: {act.verdict} — {act.reason}{_RESET}")

                self.state_mgr.save_round(state, round_dir)
                round_reports.append(self._build_round_report(state, plan, do_output, check, act))

                # 自动 checkpoint
                self.state_mgr.checkpoint(state, output_path / "checkpoint.json")

                self.hooks.trigger(HookPoint.ON_ROUND_END, state=state, round_num=round_num, act=act)

                self.total_cost += round_cost.estimated_cost_rmb
                self._print_round_status(round_num, round_cost, state, act)

                if not act.should_continue:
                    self._log(f"\n  {_GREEN}{_BOLD}⬢ 停止: {act.reason}{_RESET}")
                    break

                if self.interactive:
                    cmd = input(f"\n  [{_BOLD}继续{_RESET}/{_DIM}停车{_RESET}/{_DIM}查看{_RESET}]: ").strip().lower()
                    if cmd in ("停车", "stop", "s"):
                        self._log(f"  {_YELLOW}Founder 决定停车{_RESET}")
                        break
                    elif cmd in ("查看", "view", "v"):
                        self._print_detailed_status(state)
                        cmd2 = input(f"\n  [{_BOLD}继续{_RESET}/{_DIM}停车{_RESET}]: ").strip().lower()
                        if cmd2 in ("停车", "stop", "s"):
                            break

                prev_state = state
        except Exception as e:
            # 异常时保存 checkpoint 以便恢复
            self.state_mgr.checkpoint(state, output_path / "checkpoint.json")
            self.hooks.trigger(HookPoint.ON_ERROR, error=e, instruction=instruction, state=state)
            raise

        self._print_final_report(state, round_reports, final_act, output_path)
        result = self._build_result(instruction, state, round_reports, final_act, output_path)

        # 跨会话 Memory
        if self.config.memory_db_path:
            try:
                memory = MemoryStore(self.config.memory_db_path)
                memory.save_session(state, result)
                memory.close()
            except Exception:
                pass

        self.hooks.trigger(HookPoint.ON_COMPLETE, instruction=instruction, state=state, result=result)
        return result

    # ── Tau Mode ───────────────────────────────────────────────────

    def _run_tau(self, instruction: str, output_dir: str | Path | None,
                 resume_from: Path | None = None) -> dict:
        """Execute instruction using Tau hierarchical decomposition mode."""

        # Build agent/tool dicts compatible with TauOrchestrator
        agents = self.protocol.agents.copy()
        tools = self.protocol.tools.copy()

        # Get LLM callable from the backend if available
        def llm_call(system_prompt: str, user_prompt: str) -> str:
            if self.protocol.llm:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return self.protocol.llm.chat(
                    messages=messages,
                    model=self.protocol.model,
                    max_tokens=self.config.default_max_tokens,
                    temperature=self.config.default_temperature,
                ).content
            return "[LLM UNAVAILABLE]"

        tau = TauOrchestrator(
            agents=agents,
            tools=tools,
            llm_call=llm_call,
            max_iterations=self.max_rounds,
            verbose=self.verbose,
            cost_tracker=self.cost_tracker,
            skills=self.protocol.skill_cache.copy(),
        )

        state = tau.run(instruction)
        result = self._convert_tau_result(state, instruction, output_dir)
        result["cost_summary"] = state.cost_summary or self.cost_tracker.total_summary()
        return result

    def _convert_tau_result(self, state, instruction: str,
                            output_dir: str | Path | None) -> dict:
        """Convert TauState to Sigma-compatible result dict."""
        params: dict[str, float] = {}
        for r in state.subtask_results.values():
            params.update(r.interface_params)

        total_conflicts = sum(len(cr.conflicts) for cr in state.conflict_history)
        total_resolutions = len(state.resolution_history)

        tau_trace = {
            "mode": "tau",
            "subtask_count": len(state.task_graph.subtasks) if state.task_graph else 0,
            "total_iterations": state.iteration,
            "total_conflicts": total_conflicts,
            "total_resolutions": total_resolutions,
            "conflict_history": [
                {
                    "iteration": i + 1,
                    "conflicts": len(cr.conflicts),
                    "max_severity": cr.max_severity,
                    "resolved_params": len(cr.resolved_params),
                }
                for i, cr in enumerate(state.conflict_history)
            ],
            "subtask_breakdown": [
                {
                    "id": st.id,
                    "description": st.description,
                    "assigned_agents": st.assigned_agents,
                    "success": state.subtask_results[st.id].success if st.id in state.subtask_results else False,
                }
                for st in (state.task_graph.subtasks if state.task_graph else [])
            ],
        }

        return {
            "instruction": instruction or state.instruction,
            "framework": "Sigma/Tau",
            "tau_mode": True,
            "timestamp": datetime.now().isoformat(),
            "total_rounds": state.iteration,
            "final_verdict": state.final_verdict,
            "completed": state.completed,
            "parameters": params,
            "tau_trace": tau_trace,
            "decisions": [
                r.director_decision for r in state.resolution_history
                if r.director_decision
            ],
            "alarm_flags": [],
            "consensus": [
                {
                    "parameter": k,
                    "recommended": v,
                    "confidence": "MEDIUM",
                    "unit": "",
                    "individual": {k: v},
                }
                for k, v in params.items()
            ],
            "cost_summary": "",
            "output_dir": str(output_dir) if output_dir else "",
        }

    def _print_banner(self) -> None:
        project = self.config.project_name or "Sigma"
        self._log(f"\n{_YELLOW}{_BOLD}  {project} — Σ (Sigma) AERC 多智能体协同框架{_RESET}")
        self._log(f"{_DIM}  底线质量 · 极致效率 · 可控成本{_RESET}")

    def _print_round_status(
        self, round_num: int, cost: RoundCost, state: SharedState, act: ActOutput,
    ) -> None:
        verdict_colors = {
            "converged": _GREEN, "converging": _CYAN,
            "slow": _YELLOW, "oscillating": _RED, "stalled": _RED,
        }
        vc = verdict_colors.get(act.verdict, _DIM)
        tier_tag = f" [{state.complexity_tier.upper()}]" if state.complexity_tier else ""
        self._log(f"\n{vc}{_BOLD}███ Round {round_num}/{state.max_rounds} | {act.verdict.upper()}{tier_tag} ███{_RESET}")
        self._log(f"  {_DIM}{self.cost_tracker.round_summary(cost)}{_RESET}")

        if state.history and len(state.history) >= 2:
            prev_hist = state.history[-2]
            prev_params = {}
            if hasattr(prev_hist, 'phase_outputs'):
                prev_params = prev_hist.phase_outputs.get("do", {}).get("tool_results", {})
            for key, val in list(state.task_params.items())[:5]:
                self._log(f"  {_DIM}{key}: {val}{_RESET}")

        if act.consensus:
            for ce in act.consensus:
                tag = f"[{ce.confidence}]" if ce.confidence == "LOW" else ""
                self._log(
                    f"  {_YELLOW}├ 共识: {ce.parameter} = {ce.recommended:.1f} {ce.unit} "
                    f"[{ce.min_val:.1f}–{ce.max_val:.1f}] {tag}{_RESET}"
                )

    def _print_detailed_status(self, state: SharedState) -> None:
        self._log(f"\n{_CYAN}{'─'*60}{_RESET}")
        self._log(state.to_context())
        if state.history:
            record = state.history[-1]
            if record.tool_results:
                self._log(f"\n{_YELLOW}工具结果:{_RESET}")
                for tn, tr in record.tool_results.items():
                    perf = tr.get("performance", {}) if isinstance(tr, dict) else {}
                    self._log(f"  {tn}: {json.dumps(perf, ensure_ascii=False)}")
        self._log(f"{_CYAN}{'─'*60}{_RESET}")

    def _build_round_report(
        self, state: SharedState, plan: PlanOutput,
        do_output: DoOutput, check: CheckOutput, act: ActOutput,
    ) -> str:
        lines = [f"# Round {state.round_num} 报告"]
        lines.append(f"\n## 状态: {act.verdict}")
        lines.append(f"\n{act.reason}")

        lines.append("\n## 智能体分析摘要")
        for name, analysis in plan.agent_analyses.items():
            lines.append(f"\n### {name}")
            lines.append(str(analysis)[:600])

        if do_output.tool_results:
            lines.append("\n## 工具计算结果")
            for tn, tr in do_output.tool_results.items():
                lines.append(f"\n### {tn}")
                lines.append(json.dumps(tr, indent=2, ensure_ascii=False))

        if check.devil_advocate:
            lines.append("\n## 魔鬼代言人审查")
            lines.append(check.devil_advocate[:800])

        if act.decisions:
            lines.append("\n## 本轮决策")
            for d in act.decisions:
                lines.append(f"- {d.decision}")

        if act.alarm_flags:
            lines.append("\n## 告警")
            for a in act.alarm_flags:
                lines.append(f"- [{a.flag_type}] {a.message}")

        if act.consensus:
            lines.append("\n## 多角色共识估算")
            for ce in act.consensus:
                lines.append(f"\n### {ce.parameter} [{ce.confidence}]")
                lines.append(f"- 推荐值: **{ce.recommended} {ce.unit}**")
                lines.append(f"- 范围: {ce.min_val}–{ce.max_val} {ce.unit}")
                lines.append(f"- 各角色估算:")
                for name, est in ce.individual.items():
                    lines.append(
                        f"  - {name}: {est['value']} {ce.unit} "
                        f"(置信度: {est.get('confidence', '?')})"
                    )
                lines.append(f"- 依据: {ce.basis[:300]}")

        return "\n".join(lines)

    def _print_final_report(
        self, state: SharedState, round_reports: list[str],
        act: Optional[ActOutput], output_path: Path,
    ) -> None:
        self._log(f"\n{_GREEN}{_BOLD}{'═'*60}{_RESET}")
        self._log(f"{_GREEN}{_BOLD}  ✅ 任务完成{_RESET}")
        self._log(f"{_GREEN}{'═'*60}{_RESET}")

        report = self._build_final_markdown(state, round_reports, act)
        report_path = output_path / "REPORT.md"
        report_path.write_text(report, encoding="utf-8")
        self._log(f"  {_GREEN}📂 完整报告:{_RESET} {report_path}")

        result = self._build_result("", state, round_reports, act, output_path)
        result_path = output_path / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        self._log(f"  {_DIM}📋 结构化数据:{_RESET} {result_path}")

        cost_summary = self.cost_tracker.total_summary()
        self._log(f"  {_DIM}💰 {cost_summary}{_RESET}")

    def _build_final_markdown(
        self, state: SharedState, round_reports: list[str],
        act: Optional[ActOutput],
    ) -> str:
        project = self.config.project_name or "Sigma"
        lines = [
            f"# {project} — Σ 框架执行报告",
            "",
            f"**任务指令**: {state.task_instruction}",
            f"**复杂度分层**: {state.complexity_tier.upper()} (评分 {state.complexity_assessment.get('score', 0):.1f}/10)",
            f"**执行时间**: {datetime.now().isoformat()}",
            f"**总轮次**: {state.round_num}",
            "",
            "---",
        ]

        for i, report in enumerate(round_reports):
            lines.append(report)
            lines.append("\n---\n")

        lines.append(f"\n## 成本摘要\n{self.cost_tracker.total_summary()}")

        if state.alarm_flags:
            lines.append("\n## 未解决告警")
            for a in state.alarm_flags:
                if not a.resolved:
                    lines.append(f"- [{a.flag_type}] {a.message}")

        return "\n".join(lines)

    def _build_result(
        self, instruction: str, state: SharedState,
        round_reports: list[str], act: Optional[ActOutput],
        output_path: Path,
    ) -> dict:
        consensus_data = []
        if act and act.consensus:
            for ce in act.consensus:
                consensus_data.append({
                    "parameter": ce.parameter,
                    "recommended": ce.recommended,
                    "min": ce.min_val,
                    "max": ce.max_val,
                    "confidence": ce.confidence,
                    "unit": ce.unit,
                    "individual": ce.individual,
                })
        agent_analyses = {}
        if state.history:
            agent_analyses = state.history[-1].agent_analyses
        return {
            "instruction": instruction or state.task_instruction,
            "framework": "Sigma AERC",
            "timestamp": datetime.now().isoformat(),
            "total_rounds": state.round_num,
            "final_verdict": act.verdict if act else "unknown",
            "parameters": state.task_params,
            "decisions": [d.decision for d in state.decisions],
            "alarm_flags": [a.message for a in state.alarm_flags if not a.resolved],
            "consensus": consensus_data,
            "agent_analyses": agent_analyses,
            "cost_summary": state.cost_summary,
            "output_dir": str(output_path),
        }

    def _detect_mass_llm_failure(self, analyses: dict[str, str]) -> bool:
        """检测是否所有/大多数智能体分析都是 LLM 调用失败的产物."""
        if not analyses:
            return True
        error_count = sum(
            1 for v in analyses.values()
            if str(v).startswith("[LLM_ERROR") or str(v).startswith("[ERROR")
        )
        return error_count >= len(analyses) or error_count >= max(2, len(analyses) * 0.5)

    def _log(self, msg: str) -> None:
        if self.verbose:
            from sigma.log import get_logger
            get_logger("sigma.orchestrator").info(msg)


def _next_version(base_dir: Path) -> str:
    """自动确定下一个版本编号."""
    base_dir.mkdir(exist_ok=True)
    existing = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("v"):
            try:
                existing.append(int(entry.name[1:]))
            except ValueError:
                pass
    return f"v{max(existing, default=0) + 1}"
