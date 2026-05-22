"""
Human-in-the-loop gate — optional Founder approval at AERC phase transitions.

Insert at 4 AERC boundaries: AFTER_PLAN, AFTER_DO, AFTER_CHECK, AFTER_ACT.
Founder can APPROVE (continue), REVISE (re-run phase with feedback),
OVERRIDE (modify output), or REJECT (stop).

Enabled via SigmaConfig.enable_human_gate. Works interactively (input()),
or programmatically via a callback (human_gate_callback).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class GateAction(Enum):
    APPROVE = "approve"
    REVISE = "revise"
    OVERRIDE = "override"
    REJECT = "reject"


@dataclass
class GateDecision:
    action: GateAction
    feedback: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def should_continue(self) -> bool:
        return self.action != GateAction.REJECT

    @property
    def should_retry(self) -> bool:
        return self.action == GateAction.REVISE


class HumanGate:
    """Optional gate at each AERC phase transition.

    Usage:
        gate = HumanGate(interactive=True)
        decision = gate.check("after_plan", plan=plan, state=state)
        if decision.action == GateAction.REVISE:
            # Re-run plan phase with decision.feedback
            ...
    """

    PHASES = ("after_plan", "after_do", "after_check", "after_act")
    PHASE_LABELS = {
        "after_plan": "P→D (Analyze→Execute)",
        "after_do": "D→C (Execute→Review)",
        "after_check": "C→A (Review→Converge)",
        "after_act": "A→Next (Converge→Iterate)",
    }

    def __init__(
        self,
        interactive: bool = True,
        callback: Callable[[str, dict[str, Any]], GateDecision] | None = None,
        verbose: bool = True,
    ):
        self.interactive = interactive
        self.callback = callback
        self.verbose = verbose

    def check(self, phase: str, **context) -> GateDecision:
        """Present phase output to Founder and collect the gate decision.

        Args:
            phase: One of 'after_plan', 'after_do', 'after_check', 'after_act'.
            **context: Phase-specific data (plan, do_output, check, act, state).

        Returns:
            GateDecision with the Founder's action.
        """
        if phase not in self.PHASES:
            return GateDecision(action=GateAction.APPROVE, feedback="unknown phase")

        label = self.PHASE_LABELS.get(phase, phase)

        if self.callback:
            return self.callback(phase, context)

        if self.interactive:
            return self._interactive_check(phase, label, context)

        return GateDecision(action=GateAction.APPROVE)

    def _interactive_check(self, phase: str, label: str, context: dict) -> GateDecision:
        self._print_context(phase, label, context)

        while True:
            try:
                cmd = input(
                    f"\n  [HumanGate {label}]\n"
                    f"  [{_BOLD}A{_RESET}]pprove  "
                    f"[{_BOLD}R{_RESET}]evise  "
                    f"[{_BOLD}O{_RESET}]verride  "
                    f"[{_BOLD}X{_RESET}]eject\n"
                    f"  > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return GateDecision(action=GateAction.REJECT, feedback="Founder interrupted")

            if cmd in ("a", "approve", ""):
                return GateDecision(action=GateAction.APPROVE)
            elif cmd in ("r", "revise"):
                feedback = self._read_multiline("修订意见")
                return GateDecision(action=GateAction.REVISE, feedback=feedback)
            elif cmd in ("o", "override"):
                feedback = self._read_multiline("覆盖说明")
                overrides = self._read_overrides()
                return GateDecision(action=GateAction.OVERRIDE, feedback=feedback, overrides=overrides)
            elif cmd in ("x", "reject"):
                reason = input("  停止原因: ").strip()
                return GateDecision(action=GateAction.REJECT, feedback=reason or "Founder rejected")
            else:
                print(f"  无效输入 '{cmd}'，请重试 (A/R/O/X)")

    def _print_context(self, phase: str, label: str, context: dict) -> None:
        print(f"\n{_CYAN}{'─'*50}{_RESET}")
        print(f"{_YELLOW}  HumanGate: {label}{_RESET}")

        if phase == "after_plan":
            plan = context.get("plan")
            if plan:
                agents = list(plan.agent_analyses.keys()) if hasattr(plan, "agent_analyses") else []
                conflicts = len(plan.conflicts) if hasattr(plan, "conflicts") else 0
                tools = len(plan.tool_requests) if hasattr(plan, "tool_requests") else 0
                print(f"  角色: {', '.join(agents[:5])}{'...' if len(agents) > 5 else ''}")
                print(f"  冲突: {conflicts}  工具请求: {tools}")

        elif phase == "after_do":
            do_out = context.get("do_output")
            if do_out:
                tools = list(do_out.tool_results.keys()) if hasattr(do_out, "tool_results") else []
                for t in tools:
                    r = do_out.tool_results[t]
                    if isinstance(r, dict):
                        ok = r.get("success", "?")
                        print(f"  {t}: success={ok}")

        elif phase == "after_check":
            check = context.get("check")
            if check:
                dev = check.devil_advocate if hasattr(check, "devil_advocate") else ""
                if dev:
                    print(f"  魔鬼代言人: {dev[:200]}...")
                data_devs = len(check.data_deviations) if hasattr(check, "data_deviations") else 0
                print(f"  数据偏差: {data_devs}")

        elif phase == "after_act":
            act = context.get("act")
            state = context.get("state")
            if act:
                print(f"  裁决: {act.verdict.upper() if hasattr(act, 'verdict') else '?'}")
                if hasattr(act, "consensus") and act.consensus:
                    for ce in act.consensus:
                        print(f"  {ce.parameter}: {ce.recommended} {ce.unit} [{ce.confidence}]")
            if state:
                print(f"  Round: {state.round_num}/{state.max_rounds}")

        print(f"{_CYAN}{'─'*50}{_RESET}")

    def _read_multiline(self, prompt: str) -> str:
        print(f"  {prompt} (空行结束):")
        lines = []
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        return "\n".join(lines)

    def _read_overrides(self) -> dict[str, Any]:
        print("  覆盖参数 (key=value, 空行结束):")
        overrides = {}
        while True:
            try:
                line = input("    ").strip()
                if not line:
                    break
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        overrides[k] = float(v)
                    except ValueError:
                        overrides[k] = v
            except (EOFError, KeyboardInterrupt):
                break
        return overrides


# ANSI helpers
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
