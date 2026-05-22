"""TauResolver — graduated conflict resolution for Tau-led execution.

Three escalation levels mirror real organizations:
  Level 1 (DIRECT)  — conflicting agents discuss openly, reconcile differences
  Level 2 (SIGMA)   — focused AERC blind review with relevant agents
  Level 3 (DIRECTOR) — Director makes directional decision, records for tracking

Escalation logic:
  severity < 2.0  → DIRECT discussion first
  severity 2.0-5.0 → SIGMA AERC immediately
  severity > 5.0  → DIRECTOR decision immediately
  DIRECT fails     → escalate to SIGMA
  SIGMA fails      → escalate to DIRECTOR
  iteration >= 3   → skip to DIRECTOR decision
"""

from sigma.tau.types import (
    SubtaskResult, InterfaceConflict, ConflictReport, ResolutionResult,
    TaskGraph,
)
from sigma.tau.config import TauConfig
from sigma.log import get_logger

_log = get_logger("sigma.tau.resolver")


class TauResolver:
    """Resolves interface conflicts with graduated escalation."""

    def __init__(self, agents: dict, llm_call, verbose: bool = True,
                 skills: dict[str, str] | None = None,
                 config: TauConfig | None = None):
        """
        Args:
            agents: dict[name, AgentSpec] — all available agents
            llm_call: callable(system_prompt, user_prompt) -> str
            verbose: whether to log progress
            skills: optional skill_name → skill_content for domain knowledge injection
            config: TauConfig (thresholds, prompts, etc.)
        """
        self.agents = agents
        self.llm_call = llm_call
        self.verbose = verbose
        self.skills = skills or {}
        self.config = config or TauConfig()

    def resolve(
        self,
        conflict_report: ConflictReport,
        results: dict[str, SubtaskResult],
        task_graph: TaskGraph,
        iteration: int = 0,
    ) -> ResolutionResult:
        """Main entry point. Routes conflicts to appropriate escalation level.

        Returns ResolutionResult with resolved params, unresolved params,
        consensus values, and director decisions.
        """
        if not conflict_report.has_conflicts:
            return ResolutionResult(
                resolved=list(conflict_report.resolved_params),
                unresolved=[],
                round_count=0,
            )

        resolved: list[str] = list(conflict_report.resolved_params)
        unresolved: list[str] = []
        consensus_values: dict[str, float] = {}
        director_decisions: list[str] = []
        involved: set[str] = set()
        total_rounds = 0

        for conflict in conflict_report.conflicts:
            involved.add(conflict.subtask_a)
            involved.add(conflict.subtask_b)

            escalation = self._pick_level(conflict, iteration)
            if self.verbose:
                print(f"  [Resolver] {conflict.param_key}: severity={conflict.severity:.1f} → {escalation}")

            if escalation == "DIRECT":
                result = self._resolve_direct(conflict, results, task_graph)
                total_rounds += 1
                if result["resolved"]:
                    resolved.append(conflict.param_key)
                    consensus_values[conflict.param_key] = result["value"]
                    continue
                # DIRECT failed → escalate to SIGMA
                if self.verbose:
                    print(f"  [Resolver] {conflict.param_key}: DIRECT failed → SIGMA")
                result = self._resolve_sigma(conflict, results, task_graph)
                total_rounds += 1
                if result["resolved"]:
                    resolved.append(conflict.param_key)
                    consensus_values[conflict.param_key] = result["value"]
                    continue
                # SIGMA failed → escalate to DIRECTOR
                if self.verbose:
                    print(f"  [Resolver] {conflict.param_key}: SIGMA failed → DIRECTOR")

            elif escalation == "SIGMA":
                result = self._resolve_sigma(conflict, results, task_graph)
                total_rounds += 1
                if result["resolved"]:
                    resolved.append(conflict.param_key)
                    consensus_values[conflict.param_key] = result["value"]
                    continue
                # SIGMA failed → escalate to DIRECTOR
                if self.verbose:
                    print(f"  [Resolver] {conflict.param_key}: SIGMA failed → DIRECTOR")

            # Level 3: DIRECTOR decision
            decision = self._resolve_director_decision(conflict, results, task_graph)
            total_rounds += 1
            consensus_values[conflict.param_key] = decision["value"]
            director_decisions.append(
                f"{conflict.param_key}: {decision['value']} — {decision['rationale']}"
            )
            resolved.append(conflict.param_key)

        return ResolutionResult(
            resolved=resolved,
            unresolved=unresolved,
            consensus_values=consensus_values,
            director_decision="; ".join(director_decisions) if director_decisions else "",
            round_count=total_rounds,
            involved_agents=sorted(involved),
        )

    def _pick_level(self, conflict: InterfaceConflict, iteration: int) -> str:
        """Determine escalation level based on severity and iteration count."""
        if iteration >= self.config.max_light_iter:
            if conflict.severity > self.config.sigma_threshold:
                return "DIRECTOR"
            return "SIGMA"

        if conflict.severity < self.config.light_threshold:
            return "DIRECT"
        elif conflict.severity < self.config.sigma_threshold:
            return "SIGMA"
        else:
            return "DIRECTOR"

    # ── Level 1: Direct Discussion (Multi-Turn) ─────────────────────

    def _resolve_direct(
        self,
        conflict: InterfaceConflict,
        results: dict[str, SubtaskResult],
        task_graph: TaskGraph,
    ) -> dict:
        """Level 1: conflicting agents discuss freely for up to N rounds.

        Each agent sees the other's position and responds. If either signals
        accept_avg or the relative difference drops below the convergence
        threshold, the conflict is resolved.
        """
        r_a = results.get(conflict.subtask_a)
        r_b = results.get(conflict.subtask_b)
        if not r_a or not r_b:
            return {"resolved": False, "value": 0.0}

        agent_a = self._pick_agent_for(conflict.subtask_a, results)
        agent_b = self._pick_agent_for(conflict.subtask_b, results)

        if not agent_a or not agent_b:
            return self._resolve_direct_coordinator(conflict, results, task_graph)

        value_a = conflict.value_a
        value_b = conflict.value_b
        msg_a = self._get_agent_analysis(conflict.subtask_a, agent_a, results)
        msg_b = self._get_agent_analysis(conflict.subtask_b, agent_b, results)

        for round_num in range(self.config.direct_discussion_rounds):
            # Agent A responds to B
            turn_a = self._discuss_turn(
                agent_a, agent_b, conflict,
                value_a, value_b, msg_b,
            )
            if turn_a is None:
                break
            value_a = turn_a["value"]
            msg_a = turn_a["message"]
            if turn_a.get("accept_avg"):
                return self._consensus_result(value_a, value_b)

            # Agent B responds to A
            turn_b = self._discuss_turn(
                agent_b, agent_a, conflict,
                value_b, value_a, msg_a,
            )
            if turn_b is None:
                break
            value_b = turn_b["value"]
            msg_b = turn_b["message"]
            if turn_b.get("accept_avg"):
                return self._consensus_result(value_a, value_b)

            # Check convergence
            avg = (value_a + value_b) / 2
            if abs(avg) > 0.001:
                if abs(value_a - value_b) / abs(avg) < self.config.direct_convergence_ratio:
                    return self._consensus_result(value_a, value_b)

        # Relaxed check after max rounds
        avg = (value_a + value_b) / 2
        if abs(avg) > 0.001:
            if abs(value_a - value_b) / abs(avg) < self.config.direct_convergence_ratio * 1.5:
                return self._consensus_result(value_a, value_b)

        return {"resolved": False, "value": 0.0}

    def _discuss_turn(
        self,
        my_name: str,
        other_name: str,
        conflict: InterfaceConflict,
        my_value: float,
        other_value: float,
        other_message: str,
    ) -> dict | None:
        """One turn of direct discussion. Agent sees the other's estimate
        and responds with an updated value, message, and accept_avg flag.
        """
        agent = self.agents.get(my_name)
        if not agent:
            return None

        system = self.config.format_prompt(
            self.config.direct_agent_discussion_prompt,
            param_key=conflict.param_key,
            my_role=self._agent_role(my_name),
            other_role=self._agent_role(other_name),
            my_value=f"{my_value:.4g}",
            other_value=f"{other_value:.4g}",
            other_message=other_message[:800],
        )

        user = (
            f"参数：{conflict.param_key}\n"
            f"你的当前估算：{my_value:.4g}\n"
            f"对方估算：{other_value:.4g}\n"
            f"对方意见：{other_message[:600]}\n\n"
            "请基于对方的分析调整你的估算，或坚持并说明理由。"
        )

        try:
            resp = self.llm_call(system, user)
            data = self._parse_json(resp)
            if data and "value" in data:
                return {
                    "value": float(data["value"]),
                    "message": data.get("message", ""),
                    "accept_avg": data.get("accept_avg", False),
                }
        except Exception as e:
            _log.debug("Discuss turn failed for %s: %s", my_name, e)

        return None

    def _resolve_direct_coordinator(
        self,
        conflict: InterfaceConflict,
        results: dict[str, SubtaskResult],
        task_graph: TaskGraph,
    ) -> dict:
        """Fallback: neutral coordinator resolves without agent dialogue.

        Used when no agent specs are available for the conflicting subtasks.
        """
        r_a = results.get(conflict.subtask_a)
        r_b = results.get(conflict.subtask_b)
        if not r_a or not r_b:
            return {"resolved": False, "value": 0.0}

        analyses_a = r_a.agent_analyses
        analyses_b = r_b.agent_analyses

        system = self.config.direct_system_prompt

        user = (
            f"参数：{conflict.param_key}\n"
            f"部门A（{conflict.subtask_a}）的估算值：{conflict.value_a}\n"
            f"部门A的分析：\n"
        )
        for aname, analysis in analyses_a.items():
            user += f"  [{aname}]: {analysis[:800]}\n"
        user += (
            f"\n部门B（{conflict.subtask_b}）的估算值：{conflict.value_b}\n"
            f"部门B的分析：\n"
        )
        for aname, analysis in analyses_b.items():
            user += f"  [{aname}]: {analysis[:800]}\n"

        user += (
            f"\n差异：{conflict.relative_diff:.1%}\n"
            "请给出一个折中的共识值，并简要说明理由。"
        )

        try:
            resp = self.llm_call(system, user)
            data = self._parse_json(resp)
            if data and "value" in data:
                value = float(data["value"])
                avg = (conflict.value_a + conflict.value_b) / 2
                if abs(value - avg) / max(abs(avg), 0.001) < 0.5:
                    return {"resolved": True, "value": value}
        except Exception as e:
            _log.debug("DIRECT coordinator failed for %s: %s", conflict.param_key, e)

        return {"resolved": False, "value": 0.0}

    def _pick_agent_for(
        self,
        subtask_id: str,
        results: dict[str, SubtaskResult],
    ) -> str | None:
        """Pick the first available agent for a subtask from its results."""
        r = results.get(subtask_id)
        if not r or not r.agent_analyses:
            return None
        for name in r.agent_analyses:
            if name in self.agents:
                return name
        return next(iter(r.agent_analyses.keys()), None)

    def _get_agent_analysis(
        self,
        subtask_id: str,
        agent_name: str,
        results: dict[str, SubtaskResult],
    ) -> str:
        """Get the analysis text from an agent in a subtask result."""
        r = results.get(subtask_id)
        if not r:
            return ""
        return r.agent_analyses.get(agent_name, "")

    def _agent_role(self, agent_name: str) -> str:
        """Extract a short role label from an agent name."""
        agent = self.agents.get(agent_name)
        if agent and getattr(agent, "role", ""):
            return agent.role
        # Fallback: use the last word of the name as role
        return agent_name.split()[-1] if " " in agent_name else agent_name

    def _consensus_result(self, value_a: float, value_b: float) -> dict:
        """Return average as resolved consensus value."""
        return {"resolved": True, "value": (value_a + value_b) / 2}

    # ── Level 2: Sigma AERC Blind Review ───────────────────────────

    def _resolve_sigma(
        self,
        conflict: InterfaceConflict,
        results: dict[str, SubtaskResult],
        task_graph: TaskGraph,
    ) -> dict:
        """Level 2: focused Sigma AERC blind review with relevant agents.

        Agents independently estimate the parameter (blind to each other),
        then cross-review, then converge on consensus.
        """
        # Determine which agents to involve
        involved_agents: set[str] = set()
        for sid in [conflict.subtask_a, conflict.subtask_b]:
            for st in task_graph.subtasks:
                if st.id == sid:
                    involved_agents.update(st.assigned_agents)

        # Also include agents who handle this parameter in other subtasks
        for sid in task_graph.interface_map.get(conflict.param_key, []):
            for st in task_graph.subtasks:
                if st.id == sid:
                    involved_agents.update(st.assigned_agents)

        if len(involved_agents) < 2:
            return {"resolved": False, "value": 0.0}

        # Phase 1: Independent blind estimates
        estimates: dict[str, float] = {}
        reasonings: dict[str, str] = {}
        system_blind = self.config.format_prompt(
            self.config.sigma_blind_system_prompt,
            param_key=conflict.param_key,
        )
        user_blind = (
            f"参数：{conflict.param_key}\n"
            f"上下文：{task_graph.instruction}\n"
            f"已知：部门A估算={conflict.value_a}，部门B估算={conflict.value_b}\n"
            "请独立给出你的估算值。"
        )

        for agent_name in involved_agents:
            agent = self.agents.get(agent_name)
            if not agent:
                continue
            # Inject relevant skill content into agent's system prompt
            system = agent.backstory
            if self.skills:
                skill_context = self._get_skill_context(agent_name)
                if skill_context:
                    system = f"{agent.backstory}\n\n可用专业知识：\n{skill_context}"
            try:
                resp = self.llm_call(system, user_blind)
                data = self._parse_json(resp)
                if data and "value" in data:
                    estimates[agent_name] = float(data["value"])
                    reasonings[agent_name] = data.get("reasoning", "")
            except Exception as e:
                _log.debug("Blind estimate failed for %s: %s", agent_name, e)

        if len(estimates) < 2:
            return {"resolved": False, "value": 0.0}

        # Phase 2: Cross-review
        reviews: dict[str, dict] = {}
        for reviewer in involved_agents:
            if reviewer not in estimates:
                continue
            agent = self.agents.get(reviewer)
            if not agent:
                continue
            others = {k: v for k, v in estimates.items() if k != reviewer}
            review_prompt = self.config.format_prompt(
                self.config.sigma_review_prompt,
                param_key=conflict.param_key,
                estimates="\n".join(f"  {k}: {v}" for k, v in others.items()),
            )
            try:
                resp = self.llm_call(agent.backstory, review_prompt)
                data = self._parse_json(resp)
                if data and "agreed_value" in data:
                    reviews[reviewer] = data
            except Exception as e:
                _log.debug("Cross-review failed for %s: %s", reviewer, e)

        # Phase 3: Compute consensus (median of cross-review agreed values)
        if reviews:
            values = [r["agreed_value"] for r in reviews.values() if "agreed_value" in r]
            if values:
                values.sort()
                median = values[len(values) // 2]
                # Check convergence: range / median < 30%
                value_range = max(values) - min(values)
                if median > 0.001 and value_range / median < self.config.consensus_convergence_ratio:
                    return {"resolved": True, "value": median}

        # Without cross-review consensus, use median of blind estimates
        est_values = sorted(estimates.values())
        median = est_values[len(est_values) // 2]
        est_range = max(est_values) - min(est_values)
        if median > 0.001 and est_range / median < self.config.consensus_convergence_ratio:
            return {"resolved": True, "value": median}

        return {"resolved": False, "value": median}

    # ── Level 3: Director Decision ─────────────────────────────────

    def _resolve_director_decision(
        self,
        conflict: InterfaceConflict,
        results: dict[str, SubtaskResult],
        task_graph: TaskGraph,
    ) -> dict:
        """Level 3: Director reviews all evidence and makes directional decision.

        This is the final escalation — Director picks a direction.
        The decision is recorded for post-mortem tracking.
        """
        system = self.config.director_decision_system_prompt

        user = (
            f"参数：{conflict.param_key}\n"
            f"任务：{task_graph.instruction}\n"
            f"部门A（{conflict.subtask_a}）值：{conflict.value_a}\n"
            f"部门B（{conflict.subtask_b}）值：{conflict.value_b}\n"
            f"分歧严重程度：{conflict.severity:.1f}/10\n\n"
            "两个部门的分析摘要：\n"
        )
        for sid in [conflict.subtask_a, conflict.subtask_b]:
            r = results.get(sid)
            if r:
                for agent_name, analysis in r.agent_analyses.items():
                    user += f"  [{agent_name}]: {analysis[:600]}\n"

        user += (
            "\n作为总监，请做出方向性决策：选择一个值并说明理由。"
            "这个决策将被记录追踪，后续可以根据实际结果调整。"
        )

        try:
            resp = self.llm_call(system, user)
            data = self._parse_json(resp)
            if data and "value" in data:
                return {
                    "value": float(data["value"]),
                    "rationale": data.get("rationale", "Director decision"),
                }
        except Exception as e:
            _log.debug("Director decision failed for %s: %s", conflict.param_key, e)

        # Absolute fallback: take the average
        return {
            "value": (conflict.value_a + conflict.value_b) / 2,
            "rationale": "Fallback: arithmetic mean (LLM unavailable)",
        }

    # ── Helpers ────────────────────────────────────────────────────

    def _get_skill_context(self, agent_name: str) -> str:
        """Get skill content relevant to an agent based on role keyword matching."""
        if not self.skills:
            return ""
        role_keywords = {
            "Propulsion": ["推进", "发动机", "propellant", "thrust", "燃烧", "比冲"],
            "Structures": ["结构", "质量", "强度", "material", "mass", "tube", "铝管"],
            "GNC": ["飞控", "控制", "传感器", "姿态", "stability", "guidance"],
            "Sim": ["仿真", "气动", "弹道", "aerodynamic", "trajectory", "simulation"],
            "Supply": ["采购", "材料", "成本", "supply", "BOM", "procurement"],
            "Safety": ["安全", "法规", "safety", "regulation", "安全距离"],
        }
        relevant = []
        for name, content in self.skills.items():
            for role, keywords in role_keywords.items():
                if role.lower() in agent_name.lower():
                    if any(kw in name or kw.lower() in content[:500].lower() for kw in keywords):
                        summary = content[:400] if len(content) > 400 else content
                        relevant.append(summary)
                        break
        if relevant:
            return "\n---\n".join(relevant[:3])  # Cap at 3 skill snippets
        return ""

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from text."""
        import json
        if not text:
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
