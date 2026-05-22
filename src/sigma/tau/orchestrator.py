"""TauOrchestrator — top-level Tau-led hierarchical execution.

Flow:
  Decompose → Execute → Detect → Resolve → (Iterate if unresolved)

Tau mirrors a real engineering organization:
  - Tau (总负责人) decomposes the task and assigns subtasks to departments
  - Departments execute independently with interface "contracts"
  - Conflicts at interfaces are resolved with graduated escalation
  - Tau makes the final call when departments can't agree
"""

from sigma.tau.types import (
    SubTask, TaskGraph, SubtaskResult, InterfaceConflict,
    ConflictReport, ResolutionResult, TauState,
)
from sigma.tau.config import TauConfig
from sigma.tau.decomposer import TauDecomposer
from sigma.tau.executor import IndependentExecutor, ExecutorConfig
from sigma.tau.detector import InterfaceConflictDetector
from sigma.tau.resolver import TauResolver
from sigma.tau.hooks import TauHookSystem, TauHookPoint
from sigma.tau.capability import CapabilityRegistry
from sigma.cost_tracker import CostTracker, RoundCost, DEEPSEEK_PRICING
from sigma.learning import LearningStore, ExecutionRecord, record_from_tau_state
from sigma.observability import _auto_init, tracing_span, get_tracer, is_initialized


class TauOrchestrator:
    """Orchestrates Tau-led hierarchical task execution.

    Decompose → Execute → Detect → Resolve → Iterate
    """

    def __init__(
        self,
        agents: dict,       # name → AgentSpec
        tools: dict,         # name → ToolSpec
        llm_call,            # callable(system, user) -> str
        max_iterations: int = 5,
        verbose: bool = True,
        cost_tracker: CostTracker | None = None,
        skills: dict[str, str] | None = None,
        hooks: TauHookSystem | None = None,
        config: TauConfig | None = None,
        capabilities: CapabilityRegistry | None = None,
        learning_store: LearningStore | None = None,
    ):
        self.agents = agents
        self.tools = tools
        self.llm_call = llm_call
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.cost_tracker = cost_tracker
        self.skills = skills or {}
        self.hooks = hooks or TauHookSystem()
        self.config = config or TauConfig()
        self.capabilities = capabilities
        self.learning_store = learning_store

        self.decomposer = TauDecomposer(config=self.config)
        self.executor = IndependentExecutor(
            config=ExecutorConfig(max_workers=self.config.executor_max_workers,
                                  timeout_per_subtask=self.config.executor_timeout_per_subtask),
            guardrails=self.config.guardrails,
        )
        self.detector = InterfaceConflictDetector(threshold=self.config.detection_threshold)
        self.resolver = TauResolver(
            agents=agents, llm_call=self._tracked_call, verbose=verbose,
            skills=self.skills, config=self.config,
        )

    def _tracked_call(self, system: str, user: str) -> str:
        """LLM call with cost tracking."""
        if self.cost_tracker:
            inputs = self._estimate_tokens(system) + self._estimate_tokens(user)
            result = self.llm_call(system, user)
            outputs = self._estimate_tokens(result)
            self.cost_tracker.record_call(
                self._round_cost, inputs, outputs,
            )
            return result
        return self.llm_call(system, user)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Crude token estimation. ~3.5 chars/token for mixed Chinese-English text."""
        if not text:
            return 0
        return max(1, len(text) // 3)

    def run(self, instruction: str) -> TauState:
        """Execute a high-level instruction using the Tau pattern.

        Returns TauState with full execution trace.
        """
        _auto_init()
        state = TauState(instruction=instruction, max_iterations=self.max_iterations)

        # Start cost tracking round
        if self.cost_tracker:
            self._round_cost = self.cost_tracker.start_round(state.iteration + 1)

        import time
        t0 = time.time()

        self.hooks.fire(TauHookPoint.ON_START, instruction=instruction, state=state)

        trace_attrs = {"instruction": instruction[:120], "max_iterations": self.max_iterations}
        with tracing_span("TauOrchestrator.run", trace_attrs):
            try:
                state = self._run_impl(instruction, state)
            except Exception as e:
                self.hooks.fire(TauHookPoint.ON_ERROR, error=e, state=state,
                               instruction=instruction)
                raise
            finally:
                state.duration_ms = (time.time() - t0) * 1000
                if self.learning_store and state.completed:
                    rec = record_from_tau_state(state, instruction, state.duration_ms)
                    self.learning_store.record(rec)

        return state

    def _run_impl(self, instruction: str, state: TauState) -> TauState:
        """Internal implementation of the Tau execution loop."""
        task_graph = state.task_graph

        # Step 1: Decompose
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Tau: 拆解任务 → {len(self.agents)} 个可用部门")
            print(f"{'='*60}")

        agent_names = list(self.agents.keys())
        self.hooks.fire(TauHookPoint.ON_DECOMPOSE_START, instruction=instruction, agents=agent_names)
        lessons = ""
        if self.learning_store:
            lessons = self.learning_store.lessons_for(instruction)
        with tracing_span("decompose", {"agent_count": len(agent_names)}):
            task_graph = self.decomposer.decompose(
                instruction, agent_names, self._tracked_call,
                skills=self.skills, capabilities=self.capabilities,
                lessons=lessons,
            )
        self.hooks.fire(TauHookPoint.ON_DECOMPOSE_END, task_graph=task_graph)
        state.task_graph = task_graph

        if self.verbose:
            print(f"  拆解完成：{len(task_graph.subtasks)} 个子任务")
            for st in task_graph.subtasks:
                deps = f" (依赖: {', '.join(st.dependencies)})" if st.dependencies else ""
                print(f"    {st.id}: {st.description[:60]}... → {st.assigned_agents}{deps}")

        # Step 2-4: Execute → Detect → Resolve loop
        while state.iteration < state.max_iterations and not state.completed:
            state.iteration += 1

            if self.verbose:
                print(f"\n{'─'*40}")
                print(f"  Round {state.iteration}/{state.max_iterations}")
                print(f"{'─'*40}")

            self.hooks.fire(TauHookPoint.ON_ITERATION_START,
                           iteration=state.iteration, state=state)

            # Execute
            with tracing_span("execute", {"iteration": state.iteration, "subtasks": len(task_graph.subtasks)}):
                if self.verbose:
                    print("  [Execute] 并行执行子任务...")
                for st in task_graph.subtasks:
                    self.hooks.fire(TauHookPoint.ON_SUBTASK_START,
                                   subtask=st, iteration=state.iteration)
                results = self.executor.run_all(
                    task_graph=task_graph,
                    agents=self.agents,
                    tools=self.tools,
                    llm_call=self._tracked_call,
                    verbose=self.verbose,
                )
                state.subtask_results = results
                for st in task_graph.subtasks:
                    r = results.get(st.id)
                    self.hooks.fire(TauHookPoint.ON_SUBTASK_END,
                                   subtask=st, result=r, iteration=state.iteration)
                self.hooks.fire(TauHookPoint.ON_ALL_SUBTASKS_COMPLETE,
                               results=results, iteration=state.iteration)

                # Guardrail report
                if self.verbose:
                    for st in task_graph.subtasks:
                        r = results.get(st.id)
                        if r and r.guardrails_report:
                            gr = r.guardrails_report
                            if not gr.get("blocked", False) and gr.get("summary", "").endswith("PASS"):
                                continue
                            print(f"  [Guard] {st.id}: {gr['summary']}")
                            for item in gr.get("results", []):
                                if item["severity"] != "PASS":
                                    print(f"    {item['severity']}: {item['message']}")

            # Detect conflicts
            with tracing_span("detect", {"iteration": state.iteration}):
                if self.verbose:
                    print("  [Detect] 检测接口冲突...")
                self.hooks.fire(TauHookPoint.ON_DETECT_START,
                               results=results, task_graph=task_graph)
                conflict_report = self.detector.detect(task_graph, results)
                state.conflict_history.append(conflict_report)
                self.hooks.fire(TauHookPoint.ON_DETECT_END,
                               report=conflict_report)

            if not conflict_report.has_conflicts:
                state.completed = True
                if self.verbose:
                    print(f"  [OK] 所有接口参数一致 ({len(conflict_report.resolved_params)} 个参数)")
                break

            if self.verbose:
                print(f"  [WARN] {len(conflict_report.conflicts)} 个冲突:")
                for c in conflict_report.conflicts:
                    print(f"    {c.param_key}: {c.subtask_a}={c.value_a:.2f} vs {c.subtask_b}={c.value_b:.2f} (severity={c.severity:.1f})")

            # Resolve
            with tracing_span("resolve", {"iteration": state.iteration, "conflicts": len(conflict_report.conflicts)}):
                if self.verbose:
                    print("  [Resolve] 分级调解...")
                self.hooks.fire(TauHookPoint.ON_RESOLVE_START,
                               conflict_report=conflict_report, iteration=state.iteration)
                resolution = self.resolver.resolve(
                    conflict_report, results, task_graph, state.iteration,
                )
                state.resolution_history.append(resolution)
                self.hooks.fire(TauHookPoint.ON_RESOLVE_END,
                               resolution=resolution, iteration=state.iteration)

            if self.verbose:
                print(f"  结果: {len(resolution.resolved)} 已解决, {len(resolution.unresolved)} 未解决")
                if resolution.director_decision:
                    print(f"  总监决策: {resolution.director_decision}")

            # Update results with consensus values (immutable pattern)
            for param, value in resolution.consensus_values.items():
                for st in task_graph.subtasks:
                    if param in st.interface_params:
                        r = results.get(st.id)
                        if r:
                            new_params = {**r.interface_params, param: value}
                            results[st.id] = SubtaskResult(
                                subtask_id=r.subtask_id,
                                success=r.success,
                                agent_analyses=r.agent_analyses,
                                tool_results=r.tool_results,
                                interface_params=new_params,
                                param_confidence=r.param_confidence,
                                error=r.error,
                            )

            # If all resolved, done
            if not resolution.unresolved:
                state.completed = True
                break

            if self.verbose:
                print(f"  → 下一轮迭代（更新了 {len(resolution.consensus_values)} 个参数值）")

            self.hooks.fire(TauHookPoint.ON_ITERATION_END,
                           iteration=state.iteration, state=state)

        # Final verdict + cost summary
        if state.completed:
            state.final_verdict = self._build_final_verdict(state)
        else:
            state.final_verdict = (
                f"达到最大迭代次数 ({state.max_iterations})，"
                f"仍有 {len(state.conflict_history[-1].conflicts) if state.conflict_history else '?'} 个未解决的冲突"
            )
        if self.cost_tracker:
            state.cost_summary = self.cost_tracker.total_summary()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Tau 执行完成: {state.final_verdict[:80]}...")
            print(f"{'='*60}")

        self.hooks.fire(TauHookPoint.ON_COMPLETE, state=state)
        return state

    def checkpoint(self, state: TauState, path: str) -> None:
        """Save execution state to a JSON checkpoint file."""
        import json
        data = state.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def resume(
        cls,
        checkpoint_path: str,
        agents: dict,
        tools: dict,
        llm_call,
        max_iterations: int = 5,
        verbose: bool = True,
        cost_tracker: CostTracker | None = None,
        skills: dict[str, str] | None = None,
    ) -> "TauState | None":
        """Resume execution from a checkpoint file.

        Returns the TauState if loaded successfully, None otherwise.
        The caller should re-run from the loaded state.
        """
        import json
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TauState.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _build_final_verdict(self, state: TauState) -> str:
        """Synthesize final verdict from execution results."""
        results = state.subtask_results
        success_count = sum(1 for r in results.values() if r.success)
        total = len(results)

        params_summary = []
        for r in results.values():
            for k, v in r.interface_params.items():
                params_summary.append(f"{k}={v:.2f}")

        verdict = (
            f"任务完成：{success_count}/{total} 个子任务成功。"
            f"迭代 {state.iteration} 轮。"
        )
        if params_summary:
            verdict += f" 关键参数：{', '.join(params_summary[:10])}"
        if state.resolution_history:
            last = state.resolution_history[-1]
            if last.director_decision:
                verdict += f" 总监决策：{last.director_decision}"

        return verdict
