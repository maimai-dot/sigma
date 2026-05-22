"""IndependentExecutor — runs subtasks in parallel with dependency awareness.

Each department works independently on their assigned subtask.
Results include interface parameter values with confidence levels.

Sync: IndependentExecutor (ThreadPoolExecutor)
Async: AsyncIndependentExecutor (asyncio.gather)
"""

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sigma.tau.types import SubTask, SubtaskResult
from sigma.hooks import HookSystem, HookPoint
from sigma.guardrails import GuardrailSet, GuardrailReport


@dataclass
class ExecutorConfig:
    max_workers: int = 6
    timeout_per_subtask: float = 300  # seconds

    # Retry
    max_retries: int = 2
    """Max retry attempts per subtask (0 = no retry)."""
    retry_delay_base: float = 1.0
    """Base delay in seconds for exponential backoff between retries."""


class IndependentExecutor:
    """Runs subtasks independently, respecting dependency graph."""

    def __init__(self, config: ExecutorConfig | None = None,
                 hooks: "HookSystem | None" = None,
                 human_input_callback: callable | None = None,
                 guardrails: "GuardrailSet | None" = None):
        """
        Args:
            config: Executor configuration.
            hooks: Optional HookSystem for task-level lifecycle callbacks.
            human_input_callback: Optional callable(task_id, prompt) -> str
                                  for human-in-the-loop support.
            guardrails: Optional GuardrailSet for output validation.
        """
        self.config = config or ExecutorConfig()
        self.hooks = hooks
        self._human_input = human_input_callback
        self.guardrails = guardrails

    def run_all(
        self,
        task_graph,     # TaskGraph
        agents: dict,   # name → AgentSpec
        tools: dict,    # name → ToolSpec
        llm_call,       # callable(system, user) -> str
        verbose: bool = True,
    ) -> dict[str, SubtaskResult]:
        """Execute all subtasks respecting dependency order.

        Subtasks with no deps run in parallel first.
        Subtasks that depend on others wait for their deps to complete.
        """
        results: dict[str, SubtaskResult] = {}
        pending = list(task_graph.subtasks)

        while pending:
            ready = [
                s for s in pending
                if all(d in results and results[d].success for d in s.dependencies)
            ]
            if not ready:
                # Deadlock or all remaining have failed deps
                for s in pending:
                    if s.id not in results:
                        results[s.id] = SubtaskResult(
                            subtask_id=s.id, success=False,
                            error=f"依赖未满足: {s.dependencies}",
                        )
                break

            # Run all ready subtasks in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.config.max_workers, max(1, len(ready))),
            ) as pool:
                futures = {
                    pool.submit(self._run_one, st, agents, tools, llm_call, verbose, results,
                                task_graph.instruction): st
                    for st in ready
                }
                for fut in concurrent.futures.as_completed(futures):
                    st = futures[fut]
                    try:
                        results[st.id] = fut.result(timeout=self.config.timeout_per_subtask)
                    except Exception as e:
                        results[st.id] = SubtaskResult(
                            subtask_id=st.id, success=False, error=str(e),
                        )

            pending = [s for s in pending if s.id not in results]

        return results

    def _run_one(
        self, st: SubTask, agents: dict, tools: dict,
        llm_call, verbose: bool,
        results: dict[str, "SubtaskResult"] | None = None,
        instruction: str = "",
    ) -> SubtaskResult:
        """Execute a single subtask.

        Upstream interface_params are injected as contract constraints
        (progressive disclosure) so downstream departments work within
        commitments already made by upstream departments.
        """
        t0 = time.perf_counter()

        # Conditional execution: skip if condition returns False
        if st.condition and results:
            try:
                if not st.condition(results):
                    return SubtaskResult(
                        subtask_id=st.id, success=True,
                        error="条件不满足，已跳过",
                    )
            except Exception as e:
                return SubtaskResult(
                    subtask_id=st.id, success=False,
                    error=f"条件评估失败: {e}",
                )

        # Human-in-the-loop
        human_response = ""
        if st.human_input and self._human_input:
            try:
                human_response = self._human_input(st.id, st.human_input)
            except Exception as e:
                return SubtaskResult(
                    subtask_id=st.id, success=False,
                    error=f"人工输入失败: {e}",
                )

        if self.hooks:
            self.hooks.trigger(
                HookPoint.BEFORE_TASK,
                task_id=st.id, task_desc=st.description,
                agent_names=st.assigned_agents,
            )
        analyses = {}
        for name in st.assigned_agents:
            agent = agents.get(name)
            if not agent:
                continue
            system = agent.backstory
            params_block = ""
            if st.interface_params:
                params_block = (
                    "\n\n【接口参数输出区 — 必须在此明确给出数值】\n"
                    + "\n".join(f"{p} = <数值>" for p in st.interface_params)
                    + "\n请用格式 PARAM:{参数名}={数值} 明确标注每个接口参数。"
                )
            instruction_ctx = (
                f"总任务背景：{instruction}\n\n" if instruction else ""
            )
            user = (
                f"{instruction_ctx}"
                f"你的子任务：{st.description}\n\n"
                f"请从你的专业视角分析并给出具体数值和推理过程。注意总任务的规模和约束条件。\n"
                f"预期产出：{', '.join(st.expected_outputs) if st.expected_outputs else '由你判断'}"
                f"{params_block}"
            )
            # Progressive disclosure: inject upstream contract constraints
            upstream = self._upstream_constraints(st, results)
            if upstream:
                user += "\n\n" + upstream
            if human_response:
                user += f"\n\n人工输入（请以此为重要参考）：\n{human_response}"
            if agent.tool_names:
                user += f"\n可用工具：{', '.join(agent.tool_names)}"
            try:
                analyses[name] = llm_call(system, user)
            except Exception as e:
                # Retry with exponential backoff
                analyses[name] = self._retry_call(
                    llm_call, system, user, name, str(e),
                )

        all_errors = all(
            str(v).startswith("[ERROR:") for v in analyses.values()
        ) if analyses else True

        result = SubtaskResult(
            subtask_id=st.id,
            success=len(analyses) > 0 and not all_errors,
            agent_analyses=analyses,
            interface_params=self._extract_params(analyses, st.interface_params),
            param_confidence=self._infer_confidence(analyses),
        )

        # ── Guardrails check ──
        if self.guardrails and result.interface_params:
            gr = self.guardrails.check_all(result.interface_params)
            result = SubtaskResult(
                subtask_id=result.subtask_id,
                success=result.success and not gr.blocked,
                agent_analyses=result.agent_analyses,
                tool_results=result.tool_results,
                interface_params=result.interface_params,
                param_confidence=result.param_confidence,
                error=result.error + f"; 护栏阻止: {gr.summary}" if gr.blocked and result.error else (f"护栏阻止: {gr.summary}" if gr.blocked else result.error),
                guardrails_report=gr.to_dict(),
            )

        duration_ms = (time.perf_counter() - t0) * 1000
        if self.hooks:
            if result.success:
                self.hooks.trigger(
                    HookPoint.AFTER_TASK,
                    task_id=st.id, result=result, duration_ms=duration_ms,
                )
            else:
                self.hooks.trigger(
                    HookPoint.ON_TASK_ERROR,
                    task_id=st.id, error=result.error, attempt=0,
                )

        return result

    def _extract_params(
        self, analyses: dict[str, str], param_keys: list[str],
    ) -> dict[str, float]:
        """Extract numeric parameter values from agent analyses.

        Strategy (best-effort):
          1. PARAM:key=value format (explicit, reliable)
          2. key = value / key: value patterns
          3. Regex fallback: key near a number
        """
        import re
        params = {}
        combined = " ".join(analyses.values())
        for key in param_keys:
            value = None

            # 1. Explicit PARAM: format
            m = re.search(
                rf'PARAM:\s*{re.escape(key)}\s*=\s*(-?[\d.]+(?:[eE][+-]?\d+)?)',
                combined, re.IGNORECASE,
            )
            if m:
                value = float(m.group(1))
            else:
                # 2. key = value on its own line
                m = re.search(
                    rf'(?:^|\n)\s*{re.escape(key)}\s*[=:：]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)',
                    combined, re.IGNORECASE | re.MULTILINE,
                )
                if m:
                    value = float(m.group(1))

            if value is None:
                # 3. Regex fallback: key then a number within reasonable distance
                m = re.search(
                    rf'{re.escape(key)}[^\d]*?(\d+(?:\.\d+)?)',
                    combined, re.IGNORECASE,
                )
                if m:
                    value = float(m.group(1))

            if value is not None:
                params[key] = value
        return params

    def _retry_call(self, llm_call, system: str, user: str,
                    agent_name: str, first_error: str) -> str:
        """Call LLM with retry on failure, using exponential backoff."""
        if self.config.max_retries <= 0:
            return f"[ERROR: {first_error}]"
        for attempt in range(self.config.max_retries):
            delay = self.config.retry_delay_base * (2 ** attempt)
            if attempt > 0:
                time.sleep(delay)
            try:
                return llm_call(system, user)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return f"[ERROR: {e}] (retried {self.config.max_retries}x)"
        return f"[ERROR: {first_error}]"

    def _upstream_constraints(
        self, st: SubTask, results: dict[str, "SubtaskResult"] | None,
    ) -> str:
        """Build upstream contract constraints for progressive disclosure.

        When a subtask depends on upstream subtasks, their interface_params
        are disclosed as fixed constraints the downstream must respect.
        """
        if not results or not st.dependencies:
            return ""
        lines = []
        for dep_id in st.dependencies:
            dep_result = results.get(dep_id)
            if dep_result and dep_result.success and dep_result.interface_params:
                lines.append(f"上游部门（{dep_id}）已确定的接口参数（请以此为约束）：")
                for k, v in dep_result.interface_params.items():
                    lines.append(f"  {k} = {v}")
        return "\n".join(lines) if lines else ""

    def _infer_confidence(self, analyses: dict[str, str]) -> dict[str, str]:
        """Infer confidence based on analysis quality.

        Multiple agents agreeing → HIGH. Single agent → MEDIUM.
        Contains uncertainty keywords → LOW.
        """
        uncertainty_kw = ["不确定", "估计", "假设", "可能", "uncertain", "estimate", "approx"]
        confidences = {}
        for key in analyses:
            text = analyses[key].lower()
            if any(kw in text for kw in uncertainty_kw):
                confidences[key] = "LOW"
            elif len(analyses) >= 2:
                confidences[key] = "HIGH"
            else:
                confidences[key] = "MEDIUM"
        return confidences


class AsyncIndependentExecutor(IndependentExecutor):
    """Async variant using asyncio.gather instead of ThreadPoolExecutor.

    Inherits shared helpers (_extract_params, _upstream_constraints,
    _infer_confidence) from IndependentExecutor. Uses asyncio for I/O-bound
    LLM calls where threading adds unnecessary overhead.

    Usage:
        executor = AsyncIndependentExecutor()
        results = await executor.arun_all(task_graph, agents, tools, async_llm_call)
    """

    async def arun_all(
        self,
        task_graph,
        agents: dict,
        tools: dict,
        llm_call,        # async callable(system, user) -> str
        verbose: bool = True,
    ) -> dict[str, "SubtaskResult"]:
        """Execute all subtasks respecting dependency order (async).

        Subtasks with no deps run concurrently via asyncio.gather.
        Subtasks that depend on others wait for their deps to complete.
        """
        results: dict[str, "SubtaskResult"] = {}
        pending = list(task_graph.subtasks)

        while pending:
            ready = [
                s for s in pending
                if all(d in results and results[d].success for d in s.dependencies)
            ]
            if not ready:
                for s in pending:
                    if s.id not in results:
                        results[s.id] = SubtaskResult(
                            subtask_id=s.id, success=False,
                            error=f"依赖未满足: {s.dependencies}",
                        )
                break

            # Run all ready subtasks concurrently
            tasks = [
                self._arun_one(st, agents, tools, llm_call, verbose, results,
                               task_graph.instruction)
                for st in ready
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for st, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    results[st.id] = SubtaskResult(
                        subtask_id=st.id, success=False, error=str(result),
                    )
                else:
                    results[st.id] = result

            pending = [s for s in pending if s.id not in results]

        return results

    async def _arun_one(
        self, st: SubTask, agents: dict, tools: dict,
        llm_call, verbose: bool,
        results: dict[str, "SubtaskResult"] | None = None,
        instruction: str = "",
    ) -> SubtaskResult:
        """Execute a single subtask (async). Each assigned agent runs concurrently."""
        t0 = time.perf_counter()

        # Conditional execution
        if st.condition and results:
            try:
                if not st.condition(results):
                    return SubtaskResult(
                        subtask_id=st.id, success=True,
                        error="条件不满足，已跳过",
                    )
            except Exception as e:
                return SubtaskResult(
                    subtask_id=st.id, success=False,
                    error=f"条件评估失败: {e}",
                )

        # Human-in-the-loop
        human_response = ""
        if st.human_input and self._human_input:
            try:
                human_response = self._human_input(st.id, st.human_input)
            except Exception as e:
                return SubtaskResult(
                    subtask_id=st.id, success=False,
                    error=f"人工输入失败: {e}",
                )

        if self.hooks:
            self.hooks.trigger(
                HookPoint.BEFORE_TASK,
                task_id=st.id, task_desc=st.description,
                agent_names=st.assigned_agents,
            )
        async def call_agent(name: str):
            agent = agents.get(name)
            if not agent:
                return name, None
            system = agent.backstory
            params_block = ""
            if st.interface_params:
                params_block = (
                    "\n\n【接口参数输出区 — 必须在此明确给出数值】\n"
                    + "\n".join(f"{p} = <数值>" for p in st.interface_params)
                    + "\n请用格式 PARAM:{参数名}={数值} 明确标注每个接口参数。"
                )
            instruction_ctx = (
                f"总任务背景：{instruction}\n\n" if instruction else ""
            )
            user = (
                f"{instruction_ctx}"
                f"你的子任务：{st.description}\n\n"
                f"请从你的专业视角分析并给出具体数值和推理过程。注意总任务的规模和约束条件。\n"
                f"预期产出：{', '.join(st.expected_outputs) if st.expected_outputs else '由你判断'}"
                f"{params_block}"
            )
            upstream = self._upstream_constraints(st, results)
            if upstream:
                user += "\n\n" + upstream
            if human_response:
                user += f"\n\n人工输入（请以此为重要参考）：\n{human_response}"
            if agent.tool_names:
                user += f"\n可用工具：{', '.join(agent.tool_names)}"
            try:
                result = await llm_call(system, user)
                return name, result
            except Exception as e:
                last_error = str(e)
                for attempt in range(self.config.max_retries):
                    delay = self.config.retry_delay_base * (2 ** attempt)
                    if attempt > 0:
                        await asyncio.sleep(delay)
                    try:
                        result = await llm_call(system, user)
                        return name, result
                    except Exception as e2:
                        last_error = str(e2)
                return name, f"[ERROR: {last_error}] (retried {self.config.max_retries}x)"

        agent_tasks = [call_agent(name) for name in st.assigned_agents]
        gathered = await asyncio.gather(*agent_tasks, return_exceptions=True)

        analyses: dict[str, str] = {}
        for item in gathered:
            if isinstance(item, Exception):
                continue
            name, result = item
            if result is not None:
                analyses[name] = result

        all_errors = all(
            str(v).startswith("[ERROR:") for v in analyses.values()
        ) if analyses else True

        result = SubtaskResult(
            subtask_id=st.id,
            success=len(analyses) > 0 and not all_errors,
            agent_analyses=analyses,
            interface_params=self._extract_params(analyses, st.interface_params),
            param_confidence=self._infer_confidence(analyses),
        )

        # ── Guardrails check ──
        if self.guardrails and result.interface_params:
            gr = self.guardrails.check_all(result.interface_params)
            result = SubtaskResult(
                subtask_id=result.subtask_id,
                success=result.success and not gr.blocked,
                agent_analyses=result.agent_analyses,
                tool_results=result.tool_results,
                interface_params=result.interface_params,
                param_confidence=result.param_confidence,
                error=result.error + f"; 护栏阻止: {gr.summary}" if gr.blocked and result.error else (f"护栏阻止: {gr.summary}" if gr.blocked else result.error),
                guardrails_report=gr.to_dict(),
            )

        duration_ms = (time.perf_counter() - t0) * 1000
        if self.hooks:
            if result.success:
                self.hooks.trigger(
                    HookPoint.AFTER_TASK,
                    task_id=st.id, result=result, duration_ms=duration_ms,
                )
            else:
                self.hooks.trigger(
                    HookPoint.ON_TASK_ERROR,
                    task_id=st.id, error=result.error, attempt=0,
                )

        return result
