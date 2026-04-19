from __future__ import annotations

from time import perf_counter

from app.agent.budget import AgentBudget, AgentBudgetExceededError
from app.agent.observability import log_agent_event
from app.agent.policy import AgentPolicyViolationError, AgentToolPolicy
from app.agent.schemas import AgentResponse, AgentTraceStep, ToolCall, ToolResult
from app.agent.state import AgentRoutingDecision, AgentState
from app.agent.tools import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        budget: AgentBudget | None = None,
        policy: AgentToolPolicy | None = None,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry()
        self.budget = budget or AgentBudget()
        self.policy = policy or AgentToolPolicy()

    def create_state(
        self,
        question: str,
        session_id: str,
        request_id: str | None = None,
    ) -> AgentState:
        state = AgentState.create(
            question=question,
            session_id=session_id,
            request_id=request_id,
        )
        state.start(started_at_monotonic=perf_counter())
        self.record_step(
            state,
            AgentTraceStep(
                kind="runtime",
                status="completed",
                name="runtime_initialized",
                detail="Создано начальное состояние agent runtime.",
                metadata=self._budget_snapshot(state),
            ),
        )
        log_agent_event(
            "runtime_initialized",
            request_id=state.request_id,
            session_id=state.session_id,
            status="completed",
            metadata={
                "trace_steps": len(state.trace),
                "budget": {
                    "max_steps": self.budget.max_steps,
                    "max_tool_calls": self.budget.max_tool_calls,
                    "max_runtime_seconds": self.budget.max_runtime_seconds,
                },
            },
        )
        return state

    def apply_routing_decision(
        self,
        state: AgentState,
        decision: AgentRoutingDecision,
        selected_tool: str | None = None,
        *,
        detail: str | None = None,
        metadata: dict | None = None,
    ) -> AgentState:
        state.set_routing_decision(decision, selected_tool=selected_tool)
        trace_metadata = {"decision": decision.value, **(metadata or {})}
        self.record_step(
            state,
            AgentTraceStep(
                kind="routing",
                status="completed",
                name="routing_decision_applied",
                detail=detail or f"Маршрут запроса: {decision.value}",
                tool_name=selected_tool,
                metadata=trace_metadata,
            ),
        )
        log_agent_event(
            "routing_decision_applied",
            request_id=state.request_id,
            session_id=state.session_id,
            route=decision.value,
            tool_name=selected_tool,
            status="completed",
            metadata=trace_metadata,
        )
        return state

    def execute_tool(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict,
        *,
        optional: bool = False,
    ) -> ToolResult:
        try:
            self._ensure_tool_execution_budget(state, tool_name)
            self.policy.check_tool_execution(state, tool_name, arguments)
        except (AgentBudgetExceededError, AgentPolicyViolationError) as exc:
            if not optional:
                raise
            return self._build_optional_skip_result(state, tool_name, exc)

        started_at = perf_counter()
        result = self.tool_registry.execute(
            ToolCall(
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        duration_ms = max(int((perf_counter() - started_at) * 1000), 0)
        state.add_tool_result(result)
        self.record_step(
            state,
            AgentTraceStep(
                kind="tool",
                status="completed" if result.success else "failed",
                name="tool_executed",
                detail="Инструмент выполнен успешно." if result.success else result.error,
                tool_name=tool_name,
                duration_ms=duration_ms,
                metadata={"arguments": arguments},
            ),
        )
        log_agent_event(
            "tool_executed",
            request_id=state.request_id,
            session_id=state.session_id,
            route=state.routing_decision.value,
            tool_name=tool_name,
            status="completed" if result.success else "failed",
            duration_ms=duration_ms,
            metadata={"argument_keys": sorted(arguments.keys())},
        )
        return result

    def finalize_response(self, state: AgentState, response: AgentResponse) -> AgentState:
        state.complete(response)
        response.trace = list(state.trace)
        return state

    def fail(self, state: AgentState, error_message: str) -> AgentState:
        state.fail(error_message)
        self.record_step(
            state,
            AgentTraceStep(
                kind="runtime",
                status="failed",
                name="runtime_failed",
                detail=error_message,
            ),
            ignore_budget=True,
        )
        log_agent_event(
            "runtime_failed",
            request_id=state.request_id,
            session_id=state.session_id,
            route=state.routing_decision.value,
            status="failed",
            metadata={"error": error_message},
        )
        return state

    def record_step(
        self,
        state: AgentState,
        step: AgentTraceStep,
        *,
        ignore_budget: bool = False,
    ) -> AgentState:
        if not ignore_budget:
            self._ensure_step_budget(state)
        state.add_trace_step(step)
        return state

    def record_controlled_stop(
        self,
        state: AgentState,
        *,
        refusal_reason: str,
        detail: str,
        tool_name: str | None = None,
        metadata: dict | None = None,
    ) -> AgentState:
        is_budget_stop = refusal_reason.startswith("budget_") or refusal_reason == "workflow_timeout_exceeded"
        self.record_step(
            state,
            AgentTraceStep(
                kind="runtime" if is_budget_stop else "validation",
                status="failed",
                name="budget_exceeded" if is_budget_stop else "tool_execution_blocked",
                detail=detail,
                tool_name=tool_name,
                metadata={"reason": refusal_reason, **(metadata or {})},
            ),
            ignore_budget=True,
        )
        log_agent_event(
            "budget_exceeded" if is_budget_stop else "tool_policy_blocked",
            request_id=state.request_id,
            session_id=state.session_id,
            route=state.routing_decision.value,
            tool_name=tool_name,
            status="failed",
            refusal_reason=refusal_reason,
            duration_ms=self._elapsed_ms(state),
            metadata=metadata or {},
        )
        return state

    def _ensure_step_budget(self, state: AgentState) -> None:
        self._ensure_timeout(state)
        if len(state.trace) >= self.budget.max_steps:
            raise AgentBudgetExceededError(
                "budget_max_steps_exceeded",
                "Agent workflow остановлен: превышен лимит шагов.",
                metadata=self._budget_snapshot(state),
            )

    def _ensure_tool_execution_budget(self, state: AgentState, tool_name: str) -> None:
        self._ensure_step_budget(state)
        if len(state.tool_results) >= self.budget.max_tool_calls:
            raise AgentBudgetExceededError(
                "budget_max_tool_calls_exceeded",
                "Agent workflow остановлен: превышен лимит вызовов инструментов.",
                metadata={"attempted_tool": tool_name, **self._budget_snapshot(state)},
            )

    def _ensure_timeout(self, state: AgentState) -> None:
        if state.started_at_monotonic is None:
            return

        elapsed_seconds = perf_counter() - state.started_at_monotonic
        if elapsed_seconds > self.budget.max_runtime_seconds:
            raise AgentBudgetExceededError(
                "workflow_timeout_exceeded",
                "Agent workflow остановлен: превышено допустимое время выполнения.",
                metadata=self._budget_snapshot(state),
            )

    def _build_optional_skip_result(
        self,
        state: AgentState,
        tool_name: str,
        exc: AgentBudgetExceededError | AgentPolicyViolationError,
    ) -> ToolResult:
        refusal_reason = getattr(exc, "reason", "tool_execution_blocked")
        metadata = getattr(exc, "metadata", {})
        self.record_step(
            state,
            AgentTraceStep(
                kind="runtime",
                status="skipped",
                name="optional_tool_skipped",
                detail=str(exc),
                tool_name=tool_name,
                metadata={"reason": refusal_reason, **metadata},
            ),
            ignore_budget=True,
        )
        log_agent_event(
            "optional_tool_skipped",
            request_id=state.request_id,
            session_id=state.session_id,
            route=state.routing_decision.value,
            tool_name=tool_name,
            status="skipped",
            refusal_reason=refusal_reason,
            duration_ms=self._elapsed_ms(state),
            metadata=metadata,
        )
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output={"skipped": True, "reason": refusal_reason},
            error=str(exc),
        )

    def _budget_snapshot(self, state: AgentState) -> dict[str, int | float]:
        return {
            "step_count": len(state.trace),
            "tool_call_count": len(state.tool_results),
            "elapsed_ms": self._elapsed_ms(state),
            "max_steps": self.budget.max_steps,
            "max_tool_calls": self.budget.max_tool_calls,
            "max_runtime_seconds": self.budget.max_runtime_seconds,
        }

    @staticmethod
    def _elapsed_ms(state: AgentState) -> int:
        if state.started_at_monotonic is None:
            return 0
        return max(int((perf_counter() - state.started_at_monotonic) * 1000), 0)
