from __future__ import annotations

from time import perf_counter

from app.agent.observability import log_agent_event
from app.agent.schemas import AgentResponse, AgentTraceStep, ToolCall, ToolResult
from app.agent.state import AgentRoutingDecision, AgentState
from app.agent.tools import ToolRegistry


class AgentRuntime:
    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or ToolRegistry()

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
        state.start()
        state.add_trace_step(
            AgentTraceStep(
                kind="runtime",
                status="completed",
                name="runtime_initialized",
                detail="Создано начальное состояние agent runtime.",
            )
        )
        log_agent_event(
            "runtime_initialized",
            request_id=state.request_id,
            session_id=state.session_id,
            status="completed",
            metadata={"trace_steps": len(state.trace)},
        )
        return state

    def apply_routing_decision(
        self,
        state: AgentState,
        decision: AgentRoutingDecision,
        selected_tool: str | None = None,
    ) -> AgentState:
        state.set_routing_decision(decision, selected_tool=selected_tool)
        state.add_trace_step(
            AgentTraceStep(
                kind="routing",
                status="completed",
                name="routing_decision_applied",
                detail=f"Маршрут запроса: {decision.value}",
                tool_name=selected_tool,
            )
        )
        log_agent_event(
            "routing_decision_applied",
            request_id=state.request_id,
            session_id=state.session_id,
            route=decision.value,
            tool_name=selected_tool,
            status="completed",
        )
        return state

    def execute_tool(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict,
    ) -> ToolResult:
        started_at = perf_counter()
        result = self.tool_registry.execute(
            ToolCall(
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        duration_ms = max(int((perf_counter() - started_at) * 1000), 0)
        state.add_tool_result(result)
        state.add_trace_step(
            AgentTraceStep(
                kind="tool",
                status="completed" if result.success else "failed",
                name="tool_executed",
                detail="Инструмент выполнен успешно." if result.success else result.error,
                tool_name=tool_name,
                duration_ms=duration_ms,
                metadata={"arguments": arguments},
            )
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
        state.add_trace_step(
            AgentTraceStep(
                kind="runtime",
                status="failed",
                name="runtime_failed",
                detail=error_message,
            )
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
