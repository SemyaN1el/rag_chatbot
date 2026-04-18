from __future__ import annotations

from app.agent.schemas import AgentResponse, AgentTraceStep
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
        return state

    def finalize_response(self, state: AgentState, response: AgentResponse) -> AgentState:
        state.complete(response)
        if not response.trace:
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
        return state
