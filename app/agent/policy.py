from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from app.agent.state import AgentRoutingDecision, AgentState

SEARCH_TOOLS = frozenset({"search_vector", "search_hybrid"})
ANSWER_CONTEXT_TOOLS = frozenset({"get_cached_answer", *SEARCH_TOOLS})
ROUTE_ALLOWED_TOOLS: Mapping[AgentRoutingDecision, frozenset[str]] = {
    AgentRoutingDecision.UNDECIDED: frozenset({"get_session_memory"}),
    AgentRoutingDecision.DIRECT_ANSWER: frozenset(),
    AgentRoutingDecision.CLARIFY: frozenset(),
    AgentRoutingDecision.REFUSE: frozenset(),
    AgentRoutingDecision.RETRIEVE_VECTOR: frozenset(
        {
            "get_chat_history",
            "get_cached_answer",
            "get_session_memory",
            "search_vector",
            "set_cached_answer",
            "set_session_memory",
        }
    ),
    AgentRoutingDecision.RETRIEVE_HYBRID: frozenset(
        {
            "get_chat_history",
            "get_cached_answer",
            "get_session_memory",
            "search_hybrid",
            "set_cached_answer",
            "set_session_memory",
        }
    ),
}


class AgentPolicyViolationError(RuntimeError):
    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        tool_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.tool_name = tool_name
        self.metadata = metadata or {}


@dataclass(frozen=True, slots=True)
class AgentToolPolicy:
    blocked_tools: frozenset[str] = field(default_factory=frozenset)
    allowed_tools_by_route: Mapping[AgentRoutingDecision, frozenset[str]] = field(
        default_factory=lambda: ROUTE_ALLOWED_TOOLS
    )

    def check_tool_execution(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name in self.blocked_tools:
            raise AgentPolicyViolationError(
                "tool_blocked_by_policy",
                f"Инструмент '{normalized_tool_name}' заблокирован policy-слоем runtime.",
                tool_name=normalized_tool_name,
                metadata={"blocked_tools": sorted(self.blocked_tools)},
            )

        allowed_tools = self.allowed_tools_by_route.get(state.routing_decision, frozenset())
        if normalized_tool_name not in allowed_tools:
            raise AgentPolicyViolationError(
                "tool_not_allowed_for_route",
                (
                    f"Инструмент '{normalized_tool_name}' не разрешён для маршрута "
                    f"'{state.routing_decision.value}'."
                ),
                tool_name=normalized_tool_name,
                metadata={
                    "route": state.routing_decision.value,
                    "allowed_tools": sorted(allowed_tools),
                    "argument_keys": sorted(arguments.keys()),
                },
            )

        if normalized_tool_name == "set_cached_answer" and not _has_successful_search(state):
            raise AgentPolicyViolationError(
                "tool_write_requires_search_result",
                "Запись ответа в кэш разрешена только после успешного retrieval.",
                tool_name=normalized_tool_name,
                metadata={"route": state.routing_decision.value},
            )

        if normalized_tool_name == "set_session_memory" and not _has_answer_context(state):
            raise AgentPolicyViolationError(
                "tool_write_requires_answer_context",
                "Обновление памяти сессии разрешено только после подтверждённого ответа.",
                tool_name=normalized_tool_name,
                metadata={"route": state.routing_decision.value},
            )


def _has_successful_search(state: AgentState) -> bool:
    return any(
        result.success and result.tool_name in SEARCH_TOOLS
        for result in state.tool_results
    )


def _has_answer_context(state: AgentState) -> bool:
    for result in state.tool_results:
        if not result.success:
            continue
        if result.tool_name in SEARCH_TOOLS:
            return True
        if result.tool_name == "get_cached_answer" and bool(result.output.get("cache_hit")):
            return True
    return False
