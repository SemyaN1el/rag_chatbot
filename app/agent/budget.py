from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import AGENT_MAX_STEPS, AGENT_MAX_TOOL_CALLS, AGENT_MAX_RUNTIME_SECONDS


@dataclass(frozen=True, slots=True)
class AgentBudget:
    max_steps: int = AGENT_MAX_STEPS
    max_tool_calls: int = AGENT_MAX_TOOL_CALLS
    max_runtime_seconds: float = AGENT_MAX_RUNTIME_SECONDS

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps должен быть не меньше 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls должен быть не меньше 1")
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds должен быть больше 0")


class AgentBudgetExceededError(RuntimeError):
    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.metadata = metadata or {}
