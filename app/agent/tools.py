from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.agent.schemas import ToolCall, ToolResult

ToolHandler = Callable[[ToolCall], ToolResult]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        normalized_name = tool.name.strip()
        if not normalized_name:
            raise ValueError("Имя инструмента не может быть пустым")
        if normalized_name in self._tools:
            raise ValueError(f"Инструмент '{normalized_name}' уже зарегистрирован")
        self._tools[normalized_name] = RegisteredTool(
            name=normalized_name,
            description=tool.description.strip(),
            handler=tool.handler,
        )

    def get(self, tool_name: str) -> RegisteredTool:
        normalized_name = tool_name.strip()
        if normalized_name not in self._tools:
            raise KeyError(f"Инструмент '{normalized_name}' не зарегистрирован")
        return self._tools[normalized_name]

    def has(self, tool_name: str) -> bool:
        return tool_name.strip() in self._tools

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            tool = self.get(call.tool_name)
            return tool.handler(call)
        except Exception as exc:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                error=str(exc),
            )
