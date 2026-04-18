from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.schemas import ToolCall, ToolResult
from app.agent.tools import RegisteredTool, ToolRegistry

SearchHandler = Callable[[str], dict[str, Any]]
HistoryHandler = Callable[[int], list[dict[str, Any]]]
CacheReader = Callable[[str, str], dict[str, Any] | None]
CacheWriter = Callable[[str, str, dict[str, Any]], None]


def _require_string_argument(call: ToolCall, name: str) -> str:
    value = call.arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Аргумент '{name}' должен быть строкой")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Аргумент '{name}' не может быть пустым")
    return normalized


def _get_int_argument(
    call: ToolCall,
    name: str,
    default: int,
    *,
    min_value: int = 1,
) -> int:
    value = call.arguments.get(name, default)
    if not isinstance(value, int):
        raise ValueError(f"Аргумент '{name}' должен быть целым числом")
    if value < min_value:
        raise ValueError(f"Аргумент '{name}' должен быть не меньше {min_value}")
    return value


def _require_dict_argument(call: ToolCall, name: str) -> dict[str, Any]:
    value = call.arguments.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Аргумент '{name}' должен быть объектом")
    return value


def create_search_tool(
    *,
    tool_name: str,
    description: str,
    search_handler: SearchHandler,
    default_search_type: str,
) -> RegisteredTool:
    def handler(call: ToolCall) -> ToolResult:
        question = _require_string_argument(call, "question")
        raw_result = search_handler(question)
        sources = raw_result.get("sources", [])
        search_type = raw_result.get("search_type", default_search_type)

        return ToolResult(
            tool_name=tool_name,
            success=True,
            output={
                "question": question,
                "answer": raw_result.get("answer", ""),
                "sources": sources,
                "source_count": len(sources),
                "search_type": search_type,
            },
        )

    return RegisteredTool(
        name=tool_name,
        description=description,
        handler=handler,
    )


def create_get_chat_history_tool(
    history_handler: HistoryHandler,
) -> RegisteredTool:
    def handler(call: ToolCall) -> ToolResult:
        limit = _get_int_argument(call, "limit", default=10)
        items = history_handler(limit)
        return ToolResult(
            tool_name="get_chat_history",
            success=True,
            output={
                "items": items,
                "count": len(items),
                "limit": limit,
            },
        )

    return RegisteredTool(
        name="get_chat_history",
        description="Возвращает историю последних диалогов из хранилища.",
        handler=handler,
    )


def create_get_cached_answer_tool(
    cache_reader: CacheReader,
) -> RegisteredTool:
    def handler(call: ToolCall) -> ToolResult:
        question = _require_string_argument(call, "question")
        search_type = _require_string_argument(call, "search_type")
        cached_value = cache_reader(question, search_type)

        return ToolResult(
            tool_name="get_cached_answer",
            success=True,
            output={
                "question": question,
                "search_type": search_type,
                "cache_hit": cached_value is not None,
                "value": cached_value,
            },
        )

    return RegisteredTool(
        name="get_cached_answer",
        description="Читает сохранённый ответ из Redis-кэша по вопросу и типу поиска.",
        handler=handler,
    )


def create_set_cached_answer_tool(
    cache_writer: CacheWriter,
) -> RegisteredTool:
    def handler(call: ToolCall) -> ToolResult:
        question = _require_string_argument(call, "question")
        search_type = _require_string_argument(call, "search_type")
        result = _require_dict_argument(call, "result")
        cache_writer(question, search_type, result)

        return ToolResult(
            tool_name="set_cached_answer",
            success=True,
            output={
                "question": question,
                "search_type": search_type,
                "cached": True,
            },
        )

    return RegisteredTool(
        name="set_cached_answer",
        description="Сохраняет ответ в Redis-кэш по вопросу и типу поиска.",
        handler=handler,
    )


def register_default_tools(
    registry: ToolRegistry | None = None,
    *,
    vector_search_handler: SearchHandler | None = None,
    hybrid_search_handler: SearchHandler | None = None,
    history_handler: HistoryHandler | None = None,
    cache_reader: CacheReader | None = None,
    cache_writer: CacheWriter | None = None,
) -> ToolRegistry:
    if vector_search_handler is None or hybrid_search_handler is None:
        from app.services.rag import ask_hybrid, ask_vector

        vector_search_handler = vector_search_handler or ask_vector
        hybrid_search_handler = hybrid_search_handler or ask_hybrid

    if history_handler is None:
        from app.services.history import get_history

        history_handler = get_history

    if cache_reader is None or cache_writer is None:
        from app.services.cache import get_cached, set_cached

        cache_reader = cache_reader or get_cached
        cache_writer = cache_writer or set_cached

    tool_registry = registry or ToolRegistry()
    tool_registry.register(
        create_search_tool(
            tool_name="search_vector",
            description="Ищет ответ по документу через векторный retrieval.",
            search_handler=vector_search_handler,
            default_search_type="vector",
        )
    )
    tool_registry.register(
        create_search_tool(
            tool_name="search_hybrid",
            description="Ищет ответ по документу через гибридный retrieval.",
            search_handler=hybrid_search_handler,
            default_search_type="hybrid",
        )
    )
    tool_registry.register(create_get_chat_history_tool(history_handler))
    tool_registry.register(create_get_cached_answer_tool(cache_reader))
    tool_registry.register(create_set_cached_answer_tool(cache_writer))
    return tool_registry
