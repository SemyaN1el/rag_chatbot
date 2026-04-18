import unittest

from app.agent.schemas import ToolCall, ToolResult
from app.agent.service_tools import (
    create_get_chat_history_tool,
    register_default_tools,
)
from app.agent.tools import RegisteredTool, ToolRegistry


class ToolRegistryTestCase(unittest.TestCase):
    def test_execute_returns_handler_result(self) -> None:
        registry = ToolRegistry()
        registry.register(
            RegisteredTool(
                name="echo_tool",
                description="Возвращает аргументы как есть.",
                handler=lambda call: ToolResult(
                    tool_name=call.tool_name,
                    success=True,
                    output={"arguments": call.arguments},
                ),
            )
        )

        result = registry.execute(ToolCall(tool_name="echo_tool", arguments={"x": 1}))

        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "echo_tool")
        self.assertEqual(result.output["arguments"], {"x": 1})

    def test_execute_returns_structured_error_for_unknown_tool(self) -> None:
        registry = ToolRegistry()

        result = registry.execute(ToolCall(tool_name="missing_tool", arguments={}))

        self.assertFalse(result.success)
        self.assertEqual(result.tool_name, "missing_tool")
        self.assertIn("не зарегистрирован", result.error)

    def test_execute_returns_structured_error_for_handler_exception(self) -> None:
        registry = ToolRegistry()

        def failing_handler(call: ToolCall):
            raise RuntimeError("Сервис недоступен")

        registry.register(
            RegisteredTool(
                name="broken_tool",
                description="Всегда падает.",
                handler=failing_handler,
            )
        )

        result = registry.execute(ToolCall(tool_name="broken_tool", arguments={}))

        self.assertFalse(result.success)
        self.assertEqual(result.tool_name, "broken_tool")
        self.assertEqual(result.error, "Сервис недоступен")


class ServiceToolsTestCase(unittest.TestCase):
    def test_get_chat_history_tool_returns_normalized_output(self) -> None:
        tool = create_get_chat_history_tool(
            lambda limit: [
                {
                    "id": 1,
                    "question": "Вопрос",
                    "answer": "Ответ",
                    "search_type": "vector",
                    "created_at": "2026-04-19T01:00:00",
                }
            ][:limit]
        )

        result = tool.handler(ToolCall(tool_name="get_chat_history", arguments={"limit": 5}))

        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 1)
        self.assertEqual(result.output["limit"], 5)
        self.assertEqual(result.output["items"][0]["question"], "Вопрос")

    def test_register_default_tools_registers_agent_toolkit(self) -> None:
        cached_payload = {
            "answer": "Из кэша",
            "sources": [],
            "search_type": "vector",
        }
        captured_cache_write: list[tuple[str, str, dict]] = []

        registry = register_default_tools(
            vector_search_handler=lambda question: {
                "answer": f"vector:{question}",
                "sources": [{"page": 1, "text": "Фрагмент"}],
                "search_type": "vector",
            },
            hybrid_search_handler=lambda question: {
                "answer": f"hybrid:{question}",
                "sources": [{"rrf_score": 0.8, "text": "Фрагмент"}],
                "search_type": "hybrid",
            },
            history_handler=lambda limit: [{"id": 1, "question": "Q", "answer": "A"}][:limit],
            cache_reader=lambda question, search_type: cached_payload if search_type == "vector" else None,
            cache_writer=lambda question, search_type, result: captured_cache_write.append(
                (question, search_type, result)
            ),
        )

        self.assertEqual(
            registry.list_names(),
            [
                "get_cached_answer",
                "get_chat_history",
                "search_hybrid",
                "search_vector",
                "set_cached_answer",
            ],
        )

        vector_result = registry.execute(
            ToolCall(tool_name="search_vector", arguments={"question": "Что в документе?"})
        )
        hybrid_result = registry.execute(
            ToolCall(tool_name="search_hybrid", arguments={"question": "Что в документе?"})
        )
        history_result = registry.execute(
            ToolCall(tool_name="get_chat_history", arguments={"limit": 3})
        )
        cache_hit_result = registry.execute(
            ToolCall(
                tool_name="get_cached_answer",
                arguments={"question": "Вопрос", "search_type": "vector"},
            )
        )
        cache_set_result = registry.execute(
            ToolCall(
                tool_name="set_cached_answer",
                arguments={
                    "question": "Вопрос",
                    "search_type": "vector",
                    "result": {"answer": "Новый ответ", "sources": []},
                },
            )
        )

        self.assertTrue(vector_result.success)
        self.assertEqual(vector_result.output["search_type"], "vector")
        self.assertEqual(vector_result.output["source_count"], 1)

        self.assertTrue(hybrid_result.success)
        self.assertEqual(hybrid_result.output["search_type"], "hybrid")

        self.assertTrue(history_result.success)
        self.assertEqual(history_result.output["count"], 1)

        self.assertTrue(cache_hit_result.success)
        self.assertTrue(cache_hit_result.output["cache_hit"])
        self.assertEqual(cache_hit_result.output["value"]["answer"], "Из кэша")

        self.assertTrue(cache_set_result.success)
        self.assertTrue(cache_set_result.output["cached"])
        self.assertEqual(
            captured_cache_write,
            [("Вопрос", "vector", {"answer": "Новый ответ", "sources": []})],
        )

    def test_search_tool_reports_argument_error(self) -> None:
        registry = register_default_tools(
            vector_search_handler=lambda question: {
                "answer": question,
                "sources": [],
                "search_type": "vector",
            },
            hybrid_search_handler=lambda question: {
                "answer": question,
                "sources": [],
                "search_type": "hybrid",
            },
            history_handler=lambda limit: [],
            cache_reader=lambda question, search_type: None,
            cache_writer=lambda question, search_type, result: None,
        )

        result = registry.execute(
            ToolCall(tool_name="search_vector", arguments={"question": "   "})
        )

        self.assertFalse(result.success)
        self.assertIn("не может быть пустым", result.error)


if __name__ == "__main__":
    unittest.main()
