import unittest
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.runtime import AgentRuntime
from app.agent.schemas import ToolCall, ToolResult
from app.agent.tools import RegisteredTool, ToolRegistry
from app.routers.agent import get_agent_runtime, get_history_saver, router


def build_test_runtime(
    *,
    cache_value: dict | None = None,
    fail_vector: bool = False,
    fail_hybrid: bool = False,
    calls: dict[str, int] | None = None,
    cached_writes: list[tuple[str, str, dict]] | None = None,
) -> AgentRuntime:
    call_counters = calls if calls is not None else {}
    cache_writes = cached_writes if cached_writes is not None else []
    registry = ToolRegistry()

    def register_tool(name: str, handler: Callable[[ToolCall], ToolResult]) -> None:
        registry.register(
            RegisteredTool(
                name=name,
                description=f"Тестовый инструмент {name}",
                handler=handler,
            )
        )

    def get_cached_answer(call: ToolCall) -> ToolResult:
        call_counters["get_cached_answer"] = call_counters.get("get_cached_answer", 0) + 1
        search_type = call.arguments["search_type"]
        is_hit = cache_value is not None and search_type == cache_value.get("search_type")
        return ToolResult(
            tool_name="get_cached_answer",
            success=True,
            output={
                "question": call.arguments["question"],
                "search_type": search_type,
                "cache_hit": is_hit,
                "value": cache_value if is_hit else None,
            },
        )

    def search_vector(call: ToolCall) -> ToolResult:
        call_counters["search_vector"] = call_counters.get("search_vector", 0) + 1
        if fail_vector:
            raise RuntimeError("Vector tool unavailable")
        question = call.arguments["question"]
        return ToolResult(
            tool_name="search_vector",
            success=True,
            output={
                "question": question,
                "answer": f"vector:{question}",
                "sources": [{"page": 1, "text": "Фрагмент vector"}],
                "source_count": 1,
                "search_type": "vector",
            },
        )

    def search_hybrid(call: ToolCall) -> ToolResult:
        call_counters["search_hybrid"] = call_counters.get("search_hybrid", 0) + 1
        if fail_hybrid:
            raise RuntimeError("Hybrid tool unavailable")
        question = call.arguments["question"]
        return ToolResult(
            tool_name="search_hybrid",
            success=True,
            output={
                "question": question,
                "answer": f"hybrid:{question}",
                "sources": [{"rrf_score": 0.88, "text": "Фрагмент hybrid"}],
                "source_count": 1,
                "search_type": "hybrid",
            },
        )

    def set_cached_answer(call: ToolCall) -> ToolResult:
        call_counters["set_cached_answer"] = call_counters.get("set_cached_answer", 0) + 1
        cache_writes.append(
            (
                call.arguments["question"],
                call.arguments["search_type"],
                call.arguments["result"],
            )
        )
        return ToolResult(
            tool_name="set_cached_answer",
            success=True,
            output={"cached": True},
        )

    register_tool("get_cached_answer", get_cached_answer)
    register_tool("search_vector", search_vector)
    register_tool("search_hybrid", search_hybrid)
    register_tool("set_cached_answer", set_cached_answer)
    return AgentRuntime(registry)


class AgentRouterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router)
        self.history_calls: list[tuple[str, str, str]] = []

    def build_client(self, runtime: AgentRuntime) -> TestClient:
        self.app.dependency_overrides[get_agent_runtime] = lambda: runtime
        self.app.dependency_overrides[get_history_saver] = lambda: (
            lambda question, answer, search_type: self.history_calls.append(
                (question, answer, search_type)
            )
        )
        return TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    def test_agent_chat_vector_success(self) -> None:
        cached_writes: list[tuple[str, str, dict]] = []
        runtime = build_test_runtime(cached_writes=cached_writes)
        client = self.build_client(runtime)

        response = client.post(
            "/agent/chat",
            json={
                "question": "Какая форма аттестации?",
                "search_type": "vector",
                "session_id": "session-123",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], "session-123")
        self.assertEqual(payload["search_type"], "vector")
        self.assertFalse(payload["cached"])
        self.assertEqual(payload["answer"], "vector:Какая форма аттестации?")
        self.assertEqual(payload["citations"][0]["page"], 1)
        self.assertGreaterEqual(len(payload["trace"]), 4)
        self.assertEqual(
            self.history_calls,
            [("Какая форма аттестации?", "vector:Какая форма аттестации?", "vector")],
        )
        self.assertEqual(
            cached_writes,
            [
                (
                    "Какая форма аттестации?",
                    "vector",
                    {
                        "answer": "vector:Какая форма аттестации?",
                        "sources": [{"page": 1, "text": "Фрагмент vector"}],
                        "search_type": "vector",
                    },
                )
            ],
        )

    def test_agent_chat_hybrid_success(self) -> None:
        runtime = build_test_runtime()
        client = self.build_client(runtime)

        response = client.post(
            "/agent/chat",
            json={"question": "Что сказано в документе?", "search_type": "hybrid"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["search_type"], "hybrid")
        self.assertFalse(payload["cached"])
        self.assertEqual(payload["answer"], "hybrid:Что сказано в документе?")
        self.assertEqual(payload["citations"][0]["score"], 0.88)
        self.assertTrue(payload["request_id"])
        self.assertTrue(payload["session_id"])

    def test_agent_chat_uses_cache_before_search(self) -> None:
        calls: dict[str, int] = {}
        runtime = build_test_runtime(
            cache_value={
                "answer": "cached answer",
                "sources": [{"page": 2, "text": "Фрагмент из кэша"}],
                "search_type": "vector",
            },
            calls=calls,
        )
        client = self.build_client(runtime)

        response = client.post(
            "/agent/chat",
            json={"question": "Повторный вопрос", "search_type": "vector"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["cached"])
        self.assertEqual(payload["answer"], "cached answer")
        self.assertEqual(calls.get("search_vector", 0), 0)
        self.assertEqual(calls.get("get_cached_answer", 0), 1)

    def test_agent_chat_returns_controlled_failure_when_search_tool_breaks(self) -> None:
        runtime = build_test_runtime(fail_vector=True)
        client = self.build_client(runtime)

        response = client.post(
            "/agent/chat",
            json={"question": "Сломанный запрос", "search_type": "vector"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["refusal_reason"], "tool_execution_failed")
        self.assertEqual(payload["confidence"], 0.0)
        self.assertFalse(payload["cached"])
        self.assertTrue(any(step["status"] == "failed" for step in payload["trace"]))

    def test_agent_chat_rejects_blank_question(self) -> None:
        runtime = build_test_runtime()
        client = self.build_client(runtime)

        response = client.post(
            "/agent/chat",
            json={"question": "   ", "search_type": "vector"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Вопрос не может быть пустым")


if __name__ == "__main__":
    unittest.main()
