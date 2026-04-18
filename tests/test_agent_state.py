import unittest

from app.agent.runtime import AgentRuntime
from app.agent.schemas import AgentResponse, AgentTraceStep, ToolResult
from app.agent.state import AgentRoutingDecision, AgentState, AgentStatus


class AgentStateTestCase(unittest.TestCase):
    def test_create_state_normalizes_question(self) -> None:
        state = AgentState.create(
            question="  Какая   форма   аттестации?  ",
            session_id="session-1",
            request_id="req-1",
        )

        self.assertEqual(state.request_id, "req-1")
        self.assertEqual(state.user_question, "Какая   форма   аттестации?")
        self.assertEqual(state.normalized_question, "Какая форма аттестации?")
        self.assertEqual(state.status, AgentStatus.INITIALIZED)

    def test_state_records_trace_and_tool_result(self) -> None:
        state = AgentState.create(question="Вопрос", session_id="session-1")
        state.add_trace_step(
            AgentTraceStep(
                kind="input",
                status="completed",
                name="input_validated",
            )
        )
        state.add_tool_result(
            ToolResult(
                tool_name="search_vector",
                success=True,
                output={"hits": 3},
            )
        )

        self.assertEqual(len(state.trace), 1)
        self.assertEqual(len(state.tool_results), 1)
        self.assertIn("search_vector", state.selected_tools)

    def test_runtime_applies_routing_and_finalizes_response(self) -> None:
        runtime = AgentRuntime()
        state = runtime.create_state(
            question="Что сказано в документе?",
            session_id="session-2",
            request_id="req-2",
        )
        runtime.apply_routing_decision(
            state,
            AgentRoutingDecision.RETRIEVE_VECTOR,
            selected_tool="search_vector",
        )
        response = AgentResponse(
            answer="В документе указано ...",
            confidence=0.74,
        )

        runtime.finalize_response(state, response)

        self.assertEqual(state.status, AgentStatus.COMPLETED)
        self.assertEqual(state.routing_decision, AgentRoutingDecision.RETRIEVE_VECTOR)
        self.assertEqual(state.response.answer, "В документе указано ...")
        self.assertGreaterEqual(len(state.response.trace), 2)

    def test_runtime_failure_marks_state_as_failed(self) -> None:
        runtime = AgentRuntime()
        state = runtime.create_state(question="Вопрос", session_id="session-3")

        runtime.fail(state, "Инструмент недоступен")

        self.assertEqual(state.status, AgentStatus.FAILED)
        self.assertEqual(state.error, "Инструмент недоступен")
        self.assertEqual(state.trace[-1].status, "failed")


if __name__ == "__main__":
    unittest.main()
