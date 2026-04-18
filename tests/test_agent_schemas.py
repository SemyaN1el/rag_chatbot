import unittest

from pydantic import ValidationError

from app.agent.schemas import AgentCitation, AgentResponse, AgentTraceStep


class AgentSchemasTestCase(unittest.TestCase):
    def test_agent_response_accepts_valid_payload(self) -> None:
        response = AgentResponse(
            answer="Ответ по документу.",
            citations=[
                AgentCitation(
                    source_id="pdf_docs:1",
                    snippet="Фрагмент из документа",
                    page=2,
                    score=0.91,
                )
            ],
            confidence=0.87,
            trace=[
                AgentTraceStep(
                    kind="generation",
                    status="completed",
                    name="answer_generated",
                )
            ],
        )

        self.assertEqual(response.answer, "Ответ по документу.")
        self.assertEqual(len(response.citations), 1)
        self.assertFalse(response.is_refusal)

    def test_agent_response_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValidationError):
            AgentResponse(answer="Ответ", confidence=1.5)

    def test_agent_response_rejects_blank_answer(self) -> None:
        with self.assertRaises(ValidationError):
            AgentResponse(answer="   ")

    def test_refusal_reason_is_normalized(self) -> None:
        response = AgentResponse(
            answer="Недостаточно данных для ответа.",
            confidence=0.0,
            refusal_reason="   Недостаточно контекста   ",
        )

        self.assertTrue(response.is_refusal)
        self.assertEqual(response.refusal_reason, "Недостаточно контекста")


if __name__ == "__main__":
    unittest.main()
