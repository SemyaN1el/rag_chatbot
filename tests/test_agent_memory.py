import unittest

from app.agent.memory import (
    build_memory_augmented_question,
    should_apply_session_memory,
    update_session_memory,
)


class AgentMemoryTestCase(unittest.TestCase):
    def test_should_apply_session_memory_for_followup_question(self) -> None:
        session_memory = {
            "summary": "Q: Какая форма итоговой аттестации? | A: Экзамен",
            "recent_turns": [
                {
                    "question": "Какая форма итоговой аттестации?",
                    "answer": "Экзамен",
                    "search_type": "vector",
                }
            ],
            "turn_count": 1,
        }

        self.assertTrue(should_apply_session_memory("А сроки какие?", session_memory))
        self.assertFalse(
            should_apply_session_memory(
                "Какие дисциплины перечислены в учебном плане на втором курсе?",
                session_memory,
            )
        )

    def test_update_session_memory_appends_recent_turn_and_summary(self) -> None:
        updated_memory = update_session_memory(
            {
                "summary": "Q: Старый вопрос | A: Старый ответ",
                "recent_turns": [
                    {
                        "question": "Старый вопрос",
                        "answer": "Старый ответ",
                        "search_type": "vector",
                    }
                ],
                "turn_count": 1,
            },
            question="Новый вопрос",
            answer="Новый ответ",
            search_type="hybrid",
        )

        self.assertEqual(updated_memory["turn_count"], 2)
        self.assertEqual(updated_memory["recent_turns"][-1]["question"], "Новый вопрос")
        self.assertIn("Новый ответ", updated_memory["summary"])

    def test_build_memory_augmented_question_includes_summary_and_current_question(self) -> None:
        augmented_question = build_memory_augmented_question(
            "А сроки какие?",
            {
                "summary": "Q: Какая форма итоговой аттестации? | A: Экзамен",
                "recent_turns": [
                    {
                        "question": "Какая форма итоговой аттестации?",
                        "answer": "Экзамен",
                        "search_type": "vector",
                    }
                ],
                "turn_count": 1,
            },
        )

        self.assertIn("Краткий контекст сессии:", augmented_question)
        self.assertIn("Q: Какая форма итоговой аттестации?", augmented_question)
        self.assertIn("Текущий вопрос: А сроки какие?", augmented_question)


if __name__ == "__main__":
    unittest.main()
