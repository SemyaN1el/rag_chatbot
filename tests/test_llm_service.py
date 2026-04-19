import unittest

from app.services.llm import build_chat_messages, build_groq_request_payload


class LlmServiceTestCase(unittest.TestCase):
    def test_build_chat_messages_adds_system_message(self) -> None:
        messages = build_chat_messages(
            "Сформулируй ответ",
            system_message="Ты полезный помощник.",
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Ты полезный помощник."},
                {"role": "user", "content": "Сформулируй ответ"},
            ],
        )

    def test_build_payload_includes_schema_for_structured_output(self) -> None:
        payload = build_groq_request_payload(
            messages=[{"role": "user", "content": "Верни JSON"}],
            temperature=0.2,
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )

        self.assertEqual(payload["temperature"], 0.2)
        self.assertIn("response_format", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"]["type"], "object")


if __name__ == "__main__":
    unittest.main()
