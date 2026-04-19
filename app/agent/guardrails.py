from __future__ import annotations

import re

UNSAFE_INPUT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"ignore\s+(all\s+)?(previous|prior)?\s*instructions", re.IGNORECASE),
        "Обнаружена попытка игнорировать системные инструкции.",
    ),
    (
        re.compile(r"forget\s+(all\s+)?instructions", re.IGNORECASE),
        "Обнаружена попытка сбросить системные инструкции.",
    ),
    (
        re.compile(r"system\s+prompt|developer\s+message", re.IGNORECASE),
        "Запрос пытается получить скрытый системный контекст.",
    ),
    (
        re.compile(r"jailbreak|bypass\s+(guardrails|rules|restrictions)", re.IGNORECASE),
        "Запрос похож на попытку обойти ограничения агента.",
    ),
)


def check_input_guardrails(question: str) -> tuple[bool, str | None, str]:
    normalized = " ".join(question.split())

    for pattern, message in UNSAFE_INPUT_PATTERNS:
        if pattern.search(normalized):
            return False, "unsafe_input", message

    return True, None, "Входной запрос прошёл базовые guardrails-проверки."
