from __future__ import annotations

from app.agent.schemas import AgentChatResponse


def validate_agent_response(
    response: AgentChatResponse,
    *,
    source_count: int,
) -> tuple[bool, str | None, str]:
    if response.is_refusal:
        return True, None, "Ответ уже находится в корректном режиме отказа."

    if source_count <= 0:
        return (
            False,
            "insufficient_context",
            "Недостаточно контекста: retrieval не вернул подтверждающих источников.",
        )

    if not response.citations:
        return (
            False,
            "missing_citations",
            "Ответ не содержит валидных citations, поэтому не может считаться надёжным.",
        )

    return True, None, "Ответ прошёл базовую валидацию."
