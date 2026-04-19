from __future__ import annotations

from typing import Any

FOLLOW_UP_MARKERS = (
    "это",
    "эта",
    "этот",
    "эти",
    "его",
    "её",
    "их",
    "там",
    "тогда",
    "такие",
    "сроки",
    "подробнее",
    "ещё",
    "ранее",
    "выше",
    "предыдущ",
)


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _normalize_turn(turn: dict[str, Any]) -> dict[str, str]:
    return {
        "question": _normalize_text(str(turn.get("question", ""))),
        "answer": _normalize_text(str(turn.get("answer", ""))),
        "search_type": _normalize_text(str(turn.get("search_type", "vector"))) or "vector",
    }


def should_apply_session_memory(question: str, session_memory: dict[str, Any] | None) -> bool:
    if not session_memory:
        return False

    summary = _normalize_text(str(session_memory.get("summary", "")))
    recent_turns = session_memory.get("recent_turns", [])
    if not summary and not recent_turns:
        return False

    normalized_question = _normalize_text(question).lower()
    words = normalized_question.split()
    if len(words) <= 6:
        return True

    return any(marker in normalized_question for marker in FOLLOW_UP_MARKERS)


def build_session_summary(recent_turns: list[dict[str, Any]], *, max_turns: int = 3) -> str:
    normalized_turns = [_normalize_turn(turn) for turn in recent_turns]
    compact_turns = [
        turn for turn in normalized_turns[-max_turns:]
        if turn["question"] and turn["answer"]
    ]
    if not compact_turns:
        return ""

    parts = [
        f"Q: {turn['question']} | A: {turn['answer']}"
        for turn in compact_turns
    ]
    return " ; ".join(parts)


def update_session_memory(
    existing_memory: dict[str, Any] | None,
    *,
    question: str,
    answer: str,
    search_type: str,
    max_turns: int = 5,
) -> dict[str, Any]:
    normalized_question = _normalize_text(question)
    normalized_answer = _normalize_text(answer)
    normalized_search_type = _normalize_text(search_type) or "vector"

    previous_turns = []
    if existing_memory:
        previous_turns = [
            _normalize_turn(turn)
            for turn in existing_memory.get("recent_turns", [])
        ]

    updated_turns = [
        turn for turn in previous_turns
        if turn["question"] and turn["answer"]
    ]
    updated_turns.append(
        {
            "question": normalized_question,
            "answer": normalized_answer,
            "search_type": normalized_search_type,
        }
    )
    updated_turns = updated_turns[-max_turns:]

    return {
        "summary": build_session_summary(updated_turns),
        "recent_turns": updated_turns,
        "turn_count": len(updated_turns),
    }


def build_memory_augmented_question(
    question: str,
    session_memory: dict[str, Any],
    *,
    max_turns: int = 2,
) -> str:
    summary = _normalize_text(str(session_memory.get("summary", "")))
    recent_turns = [
        _normalize_turn(turn)
        for turn in session_memory.get("recent_turns", [])
    ]
    compact_turns = [
        turn for turn in recent_turns[-max_turns:]
        if turn["question"] and turn["answer"]
    ]

    parts = []
    if summary:
        parts.append(f"Краткий контекст сессии: {summary}")
    if compact_turns:
        parts.append("Последние ходы:")
        for turn in compact_turns:
            parts.append(f"Q: {turn['question']}")
            parts.append(f"A: {turn['answer']}")
    parts.append(f"Текущий вопрос: {_normalize_text(question)}")
    return "\n".join(parts)
