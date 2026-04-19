from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.schemas import AgentChatRequest
from app.agent.state import AgentRoutingDecision

DIRECT_ANSWER_PATTERNS = (
    "что ты умеешь",
    "что умеет агент",
    "что ты можешь",
    "какие режимы поиска",
    "какой режим поиска",
    "как ты работаешь",
    "как работает агент",
    "как пользоваться агентом",
)

OUT_OF_SCOPE_PATTERNS = (
    "анекдот",
    "погода",
    "курс доллара",
    "биткоин",
    "новости",
    "рецепт",
    "фильм",
    "сериал",
    "код на python",
    "python script",
)

DOCUMENT_ANCHORS = (
    "документ",
    "раздел",
    "практик",
    "аттеста",
    "экзам",
    "зач",
    "дисциплин",
    "обучен",
    "курс",
    "программ",
    "учебн",
    "график",
    "план",
    "требован",
)

HYBRID_PATTERNS = (
    "сравни",
    "сопостав",
    "перечисли",
    "перечень",
    "обзор",
    "суммируй",
    "резюмируй",
    "различ",
    "в чем разница",
    "какие разделы",
    "подробно",
)

CLARIFY_PATTERNS = (
    "а что еще",
    "а что ещё",
    "подробнее",
    "и что дальше",
    "это о чем",
    "это о чём",
)

FOLLOWUP_STARTERS = {"а", "и", "это", "эта", "этот", "эти", "его", "её", "их"}


@dataclass(frozen=True)
class AgentRoute:
    decision: AgentRoutingDecision
    selected_tool: str | None
    reason: str
    answer: str | None = None
    refusal_reason: str | None = None
    metadata: dict[str, Any] | None = None


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _contains_document_anchor(text: str) -> bool:
    return _contains_any(text, DOCUMENT_ANCHORS)


def _should_direct_answer(question: str) -> bool:
    return _contains_any(question, DIRECT_ANSWER_PATTERNS)


def _should_refuse_out_of_scope(question: str) -> bool:
    if _contains_document_anchor(question):
        return False
    return _contains_any(question, OUT_OF_SCOPE_PATTERNS)


def _should_clarify(question: str, session_memory: dict[str, Any] | None) -> bool:
    if session_memory:
        return False

    if _contains_any(question, CLARIFY_PATTERNS):
        return True

    words = question.split()
    if not words:
        return False

    if len(words) <= 4 and not _contains_document_anchor(question) and words[0] in FOLLOWUP_STARTERS:
        return True

    return False


def _should_prefer_hybrid(question: str, requested_search_type: str) -> bool:
    if requested_search_type == "hybrid":
        return True

    if _contains_any(question, HYBRID_PATTERNS):
        return True

    words = question.split()
    return len(words) >= 9 and _contains_document_anchor(question)


def _build_direct_answer() -> str:
    return (
        "Я помогаю разбирать загруженный документ. "
        "Поддерживаю два режима поиска: `vector` для точечных фактов и `hybrid` "
        "для более широких вопросов, сравнений и сводок. "
        "Также использую guardrails, кэш и память текущей сессии."
    )


def resolve_agent_route(
    request: AgentChatRequest,
    *,
    session_memory: dict[str, Any] | None = None,
) -> AgentRoute:
    normalized_question = _normalize_question(request.question)

    if _should_direct_answer(normalized_question):
        return AgentRoute(
            decision=AgentRoutingDecision.DIRECT_ANSWER,
            selected_tool=None,
            reason="Вопрос относится к возможностям агента, а не к содержимому документа.",
            answer=_build_direct_answer(),
            metadata={"route_kind": "direct_answer"},
        )

    if _should_refuse_out_of_scope(normalized_question):
        return AgentRoute(
            decision=AgentRoutingDecision.REFUSE,
            selected_tool=None,
            reason="Запрос выглядит как внешний по отношению к документу и agent scope.",
            answer="Я сейчас работаю только с вопросами по загруженному документу.",
            refusal_reason="out_of_scope",
            metadata={"route_kind": "refuse"},
        )

    if _should_clarify(normalized_question, session_memory):
        return AgentRoute(
            decision=AgentRoutingDecision.CLARIFY,
            selected_tool=None,
            reason="Запрос слишком короткий или контекстно-зависимый для безопасного поиска без уточнения.",
            answer="Уточни, пожалуйста, к какому разделу или теме документа относится вопрос.",
            refusal_reason="needs_clarification",
            metadata={"route_kind": "clarify"},
        )

    if _should_prefer_hybrid(normalized_question, request.search_type):
        return AgentRoute(
            decision=AgentRoutingDecision.RETRIEVE_HYBRID,
            selected_tool="search_hybrid",
            reason="Вопрос выглядит как широкий или сравнительный, поэтому выбран hybrid retrieval.",
            metadata={"route_kind": "retrieve", "retrieval_mode": "hybrid"},
        )

    return AgentRoute(
        decision=AgentRoutingDecision.RETRIEVE_VECTOR,
        selected_tool="search_vector",
        reason="Вопрос выглядит как точечный lookup, поэтому выбран vector retrieval.",
        metadata={"route_kind": "retrieve", "retrieval_mode": "vector"},
    )
