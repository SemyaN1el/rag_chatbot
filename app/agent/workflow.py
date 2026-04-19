from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from app.agent.guardrails import check_input_guardrails
from app.agent.runtime import AgentRuntime
from app.agent.schemas import AgentChatRequest, AgentChatResponse, AgentCitation, AgentTraceStep
from app.agent.service_tools import register_default_tools
from app.agent.state import AgentRoutingDecision
from app.agent.validators import validate_agent_response

HistorySaver = Callable[[str, str, str], None]


def _resolve_route(search_type: str) -> tuple[AgentRoutingDecision, str]:
    if search_type == "hybrid":
        return AgentRoutingDecision.RETRIEVE_HYBRID, "search_hybrid"
    return AgentRoutingDecision.RETRIEVE_VECTOR, "search_vector"


def _build_confidence(source_count: int, *, cached: bool) -> float:
    if source_count <= 0:
        return 0.2

    if source_count == 1:
        base = 0.6
    elif source_count == 2:
        base = 0.75
    else:
        base = 0.85

    if cached:
        base += 0.05

    return round(min(base, 0.95), 2)


def _build_citations(sources: list[dict], search_type: str) -> list[AgentCitation]:
    citations: list[AgentCitation] = []
    for index, source in enumerate(sources, start=1):
        snippet = str(source.get("text", "")).strip()
        if not snippet:
            continue

        page = source.get("page")
        if not isinstance(page, int) or page < 1:
            page = None

        raw_score = source.get("score", source.get("rrf_score"))
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None

        metadata = {
            key: value
            for key, value in source.items()
            if key not in {"text", "page", "score", "rrf_score"}
        }

        citations.append(
            AgentCitation(
                source_id=f"{search_type}:{index}",
                snippet=snippet,
                page=page,
                score=score,
                metadata=metadata,
            )
        )
    return citations


def _build_success_response(
    *,
    request: AgentChatRequest,
    request_id: str,
    session_id: str,
    payload: dict,
    trace: list[AgentTraceStep],
    cached: bool,
) -> AgentChatResponse:
    sources = payload.get("sources", [])
    citations = _build_citations(sources, request.search_type)
    return AgentChatResponse(
        request_id=request_id,
        session_id=session_id,
        search_type=request.search_type,
        cached=cached,
        answer=payload.get("answer", ""),
        citations=citations,
        confidence=_build_confidence(len(citations), cached=cached),
        refusal_reason=None,
        trace=trace,
    )


def _build_failure_response(
    *,
    request: AgentChatRequest,
    request_id: str,
    session_id: str,
    trace: list[AgentTraceStep],
) -> AgentChatResponse:
    return AgentChatResponse(
        request_id=request_id,
        session_id=session_id,
        search_type=request.search_type,
        cached=False,
        answer="Не удалось обработать запрос по документу из-за ошибки инструмента.",
        citations=[],
        confidence=0.0,
        refusal_reason="tool_execution_failed",
        trace=trace,
    )


def _build_refusal_response(
    *,
    request: AgentChatRequest,
    request_id: str,
    session_id: str,
    refusal_reason: str,
    answer: str,
    trace: list[AgentTraceStep],
    cached: bool = False,
) -> AgentChatResponse:
    return AgentChatResponse(
        request_id=request_id,
        session_id=session_id,
        search_type=request.search_type,
        cached=cached,
        answer=answer,
        citations=[],
        confidence=0.0,
        refusal_reason=refusal_reason,
        trace=trace,
    )


def execute_agent_chat(
    request: AgentChatRequest,
    *,
    runtime: AgentRuntime | None = None,
    history_saver: HistorySaver | None = None,
) -> AgentChatResponse:
    if not request.question.strip():
        raise ValueError("Вопрос не может быть пустым")

    session_id = request.session_id or f"session-{uuid4()}"
    agent_runtime = runtime or AgentRuntime(register_default_tools())

    if history_saver is None:
        from app.services.history import save_to_history

        history_saver = save_to_history

    state = agent_runtime.create_state(
        question=request.question,
        session_id=session_id,
    )
    state.add_trace_step(
        AgentTraceStep(
            kind="input",
            status="completed",
            name="input_validated",
            detail="Входной запрос прошёл базовую валидацию.",
        )
    )
    input_allowed, guard_reason, guard_message = check_input_guardrails(request.question)
    state.add_trace_step(
        AgentTraceStep(
            kind="validation",
            status="completed" if input_allowed else "failed",
            name="input_guardrails_checked",
            detail=guard_message,
        )
    )
    if not input_allowed:
        response = _build_refusal_response(
            request=request,
            request_id=state.request_id,
            session_id=state.session_id,
            refusal_reason=guard_reason or "unsafe_input",
            answer="Запрос отклонён политикой безопасности агента.",
            trace=list(state.trace),
        )
        agent_runtime.finalize_response(state, response)
        return response

    routing_decision, search_tool = _resolve_route(request.search_type)
    agent_runtime.apply_routing_decision(
        state,
        routing_decision,
        selected_tool=search_tool,
    )

    cache_result = agent_runtime.execute_tool(
        state,
        "get_cached_answer",
        {"question": request.question, "search_type": request.search_type},
    )
    if cache_result.success and cache_result.output.get("cache_hit"):
        cached_payload = cache_result.output.get("value") or {}
        response = _build_success_response(
            request=request,
            request_id=state.request_id,
            session_id=state.session_id,
            payload=cached_payload,
            trace=list(state.trace),
            cached=True,
        )
        is_valid, refusal_reason, validation_message = validate_agent_response(
            response,
            source_count=len(cached_payload.get("sources", [])),
        )
        state.add_trace_step(
            AgentTraceStep(
                kind="validation",
                status="completed" if is_valid else "failed",
                name="response_validated",
                detail=validation_message,
            )
        )
        if not is_valid:
            response = _build_refusal_response(
                request=request,
                request_id=state.request_id,
                session_id=state.session_id,
                refusal_reason=refusal_reason or "insufficient_context",
                answer=(
                    "В документе недостаточно подтверждённого контекста для безопасного ответа."
                    if refusal_reason == "insufficient_context"
                    else "Не удалось подтвердить ответ корректными цитатами из документа."
                ),
                trace=list(state.trace),
                cached=True,
            )
            agent_runtime.finalize_response(state, response)
            return response
        history_saver(request.question, response.answer, request.search_type)
        agent_runtime.finalize_response(state, response)
        return response

    search_result = agent_runtime.execute_tool(
        state,
        search_tool,
        {"question": request.question},
    )
    if not search_result.success:
        agent_runtime.fail(state, search_result.error or "Ошибка инструмента")
        return _build_failure_response(
            request=request,
            request_id=state.request_id,
            session_id=state.session_id,
            trace=list(state.trace),
        )

    payload = {
        "answer": search_result.output.get("answer", ""),
        "sources": search_result.output.get("sources", []),
        "search_type": request.search_type,
    }
    cache_write_result = agent_runtime.execute_tool(
        state,
        "set_cached_answer",
        {
            "question": request.question,
            "search_type": request.search_type,
            "result": payload,
        },
    )
    if not cache_write_result.success:
        state.add_trace_step(
            AgentTraceStep(
                kind="runtime",
                status="skipped",
                name="cache_write_failed_but_ignored",
                detail=cache_write_result.error,
            )
        )

    response = _build_success_response(
        request=request,
        request_id=state.request_id,
        session_id=state.session_id,
        payload=payload,
        trace=list(state.trace),
        cached=False,
    )
    is_valid, refusal_reason, validation_message = validate_agent_response(
        response,
        source_count=len(payload.get("sources", [])),
    )
    state.add_trace_step(
        AgentTraceStep(
            kind="validation",
            status="completed" if is_valid else "failed",
            name="response_validated",
            detail=validation_message,
        )
    )
    if not is_valid:
        response = _build_refusal_response(
            request=request,
            request_id=state.request_id,
            session_id=state.session_id,
            refusal_reason=refusal_reason or "insufficient_context",
            answer=(
                "В документе недостаточно подтверждённого контекста для безопасного ответа."
                if refusal_reason == "insufficient_context"
                else "Не удалось подтвердить ответ корректными цитатами из документа."
            ),
            trace=list(state.trace),
            cached=False,
        )
        agent_runtime.finalize_response(state, response)
        return response
    history_saver(request.question, response.answer, request.search_type)
    agent_runtime.finalize_response(state, response)
    return response
