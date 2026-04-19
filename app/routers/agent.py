from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException

from app.agent import AgentChatRequest, AgentChatResponse, AgentRuntime, register_default_tools
from app.agent.workflow import execute_agent_chat
from app.services.history import save_to_history

router = APIRouter(prefix="/agent", tags=["agent"])

HistorySaver = Callable[[str, str, str], None]


def get_agent_runtime() -> AgentRuntime:
    return AgentRuntime(register_default_tools())


def get_history_saver() -> HistorySaver:
    return save_to_history


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(
    request: AgentChatRequest,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    history_saver: HistorySaver = Depends(get_history_saver),
):
    try:
        return execute_agent_chat(
            request,
            runtime=runtime,
            history_saver=history_saver,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
