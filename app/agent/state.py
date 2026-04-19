from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.schemas import AgentResponse, AgentTraceStep, ToolResult


class AgentStatus(str, Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRoutingDecision(str, Enum):
    UNDECIDED = "undecided"
    DIRECT_ANSWER = "direct_answer"
    CLARIFY = "clarify"
    RETRIEVE_VECTOR = "retrieve_vector"
    RETRIEVE_HYBRID = "retrieve_hybrid"
    REFUSE = "refuse"


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_question: str
    normalized_question: str
    status: AgentStatus = AgentStatus.INITIALIZED
    routing_decision: AgentRoutingDecision = AgentRoutingDecision.UNDECIDED
    selected_tools: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    trace: list[AgentTraceStep] = Field(default_factory=list)
    context_chunks: list[str] = Field(default_factory=list)
    response: AgentResponse | None = None
    error: str | None = None

    @field_validator("session_id", "user_question", "normalized_question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("обязательное строковое поле не может быть пустым")
        return normalized

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @classmethod
    def create(
        cls,
        question: str,
        session_id: str,
        request_id: str | None = None,
    ) -> "AgentState":
        normalized_question = " ".join(question.split())
        payload = {
            "session_id": session_id,
            "user_question": question,
            "normalized_question": normalized_question,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        return cls(**payload)

    def start(self) -> None:
        self.status = AgentStatus.RUNNING

    def set_routing_decision(
        self,
        decision: AgentRoutingDecision,
        selected_tool: str | None = None,
    ) -> None:
        self.routing_decision = decision
        if selected_tool and selected_tool not in self.selected_tools:
            self.selected_tools.append(selected_tool)

    def add_trace_step(self, step: AgentTraceStep) -> None:
        self.trace.append(step)

    def add_tool_result(self, result: ToolResult) -> None:
        if result.tool_name not in self.selected_tools:
            self.selected_tools.append(result.tool_name)
        self.tool_results.append(result)

    def set_context(self, chunks: list[str]) -> None:
        self.context_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def complete(self, response: AgentResponse) -> None:
        self.response = response
        self.status = AgentStatus.COMPLETED
        self.error = None

    def fail(self, error_message: str) -> None:
        self.status = AgentStatus.FAILED
        self.error = error_message.strip()
