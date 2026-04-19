from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name не может быть пустым")
        return normalized


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name не может быть пустым")
        return normalized

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AgentCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    snippet: str
    page: int | None = Field(default=None, ge=1)
    score: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", "snippet")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("текстовое поле не может быть пустым")
        return normalized


class AgentTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["input", "routing", "tool", "generation", "validation", "runtime"]
    status: Literal["pending", "completed", "failed", "skipped"]
    name: str
    detail: str | None = None
    tool_name: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name не может быть пустым")
        return normalized

    @field_validator("detail", "tool_name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[AgentCitation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_reason: str | None = None
    trace: list[AgentTraceStep] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer не может быть пустым")
        return normalized

    @field_validator("refusal_reason")
    @classmethod
    def normalize_refusal_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @property
    def is_refusal(self) -> bool:
        return self.refusal_reason is not None


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    search_type: Literal["vector", "hybrid"] = "vector"
    session_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AgentChatResponse(AgentResponse):
    request_id: str
    session_id: str
    search_type: Literal["vector", "hybrid"]
    cached: bool = False
