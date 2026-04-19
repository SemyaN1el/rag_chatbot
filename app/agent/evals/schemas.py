from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.schemas import AgentChatRequest


class AgentEvalBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=16, ge=1)
    max_tool_calls: int = Field(default=6, ge=1)
    max_runtime_seconds: float = Field(default=15.0, gt=0.0)


class AgentEvalSearchFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    search_type: Literal["vector", "hybrid"] | None = None

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer не может быть пустым")
        return normalized


class AgentEvalCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    search_type: Literal["vector", "hybrid"]
    value: dict[str, Any]

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("question не может быть пустым")
        return normalized


class AgentEvalCaseFixtures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_memory: dict[str, Any] | None = None
    cache_entries: list[AgentEvalCacheEntry] = Field(default_factory=list)
    history_items: list[dict[str, Any]] = Field(default_factory=list)
    search_outputs: dict[Literal["vector", "hybrid"], AgentEvalSearchFixture] = Field(default_factory=dict)
    blocked_tools: list[str] = Field(default_factory=list)
    budget: AgentEvalBudgetConfig | None = None
    tool_delays_ms: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    @field_validator("blocked_tools")
    @classmethod
    def validate_blocked_tools(cls, value: list[str]) -> list[str]:
        return [tool.strip() for tool in value if tool.strip()]

    @field_validator("tool_delays_ms")
    @classmethod
    def validate_tool_delays(cls, value: dict[str, int]) -> dict[str, int]:
        validated: dict[str, int] = {}
        for tool_name, delay_ms in value.items():
            normalized_tool_name = tool_name.strip()
            if not normalized_tool_name:
                raise ValueError("Имя инструмента в tool_delays_ms не может быть пустым")
            if delay_ms < 0:
                raise ValueError("Задержка инструмента не может быть отрицательной")
            validated[normalized_tool_name] = delay_ms
        return validated


class AgentEvalExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal[
        "direct_answer",
        "clarify",
        "retrieve_vector",
        "retrieve_hybrid",
        "refuse",
    ] | None = None
    outcome: Literal["success", "refusal", "direct_answer"]
    search_type: Literal["vector", "hybrid"] | None = None
    refusal_reason: str | None = None
    exact_tool_names: list[str] | None = None
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    min_citation_count: int | None = Field(default=None, ge=0)
    max_citation_count: int | None = Field(default=None, ge=0)
    answer_contains: list[str] = Field(default_factory=list)
    cached: bool | None = None
    memory_applied: bool | None = None

    @model_validator(mode="after")
    def validate_citation_bounds(self) -> "AgentEvalExpected":
        if (
            self.min_citation_count is not None
            and self.max_citation_count is not None
            and self.min_citation_count > self.max_citation_count
        ):
            raise ValueError("min_citation_count не может быть больше max_citation_count")
        return self

    @field_validator("refusal_reason")
    @classmethod
    def normalize_refusal_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("exact_tool_names", "required_tools", "forbidden_tools", "answer_contains")
    @classmethod
    def normalize_string_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [item.strip() for item in value if item.strip()]


class AgentEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    description: str | None = None
    request: AgentChatRequest
    expected: AgentEvalExpected
    fixtures: AgentEvalCaseFixtures = Field(default_factory=AgentEvalCaseFixtures)

    @field_validator("id", "category")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("текстовое поле не может быть пустым")
        return normalized


class AgentEvalObserved(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: str | None = None
    outcome: str
    search_type: str
    refusal_reason: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    citation_count: int = Field(default=0, ge=0)
    cached: bool = False
    memory_applied: bool = False
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class AgentEvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str
    passed: bool
    observed: AgentEvalObserved
    checks: dict[str, bool] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)


class AgentEvalMetricThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_value: float | None = None
    max_value: float | None = None


class AgentEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dataset_path: str
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, AgentEvalMetricThreshold] = Field(default_factory=dict)
    threshold_failures: list[str] = Field(default_factory=list)
    cases: list[AgentEvalCaseResult] = Field(default_factory=list)
