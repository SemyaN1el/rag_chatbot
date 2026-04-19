from __future__ import annotations

import json
import time
from pathlib import Path
from time import perf_counter
from typing import Any

from app.agent.budget import AgentBudget
from app.agent.evals.metrics import (
    DEFAULT_AGENT_EVAL_THRESHOLDS,
    calculate_agent_eval_metrics,
    evaluate_thresholds,
)
from app.agent.evals.schemas import (
    AgentEvalCase,
    AgentEvalCaseResult,
    AgentEvalObserved,
    AgentEvalReport,
    AgentEvalSearchFixture,
)
from app.agent.policy import AgentToolPolicy
from app.agent.runtime import AgentRuntime
from app.agent.schemas import ToolCall, ToolResult
from app.agent.tools import RegisteredTool, ToolRegistry
from app.agent.workflow import execute_agent_chat

DEFAULT_AGENT_EVAL_DATASET_PATH = Path("data/agent_eval_cases.json")
DEFAULT_AGENT_EVAL_REPORT_PATH = Path("data/agent_eval_report.json")


def load_agent_eval_cases(dataset_path: str | Path) -> list[AgentEvalCase]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл agent eval cases не найден: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Agent eval dataset должен быть непустым JSON-массивом.")

    return [AgentEvalCase.model_validate(item) for item in payload]


def run_agent_eval_suite(
    cases: list[AgentEvalCase],
    *,
    dataset_path: str | Path = DEFAULT_AGENT_EVAL_DATASET_PATH,
    output_path: str | Path | None = DEFAULT_AGENT_EVAL_REPORT_PATH,
) -> AgentEvalReport:
    case_results = [run_agent_eval_case(case) for case in cases]
    passed_cases = sum(1 for result in case_results if result.passed)
    failed_cases = len(case_results) - passed_cases
    metrics = calculate_agent_eval_metrics(case_results)
    threshold_failures = evaluate_thresholds(metrics, DEFAULT_AGENT_EVAL_THRESHOLDS)

    report = AgentEvalReport(
        dataset_path=str(Path(dataset_path)),
        total_cases=len(case_results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=round(passed_cases / len(case_results), 3) if case_results else 0.0,
        metrics=metrics,
        thresholds=DEFAULT_AGENT_EVAL_THRESHOLDS,
        threshold_failures=threshold_failures,
        cases=case_results,
    )
    if output_path is not None:
        write_agent_eval_report(report, output_path)
    return report


def run_agent_eval_case(case: AgentEvalCase) -> AgentEvalCaseResult:
    runtime, invoked_tool_names = _build_runtime_for_case(case)
    execution_started_at = perf_counter()
    response = execute_agent_chat(
        case.request,
        runtime=runtime,
        history_saver=lambda question, answer, search_type: None,
    )
    latency_ms = max(int((perf_counter() - execution_started_at) * 1000), 0)

    observed = AgentEvalObserved(
        route=_extract_route_from_trace(response.trace),
        outcome=_derive_outcome(response),
        search_type=response.search_type,
        refusal_reason=response.refusal_reason,
        tool_names=invoked_tool_names,
        citation_count=len(response.citations),
        cached=response.cached,
        memory_applied=any(step.name == "session_memory_applied" for step in response.trace),
        latency_ms=latency_ms,
        estimated_cost_usd=case.fixtures.estimated_cost_usd,
    )

    checks: dict[str, bool] = {}
    failures: list[str] = []

    if case.expected.route is not None:
        route_match = observed.route == case.expected.route
        checks["route"] = route_match
        if not route_match:
            failures.append(
                f"route: ожидался '{case.expected.route}', получен '{observed.route}'"
            )

    outcome_match = observed.outcome == case.expected.outcome
    checks["outcome"] = outcome_match
    if not outcome_match:
        failures.append(
            f"outcome: ожидался '{case.expected.outcome}', получен '{observed.outcome}'"
        )

    if case.expected.search_type is not None:
        search_type_match = observed.search_type == case.expected.search_type
        checks["search_type"] = search_type_match
        if not search_type_match:
            failures.append(
                f"search_type: ожидался '{case.expected.search_type}', получен '{observed.search_type}'"
            )

    if case.expected.refusal_reason is not None:
        refusal_reason_match = observed.refusal_reason == case.expected.refusal_reason
        checks["refusal_reason"] = refusal_reason_match
        if not refusal_reason_match:
            failures.append(
                "refusal_reason: ожидался "
                f"'{case.expected.refusal_reason}', получен '{observed.refusal_reason}'"
            )

    tool_selection_match = _match_tools(case.expected, observed.tool_names)
    checks["tool_selection"] = tool_selection_match
    if not tool_selection_match:
        failures.append(
            f"tool_selection: ожидались {case.expected.exact_tool_names or case.expected.required_tools}, "
            f"получены {observed.tool_names}"
        )

    citations_match = _match_citations(case, observed.citation_count)
    checks["citations"] = citations_match
    if not citations_match:
        failures.append(
            f"citations: ожидался диапазон "
            f"[{case.expected.min_citation_count}, {case.expected.max_citation_count}], "
            f"получено {observed.citation_count}"
        )

    if case.expected.cached is not None:
        cached_match = observed.cached == case.expected.cached
        checks["cached"] = cached_match
        if not cached_match:
            failures.append(
                f"cached: ожидалось '{case.expected.cached}', получено '{observed.cached}'"
            )

    if case.expected.memory_applied is not None:
        memory_match = observed.memory_applied == case.expected.memory_applied
        checks["memory_applied"] = memory_match
        if not memory_match:
            failures.append(
                "memory_applied: ожидалось "
                f"'{case.expected.memory_applied}', получено '{observed.memory_applied}'"
            )

    answer_contains_match = _match_answer_contains(case.expected.answer_contains, response.answer)
    checks["answer_contains"] = answer_contains_match
    if not answer_contains_match:
        failures.append(
            f"answer_contains: ответ не содержит ожидаемые фрагменты {case.expected.answer_contains}"
        )

    return AgentEvalCaseResult(
        case_id=case.id,
        category=case.category,
        passed=not failures,
        observed=observed,
        checks=checks,
        failures=failures,
    )


def write_agent_eval_report(report: AgentEvalReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_agent_eval_report(report: AgentEvalReport) -> str:
    lines = [
        "=" * 60,
        "AGENT EVAL HARNESS",
        "=" * 60,
        f"Dataset: {report.dataset_path}",
        f"Cases: {report.total_cases}",
        f"Passed: {report.passed_cases}",
        f"Failed: {report.failed_cases}",
        f"Pass rate: {report.pass_rate:.3f}",
        "",
        "Metrics:",
    ]
    for metric_name in sorted(report.metrics):
        lines.append(f"  {metric_name:<28} {report.metrics[metric_name]:>8.3f}")

    if report.threshold_failures:
        lines.extend(["", "Threshold failures:"])
        lines.extend(f"  - {failure}" for failure in report.threshold_failures)

    failed_cases = [result for result in report.cases if not result.passed]
    if failed_cases:
        lines.extend(["", "Failed cases:"])
        for case_result in failed_cases:
            lines.append(f"  - {case_result.case_id}")
            lines.extend(f"      {failure}" for failure in case_result.failures)
    return "\n".join(lines)


def _build_runtime_for_case(case: AgentEvalCase) -> tuple[AgentRuntime, list[str]]:
    fixtures = case.fixtures
    cache_entries = {
        (entry.question, entry.search_type): entry.value
        for entry in fixtures.cache_entries
    }
    search_outputs = fixtures.search_outputs
    invoked_tool_names: list[str] = []
    session_memory_store = fixtures.session_memory

    def register_tool(
        registry: ToolRegistry,
        tool_name: str,
        handler,
    ) -> None:
        registry.register(
            RegisteredTool(
                name=tool_name,
                description=f"Agent eval tool: {tool_name}",
                handler=handler,
            )
        )

    def get_cached_answer(call: ToolCall) -> ToolResult:
        invoked_tool_names.append("get_cached_answer")
        question = str(call.arguments.get("question", "")).strip()
        search_type = str(call.arguments.get("search_type", "")).strip()
        cached_value = cache_entries.get((question, search_type))
        return ToolResult(
            tool_name="get_cached_answer",
            success=True,
            output={
                "question": question,
                "search_type": search_type,
                "cache_hit": cached_value is not None,
                "value": cached_value,
            },
        )

    def get_session_memory(call: ToolCall) -> ToolResult:
        invoked_tool_names.append("get_session_memory")
        session_id = str(call.arguments.get("session_id", "")).strip()
        return ToolResult(
            tool_name="get_session_memory",
            success=True,
            output={
                "session_id": session_id,
                "memory_found": session_memory_store is not None,
                "value": session_memory_store,
            },
        )

    def make_search_tool(search_type: str):
        def handler(call: ToolCall) -> ToolResult:
            invoked_tool_names.append(f"search_{search_type}")
            _maybe_sleep(fixtures.tool_delays_ms.get(f"search_{search_type}", 0))
            payload = _build_search_output(
                search_outputs.get(search_type),
                question=str(call.arguments.get("question", "")).strip(),
                search_type=search_type,
            )
            return ToolResult(
                tool_name=f"search_{search_type}",
                success=True,
                output=payload,
            )

        return handler

    def set_cached_answer(call: ToolCall) -> ToolResult:
        invoked_tool_names.append("set_cached_answer")
        question = str(call.arguments.get("question", "")).strip()
        search_type = str(call.arguments.get("search_type", "")).strip()
        cache_entries[(question, search_type)] = call.arguments.get("result", {})
        return ToolResult(
            tool_name="set_cached_answer",
            success=True,
            output={"cached": True},
        )

    def set_session_memory(call: ToolCall) -> ToolResult:
        nonlocal session_memory_store
        invoked_tool_names.append("set_session_memory")
        session_memory_store = call.arguments.get("memory")
        return ToolResult(
            tool_name="set_session_memory",
            success=True,
            output={"stored": True},
        )

    registry = ToolRegistry()
    register_tool(registry, "get_cached_answer", get_cached_answer)
    register_tool(registry, "get_session_memory", get_session_memory)
    register_tool(registry, "search_vector", make_search_tool("vector"))
    register_tool(registry, "search_hybrid", make_search_tool("hybrid"))
    register_tool(registry, "set_cached_answer", set_cached_answer)
    register_tool(registry, "set_session_memory", set_session_memory)

    budget_config = fixtures.budget or AgentBudget(
        max_steps=32,
        max_tool_calls=10,
        max_runtime_seconds=30.0,
    )
    if isinstance(budget_config, AgentBudget):
        budget = budget_config
    else:
        budget = AgentBudget(**budget_config.model_dump())

    runtime = AgentRuntime(
        registry,
        budget=budget,
        policy=AgentToolPolicy(blocked_tools=frozenset(fixtures.blocked_tools)),
    )
    return runtime, invoked_tool_names


def _build_search_output(
    fixture: AgentEvalSearchFixture | None,
    *,
    question: str,
    search_type: str,
) -> dict[str, Any]:
    if fixture is None:
        sources = [{"page": 1, "text": f"Фрагмент {search_type} по вопросу: {question}"}]
        return {
            "question": question,
            "answer": f"{search_type}:{question}",
            "sources": sources,
            "source_count": len(sources),
            "search_type": search_type,
        }

    sources = fixture.sources
    return {
        "question": question,
        "answer": fixture.answer,
        "sources": sources,
        "source_count": len(sources),
        "search_type": fixture.search_type or search_type,
    }


def _extract_route_from_trace(trace_steps) -> str | None:
    for step in trace_steps:
        if step.name == "routing_decision_applied":
            decision = step.metadata.get("decision")
            return str(decision) if decision else None
    return None


def _derive_outcome(response) -> str:
    route = _extract_route_from_trace(response.trace)
    if response.refusal_reason is not None:
        return "refusal"
    if route == "direct_answer":
        return "direct_answer"
    return "success"


def _match_tools(expected, observed_tool_names: list[str]) -> bool:
    if expected.exact_tool_names is not None and observed_tool_names != expected.exact_tool_names:
        return False

    for tool_name in expected.required_tools:
        if tool_name not in observed_tool_names:
            return False

    for tool_name in expected.forbidden_tools:
        if tool_name in observed_tool_names:
            return False

    return True


def _match_citations(case: AgentEvalCase, citation_count: int) -> bool:
    min_citation_count = case.expected.min_citation_count
    max_citation_count = case.expected.max_citation_count

    if min_citation_count is not None and citation_count < min_citation_count:
        return False
    if max_citation_count is not None and citation_count > max_citation_count:
        return False
    return True


def _match_answer_contains(expected_fragments: list[str], answer: str) -> bool:
    if not expected_fragments:
        return True

    normalized_answer = answer.lower()
    return all(fragment.lower() in normalized_answer for fragment in expected_fragments)


def _maybe_sleep(delay_ms: int) -> None:
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)
