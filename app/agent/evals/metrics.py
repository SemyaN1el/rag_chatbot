from __future__ import annotations

import math
from typing import Iterable

from app.agent.evals.schemas import AgentEvalCaseResult, AgentEvalMetricThreshold

DEFAULT_AGENT_EVAL_THRESHOLDS: dict[str, AgentEvalMetricThreshold] = {
    "route_accuracy": AgentEvalMetricThreshold(min_value=0.95),
    "tool_selection_accuracy": AgentEvalMetricThreshold(min_value=0.95),
    "refusal_reason_accuracy": AgentEvalMetricThreshold(min_value=0.90),
    "citation_validity": AgentEvalMetricThreshold(min_value=1.0),
    "task_success_rate": AgentEvalMetricThreshold(min_value=0.90),
    "cache_hit_rate": AgentEvalMetricThreshold(min_value=0.95),
    "latency_ms_p95": AgentEvalMetricThreshold(max_value=1000.0),
    "estimated_cost_usd_mean": AgentEvalMetricThreshold(max_value=0.0),
}


def calculate_agent_eval_metrics(case_results: list[AgentEvalCaseResult]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "task_success_rate": _mean(result.passed for result in case_results),
        "route_accuracy": _mean(
            result.checks.get("route", True)
            for result in case_results
            if "route" in result.checks
        ),
        "tool_selection_accuracy": _mean(
            result.checks.get("tool_selection", True)
            for result in case_results
            if "tool_selection" in result.checks
        ),
        "refusal_reason_accuracy": _mean(
            result.checks.get("refusal_reason", True)
            for result in case_results
            if "refusal_reason" in result.checks
        ),
        "citation_validity": _mean(
            result.checks.get("citations", True)
            for result in case_results
            if "citations" in result.checks
        ),
        "cache_hit_rate": _mean(
            result.checks.get("cached", True)
            for result in case_results
            if "cached" in result.checks
        ),
        "search_type_accuracy": _mean(
            result.checks.get("search_type", True)
            for result in case_results
            if "search_type" in result.checks
        ),
        "memory_usage_accuracy": _mean(
            result.checks.get("memory_applied", True)
            for result in case_results
            if "memory_applied" in result.checks
        ),
    }

    latencies = [result.observed.latency_ms for result in case_results]
    costs = [result.observed.estimated_cost_usd for result in case_results]
    metrics["latency_ms_p50"] = _percentile(latencies, 50)
    metrics["latency_ms_p95"] = _percentile(latencies, 95)
    metrics["estimated_cost_usd_mean"] = _rounded(sum(costs) / len(costs)) if costs else 0.0
    metrics["estimated_cost_usd_total"] = _rounded(sum(costs))
    return metrics


def evaluate_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, AgentEvalMetricThreshold],
) -> list[str]:
    failures: list[str] = []
    for metric_name, threshold in thresholds.items():
        metric_value = metrics.get(metric_name)
        if metric_value is None or math.isnan(metric_value):
            failures.append(f"{metric_name}: значение метрики отсутствует")
            continue

        if threshold.min_value is not None and metric_value < threshold.min_value:
            failures.append(
                f"{metric_name}: {metric_value:.3f} < {threshold.min_value:.3f}"
            )
        if threshold.max_value is not None and metric_value > threshold.max_value:
            failures.append(
                f"{metric_name}: {metric_value:.3f} > {threshold.max_value:.3f}"
            )
    return failures


def _mean(values: Iterable[bool]) -> float:
    materialized = list(values)
    if not materialized:
        return 1.0
    return _rounded(sum(1.0 for value in materialized if value) / len(materialized))


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return _rounded(ordered[0])

    rank = (percentile / 100) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return _rounded(ordered[lower_index])

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    interpolated = lower_value + (upper_value - lower_value) * (rank - lower_index)
    return _rounded(interpolated)


def _rounded(value: float) -> float:
    return round(value, 3)
