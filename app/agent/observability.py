from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

AGENT_LOGGER_NAME = "app.agent"


def get_agent_logger() -> logging.Logger:
    logger = logging.getLogger(AGENT_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        cleaned[key] = value
    return cleaned


def log_agent_event(
    event: str,
    *,
    request_id: str,
    session_id: str,
    level: str = "info",
    search_type: str | None = None,
    route: str | None = None,
    tool_name: str | None = None,
    status: str | None = None,
    outcome: str | None = None,
    refusal_reason: str | None = None,
    cached: bool | None = None,
    confidence: float | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = _clean_payload(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "request_id": request_id,
            "session_id": session_id,
            "search_type": search_type,
            "route": route,
            "tool_name": tool_name,
            "status": status,
            "outcome": outcome,
            "refusal_reason": refusal_reason,
            "cached": cached,
            "confidence": confidence,
            "duration_ms": duration_ms,
            "metadata": metadata or {},
        }
    )
    logger = get_agent_logger()
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(json.dumps(payload, ensure_ascii=False, sort_keys=True))
