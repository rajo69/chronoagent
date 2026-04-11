"""Signals router — GET /api/v1/agents/{agent_id}/signals.

Returns the most-recent *window* behavioral signal records for an agent,
ordered oldest-first so callers can draw time-series charts directly.

``GET /api/v1/agents/{agent_id}/signals?window=50``
    Returns up to *window* rows from ``agent_signal_records`` for *agent_id*,
    ordered by timestamp ascending.  Returns an empty list (not 404) if the
    agent exists in the system but has no recorded signals yet.  Returns HTTP
    404 only when *agent_id* is an empty string.

Response shape::

    {
        "agent_id": "security_reviewer",
        "window": 50,
        "count": 12,
        "signals": [
            {
                "timestamp": "2026-04-10T12:00:00+00:00",
                "task_id": "pr-1",
                "total_latency_ms": 123.4,
                "retrieval_count": 5,
                "token_count": 200,
                "kl_divergence": 0.3,
                "tool_calls": 2,
                "memory_query_entropy": 0.7
            }
        ]
    }
"""

from __future__ import annotations

import datetime
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from chronoagent.api.deps import get_db
from chronoagent.db.models import AgentSignalRecord
from chronoagent.retry import db_retry

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["signals"])

# Type alias so Depends(get_db) never appears in a function-argument default.
DBSession = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SignalPoint(BaseModel):
    """One row of the signal time-series for a single pipeline step.

    Attributes:
        timestamp: UTC timestamp of the step completion.
        task_id: Optional work-unit identifier (e.g. PR id).
        total_latency_ms: End-to-end wall-clock time in ms.
        retrieval_count: Total ChromaDB documents retrieved.
        token_count: Approximate LLM input token count.
        kl_divergence: KL divergence from clean retrieval baseline.
        tool_calls: Number of discrete tool / retrieval calls.
        memory_query_entropy: Normalised Shannon entropy of similarity scores.
    """

    timestamp: datetime.datetime
    task_id: str | None
    total_latency_ms: float
    retrieval_count: int
    token_count: int
    kl_divergence: float
    tool_calls: int
    memory_query_entropy: float


class SignalsResponse(BaseModel):
    """Time-series signal response for one agent.

    Attributes:
        agent_id: The queried agent identifier.
        window: The *window* parameter used in the query.
        count: Number of signal points returned (≤ *window*).
        signals: Signal points ordered oldest-first.
    """

    agent_id: str
    window: int
    count: int
    signals: list[SignalPoint]


# ---------------------------------------------------------------------------
# Retry-wrapped DB helper
# ---------------------------------------------------------------------------


@db_retry
def _fetch_signal_rows(session: Session, stmt: Any) -> list[AgentSignalRecord]:
    return list(session.execute(stmt).scalars().all())


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/agents/{agent_id}/signals",
    response_model=SignalsResponse,
    summary="Get agent signal time-series",
    description=(
        "Returns the most-recent *window* behavioral signal records for the "
        "specified agent, ordered oldest-first."
    ),
)
def get_agent_signals(
    agent_id: str,
    session: DBSession,
    window: Annotated[
        int,
        Query(ge=1, le=10_000, description="Maximum number of signal points to return."),
    ] = 50,
) -> SignalsResponse:
    """Return recent signal records for *agent_id*.

    Args:
        agent_id: Agent identifier (path parameter).
        session: SQLAlchemy session (injected by :func:`~chronoagent.api.deps.get_db`).
        window: Maximum number of most-recent records to return (query param,
            default 50, max 10 000).

    Returns:
        :class:`SignalsResponse` with signal time-series.

    Raises:
        HTTPException 422: If *window* is < 1 or > 10 000 (handled by FastAPI).
    """
    if not agent_id:
        raise HTTPException(status_code=404, detail="agent_id must not be empty")

    stmt = (
        select(AgentSignalRecord)
        .where(AgentSignalRecord.agent_id == agent_id)
        .order_by(AgentSignalRecord.timestamp.desc())
        .limit(window)
    )
    rows = _fetch_signal_rows(session, stmt)
    # Reverse to return oldest-first
    rows.reverse()

    points = [
        SignalPoint(
            timestamp=row.timestamp,
            task_id=row.task_id,
            total_latency_ms=row.total_latency_ms,
            retrieval_count=row.retrieval_count,
            token_count=row.token_count,
            kl_divergence=row.kl_divergence,
            tool_calls=row.tool_calls,
            memory_query_entropy=row.memory_query_entropy,
        )
        for row in rows
    ]

    logger.debug(
        "signals.queried",
        agent_id=agent_id,
        window=window,
        count=len(points),
    )

    return SignalsResponse(
        agent_id=agent_id,
        window=window,
        count=len(points),
        signals=points,
    )
