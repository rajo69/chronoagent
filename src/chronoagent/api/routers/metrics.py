"""Prometheus scrape endpoint (Phase 8 task 8.3).

Exposes a single endpoint:

``GET /metrics``
    Renders the current value of every metric registered on
    ``app.state.metrics`` in the Prometheus text exposition format.

The endpoint also refreshes the poll-driven gauges (pending escalations,
quarantine size, memory baseline state, mean system health) from the
matching ``app.state`` objects before rendering.  Event-driven metrics
(health-update counters, allocation counters, escalation counters) are
populated by bus subscribers wired up in :mod:`chronoagent.main`.

The router is not included in the OpenAPI schema because the payload is
not JSON; tools like Prometheus discover it by scraping a fixed URL.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.db.models import EscalationRecord
from chronoagent.memory.integrity import MemoryIntegrityModule
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.observability.metrics import CONTENT_TYPE_LATEST, ChronoAgentMetrics
from chronoagent.retry import db_retry
from chronoagent.scorer.health_scorer import HealthUpdate, TemporalHealthScorer

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(tags=["metrics"])


def _refresh_poll_gauges(request: Request, metrics: ChronoAgentMetrics) -> None:
    """Snapshot state-backed values into the metrics gauges.

    Event-driven metrics (counters, histograms, per-agent gauges) are
    kept up to date by bus subscribers.  The gauges in this function
    are not event-driven, so we refresh them on every scrape.

    Missing dependencies on ``app.state`` are tolerated silently: the
    corresponding gauge simply keeps its last value.  This keeps
    ``/metrics`` scrapeable even during partial-startup states.

    Args:
        request: The incoming FastAPI request (for ``app.state``).
        metrics: The :class:`ChronoAgentMetrics` sink to refresh.
    """
    state = request.app.state

    scorer: TemporalHealthScorer | None = getattr(state, "health_scorer", None)
    if scorer is not None:
        health_map: dict[str, HealthUpdate] = scorer.get_all_health()
        if health_map:
            mean = sum(u.health for u in health_map.values()) / len(health_map)
            metrics.set_system_health(mean)
        else:
            metrics.set_system_health(1.0)

    quarantine_store: QuarantineStore | None = getattr(state, "quarantine_store", None)
    if quarantine_store is not None:
        metrics.set_quarantine_size(quarantine_store.count)

    integrity_module: MemoryIntegrityModule | None = getattr(state, "integrity_module", None)
    if integrity_module is not None:
        metrics.set_memory_baseline_size(integrity_module.baseline_size)
        metrics.set_memory_pending_refit(integrity_module.pending_refit_count)

    session_factory: sessionmaker[Session] | None = getattr(state, "session_factory", None)
    if session_factory is not None:
        pending = _fetch_pending_count(session_factory)
        metrics.set_pending_escalations(pending)


@db_retry
def _fetch_pending_count(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as session:
        pending: int = session.execute(
            select(func.count())
            .select_from(EscalationRecord)
            .where(EscalationRecord.status == "pending")
        ).scalar_one()
        return pending


@router.get(
    "/metrics",
    summary="Prometheus exposition-format metrics",
    description=(
        "Renders the current value of every ChronoAgent metric in the "
        "Prometheus text exposition format.  Event-driven metrics are "
        "populated by bus subscribers; state-backed gauges are refreshed "
        "on every scrape."
    ),
    include_in_schema=False,
)
def get_metrics(request: Request) -> Response:
    """Return the Prometheus exposition-format payload.

    Args:
        request: Incoming FastAPI request; ``app.state.metrics`` must
            already be initialised by the lifespan.

    Returns:
        :class:`~fastapi.Response` whose body is the exposition payload
        and whose ``Content-Type`` header is the canonical Prometheus
        text format.

    Raises:
        :class:`~fastapi.HTTPException` 503 if the metrics sink has not
        yet been initialised on ``app.state``.
    """
    metrics: ChronoAgentMetrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics sink not initialised")

    _refresh_poll_gauges(request, metrics)

    payload = metrics.render()
    logger.debug("metrics.scraped", bytes=len(payload))
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
