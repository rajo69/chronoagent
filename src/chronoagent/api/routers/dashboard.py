"""Dashboard backend router (Phase 8 task 8.1).

Exposes read-only endpoints for the observability dashboard frontend:

``GET /dashboard/api/agents``
    List all known agents with their current health score, BOCPD/Chronos
    components, and the timestamp of their most recent signal record.
    Ordered most-at-risk first (ascending health).

``GET /dashboard/api/agents/{agent_id}/timeline``
    Time-series of behavioral signals for one agent, drawn from
    :class:`~chronoagent.db.models.AgentSignalRecord`.  Accepts ``limit``
    (max rows, 1..1000, default 50) and ``since`` (ISO 8601 datetime filter)
    query params.  Returns empty list for unknown agents rather than 404.

``GET /dashboard/api/allocations``
    Recent task-allocation decisions with full bid ledger, health snapshot,
    and rationale.  Ordered newest-first.  Accepts ``limit`` (1..200,
    default 20).

``GET /dashboard/api/memory``
    Current state of the memory integrity subsystem: baseline status, signal
    weights, retrieval stats, quarantine count and IDs.

``GET /dashboard/api/escalations``
    Combined view of the escalation queue: all pending records plus the most
    recently resolved records.  Accepts ``limit`` (default 20) and an optional
    ``status`` filter; when a filter is supplied only the matching list is
    populated.

``WebSocket /dashboard/ws/live``
    Pushes a :class:`LiveUpdate` JSON frame every 2 seconds.  Frame contains
    system-level aggregates (health, agent count, pending escalations,
    quarantine count) plus the per-agent summary list so the frontend can
    drive all gauges from a single socket.

All endpoints read from ``app.state``; the lifespan in
:mod:`chronoagent.main` initialises every dependency before the first
request arrives.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Annotated, Any

import structlog
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.dashboard import INDEX_HTML
from chronoagent.db.models import AgentSignalRecord, AllocationAuditRecord, EscalationRecord
from chronoagent.memory.integrity import MemoryIntegrityModule
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.scorer.health_scorer import HealthUpdate, TemporalHealthScorer

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AgentSummary(BaseModel):
    """Health + last-seen summary for one agent.

    Attributes:
        agent_id: Agent identifier.
        health: Current health score in [0, 1]; ``None`` before first update.
        bocpd_score: Raw BOCPD changepoint probability, or ``None``.
        chronos_score: Raw Chronos anomaly score, or ``None``.
        components: Which scorer components contributed: ``"ensemble"``,
            ``"bocpd_only"``, ``"chronos_only"``, or ``"none"``.
        last_signal_at: UTC timestamp of the agent's most recent signal
            record; ``None`` if no signals have been persisted yet.
    """

    agent_id: str
    health: float | None
    bocpd_score: float | None
    chronos_score: float | None
    components: str
    last_signal_at: datetime.datetime | None


class AgentListResponse(BaseModel):
    """All known agents with aggregate system health.

    Attributes:
        agents: Per-agent summaries, ordered ascending by health
            (most-at-risk first; agents with ``None`` health sorted last).
        count: Number of agents in the list.
        system_health: Mean health across agents that have a score;
            ``1.0`` when no scores are available yet.
    """

    agents: list[AgentSummary]
    count: int
    system_health: float


class TimelinePoint(BaseModel):
    """One row of behavioral-signal history for an agent.

    Attributes:
        timestamp: UTC time the step completed.
        total_latency_ms: End-to-end wall-clock time in milliseconds.
        retrieval_count: ChromaDB documents retrieved.
        token_count: Approximate LLM input tokens.
        kl_divergence: KL divergence from clean retrieval baseline.
        tool_calls: Discrete tool / retrieval calls.
        memory_query_entropy: Normalised Shannon entropy of similarity scores.
    """

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime.datetime
    total_latency_ms: float
    retrieval_count: int
    token_count: int
    kl_divergence: float
    tool_calls: int
    memory_query_entropy: float


class TimelineResponse(BaseModel):
    """Behavioral-signal time-series for one agent.

    Attributes:
        agent_id: Agent identifier.
        points: Signal records ordered chronologically (oldest first).
        count: Number of points returned.
    """

    agent_id: str
    points: list[TimelinePoint]
    count: int


class AllocationEntry(BaseModel):
    """Single task-allocation decision with full bid ledger.

    Attributes:
        id: Auto-increment primary key.
        task_id: Opaque task identifier.
        task_type: One of the registered task types.
        assigned_agent: Winning agent ID, or ``None`` if escalated.
        escalated: Whether the round triggered human escalation.
        all_bids: Full bid ledger in agent-ID order.
        health_snapshot: Agent health scores at decision time.
        rationale: Human-readable explanation of the outcome.
        threshold: Minimum-bid threshold that was in force.
        timestamp: UTC timestamp of the decision.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: str
    task_type: str
    assigned_agent: str | None
    escalated: bool
    all_bids: list[dict[str, Any]]
    health_snapshot: dict[str, float]
    rationale: str
    threshold: float
    timestamp: datetime.datetime


class AllocationListResponse(BaseModel):
    """Recent allocation decisions, newest first.

    Attributes:
        allocations: List of allocation entries.
        count: Number of entries returned.
    """

    allocations: list[AllocationEntry]
    count: int


class MemoryStatusResponse(BaseModel):
    """Current state of the memory integrity subsystem.

    Attributes:
        baseline_fitted: Whether the IsolationForest baseline has been fitted.
        baseline_size: Number of clean embeddings the baseline was fitted on.
        pending_refit_count: New clean docs buffered since the last refit.
        total_retrievals: Lifetime count of retrieval events recorded.
        flag_threshold: Aggregate score at which a document is flagged.
        weights: Normalised signal weights, keyed by signal name.
        quarantine_count: Documents currently in the quarantine collection.
        quarantined_ids: ChromaDB IDs of every quarantined document.
    """

    baseline_fitted: bool
    baseline_size: int
    pending_refit_count: int
    total_retrievals: int
    flag_threshold: float
    weights: dict[str, float]
    quarantine_count: int
    quarantined_ids: list[str]


class EscalationEntry(BaseModel):
    """Single escalation record for the dashboard queue view.

    Attributes:
        id: UUID4 hex primary key.
        agent_id: Agent that triggered the escalation.
        trigger: ``"low_health"`` or ``"quarantine_event"``.
        status: Current status (``"pending"``, ``"approved"``, etc.).
        context: JSON context snapshot from escalation time.
        resolution_notes: Human-provided notes, or ``None``.
        created_at: UTC creation timestamp.
        resolved_at: UTC resolution timestamp, or ``None`` if still pending.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    trigger: str
    status: str
    context: dict[str, Any]
    resolution_notes: str | None
    created_at: datetime.datetime
    resolved_at: datetime.datetime | None


class EscalationQueueResponse(BaseModel):
    """Combined escalation queue: pending items plus recently resolved.

    Attributes:
        pending: Records with ``status == "pending"``, newest first.
        recent_resolved: Records with any resolved status, ordered by
            ``resolved_at`` descending.
        pending_count: Length of the ``pending`` list.
        resolved_count: Length of the ``recent_resolved`` list.
    """

    pending: list[EscalationEntry]
    recent_resolved: list[EscalationEntry]
    pending_count: int
    resolved_count: int


class LiveUpdate(BaseModel):
    """Single WebSocket push frame for the live dashboard feed.

    Attributes:
        timestamp: UTC time the frame was assembled.
        system_health: Mean health across agents that have a score.
        agent_count: Total number of tracked agents.
        pending_escalations: Count of pending escalation records.
        quarantine_count: Documents currently quarantined.
        agents: Per-agent summaries (``last_signal_at`` omitted for speed).
    """

    timestamp: datetime.datetime
    system_health: float
    agent_count: int
    pending_escalations: int
    quarantine_count: int
    agents: list[AgentSummary]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _components_label(update: HealthUpdate) -> str:
    """Return a human-readable label for which scorer components contributed."""
    have_b = update.bocpd_score is not None
    have_c = update.chronos_score is not None
    if have_b and have_c:
        return "ensemble"
    if have_b:
        return "bocpd_only"
    if have_c:
        return "chronos_only"
    return "none"


def _update_to_summary(
    update: HealthUpdate, last_signal_at: datetime.datetime | None
) -> AgentSummary:
    """Convert a :class:`HealthUpdate` to an :class:`AgentSummary`."""
    return AgentSummary(
        agent_id=update.agent_id,
        health=update.health,
        bocpd_score=update.bocpd_score,
        chronos_score=update.chronos_score,
        components=_components_label(update),
        last_signal_at=last_signal_at,
    )


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_session_factory(request: Request) -> sessionmaker[Session]:
    """Inject the shared SQLAlchemy session factory from ``app.state``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not yet initialised.
    """
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="Session factory not initialised")
    return factory  # type: ignore[no-any-return]


def _get_scorer(request: Request) -> TemporalHealthScorer:
    """Inject the :class:`~chronoagent.scorer.health_scorer.TemporalHealthScorer` from app.state.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not yet initialised.
    """
    scorer = getattr(request.app.state, "health_scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Health scorer not initialised")
    return scorer  # type: ignore[no-any-return]


def _get_integrity_module(request: Request) -> MemoryIntegrityModule:
    """Inject the :class:`~chronoagent.memory.integrity.MemoryIntegrityModule` from ``app.state``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not yet initialised.
    """
    module = getattr(request.app.state, "integrity_module", None)
    if module is None:
        raise HTTPException(status_code=503, detail="Integrity module not initialised")
    return module  # type: ignore[no-any-return]


def _get_quarantine_store(request: Request) -> QuarantineStore:
    """Inject the :class:`~chronoagent.memory.quarantine.QuarantineStore` from ``app.state``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not yet initialised.
    """
    store = getattr(request.app.state, "quarantine_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Quarantine store not initialised")
    return store  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_class=HTMLResponse,
    summary="Serve the dashboard single-page HTML",
    description=(
        "Returns the self-contained dashboard HTML (Chart.js via CDN, no build step). "
        "All data is fetched client-side from the ``/dashboard/api/*`` endpoints and "
        "the ``/dashboard/ws/live`` WebSocket."
    ),
    include_in_schema=False,
)
def dashboard_index() -> FileResponse:
    """Return the dashboard single-page HTML asset.

    Returns:
        :class:`~fastapi.responses.FileResponse` pointing at the bundled
        ``src/chronoagent/dashboard/static/index.html`` file.

    Raises:
        :class:`~fastapi.HTTPException` 500 if the bundled asset is missing
        (indicates a packaging error).
    """
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=500, detail="Dashboard index.html missing from package")
    return FileResponse(INDEX_HTML, media_type="text/html")


@router.get(
    "/api/agents",
    response_model=AgentListResponse,
    summary="List all agents with current health",
    description=(
        "Returns every known agent (from scorer state and signal history) with "
        "current health score and last-seen timestamp.  Ordered most-at-risk first."
    ),
)
def list_agents(
    session_factory: Annotated[sessionmaker[Session], Depends(_get_session_factory)],
    scorer: Annotated[TemporalHealthScorer, Depends(_get_scorer)],
) -> AgentListResponse:
    """List all tracked agents with health summary.

    Merges agent IDs known to the health scorer with agent IDs that have
    persisted signal records, so agents that have emitted signals but not
    yet received a health update (or vice versa) are both included.

    Args:
        session_factory: Shared SQLAlchemy session factory (injected).
        scorer: Temporal health scorer (injected).

    Returns:
        :class:`AgentListResponse` with per-agent summaries sorted ascending
        by health (most-at-risk first; agents without a score come last).
    """
    health_map: dict[str, HealthUpdate] = scorer.get_all_health()

    # Fetch distinct agent IDs from DB and their most-recent signal timestamp.
    with session_factory() as session:
        last_seen_rows = session.execute(
            select(
                AgentSignalRecord.agent_id,
                func.max(AgentSignalRecord.timestamp).label("last_at"),
            ).group_by(AgentSignalRecord.agent_id)
        ).all()

    last_seen: dict[str, datetime.datetime] = {row.agent_id: row.last_at for row in last_seen_rows}

    all_ids = set(health_map) | set(last_seen)

    summaries: list[AgentSummary] = []
    for agent_id in all_ids:
        update = health_map.get(agent_id)
        if update is not None:
            summaries.append(_update_to_summary(update, last_seen.get(agent_id)))
        else:
            # Agent has signal records but no health update yet.
            summaries.append(
                AgentSummary(
                    agent_id=agent_id,
                    health=None,
                    bocpd_score=None,
                    chronos_score=None,
                    components="none",
                    last_signal_at=last_seen.get(agent_id),
                )
            )

    # Sort ascending by health so most-at-risk surfaces first; None scores last.
    summaries.sort(key=lambda s: (s.health is None, s.health or 0.0))

    scored = [s for s in summaries if s.health is not None]
    system_health = (
        sum(s.health for s in scored if s.health is not None) / len(scored) if scored else 1.0
    )

    logger.debug("dashboard.agents.listed", count=len(summaries))
    return AgentListResponse(
        agents=summaries,
        count=len(summaries),
        system_health=round(system_health, 4),
    )


@router.get(
    "/api/agents/{agent_id}/timeline",
    response_model=TimelineResponse,
    summary="Get agent behavioral signal timeline",
    description=(
        "Returns a time-series of behavioral signals for one agent from the DB. "
        "Returns an empty list for unknown agents rather than 404."
    ),
)
def get_agent_timeline(
    agent_id: str,
    session_factory: Annotated[sessionmaker[Session], Depends(_get_session_factory)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    since: Annotated[datetime.datetime | None, Query()] = None,
) -> TimelineResponse:
    """Return behavioral signal history for ``agent_id``.

    Args:
        agent_id: Agent identifier (path parameter).
        session_factory: Shared SQLAlchemy session factory (injected).
        limit: Maximum number of points to return (1..1000, default 50).
        since: Optional lower bound on ``timestamp`` (inclusive).

    Returns:
        :class:`TimelineResponse` with signal records ordered chronologically
        (oldest first).  Empty list when the agent has no records.
    """
    stmt = select(AgentSignalRecord).where(AgentSignalRecord.agent_id == agent_id)
    if since is not None:
        stmt = stmt.where(AgentSignalRecord.timestamp >= since)
    stmt = stmt.order_by(AgentSignalRecord.timestamp.desc()).limit(limit)

    with session_factory() as session:
        rows = list(session.execute(stmt).scalars().all())

    # Reverse so the response is chronological (oldest first).
    rows.reverse()
    points = [TimelinePoint.model_validate(r) for r in rows]

    logger.debug("dashboard.timeline.queried", agent_id=agent_id, count=len(points))
    return TimelineResponse(agent_id=agent_id, points=points, count=len(points))


@router.get(
    "/api/allocations",
    response_model=AllocationListResponse,
    summary="List recent allocation decisions",
    description="Recent task-allocation decisions with full bid ledger, newest first.",
)
def list_allocations(
    session_factory: Annotated[sessionmaker[Session], Depends(_get_session_factory)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> AllocationListResponse:
    """Return recent task-allocation decisions.

    Args:
        session_factory: Shared SQLAlchemy session factory (injected).
        limit: Maximum rows to return (1..200, default 20).

    Returns:
        :class:`AllocationListResponse` with decisions ordered newest-first.
    """
    stmt = (
        select(AllocationAuditRecord).order_by(AllocationAuditRecord.timestamp.desc()).limit(limit)
    )
    with session_factory() as session:
        rows = list(session.execute(stmt).scalars().all())

    allocations = [AllocationEntry.model_validate(r) for r in rows]
    logger.debug("dashboard.allocations.listed", count=len(allocations))
    return AllocationListResponse(allocations=allocations, count=len(allocations))


@router.get(
    "/api/memory",
    response_model=MemoryStatusResponse,
    summary="Get memory integrity status",
    description="Returns the current state of the memory integrity subsystem and quarantine store.",
)
def get_memory_status(
    integrity_module: Annotated[MemoryIntegrityModule, Depends(_get_integrity_module)],
    quarantine_store: Annotated[QuarantineStore, Depends(_get_quarantine_store)],
) -> MemoryStatusResponse:
    """Return current memory integrity and quarantine status.

    Args:
        integrity_module: Memory integrity module (injected).
        quarantine_store: Quarantine store (injected).

    Returns:
        :class:`MemoryStatusResponse` with baseline state, signal weights,
        retrieval stats, and quarantine summary.
    """
    logger.debug("dashboard.memory.queried")
    return MemoryStatusResponse(
        baseline_fitted=integrity_module.baseline_fitted,
        baseline_size=integrity_module.baseline_size,
        pending_refit_count=integrity_module.pending_refit_count,
        total_retrievals=integrity_module.total_retrievals,
        flag_threshold=integrity_module.flag_threshold,
        weights=integrity_module.weights,
        quarantine_count=quarantine_store.count,
        quarantined_ids=quarantine_store.list_ids(),
    )


@router.get(
    "/api/escalations",
    response_model=EscalationQueueResponse,
    summary="Get escalation queue",
    description=(
        "Returns pending escalations plus recently resolved ones. "
        "Supply ``?status=<value>`` to filter to a single status."
    ),
)
def get_escalation_queue(
    session_factory: Annotated[sessionmaker[Session], Depends(_get_session_factory)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    status: Annotated[str | None, Query()] = None,
) -> EscalationQueueResponse:
    """Return the escalation queue split into pending and recently resolved.

    When ``status`` is ``None`` (default) both lists are populated up to
    ``limit`` entries each.  When ``status`` is provided only the matching
    list is populated; the other list is returned empty.

    Args:
        session_factory: Shared SQLAlchemy session factory (injected).
        limit: Maximum rows per list (1..200, default 20).
        status: Optional exact status filter (e.g. ``"approved"``).

    Returns:
        :class:`EscalationQueueResponse` with pending and resolved lists.
    """
    pending: list[EscalationEntry] = []
    recent_resolved: list[EscalationEntry] = []

    with session_factory() as session:
        if status is None or status == "pending":
            pending_stmt = (
                select(EscalationRecord)
                .where(EscalationRecord.status == "pending")
                .order_by(EscalationRecord.created_at.desc())
                .limit(limit)
            )
            pending_rows = list(session.execute(pending_stmt).scalars().all())
            pending = [EscalationEntry.model_validate(r) for r in pending_rows]

        if status is None or status != "pending":
            resolved_filter = (
                EscalationRecord.status == status
                if status is not None
                else EscalationRecord.status != "pending"
            )
            resolved_stmt = (
                select(EscalationRecord)
                .where(resolved_filter)
                .order_by(EscalationRecord.resolved_at.desc())
                .limit(limit)
            )
            resolved_rows = list(session.execute(resolved_stmt).scalars().all())
            recent_resolved = [EscalationEntry.model_validate(r) for r in resolved_rows]

    logger.debug(
        "dashboard.escalations.queried",
        status_filter=status,
        pending=len(pending),
        resolved=len(recent_resolved),
    )
    return EscalationQueueResponse(
        pending=pending,
        recent_resolved=recent_resolved,
        pending_count=len(pending),
        resolved_count=len(recent_resolved),
    )


# ---------------------------------------------------------------------------
# WebSocket: live feed
# ---------------------------------------------------------------------------


def _build_live_update(state: Any) -> LiveUpdate | None:
    """Assemble a :class:`LiveUpdate` frame from current ``app.state``.

    Returns ``None`` when any required dependency is not yet initialised,
    which causes the WebSocket handler to send an error frame and close.

    Args:
        state: The ``app.state`` object from a :class:`~fastapi.FastAPI` instance.

    Returns:
        :class:`LiveUpdate` frame, or ``None`` if state is not ready.
    """
    scorer: TemporalHealthScorer | None = getattr(state, "health_scorer", None)
    quarantine_store: QuarantineStore | None = getattr(state, "quarantine_store", None)
    session_factory: sessionmaker[Session] | None = getattr(state, "session_factory", None)

    if scorer is None or quarantine_store is None or session_factory is None:
        return None

    health_map: dict[str, HealthUpdate] = scorer.get_all_health()
    agents = sorted(
        [
            AgentSummary(
                agent_id=u.agent_id,
                health=u.health,
                bocpd_score=u.bocpd_score,
                chronos_score=u.chronos_score,
                components=_components_label(u),
                last_signal_at=None,  # omitted for speed on live ticks
            )
            for u in health_map.values()
        ],
        key=lambda s: (s.health is None, s.health or 0.0),
    )

    scored = [s for s in agents if s.health is not None]
    system_health = (
        sum(s.health for s in scored if s.health is not None) / len(scored) if scored else 1.0
    )

    with session_factory() as session:
        pending_count: int = session.execute(
            select(func.count())
            .select_from(EscalationRecord)
            .where(EscalationRecord.status == "pending")
        ).scalar_one()

    return LiveUpdate(
        timestamp=datetime.datetime.now(datetime.UTC),
        system_health=round(system_health, 4),
        agent_count=len(agents),
        pending_escalations=pending_count,
        quarantine_count=quarantine_store.count,
        agents=agents,
    )


@router.websocket("/ws/live")
async def live_updates(websocket: WebSocket) -> None:
    """Push live dashboard update frames every 2 seconds.

    Uses async polling rather than bus subscription to avoid threading
    hazards with the synchronous :class:`~chronoagent.messaging.local_bus.LocalBus`.
    Each frame is a :class:`LiveUpdate` serialised to JSON.

    If any required ``app.state`` dependency is not yet initialised the
    handler sends ``{"error": "not_ready"}`` and closes the connection.

    Args:
        websocket: The incoming WebSocket connection.
    """
    await websocket.accept()
    state = websocket.app.state
    try:
        while True:
            frame = _build_live_update(state)
            if frame is None:
                await websocket.send_json({"error": "not_ready"})
                break
            await websocket.send_json(frame.model_dump(mode="json"))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
