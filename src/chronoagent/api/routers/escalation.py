"""Human escalation API router (Phase 7 task 7.3).

Exposes two endpoints:

``GET /api/v1/escalations``
    List escalation records, optionally filtered by status.  Default filter
    is ``status=pending`` so the operator sees only actionable items.
    Pass ``status=all`` to return every record regardless of status.

``POST /api/v1/escalations/{escalation_id}/resolve``
    Resolve a pending escalation with one of three resolutions:
    ``"approve"``, ``"reject"``, or ``"modify"``.  Accepts optional
    free-form notes.  Publishing a ``"escalation.resolved"`` bus event after
    each successful resolution allows downstream components to react.

Both endpoints read ``app.state.session_factory``, ``app.state.bus``, and
``app.state.audit_logger``; the lifespan in :mod:`chronoagent.main` initialises
these before the first request arrives.
"""

from __future__ import annotations

import datetime
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.db.models import EscalationRecord
from chronoagent.escalation.audit import AuditTrailLogger
from chronoagent.escalation.escalation_manager import RESOLVED_CHANNEL
from chronoagent.messaging.bus import MessageBus
from chronoagent.retry import db_retry

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["escalations"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class EscalationResponse(BaseModel):
    """Single escalation record returned by the API.

    Attributes:
        id: UUID4 hex string primary key.
        agent_id: Agent that triggered the escalation.
        trigger: ``"low_health"`` or ``"quarantine_event"``.
        status: Current status of the escalation record.
        context: JSON context snapshot assembled at escalation time.
        resolution_notes: Human-provided notes from the resolver.
        created_at: UTC timestamp of escalation creation.
        resolved_at: UTC timestamp of resolution, or ``None`` if still pending.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    trigger: str
    status: str
    context: dict  # type: ignore[type-arg]
    resolution_notes: str | None
    created_at: datetime.datetime
    resolved_at: datetime.datetime | None


class EscalationListResponse(BaseModel):
    """Paginated list of escalation records.

    Attributes:
        escalations: List of matching records ordered newest-first.
        count: Total number of records in this response (not total in DB).
    """

    escalations: list[EscalationResponse]
    count: int


class ResolveRequest(BaseModel):
    """Request body for POST /api/v1/escalations/{id}/resolve.

    Attributes:
        resolution: Human decision.
        notes: Optional free-form text (max 2048 characters).
    """

    resolution: Literal["approve", "reject", "modify"]
    notes: str | None = Field(default=None, max_length=2048)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_session_factory(request: Request) -> sessionmaker[Session]:
    """Inject the shared SQLAlchemy session factory.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`~sqlalchemy.orm.sessionmaker` stored on
        ``app.state.session_factory``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not initialised.
    """
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="Session factory not initialised")
    return factory  # type: ignore[no-any-return]


def _get_bus(request: Request) -> MessageBus:
    """Inject the shared message bus.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`~chronoagent.messaging.bus.MessageBus` stored on
        ``app.state.bus``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not initialised.
    """
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        raise HTTPException(status_code=503, detail="Message bus not initialised")
    return bus  # type: ignore[no-any-return]


def _get_audit_logger(request: Request) -> AuditTrailLogger:
    """Inject the shared audit trail logger.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`~chronoagent.escalation.audit.AuditTrailLogger` stored on
        ``app.state.audit_logger``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if not initialised.
    """
    audit = getattr(request.app.state, "audit_logger", None)
    if audit is None:
        raise HTTPException(status_code=503, detail="Audit logger not initialised")
    return audit  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Resolution status map
# ---------------------------------------------------------------------------

_RESOLUTION_TO_STATUS: dict[str, str] = {
    "approve": "approved",
    "reject": "rejected",
    "modify": "modified",
}

# ---------------------------------------------------------------------------
# Retry-wrapped DB helpers
# ---------------------------------------------------------------------------
#
# Each helper opens its own session and is decorated with ``db_retry`` so
# only the database transaction retries on transient ``OperationalError``
# failures -- never the surrounding bus publish or audit log side effects.


@db_retry
def _list_escalations_db(
    session_factory: sessionmaker[Session],
    stmt: Any,
) -> list[EscalationRecord]:
    with session_factory() as session:
        return list(session.execute(stmt).scalars().all())


@db_retry
def _apply_resolution(
    session_factory: sessionmaker[Session],
    *,
    escalation_id: str,
    new_status: str,
    notes: str | None,
    resolved_at: datetime.datetime,
) -> tuple[EscalationResponse, str]:
    with session_factory() as session:
        row = session.get(EscalationRecord, escalation_id)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Escalation {escalation_id!r} not found",
            )
        if row.status != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Escalation {escalation_id!r} is already resolved (status={row.status!r})",
            )

        row.status = new_status
        row.resolution_notes = notes
        row.resolved_at = resolved_at
        session.commit()

        resolved_row = EscalationResponse.model_validate(row)
        agent_id = row.agent_id
    return resolved_row, agent_id


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/escalations", response_model=EscalationListResponse)
def list_escalations(
    session_factory: Annotated[sessionmaker[Session], Depends(_get_session_factory)],
    status: str | None = Query(default="pending"),
) -> EscalationListResponse:
    """List escalation records, optionally filtered by status.

    Pass ``?status=pending`` (the default) to show only actionable items.
    Pass ``?status=all`` or omit to return all records regardless of status.
    Any other value is used as an exact status filter (e.g. ``?status=approved``).

    Args:
        session_factory: Shared SQLAlchemy session factory (injected).
        status: Status filter.  ``"pending"`` is the default.
            ``"all"`` disables filtering.

    Returns:
        :class:`EscalationListResponse` with matching records ordered newest-first.
    """
    stmt = select(EscalationRecord).order_by(EscalationRecord.created_at.desc())
    if status and status != "all":
        stmt = stmt.where(EscalationRecord.status == status)

    rows = _list_escalations_db(session_factory, stmt)
    escalations = [EscalationResponse.model_validate(r) for r in rows]
    logger.debug("escalations.list", status_filter=status, count=len(escalations))
    return EscalationListResponse(escalations=escalations, count=len(escalations))


@router.post(
    "/escalations/{escalation_id}/resolve",
    response_model=EscalationResponse,
)
def resolve_escalation(
    escalation_id: str,
    body: ResolveRequest,
    session_factory: Annotated[sessionmaker[Session], Depends(_get_session_factory)],
    bus: Annotated[MessageBus, Depends(_get_bus)],
    audit_logger: Annotated[AuditTrailLogger, Depends(_get_audit_logger)],
) -> EscalationResponse:
    """Resolve a pending escalation.

    Transitions the escalation from ``"pending"`` to the appropriate resolved
    status, records the resolution timestamp and optional notes, writes an
    ``"approval"`` audit event, and publishes a ``"escalation.resolved"`` bus
    event so downstream components can react.

    Args:
        escalation_id: Primary key of the escalation to resolve.
        body: Resolution decision and optional notes.
        session_factory: Shared SQLAlchemy session factory (injected).
        bus: Shared message bus (injected).
        audit_logger: Shared audit trail logger (injected).

    Returns:
        :class:`EscalationResponse` with the updated record.

    Raises:
        :class:`~fastapi.HTTPException` 404 if ``escalation_id`` is not found.
        :class:`~fastapi.HTTPException` 409 if the escalation is already resolved.
    """
    new_status = _RESOLUTION_TO_STATUS[body.resolution]
    resolved_at = datetime.datetime.now(datetime.UTC)
    resolved_row, agent_id = _apply_resolution(
        session_factory,
        escalation_id=escalation_id,
        new_status=new_status,
        notes=body.notes,
        resolved_at=resolved_at,
    )

    # Audit log (outside the session so the session is already committed)
    audit_logger.log_event(
        "approval",
        agent_id,
        {
            "escalation_id": escalation_id,
            "resolution": body.resolution,
            "new_status": new_status,
            "notes": body.notes,
        },
    )

    # Publish resolved event
    bus.publish(
        RESOLVED_CHANNEL,
        {
            "id": escalation_id,
            "agent_id": agent_id,
            "status": new_status,
            "resolution": body.resolution,
            "notes": body.notes,
        },
    )

    logger.info(
        "escalation.resolved",
        escalation_id=escalation_id,
        agent_id=agent_id,
        resolution=body.resolution,
        new_status=new_status,
    )
    return resolved_row
