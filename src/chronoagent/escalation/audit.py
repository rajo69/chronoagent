"""Append-only audit trail logger (Phase 7 task 7.2).

:class:`AuditTrailLogger` writes one :class:`~chronoagent.db.models.AuditEvent`
row per call, covering the five event types that make up ChronoAgent's
human-escalation audit trail:

- ``"allocation"`` -- contract-net allocation decision recorded.
- ``"health_update"`` -- per-agent health score changed.
- ``"escalation"`` -- auto-escalation triggered.
- ``"quarantine"`` -- one or more documents moved to quarantine.
- ``"approval"`` -- human resolved an escalation.

All timestamps are UTC.  The logger is safe to call from multiple threads
because each :meth:`log_event` call opens and closes its own session.
"""

from __future__ import annotations

import datetime
from typing import Any

import structlog
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.db.models import AuditEvent
from chronoagent.retry import db_retry

logger: structlog.BoundLogger = structlog.get_logger(__name__)

ALLOWED_EVENT_TYPES: frozenset[str] = frozenset(
    {"allocation", "health_update", "escalation", "quarantine", "approval"}
)


class AuditTrailLogger:
    """Write append-only audit events to the ``audit_events`` table.

    Parameters
    ----------
    session_factory:
        A :class:`~sqlalchemy.orm.sessionmaker` bound to the application engine.
        Each :meth:`log_event` call opens an independent session so the logger
        is thread-safe.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @db_retry
    def log_event(
        self,
        event_type: str,
        agent_id: str | None,
        payload: dict[str, Any],
    ) -> int:
        """Persist one audit event and return its auto-increment primary key.

        Args:
            event_type: Must be one of ``ALLOWED_EVENT_TYPES``.
            agent_id: Agent the event concerns, or ``None`` for system-level
                events (e.g. a startup allocation with no assigned agent).
            payload: Arbitrary JSON-serialisable dict with event context.
                Stored verbatim; keys must be strings.

        Returns:
            The integer primary key assigned by the database.

        Raises:
            ValueError: If ``event_type`` is not in :data:`ALLOWED_EVENT_TYPES`.
        """
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"invalid event_type {event_type!r}; must be one of {sorted(ALLOWED_EVENT_TYPES)}"
            )
        row = AuditEvent(
            event_type=event_type,
            agent_id=agent_id,
            payload=dict(payload),
            timestamp=datetime.datetime.now(datetime.UTC),
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()
            event_id = int(row.id)
        logger.debug(
            "audit.log_event",
            event_type=event_type,
            agent_id=agent_id,
            event_id=event_id,
        )
        return event_id
