"""Escalation handler (Phase 7 task 7.1).

:class:`EscalationHandler` monitors agent health and quarantine events,
auto-escalates when thresholds are breached, enforces a per-agent cooldown
to prevent notification spam, assembles rich context for human reviewers,
persists each escalation to the ``escalation_records`` table, and publishes
events on the message bus.

Channel conventions
-------------------
- Inbound:  ``"health_updates"``     payload: ``HealthUpdate`` dict
- Inbound:  ``"memory.quarantine"``  payload: ``{"agent_id": str, "ids": list[str], ...}``
- Outbound: ``"escalations"``        payload: escalation dict
- Outbound: ``"escalation.resolved"``  payload: resolution dict (published by the API router)

Cooldown semantics
------------------
Cooldown is tracked in-memory per agent.  A process restart resets the
cooldown clock.  This is intentional: the rate-limit is a soft guard against
notification storms, not a security boundary.  One extra escalation on restart
is acceptable.
"""

from __future__ import annotations

import datetime
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import desc, select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.db.models import AllocationAuditRecord
from chronoagent.db.models import EscalationRecord as EscalationRecordORM
from chronoagent.escalation.audit import AuditTrailLogger
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.messaging.bus import MessageBus
from chronoagent.observability.logging import get_logger
from chronoagent.retry import db_retry
from chronoagent.scorer.health_scorer import HealthUpdate, TemporalHealthScorer

logger = get_logger(__name__)

ESCALATION_CHANNEL = "escalations"
RESOLVED_CHANNEL = "escalation.resolved"

DEFAULT_THRESHOLD: float = 0.3
DEFAULT_COOLDOWN_SECONDS: float = 3600.0
DEFAULT_ALLOCATION_HISTORY: int = 10


@dataclass(frozen=True)
class EscalationOutcome:
    """Return value of :meth:`EscalationHandler.maybe_escalate`.

    Wraps the persisted escalation record primary key and the assembled
    context dict so callers do not need a round-trip DB read.

    Attributes:
        escalation_id: UUID4 hex string primary key of the persisted row.
        agent_id: Agent that triggered the escalation.
        trigger: ``"low_health"`` or ``"quarantine_event"``.
        context: JSON-serialisable snapshot assembled at escalation time.
        created_at: UTC timestamp of the escalation.
    """

    escalation_id: str
    agent_id: str
    trigger: str
    context: dict[str, Any]
    created_at: datetime.datetime


class EscalationHandler:
    """Auto-escalate on low health or quarantine events with cooldown and audit trail.

    Parameters
    ----------
    bus:
        Message bus for subscribing to inbound events and publishing outbound ones.
    health_scorer:
        Shared :class:`~chronoagent.scorer.health_scorer.TemporalHealthScorer`
        used to pull per-agent health components into escalation context.
    quarantine_store:
        Shared :class:`~chronoagent.memory.quarantine.QuarantineStore` used to
        include the current quarantine count in escalation context.
    session_factory:
        SQLAlchemy session factory for persisting escalation records and
        querying recent allocation history.
    audit_logger:
        :class:`~chronoagent.escalation.audit.AuditTrailLogger` for writing
        ``"escalation"`` audit events.
    threshold:
        Health score below which an agent is considered unhealthy.  Default 0.3.
    cooldown_seconds:
        Minimum seconds between successive escalations for the same agent.
        Default 3600 (1 hour).
    allocation_history_limit:
        How many recent allocation task IDs to include in the context snapshot.
        Default 10.
    now_fn:
        Callable returning the current Unix timestamp.  Defaults to
        :func:`time.time`.  Inject a deterministic clock in tests.
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        health_scorer: TemporalHealthScorer,
        quarantine_store: QuarantineStore,
        session_factory: sessionmaker[Session],
        audit_logger: AuditTrailLogger,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        allocation_history_limit: int = DEFAULT_ALLOCATION_HISTORY,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._bus = bus
        self._health_scorer = health_scorer
        self._quarantine_store = quarantine_store
        self._session_factory = session_factory
        self._audit_logger = audit_logger
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._allocation_history_limit = allocation_history_limit
        self._now_fn = now_fn

        # Per-agent last-escalation Unix timestamp.  In-memory; resets on restart.
        self._last_escalated: dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Bus event handlers
    # ------------------------------------------------------------------

    def on_health_update(self, _channel: str, message: Any) -> None:
        """Bus subscriber for ``"health_updates"``.

        Parses the inbound :class:`~chronoagent.scorer.health_scorer.HealthUpdate`
        dict and calls :meth:`maybe_escalate` when health falls below threshold.

        Args:
            _channel: Channel name (unused; required by :class:`MessageHandler` sig).
            message: Dict payload published by
                :class:`~chronoagent.scorer.health_scorer.TemporalHealthScorer`.
        """
        try:
            if isinstance(message, dict):
                update = HealthUpdate(**message)
            elif isinstance(message, HealthUpdate):
                update = message
            else:
                logger.warning(
                    "escalation.on_health_update.unexpected_type", msg_type=type(message).__name__
                )
                return
        except (TypeError, KeyError) as exc:
            logger.warning("escalation.on_health_update.malformed", error=str(exc))
            return

        try:
            if update.health < self._threshold:
                self.maybe_escalate(
                    update.agent_id,
                    trigger="low_health",
                    health_score=update.health,
                )
        except Exception:
            logger.exception("escalation.on_health_update.error", agent_id=update.agent_id)

    def on_quarantine_event(self, _channel: str, message: Any) -> None:
        """Bus subscriber for ``"memory.quarantine"``.

        Calls :meth:`maybe_escalate` immediately; quarantine events always
        attempt escalation regardless of health score.

        Args:
            _channel: Channel name (unused).
            message: Dict with at minimum ``"agent_id"`` and ``"ids"`` keys.
        """
        try:
            if not isinstance(message, dict):
                logger.warning(
                    "escalation.on_quarantine_event.unexpected_type",
                    msg_type=type(message).__name__,
                )
                return
            agent_id: str = str(message.get("agent_id", "unknown"))
            ids: list[str] = list(message.get("ids", []))
            reason: str | None = message.get("reason")
            extra: dict[str, Any] = {}
            if reason is not None:
                extra["quarantine_reason"] = reason
        except Exception as exc:
            logger.warning("escalation.on_quarantine_event.malformed", error=str(exc))
            return

        try:
            self.maybe_escalate(
                agent_id,
                trigger="quarantine_event",
                health_score=None,
                flagged_doc_ids=ids,
                extra=extra,
            )
        except Exception:
            logger.exception("escalation.on_quarantine_event.error", agent_id=agent_id)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def maybe_escalate(
        self,
        agent_id: str,
        *,
        trigger: Literal["low_health", "quarantine_event"],
        health_score: float | None,
        flagged_doc_ids: list[str] | None = None,
        task_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EscalationOutcome | None:
        """Check cooldown and thresholds, then create and publish an escalation.

        For ``"low_health"`` triggers the call is suppressed when
        ``health_score`` is ``None`` or ``>= threshold``.  For
        ``"quarantine_event"`` triggers the threshold check is skipped; any
        quarantine event is considered worth escalating.  Both trigger types
        respect the per-agent cooldown.

        Args:
            agent_id: Agent to potentially escalate.
            trigger: Reason for the escalation check.
            health_score: Current health score in ``[0, 1]``, or ``None``.
            flagged_doc_ids: Document IDs flagged by the integrity module.
            task_id: Optional task identifier for context.
            extra: Additional key-value pairs included in the context snapshot.

        Returns:
            :class:`EscalationOutcome` when an escalation was created, or
            ``None`` when suppressed by cooldown or threshold.
        """
        now = self._now_fn()

        # --- Cooldown check (lock only around dict read/write, not DB) ---
        with self._cooldown_lock:
            last = self._last_escalated.get(agent_id, float("-inf"))
            if now - last < self._cooldown_seconds:
                logger.debug(
                    "escalation.cooldown_suppressed",
                    agent_id=agent_id,
                    seconds_remaining=self._cooldown_seconds - (now - last),
                )
                return None

        # --- Threshold check (low_health only) ---
        if trigger == "low_health" and (health_score is None or health_score >= self._threshold):
            return None

        # --- Assemble context ---
        doc_ids = flagged_doc_ids or []
        context = self._assemble_context(agent_id, health_score, doc_ids, task_id, extra)

        # --- Persist escalation record ---
        escalation_id = uuid.uuid4().hex
        created_at = datetime.datetime.now(datetime.UTC)
        orm_row = EscalationRecordORM(
            id=escalation_id,
            agent_id=agent_id,
            trigger=trigger,
            status="pending",
            context=context,
            resolution_notes=None,
            created_at=created_at,
            resolved_at=None,
        )
        self._persist_escalation(orm_row)

        # --- Audit log ---
        self._audit_logger.log_event(
            "escalation",
            agent_id,
            {
                "escalation_id": escalation_id,
                "trigger": trigger,
                "health_score": health_score,
                "flagged_doc_ids": doc_ids,
                "task_id": task_id,
            },
        )

        # --- Update cooldown clock BEFORE publishing to avoid reentrancy issues ---
        with self._cooldown_lock:
            self._last_escalated[agent_id] = now

        # --- Publish to bus ---
        self._bus.publish(
            ESCALATION_CHANNEL,
            {
                "id": escalation_id,
                "agent_id": agent_id,
                "trigger": trigger,
                "health_score": health_score,
                "context": context,
                "created_at": created_at.isoformat(),
            },
        )

        logger.info(
            "escalation.created",
            escalation_id=escalation_id,
            agent_id=agent_id,
            trigger=trigger,
            health_score=health_score,
        )
        return EscalationOutcome(
            escalation_id=escalation_id,
            agent_id=agent_id,
            trigger=trigger,
            context=context,
            created_at=created_at,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assemble_context(
        self,
        agent_id: str,
        health_score: float | None,
        flagged_doc_ids: list[str],
        task_id: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the JSON-serialisable context snapshot for an escalation.

        Args:
            agent_id: Agent being escalated.
            health_score: Current health score, or ``None``.
            flagged_doc_ids: Flagged document IDs.
            task_id: Optional task context.
            extra: Caller-supplied extra fields.

        Returns:
            A JSON-serialisable dict with keys: ``health_score``,
            ``health_components``, ``flagged_doc_ids``, ``quarantine_count``,
            ``recent_allocation_task_ids``, ``task_id``, and any ``extra`` keys.
        """
        health_update = self._health_scorer.get_health(agent_id)
        health_components: dict[str, float | None] = {
            "bocpd_score": health_update.bocpd_score if health_update else None,
            "chronos_score": health_update.chronos_score if health_update else None,
        }
        ctx: dict[str, Any] = {
            "health_score": health_score,
            "health_components": health_components,
            "flagged_doc_ids": flagged_doc_ids,
            "quarantine_count": self._quarantine_store.count,
            "recent_allocation_task_ids": self._recent_allocation_task_ids(agent_id),
            "task_id": task_id,
        }
        if extra:
            ctx.update(extra)
        return ctx

    @db_retry
    def _persist_escalation(self, orm_row: EscalationRecordORM) -> None:
        """Insert an :class:`EscalationRecordORM` row in its own transaction.

        Isolated so the :func:`db_retry` policy re-runs only the database
        write on transient ``OperationalError``, not the surrounding bus
        publish or cooldown bookkeeping in :meth:`maybe_escalate`.
        """
        with self._session_factory() as session:
            session.add(orm_row)
            session.commit()

    @db_retry
    def _recent_allocation_task_ids(self, agent_id: str) -> list[str]:
        """Return up to ``allocation_history_limit`` recent task IDs for agent.

        Args:
            agent_id: Agent to query.

        Returns:
            List of task ID strings ordered newest-first.
        """
        stmt = (
            select(AllocationAuditRecord.task_id)
            .where(AllocationAuditRecord.assigned_agent == agent_id)
            .order_by(desc(AllocationAuditRecord.timestamp))
            .limit(self._allocation_history_limit)
        )
        with self._session_factory() as session:
            rows = session.execute(stmt).scalars().all()
        return list(rows)
