"""Unit tests for EscalationHandler (Phase 7 task 7.1).

Covers: low health triggers escalation, high health suppresses, cooldown
blocks repeat, cooldown expiry allows re-escalation, quarantine events
bypass threshold check, context assembly, audit event written, and
bus handler helpers.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Generator
from typing import Any

import chromadb
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.config import Settings
from chronoagent.db.models import AllocationAuditRecord, AuditEvent, Base, EscalationRecord
from chronoagent.db.session import make_engine
from chronoagent.escalation.audit import AuditTrailLogger
from chronoagent.escalation.escalation_manager import (
    ESCALATION_CHANNEL,
    EscalationHandler,
    EscalationOutcome,
)
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.scorer.health_scorer import TemporalHealthScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_factory() -> Generator[sessionmaker[Session], None, None]:
    """In-memory SQLite session factory with all tables created."""
    settings = Settings(database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def bus() -> LocalBus:
    return LocalBus()


@pytest.fixture()
def captured_escalations(bus: LocalBus) -> list[tuple[str, Any]]:
    """Capture messages published to the escalations channel."""
    captured: list[tuple[str, Any]] = []
    bus.subscribe(ESCALATION_CHANNEL, lambda ch, msg: captured.append((ch, msg)))
    return captured


@pytest.fixture()
def clock() -> list[float]:
    """Injectable clock.  Mutate clock[0] to advance time."""
    return [1_000_000.0]


@pytest.fixture()
def handler(
    bus: LocalBus,
    session_factory: sessionmaker[Session],
    clock: list[float],
) -> Generator[EscalationHandler, None, None]:
    """EscalationHandler with 0.3 threshold, 100 s cooldown, injectable clock.

    Subscribes to ``"health_updates"`` and ``"memory.quarantine"`` channels
    (mirroring what ``main.py`` lifespan does) so bus-handler tests work.
    """
    qstore = QuarantineStore(
        chromadb.EphemeralClient().get_or_create_collection(f"q_{uuid.uuid4().hex}")
    )
    scorer = TemporalHealthScorer(bus=bus)
    audit = AuditTrailLogger(session_factory)
    h = EscalationHandler(
        bus=bus,
        health_scorer=scorer,
        quarantine_store=qstore,
        session_factory=session_factory,
        audit_logger=audit,
        threshold=0.3,
        cooldown_seconds=100.0,
        now_fn=lambda: clock[0],
    )
    bus.subscribe("health_updates", h.on_health_update)
    bus.subscribe("memory.quarantine", h.on_quarantine_event)
    yield h
    bus.unsubscribe("health_updates", h.on_health_update)
    bus.unsubscribe("memory.quarantine", h.on_quarantine_event)


# ---------------------------------------------------------------------------
# Low-health trigger
# ---------------------------------------------------------------------------


class TestLowHealthTrigger:
    def test_low_health_returns_outcome(
        self,
        handler: EscalationHandler,
        captured_escalations: list[tuple[str, Any]],
        session_factory: sessionmaker[Session],
    ) -> None:
        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        assert isinstance(outcome, EscalationOutcome)
        assert outcome.agent_id == "agent_a"
        assert outcome.trigger == "low_health"
        # DB persisted
        with session_factory() as session:
            row = session.get(EscalationRecord, outcome.escalation_id)
        assert row is not None
        assert row.status == "pending"

    def test_low_health_publishes_bus_event(
        self,
        handler: EscalationHandler,
        captured_escalations: list[tuple[str, Any]],
    ) -> None:
        handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        assert len(captured_escalations) == 1
        _ch, msg = captured_escalations[0]
        assert msg["agent_id"] == "agent_a"
        assert msg["trigger"] == "low_health"

    def test_high_health_returns_none(
        self,
        handler: EscalationHandler,
        captured_escalations: list[tuple[str, Any]],
    ) -> None:
        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.9)
        assert outcome is None
        assert captured_escalations == []

    def test_health_at_threshold_returns_none(
        self,
        handler: EscalationHandler,
    ) -> None:
        # threshold = 0.3; health exactly at threshold is NOT below -> suppressed
        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.3)
        assert outcome is None

    def test_none_health_score_returns_none(
        self,
        handler: EscalationHandler,
    ) -> None:
        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=None)
        assert outcome is None


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_blocks_repeat(
        self,
        handler: EscalationHandler,
        session_factory: sessionmaker[Session],
    ) -> None:
        handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        second = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.05)
        assert second is None
        with session_factory() as session:
            rows = list(session.execute(select(EscalationRecord)).scalars().all())
        assert len(rows) == 1

    def test_cooldown_expires_allows_re_escalation(
        self,
        handler: EscalationHandler,
        clock: list[float],
        session_factory: sessionmaker[Session],
    ) -> None:
        handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        clock[0] += 101.0  # past cooldown_seconds=100
        second = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.05)
        assert second is not None
        with session_factory() as session:
            rows = list(session.execute(select(EscalationRecord)).scalars().all())
        assert len(rows) == 2

    def test_cooldown_is_per_agent(
        self,
        handler: EscalationHandler,
        session_factory: sessionmaker[Session],
    ) -> None:
        handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        # Different agent should NOT be blocked
        second = handler.maybe_escalate("agent_b", trigger="low_health", health_score=0.1)
        assert second is not None
        with session_factory() as session:
            rows = list(session.execute(select(EscalationRecord)).scalars().all())
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Quarantine event trigger
# ---------------------------------------------------------------------------


class TestQuarantineEventTrigger:
    def test_quarantine_event_escalates_with_none_health(
        self,
        handler: EscalationHandler,
        captured_escalations: list[tuple[str, Any]],
    ) -> None:
        outcome = handler.maybe_escalate(
            "agent_a",
            trigger="quarantine_event",
            health_score=None,
            flagged_doc_ids=["doc_1", "doc_2"],
        )
        assert outcome is not None
        assert outcome.trigger == "quarantine_event"
        assert len(captured_escalations) == 1

    def test_quarantine_event_ignores_threshold(
        self,
        handler: EscalationHandler,
    ) -> None:
        # health_score=0.99 is above threshold but quarantine_event always escalates
        outcome = handler.maybe_escalate(
            "agent_a",
            trigger="quarantine_event",
            health_score=0.99,
            flagged_doc_ids=["doc_x"],
        )
        assert outcome is not None

    def test_quarantine_event_context_contains_flagged_ids(
        self,
        handler: EscalationHandler,
    ) -> None:
        outcome = handler.maybe_escalate(
            "agent_a",
            trigger="quarantine_event",
            health_score=None,
            flagged_doc_ids=["d1", "d2", "d3"],
        )
        assert outcome is not None
        assert outcome.context["flagged_doc_ids"] == ["d1", "d2", "d3"]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


class TestContextAssembly:
    def test_context_contains_health_components(
        self,
        handler: EscalationHandler,
        bus: LocalBus,
    ) -> None:
        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        assert outcome is not None
        assert "health_components" in outcome.context
        # No signal has been pushed so get_health returns None
        assert outcome.context["health_components"]["bocpd_score"] is None
        assert outcome.context["health_components"]["chronos_score"] is None

    def test_context_contains_quarantine_count(
        self,
        handler: EscalationHandler,
    ) -> None:
        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        assert outcome is not None
        assert "quarantine_count" in outcome.context
        assert outcome.context["quarantine_count"] == 0

    def test_context_contains_recent_allocation_task_ids(
        self,
        handler: EscalationHandler,
        session_factory: sessionmaker[Session],
    ) -> None:
        ts = datetime.datetime.now(datetime.UTC)
        with session_factory() as session:
            for i in range(3):
                session.add(
                    AllocationAuditRecord(
                        task_id=f"task_{i}",
                        task_type="security_review",
                        assigned_agent="agent_a",
                        escalated=False,
                        all_bids=[],
                        health_snapshot={},
                        rationale="test",
                        threshold=0.5,
                        timestamp=ts,
                    )
                )
            session.commit()

        outcome = handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        assert outcome is not None
        assert len(outcome.context["recent_allocation_task_ids"]) == 3

    def test_context_extra_merged(
        self,
        handler: EscalationHandler,
    ) -> None:
        outcome = handler.maybe_escalate(
            "agent_a",
            trigger="low_health",
            health_score=0.1,
            extra={"custom_key": "custom_value"},
        )
        assert outcome is not None
        assert outcome.context["custom_key"] == "custom_value"


# ---------------------------------------------------------------------------
# Audit event
# ---------------------------------------------------------------------------


class TestAuditEvent:
    def test_audit_event_written_on_escalation(
        self,
        handler: EscalationHandler,
        session_factory: sessionmaker[Session],
    ) -> None:
        handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.1)
        with session_factory() as session:
            rows = list(
                session.execute(select(AuditEvent).where(AuditEvent.event_type == "escalation"))
                .scalars()
                .all()
            )
        assert len(rows) == 1
        assert rows[0].agent_id == "agent_a"
        assert rows[0].payload["trigger"] == "low_health"

    def test_no_audit_event_when_suppressed(
        self,
        handler: EscalationHandler,
        session_factory: sessionmaker[Session],
    ) -> None:
        # Health above threshold -> suppressed
        handler.maybe_escalate("agent_a", trigger="low_health", health_score=0.9)
        with session_factory() as session:
            rows = list(
                session.execute(select(AuditEvent).where(AuditEvent.event_type == "escalation"))
                .scalars()
                .all()
            )
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Bus handler helpers
# ---------------------------------------------------------------------------


class TestBusHandlers:
    def test_on_health_update_low_triggers_escalation(
        self,
        handler: EscalationHandler,
        bus: LocalBus,
        captured_escalations: list[tuple[str, Any]],
    ) -> None:
        bus.publish(
            "health_updates",
            {
                "agent_id": "agent_a",
                "health": 0.1,
                "bocpd_score": 0.5,
                "chronos_score": None,
            },
        )
        assert len(captured_escalations) == 1

    def test_on_health_update_high_noop(
        self,
        handler: EscalationHandler,
        bus: LocalBus,
        captured_escalations: list[tuple[str, Any]],
    ) -> None:
        bus.publish(
            "health_updates",
            {
                "agent_id": "agent_a",
                "health": 0.95,
                "bocpd_score": 0.1,
                "chronos_score": None,
            },
        )
        assert captured_escalations == []

    def test_on_health_update_malformed_does_not_raise(
        self,
        handler: EscalationHandler,
        bus: LocalBus,
    ) -> None:
        # Should not raise; malformed messages are swallowed with a log warning.
        bus.publish("health_updates", "not-a-dict")

    def test_on_quarantine_event_escalates(
        self,
        handler: EscalationHandler,
        bus: LocalBus,
        captured_escalations: list[tuple[str, Any]],
    ) -> None:
        bus.publish(
            "memory.quarantine",
            {"agent_id": "agent_b", "ids": ["doc_1"], "reason": "integrity_module"},
        )
        assert len(captured_escalations) == 1
        _ch, msg = captured_escalations[0]
        assert msg["agent_id"] == "agent_b"

    def test_on_quarantine_event_malformed_does_not_raise(
        self,
        handler: EscalationHandler,
        bus: LocalBus,
    ) -> None:
        bus.publish("memory.quarantine", 42)
