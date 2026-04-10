"""Unit tests for AuditTrailLogger (Phase 7 task 7.2).

Covers: persist and return ID, all five allowed event types, invalid type
raises ValueError, null agent_id for system events, and JSON payload
round-trip fidelity.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.config import Settings
from chronoagent.db.models import AuditEvent, Base
from chronoagent.db.session import make_engine
from chronoagent.escalation.audit import ALLOWED_EVENT_TYPES, AuditTrailLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def in_memory_session_factory() -> Generator[sessionmaker[Session], None, None]:
    """In-memory SQLite session factory with all tables created."""
    settings = Settings(database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    yield factory


@pytest.fixture()
def audit_logger(
    in_memory_session_factory: sessionmaker[Session],
) -> AuditTrailLogger:
    """AuditTrailLogger wired to the in-memory factory."""
    return AuditTrailLogger(in_memory_session_factory)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogEventReturnsId:
    def test_returns_positive_integer(self, audit_logger: AuditTrailLogger) -> None:
        event_id = audit_logger.log_event("allocation", "agent_a", {"foo": 1})
        assert isinstance(event_id, int)
        assert event_id > 0

    def test_successive_ids_are_unique(self, audit_logger: AuditTrailLogger) -> None:
        id1 = audit_logger.log_event("escalation", "agent_a", {})
        id2 = audit_logger.log_event("escalation", "agent_a", {})
        assert id1 != id2


class TestLogEventPersists:
    def test_row_appears_in_db(
        self,
        audit_logger: AuditTrailLogger,
        in_memory_session_factory: sessionmaker[Session],
    ) -> None:
        event_id = audit_logger.log_event("quarantine", "agent_b", {"reason": "test"})
        with in_memory_session_factory() as session:
            row = session.get(AuditEvent, event_id)
        assert row is not None
        assert row.event_type == "quarantine"
        assert row.agent_id == "agent_b"
        assert row.payload == {"reason": "test"}
        assert row.timestamp is not None

    def test_multiple_rows_persist(
        self,
        audit_logger: AuditTrailLogger,
        in_memory_session_factory: sessionmaker[Session],
    ) -> None:
        for i in range(3):
            audit_logger.log_event("health_update", f"agent_{i}", {"health": 0.5})
        with in_memory_session_factory() as session:
            rows = list(session.execute(select(AuditEvent)).scalars().all())
        assert len(rows) == 3


class TestLogEventAllowedTypes:
    @pytest.mark.parametrize("event_type", sorted(ALLOWED_EVENT_TYPES))
    def test_each_type_accepted(self, event_type: str, audit_logger: AuditTrailLogger) -> None:
        event_id = audit_logger.log_event(event_type, "agent_x", {})
        assert event_id > 0


class TestLogEventInvalidType:
    def test_raises_value_error(self, audit_logger: AuditTrailLogger) -> None:
        with pytest.raises(ValueError, match="invalid event_type"):
            audit_logger.log_event("garbage", "agent_a", {})

    def test_error_message_lists_allowed(self, audit_logger: AuditTrailLogger) -> None:
        with pytest.raises(ValueError, match="allocation"):
            audit_logger.log_event("unknown", "agent_a", {})


class TestLogEventNullAgentId:
    def test_null_agent_id_is_accepted(
        self,
        audit_logger: AuditTrailLogger,
        in_memory_session_factory: sessionmaker[Session],
    ) -> None:
        event_id = audit_logger.log_event("allocation", None, {"system": True})
        with in_memory_session_factory() as session:
            row = session.get(AuditEvent, event_id)
        assert row is not None
        assert row.agent_id is None


class TestLogEventPayloadRoundTrip:
    def test_nested_dict_survives(
        self,
        audit_logger: AuditTrailLogger,
        in_memory_session_factory: sessionmaker[Session],
    ) -> None:
        payload: dict[str, Any] = {
            "health_score": 0.12,
            "components": {"bocpd": 0.5, "chronos": None},
            "ids": ["doc_1", "doc_2"],
        }
        event_id = audit_logger.log_event("escalation", "agent_a", payload)
        with in_memory_session_factory() as session:
            row = session.get(AuditEvent, event_id)
        assert row is not None
        assert row.payload["health_score"] == 0.12
        assert row.payload["components"]["bocpd"] == 0.5
        assert row.payload["components"]["chronos"] is None
        assert row.payload["ids"] == ["doc_1", "doc_2"]
