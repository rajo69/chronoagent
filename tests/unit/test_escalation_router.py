"""Unit tests for the escalation API router (Phase 7 task 7.3).

Covers: list returns 200 empty, list returns pending items, status filter,
default filter is pending, resolve approve/reject/modify, resolve publishes
bus event, resolve non-existent 404, already-resolved 409, resolve writes
audit event.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Generator
from typing import Any

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from chronoagent.config import Settings
from chronoagent.db.models import Base, EscalationRecord
from chronoagent.escalation.audit import AuditTrailLogger
from chronoagent.escalation.escalation_manager import EscalationHandler
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.scorer.health_scorer import TemporalHealthScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_app() -> Generator[tuple[TestClient, FastAPI], None, None]:
    """TestClient with a fully wired escalation app state."""
    from chronoagent.main import create_app

    settings = Settings(database_url="sqlite:///:memory:", env="test", llm_backend="mock")
    app = create_app(settings=settings)

    suffix = uuid.uuid4().hex
    # Use StaticPool so create_all and all session-factory connections share
    # the same in-memory SQLite database instance.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    bus = LocalBus()
    qstore = QuarantineStore(chromadb.EphemeralClient().get_or_create_collection(f"q_{suffix}"))
    scorer = TemporalHealthScorer(bus=bus)
    audit = AuditTrailLogger(factory)
    handler = EscalationHandler(
        bus=bus,
        health_scorer=scorer,
        quarantine_store=qstore,
        session_factory=factory,
        audit_logger=audit,
    )

    with TestClient(app) as tc:
        app.state.session_factory = factory
        app.state.bus = bus
        app.state.audit_logger = audit
        app.state.escalation_handler = handler
        yield tc, app


def _seed_escalation(
    app: FastAPI,
    *,
    agent_id: str = "agent_a",
    trigger: str = "low_health",
    status: str = "pending",
) -> str:
    """Insert one EscalationRecord directly and return its ID."""
    escalation_id = uuid.uuid4().hex
    row = EscalationRecord(
        id=escalation_id,
        agent_id=agent_id,
        trigger=trigger,
        status=status,
        context={"health_score": 0.1},
        resolution_notes=None,
        created_at=datetime.datetime.now(datetime.UTC),
        resolved_at=None,
    )
    with app.state.session_factory() as session:
        session.add(row)
        session.commit()
    return escalation_id


# ---------------------------------------------------------------------------
# GET /api/v1/escalations
# ---------------------------------------------------------------------------


class TestListEscalations:
    def test_returns_200_empty(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/api/v1/escalations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["escalations"] == []
        assert body["count"] == 0

    def test_returns_pending_items(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        resp = tc.get("/api/v1/escalations")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_default_filter_is_pending(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="approved")
        resp = tc.get("/api/v1/escalations")
        assert resp.json()["count"] == 1
        assert resp.json()["escalations"][0]["status"] == "pending"

    def test_status_filter_approved(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="approved")
        resp = tc.get("/api/v1/escalations?status=approved")
        assert resp.json()["count"] == 1
        assert resp.json()["escalations"][0]["status"] == "approved"

    def test_status_all_returns_everything(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="approved")
        _seed_escalation(app, status="rejected")
        resp = tc.get("/api/v1/escalations?status=all")
        assert resp.json()["count"] == 3

    def test_response_shape(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        resp = tc.get("/api/v1/escalations")
        item = resp.json()["escalations"][0]
        assert item["id"] == esc_id
        assert item["agent_id"] == "agent_a"
        assert item["trigger"] == "low_health"
        assert item["status"] == "pending"
        assert "context" in item
        assert item["resolution_notes"] is None
        assert item["resolved_at"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/escalations/{id}/resolve
# ---------------------------------------------------------------------------


class TestResolveEscalation:
    def test_resolve_approve_returns_200(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        resp = tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "approve", "notes": "looks fine"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "approved"
        assert body["resolution_notes"] == "looks fine"
        assert body["resolved_at"] is not None

    def test_resolve_reject_transitions_status(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        resp = tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "reject"},
        )
        assert resp.json()["status"] == "rejected"

    def test_resolve_modify_transitions_status(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        resp = tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "modify", "notes": "needs tweak"},
        )
        assert resp.json()["status"] == "modified"

    def test_resolve_persists_in_db(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "approve"},
        )
        with app.state.session_factory() as session:
            row = session.get(EscalationRecord, esc_id)
        assert row is not None
        assert row.status == "approved"
        assert row.resolved_at is not None

    def test_resolve_publishes_bus_event(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        captured: list[tuple[str, Any]] = []
        app.state.bus.subscribe("escalation.resolved", lambda ch, msg: captured.append((ch, msg)))
        esc_id = _seed_escalation(app)
        tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "approve"},
        )
        assert len(captured) == 1
        _ch, msg = captured[0]
        assert msg["id"] == esc_id
        assert msg["status"] == "approved"

    def test_resolve_nonexistent_404(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.post(
            f"/api/v1/escalations/{uuid.uuid4().hex}/resolve",
            json={"resolution": "approve"},
        )
        assert resp.status_code == 404

    def test_resolve_already_resolved_409(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "approve"},
        )
        resp = tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "reject"},
        )
        assert resp.status_code == 409

    def test_resolve_writes_audit_event(self, client_app: tuple[TestClient, FastAPI]) -> None:
        from sqlalchemy import select

        from chronoagent.db.models import AuditEvent

        tc, app = client_app
        esc_id = _seed_escalation(app)
        tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "approve", "notes": "approved"},
        )
        with app.state.session_factory() as session:
            rows = list(
                session.execute(select(AuditEvent).where(AuditEvent.event_type == "approval"))
                .scalars()
                .all()
            )
        assert len(rows) == 1
        assert rows[0].payload["escalation_id"] == esc_id
        assert rows[0].payload["resolution"] == "approve"

    def test_resolve_notes_optional(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        esc_id = _seed_escalation(app)
        resp = tc.post(
            f"/api/v1/escalations/{esc_id}/resolve",
            json={"resolution": "reject"},
        )
        assert resp.status_code == 200
        assert resp.json()["resolution_notes"] is None
