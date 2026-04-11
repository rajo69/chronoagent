"""Unit tests for the dashboard API router (Phase 8 task 8.1).

Covers:
- GET /dashboard/api/agents: empty, merged sources, health sort, 503
- GET /dashboard/api/agents/{id}/timeline: empty, limit, since, sort order
- GET /dashboard/api/allocations: ordering, JSON fields, limit
- GET /dashboard/api/memory: full response shape from stubs
- GET /dashboard/api/escalations: split pending/resolved, status filter
- WebSocket /dashboard/ws/live: frame shape, pending count, not-ready path
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from chronoagent.config import Settings
from chronoagent.db.models import (
    AgentSignalRecord,
    AllocationAuditRecord,
    Base,
    EscalationRecord,
)
from chronoagent.escalation.audit import AuditTrailLogger
from chronoagent.escalation.escalation_manager import EscalationHandler
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.scorer.health_scorer import HealthUpdate, TemporalHealthScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = datetime.UTC


def _now() -> datetime.datetime:
    return datetime.datetime.now(_UTC)


def _ts(offset_seconds: float = 0.0) -> datetime.datetime:
    return datetime.datetime.now(_UTC) + datetime.timedelta(seconds=offset_seconds)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_app() -> Generator[tuple[TestClient, FastAPI], None, None]:
    """TestClient backed by an in-memory SQLite DB and fresh app.state."""
    from chronoagent.main import create_app

    settings = Settings(database_url="sqlite:///:memory:", env="test", llm_backend="mock")
    app = create_app(settings=settings)

    suffix = uuid.uuid4().hex
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
        app.state.health_scorer = scorer
        app.state.quarantine_store = qstore
        app.state.audit_logger = audit
        app.state.escalation_handler = handler
        yield tc, app


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_signal(
    app: FastAPI,
    *,
    agent_id: str = "agent_a",
    timestamp: datetime.datetime | None = None,
    total_latency_ms: float = 100.0,
    retrieval_count: int = 3,
    token_count: int = 256,
    kl_divergence: float = 0.05,
    tool_calls: int = 2,
    memory_query_entropy: float = 0.7,
) -> int:
    """Insert one AgentSignalRecord and return its id."""
    row = AgentSignalRecord(
        agent_id=agent_id,
        task_id=None,
        timestamp=timestamp or _now(),
        total_latency_ms=total_latency_ms,
        retrieval_count=retrieval_count,
        token_count=token_count,
        kl_divergence=kl_divergence,
        tool_calls=tool_calls,
        memory_query_entropy=memory_query_entropy,
    )
    with app.state.session_factory() as session:
        session.add(row)
        session.commit()
        return int(row.id)


def _seed_allocation(
    app: FastAPI,
    *,
    task_id: str | None = None,
    task_type: str = "security_review",
    assigned_agent: str | None = "agent_a",
    escalated: bool = False,
    all_bids: list[dict[str, Any]] | None = None,
    health_snapshot: dict[str, float] | None = None,
    rationale: str = "highest bid",
    threshold: float = 0.5,
    timestamp: datetime.datetime | None = None,
) -> int:
    """Insert one AllocationAuditRecord and return its id."""
    row = AllocationAuditRecord(
        task_id=task_id or f"pr-{uuid.uuid4().hex[:6]}::security_review",
        task_type=task_type,
        assigned_agent=assigned_agent,
        escalated=escalated,
        all_bids=all_bids or [{"agent_id": "agent_a", "score": 0.9}],
        health_snapshot=health_snapshot or {"agent_a": 0.9},
        rationale=rationale,
        threshold=threshold,
        timestamp=timestamp or _now(),
    )
    with app.state.session_factory() as session:
        session.add(row)
        session.commit()
        return int(row.id)


def _seed_escalation(
    app: FastAPI,
    *,
    agent_id: str = "agent_a",
    trigger: str = "low_health",
    status: str = "pending",
    resolved_at: datetime.datetime | None = None,
) -> str:
    """Insert one EscalationRecord and return its id."""
    eid = uuid.uuid4().hex
    row = EscalationRecord(
        id=eid,
        agent_id=agent_id,
        trigger=trigger,
        status=status,
        context={"health_score": 0.2},
        resolution_notes=None,
        created_at=_now(),
        resolved_at=resolved_at,
    )
    with app.state.session_factory() as session:
        session.add(row)
        session.commit()
    return eid


# ---------------------------------------------------------------------------
# GET /dashboard/api/agents
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_empty_returns_200(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/agents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["agents"] == []
        assert body["count"] == 0
        assert body["system_health"] == 1.0

    def test_scorer_agent_appears(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app

        # Inject a stub scorer with one known agent.
        class StubScorer:
            def get_all_health(self) -> dict[str, HealthUpdate]:
                return {
                    "agent_a": HealthUpdate(
                        agent_id="agent_a",
                        health=0.8,
                        bocpd_score=0.2,
                        chronos_score=None,
                    )
                }

            def stop(self) -> None:
                pass

        app.state.health_scorer = StubScorer()
        resp = tc.get("/dashboard/api/agents")
        body = resp.json()
        assert body["count"] == 1
        assert body["agents"][0]["agent_id"] == "agent_a"
        assert body["agents"][0]["health"] == pytest.approx(0.8)
        assert body["agents"][0]["components"] == "bocpd_only"
        assert body["system_health"] == pytest.approx(0.8)

    def test_db_only_agent_appears(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Agent has signal records but no health update yet."""
        tc, app = client_app
        _seed_signal(app, agent_id="db_only")
        resp = tc.get("/dashboard/api/agents")
        body = resp.json()
        ids = [a["agent_id"] for a in body["agents"]]
        assert "db_only" in ids
        # health is None, components is "none"
        entry = next(a for a in body["agents"] if a["agent_id"] == "db_only")
        assert entry["health"] is None
        assert entry["components"] == "none"
        assert entry["last_signal_at"] is not None

    def test_merged_sources(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Scorer agent + DB-only agent both appear."""
        tc, app = client_app
        _seed_signal(app, agent_id="db_only")

        class StubScorer:
            def get_all_health(self) -> dict[str, HealthUpdate]:
                return {
                    "scorer_only": HealthUpdate(
                        agent_id="scorer_only",
                        health=0.9,
                        bocpd_score=0.1,
                        chronos_score=None,
                    )
                }

            def stop(self) -> None:
                pass

        app.state.health_scorer = StubScorer()
        resp = tc.get("/dashboard/api/agents")
        ids = {a["agent_id"] for a in resp.json()["agents"]}
        assert "db_only" in ids
        assert "scorer_only" in ids

    def test_sorted_most_at_risk_first(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Agents are sorted ascending by health (lowest = most at risk first)."""
        tc, app = client_app

        class StubScorer:
            def get_all_health(self) -> dict[str, HealthUpdate]:
                return {
                    "healthy": HealthUpdate(
                        agent_id="healthy", health=0.95, bocpd_score=None, chronos_score=None
                    ),
                    "sick": HealthUpdate(
                        agent_id="sick", health=0.1, bocpd_score=None, chronos_score=None
                    ),
                }

            def stop(self) -> None:
                pass

        app.state.health_scorer = StubScorer()
        resp = tc.get("/dashboard/api/agents")
        agents = resp.json()["agents"]
        assert agents[0]["agent_id"] == "sick"
        assert agents[1]["agent_id"] == "healthy"

    def test_none_health_sorted_last(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Agents without a health score appear after scored agents."""
        tc, app = client_app
        _seed_signal(app, agent_id="no_score")

        class StubScorer:
            def get_all_health(self) -> dict[str, HealthUpdate]:
                return {
                    "scored": HealthUpdate(
                        agent_id="scored", health=0.5, bocpd_score=None, chronos_score=None
                    )
                }

            def stop(self) -> None:
                pass

        app.state.health_scorer = StubScorer()
        resp = tc.get("/dashboard/api/agents")
        agents = resp.json()["agents"]
        assert agents[-1]["agent_id"] == "no_score"

    def test_503_when_scorer_missing(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        original = app.state.health_scorer
        app.state.health_scorer = None
        try:
            resp = tc.get("/dashboard/api/agents")
            assert resp.status_code == 503
        finally:
            app.state.health_scorer = original

    def test_last_signal_at_populated(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        ts = datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=_UTC)
        _seed_signal(app, agent_id="timed_agent", timestamp=ts)
        resp = tc.get("/dashboard/api/agents")
        entry = next(a for a in resp.json()["agents"] if a["agent_id"] == "timed_agent")
        assert entry["last_signal_at"] is not None

    def test_ensemble_components_label(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app

        class StubScorer:
            def get_all_health(self) -> dict[str, HealthUpdate]:
                return {
                    "both": HealthUpdate(
                        agent_id="both", health=0.7, bocpd_score=0.3, chronos_score=0.2
                    )
                }

            def stop(self) -> None:
                pass

        app.state.health_scorer = StubScorer()
        resp = tc.get("/dashboard/api/agents")
        assert resp.json()["agents"][0]["components"] == "ensemble"


# ---------------------------------------------------------------------------
# GET /dashboard/api/agents/{agent_id}/timeline
# ---------------------------------------------------------------------------


class TestAgentTimeline:
    def test_unknown_agent_returns_empty(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/agents/unknown_agent/timeline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["points"] == []
        assert body["count"] == 0
        assert body["agent_id"] == "unknown_agent"

    def test_returns_points_for_known_agent(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_signal(app, agent_id="agent_a")
        resp = tc.get("/dashboard/api/agents/agent_a/timeline")
        body = resp.json()
        assert body["count"] == 1
        p = body["points"][0]
        assert p["total_latency_ms"] == pytest.approx(100.0)
        assert p["retrieval_count"] == 3

    def test_chronological_order(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Points are returned oldest-first."""
        tc, app = client_app
        t1 = _ts(-10)
        t2 = _ts(-5)
        t3 = _ts(0)
        _seed_signal(app, agent_id="ordered", timestamp=t2, total_latency_ms=200.0)
        _seed_signal(app, agent_id="ordered", timestamp=t1, total_latency_ms=100.0)
        _seed_signal(app, agent_id="ordered", timestamp=t3, total_latency_ms=300.0)
        resp = tc.get("/dashboard/api/agents/ordered/timeline")
        latencies = [p["total_latency_ms"] for p in resp.json()["points"]]
        assert latencies == [100.0, 200.0, 300.0]

    def test_limit_respected(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        for i in range(5):
            _seed_signal(app, agent_id="limited", timestamp=_ts(-i * 10))
        resp = tc.get("/dashboard/api/agents/limited/timeline?limit=3")
        assert resp.json()["count"] == 3

    def test_since_filter(self, client_app: tuple[TestClient, FastAPI]) -> None:
        import urllib.parse

        tc, app = client_app
        cutoff = datetime.datetime(2026, 6, 1, 0, 0, 0, tzinfo=_UTC)
        _seed_signal(app, agent_id="filtered", timestamp=datetime.datetime(2026, 5, 1, tzinfo=_UTC))
        _seed_signal(app, agent_id="filtered", timestamp=datetime.datetime(2026, 7, 1, tzinfo=_UTC))
        since_param = urllib.parse.quote(cutoff.isoformat())
        resp = tc.get(f"/dashboard/api/agents/filtered/timeline?since={since_param}")
        body = resp.json()
        assert body["count"] == 1

    def test_limit_zero_returns_422(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/agents/any/timeline?limit=0")
        assert resp.status_code == 422

    def test_limit_above_max_returns_422(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/agents/any/timeline?limit=1001")
        assert resp.status_code == 422

    def test_signal_fields_present(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_signal(
            app,
            agent_id="fields_check",
            total_latency_ms=123.4,
            retrieval_count=5,
            token_count=512,
            kl_divergence=0.12,
            tool_calls=3,
            memory_query_entropy=0.88,
        )
        resp = tc.get("/dashboard/api/agents/fields_check/timeline")
        p = resp.json()["points"][0]
        assert p["total_latency_ms"] == pytest.approx(123.4)
        assert p["retrieval_count"] == 5
        assert p["token_count"] == 512
        assert p["kl_divergence"] == pytest.approx(0.12)
        assert p["tool_calls"] == 3
        assert p["memory_query_entropy"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# GET /dashboard/api/allocations
# ---------------------------------------------------------------------------


class TestListAllocations:
    def test_empty_returns_200(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/allocations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["allocations"] == []
        assert body["count"] == 0

    def test_returns_entries(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_allocation(app, task_id="pr-1::security_review")
        resp = tc.get("/dashboard/api/allocations")
        body = resp.json()
        assert body["count"] == 1
        entry = body["allocations"][0]
        assert entry["task_id"] == "pr-1::security_review"
        assert entry["task_type"] == "security_review"
        assert entry["assigned_agent"] == "agent_a"
        assert entry["escalated"] is False

    def test_ordered_newest_first(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_allocation(app, task_id="old_task", timestamp=_ts(-100))
        _seed_allocation(app, task_id="new_task", timestamp=_ts(-1))
        resp = tc.get("/dashboard/api/allocations")
        ids = [a["task_id"] for a in resp.json()["allocations"]]
        assert ids[0] == "new_task"
        assert ids[1] == "old_task"

    def test_limit_respected(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        for _ in range(5):
            _seed_allocation(app)
        resp = tc.get("/dashboard/api/allocations?limit=3")
        assert resp.json()["count"] == 3

    def test_json_fields_round_trip(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        bids = [
            {"agent_id": "agent_a", "capability": 0.9, "health": 0.85, "score": 0.87},
            {"agent_id": "agent_b", "capability": 0.7, "health": 0.6, "score": 0.65},
        ]
        snap = {"agent_a": 0.85, "agent_b": 0.6}
        _seed_allocation(app, all_bids=bids, health_snapshot=snap)
        resp = tc.get("/dashboard/api/allocations")
        entry = resp.json()["allocations"][0]
        assert entry["all_bids"] == bids
        assert entry["health_snapshot"] == snap

    def test_limit_zero_returns_422(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/allocations?limit=0")
        assert resp.status_code == 422

    def test_escalated_flag_true(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_allocation(app, escalated=True, assigned_agent=None)
        resp = tc.get("/dashboard/api/allocations")
        entry = resp.json()["allocations"][0]
        assert entry["escalated"] is True
        assert entry["assigned_agent"] is None

    def test_503_when_factory_missing(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        original = app.state.session_factory
        app.state.session_factory = None
        try:
            resp = tc.get("/dashboard/api/allocations")
            assert resp.status_code == 503
        finally:
            app.state.session_factory = original


# ---------------------------------------------------------------------------
# GET /dashboard/api/memory
# ---------------------------------------------------------------------------


class TestMemoryStatus:
    def test_returns_correct_shape(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        # Stub the integrity module.
        stub_module = SimpleNamespace(
            baseline_fitted=True,
            baseline_size=50,
            pending_refit_count=3,
            total_retrievals=120,
            flag_threshold=0.6,
            weights={
                "embedding_outlier": 0.25,
                "freshness_anomaly": 0.25,
                "retrieval_frequency": 0.25,
                "content_embedding_mismatch": 0.25,
            },
        )
        app.state.integrity_module = stub_module
        resp = tc.get("/dashboard/api/memory")
        assert resp.status_code == 200
        body = resp.json()
        assert body["baseline_fitted"] is True
        assert body["baseline_size"] == 50
        assert body["pending_refit_count"] == 3
        assert body["total_retrievals"] == 120
        assert body["flag_threshold"] == pytest.approx(0.6)
        assert "embedding_outlier" in body["weights"]

    def test_quarantine_count_and_ids(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        stub_module = SimpleNamespace(
            baseline_fitted=False,
            baseline_size=0,
            pending_refit_count=0,
            total_retrievals=0,
            flag_threshold=0.6,
            weights={},
        )
        app.state.integrity_module = stub_module
        # quarantine_store is the real QuarantineStore (empty collection).
        resp = tc.get("/dashboard/api/memory")
        body = resp.json()
        assert body["quarantine_count"] == 0
        assert body["quarantined_ids"] == []

    def test_503_when_integrity_module_missing(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        tc, app = client_app
        original = app.state.integrity_module
        app.state.integrity_module = None
        try:
            resp = tc.get("/dashboard/api/memory")
            assert resp.status_code == 503
        finally:
            app.state.integrity_module = original

    def test_503_when_quarantine_store_missing(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        tc, app = client_app
        original = app.state.quarantine_store
        app.state.quarantine_store = None
        try:
            resp = tc.get("/dashboard/api/memory")
            assert resp.status_code == 503
        finally:
            app.state.quarantine_store = original


# ---------------------------------------------------------------------------
# GET /dashboard/api/escalations
# ---------------------------------------------------------------------------


class TestEscalationQueue:
    def test_empty_db_returns_empty_lists(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        resp = tc.get("/dashboard/api/escalations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pending"] == []
        assert body["recent_resolved"] == []
        assert body["pending_count"] == 0
        assert body["resolved_count"] == 0

    def test_pending_and_resolved_split(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="approved", resolved_at=_now())
        _seed_escalation(app, status="rejected", resolved_at=_now())
        resp = tc.get("/dashboard/api/escalations")
        body = resp.json()
        assert body["pending_count"] == 1
        assert body["resolved_count"] == 2
        assert all(e["status"] == "pending" for e in body["pending"])
        assert all(e["status"] != "pending" for e in body["recent_resolved"])

    def test_status_filter_pending_only(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="approved", resolved_at=_now())
        resp = tc.get("/dashboard/api/escalations?status=pending")
        body = resp.json()
        assert body["pending_count"] == 1
        assert body["resolved_count"] == 0
        assert body["recent_resolved"] == []

    def test_status_filter_approved_only(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="approved", resolved_at=_now())
        resp = tc.get("/dashboard/api/escalations?status=approved")
        body = resp.json()
        assert body["pending_count"] == 0
        assert body["resolved_count"] == 1
        assert all(e["status"] == "approved" for e in body["recent_resolved"])

    def test_limit_respected(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        for _ in range(5):
            _seed_escalation(app, status="pending")
        resp = tc.get("/dashboard/api/escalations?limit=2")
        assert resp.json()["pending_count"] == 2

    def test_response_shape(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        eid = _seed_escalation(
            app, agent_id="agent_x", trigger="quarantine_event", status="pending"
        )
        resp = tc.get("/dashboard/api/escalations")
        entry = resp.json()["pending"][0]
        assert entry["id"] == eid
        assert entry["agent_id"] == "agent_x"
        assert entry["trigger"] == "quarantine_event"
        assert entry["status"] == "pending"
        assert entry["resolved_at"] is None

    def test_503_when_factory_missing(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        original = app.state.session_factory
        app.state.session_factory = None
        try:
            resp = tc.get("/dashboard/api/escalations")
            assert resp.status_code == 503
        finally:
            app.state.session_factory = original


# ---------------------------------------------------------------------------
# WebSocket /dashboard/ws/live
# ---------------------------------------------------------------------------


class TestLiveFeed:
    def test_frame_keys_present(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        with tc.websocket_connect("/dashboard/ws/live") as ws:
            frame = ws.receive_json()
        assert "timestamp" in frame
        assert "system_health" in frame
        assert "agent_count" in frame
        assert "pending_escalations" in frame
        assert "quarantine_count" in frame
        assert "agents" in frame

    def test_pending_escalations_reflects_db(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, app = client_app
        _seed_escalation(app, status="pending")
        _seed_escalation(app, status="pending")
        with tc.websocket_connect("/dashboard/ws/live") as ws:
            frame = ws.receive_json()
        assert frame["pending_escalations"] == 2

    def test_quarantine_count_zero_when_empty(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        with tc.websocket_connect("/dashboard/ws/live") as ws:
            frame = ws.receive_json()
        assert frame["quarantine_count"] == 0

    def test_agent_summaries_in_frame(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """When scorer has agents, they appear in the live frame."""
        tc, app = client_app

        class StubScorer:
            def get_all_health(self) -> dict[str, HealthUpdate]:
                return {
                    "live_agent": HealthUpdate(
                        agent_id="live_agent", health=0.75, bocpd_score=0.25, chronos_score=None
                    )
                }

            def stop(self) -> None:
                pass

        app.state.health_scorer = StubScorer()
        with tc.websocket_connect("/dashboard/ws/live") as ws:
            frame = ws.receive_json()
        assert frame["agent_count"] == 1
        assert frame["agents"][0]["agent_id"] == "live_agent"

    def test_not_ready_frame_when_scorer_missing(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        tc, app = client_app
        original = app.state.health_scorer
        app.state.health_scorer = None
        try:
            with tc.websocket_connect("/dashboard/ws/live") as ws:
                frame = ws.receive_json()
            assert frame == {"error": "not_ready"}
        finally:
            app.state.health_scorer = original

    def test_system_health_is_float(self, client_app: tuple[TestClient, FastAPI]) -> None:
        tc, _ = client_app
        with tc.websocket_connect("/dashboard/ws/live") as ws:
            frame = ws.receive_json()
        assert isinstance(frame["system_health"], float)
