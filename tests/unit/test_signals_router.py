"""Unit tests for GET /api/v1/agents/{agent_id}/signals (task 3.5)."""

from __future__ import annotations

import datetime
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from chronoagent.api.deps import get_db
from chronoagent.config import Settings
from chronoagent.db.models import AgentSignalRecord, Base
from chronoagent.main import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _in_memory_session_factory() -> sessionmaker[Session]:
    # StaticPool forces all connections through the same underlying SQLite connection,
    # ensuring create_all() and the session share the same in-memory database.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _make_record(
    agent_id: str = "security_reviewer",
    task_id: str | None = "pr-1",
    offset_seconds: int = 0,
    **overrides: Any,
) -> AgentSignalRecord:
    """Build an AgentSignalRecord with sensible defaults."""
    base_ts = datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
    return AgentSignalRecord(
        agent_id=agent_id,
        task_id=task_id,
        timestamp=base_ts + datetime.timedelta(seconds=offset_seconds),
        total_latency_ms=overrides.get("total_latency_ms", 100.0),
        retrieval_count=overrides.get("retrieval_count", 5),
        token_count=overrides.get("token_count", 200),
        kl_divergence=overrides.get("kl_divergence", 0.3),
        tool_calls=overrides.get("tool_calls", 2),
        memory_query_entropy=overrides.get("memory_query_entropy", 0.7),
    )


@pytest.fixture()
def client() -> Generator[tuple[TestClient, Session], None, None]:
    """TestClient wired to an in-memory SQLite DB via dependency override."""
    settings = Settings(env="test", llm_backend="mock")
    app = create_app(settings=settings)

    factory = _in_memory_session_factory()
    # Share a single session so the test can insert rows AND the router sees them
    shared_session = factory()

    def override_get_db() -> Generator[Session, None, None]:
        yield shared_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client, shared_session

    shared_session.close()


# ---------------------------------------------------------------------------
# Empty-result cases
# ---------------------------------------------------------------------------


class TestGetSignalsEmpty:
    def test_returns_200_for_unknown_agent(self, client: tuple[TestClient, Session]) -> None:
        """Agent with no rows returns HTTP 200, not 404."""
        tc, _ = client
        resp = tc.get("/api/v1/agents/nonexistent_agent/signals")
        assert resp.status_code == 200

    def test_empty_signals_list(self, client: tuple[TestClient, Session]) -> None:
        tc, _ = client
        resp = tc.get("/api/v1/agents/nobody/signals")
        body = resp.json()
        assert body["signals"] == []
        assert body["count"] == 0

    def test_empty_response_has_correct_agent_id(self, client: tuple[TestClient, Session]) -> None:
        tc, _ = client
        resp = tc.get("/api/v1/agents/ghost_agent/signals")
        assert resp.json()["agent_id"] == "ghost_agent"

    def test_empty_response_window_matches_query(self, client: tuple[TestClient, Session]) -> None:
        tc, _ = client
        resp = tc.get("/api/v1/agents/ghost/signals?window=25")
        assert resp.json()["window"] == 25


# ---------------------------------------------------------------------------
# Happy-path: records present
# ---------------------------------------------------------------------------


class TestGetSignalsWithData:
    def test_returns_inserted_signals(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        for i in range(3):
            session.add(_make_record("reviewer", offset_seconds=i))
        session.commit()

        resp = tc.get("/api/v1/agents/reviewer/signals")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["signals"]) == 3

    def test_signals_ordered_oldest_first(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        for i in range(5):
            session.add(_make_record("ordered_agent", offset_seconds=i * 10))
        session.commit()

        resp = tc.get("/api/v1/agents/ordered_agent/signals")
        signals = resp.json()["signals"]
        timestamps = [s["timestamp"] for s in signals]
        assert timestamps == sorted(timestamps)

    def test_window_limits_result_count(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        for i in range(20):
            session.add(_make_record("windowed_agent", offset_seconds=i))
        session.commit()

        resp = tc.get("/api/v1/agents/windowed_agent/signals?window=5")
        assert resp.json()["count"] == 5
        assert len(resp.json()["signals"]) == 5

    def test_window_returns_most_recent(self, client: tuple[TestClient, Session]) -> None:
        """With window=3 and 10 rows, the 3 newest rows should be returned."""
        tc, session = client
        for i in range(10):
            session.add(
                _make_record("latest_agent", offset_seconds=i, total_latency_ms=float(i * 10))
            )
        session.commit()

        resp = tc.get("/api/v1/agents/latest_agent/signals?window=3")
        latencies = [s["total_latency_ms"] for s in resp.json()["signals"]]
        # Oldest-first within the 3 most-recent: steps 7,8,9 → latency 70,80,90
        assert latencies == pytest.approx([70.0, 80.0, 90.0])

    def test_signal_field_values_correct(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        session.add(
            _make_record(
                "precise_agent",
                task_id="pr-42",
                total_latency_ms=123.4,
                retrieval_count=7,
                token_count=512,
                kl_divergence=0.42,
                tool_calls=3,
                memory_query_entropy=0.65,
            )
        )
        session.commit()

        resp = tc.get("/api/v1/agents/precise_agent/signals")
        sig = resp.json()["signals"][0]
        assert sig["task_id"] == "pr-42"
        assert sig["total_latency_ms"] == pytest.approx(123.4)
        assert sig["retrieval_count"] == 7
        assert sig["token_count"] == 512
        assert sig["kl_divergence"] == pytest.approx(0.42)
        assert sig["tool_calls"] == 3
        assert sig["memory_query_entropy"] == pytest.approx(0.65)

    def test_null_task_id_preserved(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        session.add(_make_record("null_task_agent", task_id=None))
        session.commit()

        resp = tc.get("/api/v1/agents/null_task_agent/signals")
        assert resp.json()["signals"][0]["task_id"] is None

    def test_agent_isolation(self, client: tuple[TestClient, Session]) -> None:
        """Records for agent A must not appear in agent B's response."""
        tc, session = client
        session.add(_make_record("agent_a", offset_seconds=0))
        session.add(_make_record("agent_b", offset_seconds=1))
        session.commit()

        resp_a = tc.get("/api/v1/agents/agent_a/signals")
        assert all(s["total_latency_ms"] is not None for s in resp_a.json()["signals"])
        assert resp_a.json()["count"] == 1

        resp_b = tc.get("/api/v1/agents/agent_b/signals")
        assert resp_b.json()["count"] == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGetSignalsValidation:
    def test_window_zero_returns_422(self, client: tuple[TestClient, Session]) -> None:
        """window=0 is below the minimum of 1 → HTTP 422."""
        tc, _ = client
        resp = tc.get("/api/v1/agents/any/signals?window=0")
        assert resp.status_code == 422

    def test_window_negative_returns_422(self, client: tuple[TestClient, Session]) -> None:
        tc, _ = client
        resp = tc.get("/api/v1/agents/any/signals?window=-1")
        assert resp.status_code == 422

    def test_window_too_large_returns_422(self, client: tuple[TestClient, Session]) -> None:
        tc, _ = client
        resp = tc.get("/api/v1/agents/any/signals?window=10001")
        assert resp.status_code == 422

    def test_default_window_is_50(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        for i in range(60):
            session.add(_make_record("default_window_agent", offset_seconds=i))
        session.commit()

        resp = tc.get("/api/v1/agents/default_window_agent/signals")
        assert resp.json()["window"] == 50
        assert resp.json()["count"] == 50

    def test_response_schema_has_required_fields(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        session.add(_make_record("schema_agent"))
        session.commit()

        body = tc.get("/api/v1/agents/schema_agent/signals").json()
        assert "agent_id" in body
        assert "window" in body
        assert "count" in body
        assert "signals" in body

    def test_signal_point_has_all_fields(self, client: tuple[TestClient, Session]) -> None:
        tc, session = client
        session.add(_make_record("fields_agent"))
        session.commit()

        sig = tc.get("/api/v1/agents/fields_agent/signals").json()["signals"][0]
        expected_keys = {
            "timestamp",
            "task_id",
            "total_latency_ms",
            "retrieval_count",
            "token_count",
            "kl_divergence",
            "tool_calls",
            "memory_query_entropy",
        }
        assert expected_keys.issubset(sig.keys())
