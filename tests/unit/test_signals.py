"""Unit tests for AgentSignalRecord and BehavioralCollector.persist_step (task 3.1 / 3.4).

Uses an in-memory SQLite database — no PostgreSQL required for unit tests.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from chronoagent.config import Settings
from chronoagent.db.models import AgentSignalRecord, Base
from chronoagent.db.session import make_engine, make_session_factory
from chronoagent.monitor.collector import BehavioralCollector, StepSignals

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def in_memory_session() -> Session:
    """SQLAlchemy Session backed by an in-memory SQLite DB with all tables created."""
    from sqlalchemy.orm import sessionmaker

    settings = Settings(database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    # Reuse the SAME engine so the in-memory tables are visible to the session.
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# AgentSignalRecord — model construction
# ---------------------------------------------------------------------------


class TestAgentSignalRecord:
    def test_repr_contains_agent_id(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        record = AgentSignalRecord(
            agent_id="security_reviewer",
            task_id="pr-42",
            timestamp=now,
            total_latency_ms=100.0,
            retrieval_count=5,
            token_count=200,
            kl_divergence=0.3,
            tool_calls=2,
            memory_query_entropy=0.7,
        )
        assert "security_reviewer" in repr(record)
        assert "pr-42" in repr(record)

    def test_nullable_task_id(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        record = AgentSignalRecord(
            agent_id="planner",
            task_id=None,
            timestamp=now,
            total_latency_ms=50.0,
            retrieval_count=3,
            token_count=100,
            kl_divergence=0.0,
            tool_calls=1,
            memory_query_entropy=0.0,
        )
        assert record.task_id is None


# ---------------------------------------------------------------------------
# persist_step — round-trip via DB
# ---------------------------------------------------------------------------


class TestPersistStep:
    def _make_signals(self) -> StepSignals:
        return StepSignals(
            total_latency_ms=123.4,
            retrieval_count=7,
            token_count=512,
            kl_divergence=0.42,
            tool_calls=3,
            memory_query_entropy=0.65,
        )

    def test_persist_step_adds_record(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        signals = self._make_signals()
        collector.persist_step(
            in_memory_session, signals, agent_id="security_reviewer", task_id="pr-1"
        )
        in_memory_session.commit()

        rows = in_memory_session.execute(select(AgentSignalRecord)).scalars().all()
        assert len(rows) == 1

    def test_persist_step_signal_values_round_trip(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        signals = self._make_signals()
        collector.persist_step(
            in_memory_session, signals, agent_id="style_reviewer", task_id="pr-99"
        )
        in_memory_session.commit()

        row = in_memory_session.execute(select(AgentSignalRecord)).scalars().first()
        assert row is not None
        assert row.agent_id == "style_reviewer"
        assert row.task_id == "pr-99"
        assert row.total_latency_ms == pytest.approx(123.4)
        assert row.retrieval_count == 7
        assert row.token_count == 512
        assert row.kl_divergence == pytest.approx(0.42)
        assert row.tool_calls == 3
        assert row.memory_query_entropy == pytest.approx(0.65)

    def test_persist_step_timestamp_is_utc(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        before = datetime.datetime.now(datetime.UTC)
        collector.persist_step(
            in_memory_session, self._make_signals(), agent_id="planner"
        )
        in_memory_session.commit()
        after = datetime.datetime.now(datetime.UTC)

        row = in_memory_session.execute(select(AgentSignalRecord)).scalars().first()
        assert row is not None
        # SQLite stores timezone-aware datetimes; compare naively via utctimetuple
        ts = row.timestamp
        # Strip tzinfo for comparison if SQLite returns naive datetime
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.UTC)
        assert before <= ts <= after

    def test_persist_step_without_task_id(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        collector.persist_step(
            in_memory_session, self._make_signals(), agent_id="summarizer"
        )
        in_memory_session.commit()

        row = in_memory_session.execute(select(AgentSignalRecord)).scalars().first()
        assert row is not None
        assert row.task_id is None

    def test_persist_multiple_steps(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        for i in range(5):
            signals = StepSignals(
                total_latency_ms=float(i * 10),
                retrieval_count=i,
                token_count=i * 100,
                kl_divergence=float(i) * 0.1,
                tool_calls=i,
                memory_query_entropy=float(i) * 0.01,
            )
            collector.persist_step(
                in_memory_session,
                signals,
                agent_id="security_reviewer",
                task_id=f"pr-{i}",
            )
        in_memory_session.commit()

        rows = in_memory_session.execute(select(AgentSignalRecord)).scalars().all()
        assert len(rows) == 5

    def test_persist_returns_record_instance(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        record = collector.persist_step(
            in_memory_session, self._make_signals(), agent_id="planner"
        )
        assert isinstance(record, AgentSignalRecord)

    def test_persist_different_agents(self, in_memory_session: Session) -> None:
        collector = BehavioralCollector()
        agents = ["security_reviewer", "style_reviewer", "summarizer"]
        for agent in agents:
            collector.persist_step(
                in_memory_session, self._make_signals(), agent_id=agent
            )
        in_memory_session.commit()

        rows = in_memory_session.execute(select(AgentSignalRecord)).scalars().all()
        persisted_agents = {r.agent_id for r in rows}
        assert persisted_agents == set(agents)


# ---------------------------------------------------------------------------
# make_engine / make_session_factory
# ---------------------------------------------------------------------------


class TestSessionFactory:
    def test_make_engine_sqlite(self) -> None:
        settings = Settings(database_url="sqlite:///:memory:")
        engine = make_engine(settings)
        assert engine is not None

    def test_make_session_factory_returns_callable(self) -> None:
        settings = Settings(database_url="sqlite:///:memory:")
        factory = make_session_factory(settings)
        assert callable(factory)

    def test_session_factory_creates_session(self) -> None:
        settings = Settings(database_url="sqlite:///:memory:")
        engine = make_engine(settings)
        Base.metadata.create_all(engine)
        factory = make_session_factory(settings)
        with factory() as session:
            assert isinstance(session, Session)
