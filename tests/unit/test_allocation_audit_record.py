"""Unit tests for the AllocationAuditRecord ORM model (Phase 5 task 5.5).

The model is the persistent surface for contract-net allocation
decisions.  These tests cover ORM round-trip via an in-memory SQLite
database, JSON-typed columns for the bid ledger and health snapshot,
the nullable ``assigned_agent`` for escalation rows, and the Alembic
migration upgrade/downgrade pair so the schema can be re-created on a
fresh database without :meth:`Base.metadata.create_all`.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.config import Settings
from chronoagent.db.models import AllocationAuditRecord, Base
from chronoagent.db.session import make_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def in_memory_session() -> Session:
    """SQLAlchemy Session backed by an in-memory SQLite DB with all tables created."""
    settings = Settings(database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with session_factory() as session:
        yield session


def _sample_bids() -> list[dict[str, float | str]]:
    """A minimal four-agent bid ledger in canonical AGENT_IDS order."""
    return [
        {"agent_id": "security_reviewer", "capability": 1.0, "health": 0.9, "score": 0.9},
        {"agent_id": "style_reviewer", "capability": 0.4, "health": 0.95, "score": 0.38},
        {"agent_id": "summarizer", "capability": 0.3, "health": 1.0, "score": 0.3},
        {"agent_id": "planner", "capability": 0.2, "health": 1.0, "score": 0.2},
    ]


def _sample_snapshot() -> dict[str, float]:
    return {
        "security_reviewer": 0.9,
        "style_reviewer": 0.95,
        "summarizer": 1.0,
        "planner": 1.0,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestAllocationAuditRecord:
    def test_repr_contains_identity_fields(self) -> None:
        record = AllocationAuditRecord(
            task_id="pr-42::security_review",
            task_type="security_review",
            assigned_agent="security_reviewer",
            escalated=False,
            all_bids=_sample_bids(),
            health_snapshot=_sample_snapshot(),
            rationale="assigned",
            threshold=0.25,
            timestamp=datetime.datetime.now(datetime.UTC),
        )
        text = repr(record)
        assert "pr-42::security_review" in text
        assert "security_review" in text
        assert "security_reviewer" in text
        assert "escalated=False" in text

    def test_assigned_agent_nullable_for_escalation(self) -> None:
        record = AllocationAuditRecord(
            task_id="pr-9::style_review",
            task_type="style_review",
            assigned_agent=None,
            escalated=True,
            all_bids=_sample_bids(),
            health_snapshot={k: 0.0 for k in _sample_snapshot()},
            rationale="escalated: max bid below threshold",
            threshold=0.25,
            timestamp=datetime.datetime.now(datetime.UTC),
        )
        assert record.assigned_agent is None
        assert record.escalated is True


# ---------------------------------------------------------------------------
# ORM round-trip via in-memory SQLite
# ---------------------------------------------------------------------------


class TestAllocationAuditRecordPersistence:
    def test_insert_and_query(self, in_memory_session: Session) -> None:
        ts = datetime.datetime.now(datetime.UTC)
        record = AllocationAuditRecord(
            task_id="pr-1::security_review",
            task_type="security_review",
            assigned_agent="security_reviewer",
            escalated=False,
            all_bids=_sample_bids(),
            health_snapshot=_sample_snapshot(),
            rationale="assigned to 'security_reviewer'",
            threshold=0.25,
            timestamp=ts,
        )
        in_memory_session.add(record)
        in_memory_session.commit()

        rows = in_memory_session.execute(select(AllocationAuditRecord)).scalars().all()
        assert len(rows) == 1
        loaded = rows[0]
        assert loaded.id is not None
        assert loaded.task_id == "pr-1::security_review"
        assert loaded.task_type == "security_review"
        assert loaded.assigned_agent == "security_reviewer"
        assert loaded.escalated is False
        assert loaded.threshold == pytest.approx(0.25)
        assert loaded.rationale == "assigned to 'security_reviewer'"

    def test_json_columns_round_trip(self, in_memory_session: Session) -> None:
        bids = _sample_bids()
        snapshot = _sample_snapshot()
        in_memory_session.add(
            AllocationAuditRecord(
                task_id="pr-2::security_review",
                task_type="security_review",
                assigned_agent="security_reviewer",
                escalated=False,
                all_bids=bids,
                health_snapshot=snapshot,
                rationale="assigned",
                threshold=0.25,
                timestamp=datetime.datetime.now(datetime.UTC),
            )
        )
        in_memory_session.commit()

        loaded = in_memory_session.execute(select(AllocationAuditRecord)).scalars().one()
        assert loaded.all_bids == bids
        assert loaded.health_snapshot == snapshot
        # Order is preserved (canonical AGENT_IDS order matters for replay).
        assert [b["agent_id"] for b in loaded.all_bids] == [
            "security_reviewer",
            "style_reviewer",
            "summarizer",
            "planner",
        ]

    def test_escalated_row_persists_with_null_agent(self, in_memory_session: Session) -> None:
        in_memory_session.add(
            AllocationAuditRecord(
                task_id="pr-3::style_review",
                task_type="style_review",
                assigned_agent=None,
                escalated=True,
                all_bids=_sample_bids(),
                health_snapshot={k: 0.0 for k in _sample_snapshot()},
                rationale="escalated: max bid 0.0000 below threshold 0.2500",
                threshold=0.25,
                timestamp=datetime.datetime.now(datetime.UTC),
            )
        )
        in_memory_session.commit()

        loaded = in_memory_session.execute(select(AllocationAuditRecord)).scalars().one()
        assert loaded.escalated is True
        assert loaded.assigned_agent is None
        assert loaded.rationale.startswith("escalated:")

    def test_timestamp_round_trip_is_utc(self, in_memory_session: Session) -> None:
        before = datetime.datetime.now(datetime.UTC)
        in_memory_session.add(
            AllocationAuditRecord(
                task_id="pr-4::security_review",
                task_type="security_review",
                assigned_agent="security_reviewer",
                escalated=False,
                all_bids=_sample_bids(),
                health_snapshot=_sample_snapshot(),
                rationale="assigned",
                threshold=0.25,
                timestamp=before,
            )
        )
        in_memory_session.commit()
        after = datetime.datetime.now(datetime.UTC)

        loaded = in_memory_session.execute(select(AllocationAuditRecord)).scalars().one()
        ts = loaded.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.UTC)
        assert before <= ts <= after

    def test_query_by_task_id_uses_index(self, in_memory_session: Session) -> None:
        # Two rows for the same task, one for another. The composite
        # (task_id, timestamp) index supports this lookup pattern.
        for i, task_id in enumerate(["pr-5::sec", "pr-5::sec", "pr-6::sec"]):
            in_memory_session.add(
                AllocationAuditRecord(
                    task_id=task_id,
                    task_type="security_review",
                    assigned_agent="security_reviewer",
                    escalated=False,
                    all_bids=_sample_bids(),
                    health_snapshot=_sample_snapshot(),
                    rationale=f"assigned {i}",
                    threshold=0.25,
                    timestamp=datetime.datetime.now(datetime.UTC),
                )
            )
        in_memory_session.commit()

        rows = (
            in_memory_session.execute(
                select(AllocationAuditRecord).where(AllocationAuditRecord.task_id == "pr-5::sec")
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2

    def test_table_has_composite_index(self, in_memory_session: Session) -> None:
        engine = in_memory_session.get_bind()
        inspector = inspect(engine)
        names = {idx["name"] for idx in inspector.get_indexes("allocation_audit_records")}
        assert "ix_aar_task_ts" in names


# ---------------------------------------------------------------------------
# Alembic migration round-trip
# ---------------------------------------------------------------------------


class TestAllocationAuditRecordMigration:
    """Run the Alembic upgrade/downgrade pair against a real file-backed SQLite.

    The project's ``alembic/env.py`` calls
    :func:`chronoagent.config.load_settings` and overrides
    ``sqlalchemy.url`` from the result.  Since ``load_settings`` reads
    ``configs/base.yaml`` and passes the values as init kwargs (which
    outrank environment variables in pydantic-settings), we monkeypatch
    ``load_settings`` directly so the migration runs against a tmp
    SQLite file rather than the dev database.
    """

    def _config(self) -> Config:
        cfg = Config()
        cfg.set_main_option("script_location", "alembic")
        return cfg

    def _redirect_settings(self, monkeypatch: pytest.MonkeyPatch, url: str) -> None:
        from chronoagent import config as config_module

        def _fake_load_settings(yaml_path: Path | None = None) -> Settings:
            return Settings(database_url=url)

        monkeypatch.setattr(config_module, "load_settings", _fake_load_settings)

    def test_upgrade_creates_table_and_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "audit.db"
        url = f"sqlite:///{db_path}"
        self._redirect_settings(monkeypatch, url)

        command.upgrade(self._config(), "002")

        engine = make_engine(Settings(database_url=url))
        try:
            inspector = inspect(engine)
            assert "allocation_audit_records" in inspector.get_table_names()
            cols = {c["name"] for c in inspector.get_columns("allocation_audit_records")}
            assert {
                "id",
                "task_id",
                "task_type",
                "assigned_agent",
                "escalated",
                "all_bids",
                "health_snapshot",
                "rationale",
                "threshold",
                "timestamp",
            }.issubset(cols)
            idx_names = {idx["name"] for idx in inspector.get_indexes("allocation_audit_records")}
            assert "ix_aar_task_ts" in idx_names
        finally:
            engine.dispose()

    def test_downgrade_drops_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "audit_down.db"
        url = f"sqlite:///{db_path}"
        self._redirect_settings(monkeypatch, url)

        cfg = self._config()
        command.upgrade(cfg, "002")
        command.downgrade(cfg, "001")

        engine = make_engine(Settings(database_url=url))
        try:
            inspector = inspect(engine)
            assert "allocation_audit_records" not in inspector.get_table_names()
            # 001 table is still there.
            assert "agent_signal_records" in inspector.get_table_names()
        finally:
            engine.dispose()
