"""Tests for graceful-degradation builder helpers in ``chronoagent.main``.

Phase 9 task 9.3 wraps every external dependency (Redis, Postgres, ChromaDB,
Chronos) in a ``_build_*`` helper that attempts the primary backend and falls
back to a local alternative on failure, recording the outcome on
``app.state.component_status`` as a :class:`ComponentStatus`.  These tests
exercise each helper in isolation and the wired-up app-state surface area.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import chromadb
import pytest
from fastapi.testclient import TestClient

from chronoagent import main as main_module
from chronoagent.config import Settings
from chronoagent.main import (
    _build_chromadb,
    _build_database,
    _build_message_bus,
    _probe_forecaster,
    create_app,
)
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.observability.components import ComponentStatus

# ── ComponentStatus dataclass ────────────────────────────────────────────────


class TestComponentStatus:
    """ComponentStatus is a frozen, hashable dataclass."""

    def test_is_frozen(self) -> None:
        status = ComponentStatus(name="bus", mode="primary", detail="local bus")
        with pytest.raises(FrozenInstanceError):
            status.mode = "fallback"  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        status = ComponentStatus(name="bus", mode="primary", detail="local bus")
        assert hash(status) == hash(status)

    def test_fields_preserved(self) -> None:
        status = ComponentStatus(name="database", mode="fallback", detail="sqlite")
        assert status.name == "database"
        assert status.mode == "fallback"
        assert status.detail == "sqlite"


# ── _build_message_bus ───────────────────────────────────────────────────────


class TestBuildMessageBusDevTest:
    """Dev and test envs always use LocalBus as the primary."""

    def test_dev_returns_local_bus_primary(self) -> None:
        settings = Settings(env="dev", llm_backend="mock")
        bus, status = _build_message_bus(settings)
        assert isinstance(bus, LocalBus)
        assert status.name == "bus"
        assert status.mode == "primary"
        assert "local bus" in status.detail

    def test_test_env_returns_local_bus_primary(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        bus, status = _build_message_bus(settings)
        assert isinstance(bus, LocalBus)
        assert status.mode == "primary"


class TestBuildMessageBusProd:
    """Prod env tries Redis first and falls back on failure."""

    def test_prod_redis_unreachable_falls_back_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``RedisBus`` construction/ping raises, return LocalBus + fallback status."""
        import chronoagent.messaging.redis_bus as redis_bus_module

        class FakeClient:
            def ping(self) -> None:
                raise ConnectionError("redis unreachable")

        class FakeRedisBus:
            def __init__(self, url: str) -> None:  # noqa: D401
                self._client = FakeClient()

        monkeypatch.setattr(redis_bus_module, "RedisBus", FakeRedisBus)

        settings = Settings(
            env="prod",
            llm_backend="mock",
            redis_url="redis://nowhere:6379/0",
        )
        bus, status = _build_message_bus(settings)
        assert isinstance(bus, LocalBus)
        assert status.mode == "fallback"
        assert "redis unavailable" in status.detail
        assert "ConnectionError" in status.detail

    def test_prod_redis_constructor_raises_falls_back_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An exception raised directly from ``RedisBus.__init__`` is caught too."""
        import chronoagent.messaging.redis_bus as redis_bus_module

        class FakeRedisBus:
            def __init__(self, url: str) -> None:
                raise RuntimeError("redis-py refused to construct")

        monkeypatch.setattr(redis_bus_module, "RedisBus", FakeRedisBus)

        settings = Settings(env="prod", llm_backend="mock")
        bus, status = _build_message_bus(settings)
        assert isinstance(bus, LocalBus)
        assert status.mode == "fallback"
        assert "RuntimeError" in status.detail

    def test_prod_redis_success_returns_primary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the ping succeeds, the Redis-backed bus is returned as primary."""
        import chronoagent.messaging.redis_bus as redis_bus_module

        class FakeClient:
            def ping(self) -> bool:
                return True

        class FakeRedisBus:
            def __init__(self, url: str) -> None:
                self._client = FakeClient()
                self.url = url

        monkeypatch.setattr(redis_bus_module, "RedisBus", FakeRedisBus)

        settings = Settings(
            env="prod",
            llm_backend="mock",
            redis_url="redis://primary:6379/0",
        )
        bus, status = _build_message_bus(settings)
        assert isinstance(bus, FakeRedisBus)
        assert status.mode == "primary"
        assert "redis://primary:6379/0" in status.detail


# ── _build_database ──────────────────────────────────────────────────────────


class TestBuildDatabase:
    """Primary engine is probed and falls back to in-memory SQLite on failure."""

    def test_valid_sqlite_returns_primary(self) -> None:
        settings = Settings(
            env="test",
            llm_backend="mock",
            database_url="sqlite:///:memory:",
        )
        engine, status = _build_database(settings)
        assert status.name == "database"
        assert status.mode == "primary"
        assert status.detail == "sqlite:///:memory:"
        engine.dispose()

    def test_schema_created_on_primary(self) -> None:
        """``Base.metadata.create_all`` runs against the primary engine."""
        from sqlalchemy import inspect

        settings = Settings(
            env="test",
            llm_backend="mock",
            database_url="sqlite:///:memory:",
        )
        engine, _ = _build_database(settings)
        inspector = inspect(engine)
        assert "escalation_records" in inspector.get_table_names()
        engine.dispose()

    def test_broken_url_falls_back_to_in_memory_sqlite(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the primary engine raises on connect, fall back to ``sqlite:///:memory:``."""
        from sqlalchemy import inspect

        import chronoagent.main as main_mod

        def broken_make_engine(_settings: Settings) -> Any:
            raise RuntimeError("postgres down")

        monkeypatch.setattr(main_mod, "make_engine", broken_make_engine)

        settings = Settings(
            env="prod",
            llm_backend="mock",
            database_url="postgresql://nowhere/db",
        )
        engine, status = _build_database(settings)
        assert status.mode == "fallback"
        assert "sqlite:///:memory:" in status.detail
        assert "postgresql://nowhere/db" in status.detail
        assert "RuntimeError" in status.detail
        # Fallback schema is valid.
        inspector = inspect(engine)
        assert "escalation_records" in inspector.get_table_names()
        engine.dispose()

    def test_fallback_engine_is_thread_shared(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback engine uses StaticPool so threads see the same in-memory DB."""
        import threading

        from sqlalchemy import text

        import chronoagent.main as main_mod

        def broken_make_engine(_settings: Settings) -> Any:
            raise RuntimeError("primary down")

        monkeypatch.setattr(main_mod, "make_engine", broken_make_engine)

        settings = Settings(env="test", llm_backend="mock")
        engine, _ = _build_database(settings)

        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS probe (v INTEGER)"))
            conn.execute(text("INSERT INTO probe VALUES (42)"))

        results: list[int] = []

        def reader() -> None:
            with engine.connect() as conn:
                row = conn.execute(text("SELECT v FROM probe")).scalar_one()
                results.append(int(row))

        thread = threading.Thread(target=reader)
        thread.start()
        thread.join()
        assert results == [42]
        engine.dispose()


# ── _build_chromadb ──────────────────────────────────────────────────────────


class TestBuildChromadb:
    """EphemeralClient is the primary; failure surfaces loudly (no fallback)."""

    def test_returns_ephemeral_primary(self) -> None:
        client, status = _build_chromadb()
        assert status.name == "chromadb"
        assert status.mode == "primary"
        assert "ephemeral" in status.detail
        # Smoke-test: client is usable.
        collection = client.get_or_create_collection("probe")
        assert collection.name == "probe"

    def test_construction_failure_reraises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """There is no fallback path for Chroma; the error propagates."""

        def broken() -> Any:
            raise RuntimeError("chroma broken")

        monkeypatch.setattr(chromadb, "EphemeralClient", broken)

        with pytest.raises(RuntimeError, match="chroma broken"):
            _build_chromadb()


# ── _probe_forecaster ────────────────────────────────────────────────────────


class TestProbeForecaster:
    """Chronos availability is surfaced as primary/fallback on component_status."""

    def test_reports_primary_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import chronoagent.main as main_mod

        class FakeForecaster:
            @property
            def available(self) -> bool:
                return True

        monkeypatch.setattr(main_mod, "ChronosForecaster", FakeForecaster)

        status = _probe_forecaster()
        assert status.name == "forecaster"
        assert status.mode == "primary"
        assert "Chronos" in status.detail

    def test_reports_fallback_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import chronoagent.main as main_mod

        class FakeForecaster:
            @property
            def available(self) -> bool:
                return False

        monkeypatch.setattr(main_mod, "ChronosForecaster", FakeForecaster)

        status = _probe_forecaster()
        assert status.mode == "fallback"
        assert "BOCPD-only" in status.detail


# ── app.state.component_status wiring ────────────────────────────────────────


class TestComponentStatusOnAppState:
    """Lifespan populates ``app.state.component_status`` with all four subsystems."""

    def test_default_test_env_has_four_components(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app):
            status_map = app.state.component_status
            assert set(status_map.keys()) == {"bus", "database", "chromadb", "forecaster"}
            for status in status_map.values():
                assert isinstance(status, ComponentStatus)

    def test_test_env_bus_is_primary_local(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app):
            bus_status = app.state.component_status["bus"]
            assert bus_status.mode == "primary"
            assert "local bus" in bus_status.detail

    def test_test_env_database_is_primary(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app):
            db_status = app.state.component_status["database"]
            assert db_status.mode == "primary"

    def test_broken_database_surfaces_as_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Monkeypatching ``make_engine`` to raise drives the fallback path end-to-end."""

        def broken_make_engine(_settings: Settings) -> Any:
            raise RuntimeError("simulated postgres outage")

        monkeypatch.setattr(main_module, "make_engine", broken_make_engine)

        settings = Settings(
            env="test",
            llm_backend="mock",
            database_url="postgresql://nowhere/db",
        )
        app = create_app(settings=settings)
        with TestClient(app) as client:
            db_status = app.state.component_status["database"]
            assert db_status.mode == "fallback"
            assert "RuntimeError" in db_status.detail
            # App still serves health despite the fallback.
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_broken_redis_in_prod_surfaces_as_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Prod env with unreachable Redis falls back and app still boots."""
        import chronoagent.messaging.redis_bus as redis_bus_module

        class FakeClient:
            def ping(self) -> None:
                raise ConnectionError("no redis here")

        class FakeRedisBus:
            def __init__(self, url: str) -> None:
                self._client = FakeClient()

        monkeypatch.setattr(redis_bus_module, "RedisBus", FakeRedisBus)

        settings = Settings(
            env="prod",
            llm_backend="mock",
            redis_url="redis://nowhere:6379/0",
        )
        app = create_app(settings=settings)
        with TestClient(app):
            bus_status = app.state.component_status["bus"]
            assert bus_status.mode == "fallback"
            assert isinstance(app.state.bus, LocalBus)

    def test_forecaster_status_present(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app):
            fc_status = app.state.component_status["forecaster"]
            assert fc_status.name == "forecaster"
            assert fc_status.mode in ("primary", "fallback")
