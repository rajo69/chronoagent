"""FastAPI application factory.

Lifespan initialises DB / Redis / Chroma connections and tears them down
cleanly.  Every external dependency is initialised through a ``_build_*``
helper that attempts the primary backend and falls back to a local
alternative on failure, recording the outcome on
``app.state.component_status`` for the forthcoming ``/api/v1/health``
endpoint (Phase 9 task 9.4).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import chromadb
from chromadb.api import ClientAPI
from fastapi import FastAPI
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import StaticPool

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.api.middleware import RateLimitConfig, RateLimitMiddleware
from chronoagent.config import Settings, load_settings
from chronoagent.db.models import Base
from chronoagent.db.session import make_engine, make_session_factory_from_engine
from chronoagent.escalation.audit import AuditTrailLogger
from chronoagent.escalation.escalation_manager import EscalationHandler
from chronoagent.memory.integrity import MemoryIntegrityModule
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.memory.store import MemoryStore
from chronoagent.messaging.bus import MessageBus
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.observability.components import ComponentStatus
from chronoagent.observability.logging import configure_logging, get_logger
from chronoagent.observability.metrics import ChronoAgentMetrics
from chronoagent.observability.metrics_wiring import (
    subscribe_metrics_to_bus,
    unsubscribe_metrics_from_bus,
)
from chronoagent.pipeline.graph import ReviewPipeline
from chronoagent.scorer.chronos_forecaster import ChronosForecaster
from chronoagent.scorer.health_scorer import TemporalHealthScorer

logger = get_logger(__name__)


# ── Graceful-degradation builder helpers (Phase 9 task 9.3) ───────────────────
#
# Each helper attempts the primary backend for an external dependency and
# falls back to a local alternative on any failure, returning a
# :class:`ComponentStatus` describing the outcome.  The helpers are module-
# level so tests can monkeypatch them individually to exercise fallback paths
# without standing up real infrastructure.


def _build_message_bus(settings: Settings) -> tuple[MessageBus, ComponentStatus]:
    """Construct the message bus with graceful fallback to ``LocalBus``.

    Prod environments attempt to connect to the configured Redis URL; on any
    failure (including ``ImportError`` or network errors) we log a warning and
    fall back to an in-process :class:`LocalBus`.  Dev and test environments
    skip Redis entirely and treat ``LocalBus`` as the primary choice.
    """
    if settings.env != "prod":
        return LocalBus(), ComponentStatus(
            name="bus",
            mode="primary",
            detail="local bus (dev/test)",
        )

    try:
        from chronoagent.messaging.redis_bus import RedisBus

        bus = RedisBus(url=settings.redis_url)
        # ``redis.Redis.from_url`` is lazy; ping() is the actual probe.
        bus._client.ping()  # noqa: SLF001 — single-shot probe, not a public API
    except Exception as exc:  # noqa: BLE001 — fallback must catch all failures
        logger.warning(
            "bus_fallback",
            primary="redis",
            url=settings.redis_url,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return LocalBus(), ComponentStatus(
            name="bus",
            mode="fallback",
            detail=f"local bus (redis unavailable: {type(exc).__name__})",
        )

    return bus, ComponentStatus(
        name="bus",
        mode="primary",
        detail=f"redis: {settings.redis_url}",
    )


def _build_database(settings: Settings) -> tuple[Engine, ComponentStatus]:
    """Construct the database engine with graceful fallback to in-memory SQLite.

    The primary engine is probed with ``SELECT 1`` to surface connection errors
    immediately; if the probe or the subsequent ``create_all`` fails we fall
    back to an in-memory SQLite engine wired to :class:`StaticPool` so the
    single backing database is shared across threads for the lifetime of the
    app.  All ``create_all`` calls are idempotent so re-running on the fallback
    engine is safe.
    """
    primary_url = settings.database_url
    try:
        engine = make_engine(settings)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(engine)
    except Exception as exc:  # noqa: BLE001 — fallback must catch all failures
        logger.warning(
            "database_fallback",
            primary=primary_url,
            reason=f"{type(exc).__name__}: {exc}",
        )
        fallback_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(fallback_engine)
        return fallback_engine, ComponentStatus(
            name="database",
            mode="fallback",
            detail=(
                f"sqlite:///:memory: (primary {primary_url} unavailable: {type(exc).__name__})"
            ),
        )

    return engine, ComponentStatus(
        name="database",
        mode="primary",
        detail=primary_url,
    )


def _build_chromadb() -> tuple[ClientAPI, ComponentStatus]:
    """Construct the ChromaDB client.

    ChromaDB's :class:`~chromadb.EphemeralClient` is fully in-process and has
    no external dependencies, so it is always the primary choice today.  The
    helper still routes through the component-status pipeline so the 9.4
    health endpoint can label it uniformly with the other subsystems, and so
    tests can force the ``unavailable`` branch by monkeypatching this helper.
    """
    try:
        client: ClientAPI = chromadb.EphemeralClient()
    except Exception as exc:  # noqa: BLE001 — label unavailable instead of crashing
        logger.error(
            "chromadb_unavailable",
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise

    return client, ComponentStatus(
        name="chromadb",
        mode="primary",
        detail="ephemeral (in-process)",
    )


def _probe_forecaster() -> ComponentStatus:
    """Probe the Chronos forecaster and report primary vs BOCPD-only fallback.

    :class:`ChronosForecaster` already degrades gracefully at the method level
    when the ``chronos-forecasting`` package is absent; this helper surfaces
    the same signal up to ``app.state.component_status`` so operators see a
    single source of truth for "is Chronos actually loaded".
    """
    if ChronosForecaster().available:
        return ComponentStatus(
            name="forecaster",
            mode="primary",
            detail="BOCPD + Chronos ensemble",
        )
    return ComponentStatus(
        name="forecaster",
        mode="fallback",
        detail="BOCPD-only (chronos-forecasting not installed)",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Args:
        app: The :class:`~fastapi.FastAPI` instance being started.

    Yields:
        Control to the running application.
    """
    settings: Settings = app.state.settings
    configure_logging(settings.env)

    logger.info("chronoagent starting", env=settings.env, backend=settings.llm_backend)

    # Component status registry — populated by each ``_build_*`` helper below
    # and consumed by the 9.4 health endpoint.  Keyed by logical component
    # name so routers can look up a specific component directly.
    component_status: dict[str, ComponentStatus] = {}
    app.state.component_status = component_status

    # Database: primary engine (Postgres/SQLite from settings) with
    # graceful fallback to an in-memory SQLite engine.
    engine, db_status = _build_database(settings)
    component_status[db_status.name] = db_status
    app.state.session_factory = make_session_factory_from_engine(engine)

    app.state.pipeline = ReviewPipeline.create()
    app.state.review_store = {}

    # Memory integrity subsystem: active store, quarantine store, integrity
    # module.  Chroma uses the shared in-process client built above.
    chroma_client, chroma_status = _build_chromadb()
    component_status[chroma_status.name] = chroma_status
    memory_backend = MockBackend()
    app.state.active_store = MemoryStore(
        chroma_client.get_or_create_collection("memory_active"),
        memory_backend,
    )
    app.state.quarantine_store = QuarantineStore(
        chroma_client.get_or_create_collection("memory_quarantine"),
    )
    app.state.integrity_module = MemoryIntegrityModule(memory_backend)

    # Messaging bus: Redis in prod, LocalBus in dev/test, with graceful
    # fallback to LocalBus when Redis is unreachable.
    bus, bus_status = _build_message_bus(settings)
    component_status[bus_status.name] = bus_status
    app.state.bus = bus
    app.state.health_scorer = TemporalHealthScorer(bus=bus)

    # Forecaster status: Chronos degrades to BOCPD-only gracefully; this
    # helper surfaces the effective mode on component_status.
    forecaster_status = _probe_forecaster()
    component_status[forecaster_status.name] = forecaster_status

    # Human escalation layer (Phase 7).
    app.state.audit_logger = AuditTrailLogger(app.state.session_factory)
    escalation_handler = EscalationHandler(
        bus=bus,
        health_scorer=app.state.health_scorer,
        quarantine_store=app.state.quarantine_store,
        session_factory=app.state.session_factory,
        audit_logger=app.state.audit_logger,
    )
    app.state.escalation_handler = escalation_handler
    bus.subscribe("health_updates", escalation_handler.on_health_update)
    bus.subscribe("memory.quarantine", escalation_handler.on_quarantine_event)

    # Prometheus metrics sink (Phase 8 task 8.3).  Isolated registry, wired
    # to the bus via closures in ``observability.metrics_wiring`` so the
    # sink remains passive and testable in isolation.
    app.state.metrics = ChronoAgentMetrics()
    app.state.metrics_subscribers = subscribe_metrics_to_bus(bus, app.state.metrics)

    yield

    unsubscribe_metrics_from_bus(bus, app.state.metrics_subscribers)
    bus.unsubscribe("health_updates", escalation_handler.on_health_update)
    bus.unsubscribe("memory.quarantine", escalation_handler.on_quarantine_event)
    app.state.health_scorer.stop()
    logger.info("chronoagent shutting down")


def create_app(
    settings: Settings | None = None,
    *,
    rate_limit_config: RateLimitConfig | None = None,
    rate_limit_clock: Callable[[], float] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional pre-built :class:`Settings` instance.
                  Defaults to :func:`~chronoagent.config.load_settings`.
        rate_limit_config: Optional :class:`RateLimitConfig` controlling
            the HTTP + WebSocket rate limiter installed as middleware.
            Defaults to the PLAN's policy (POST 10/min, GET 60/min,
            WS 5 concurrent) when ``None``.
        rate_limit_clock: Optional monotonic clock callable used by the
            rate limiter for bucket rollover.  Tests inject a mutable
            clock to drive the sliding window deterministically.

    Returns:
        Configured :class:`~fastapi.FastAPI` application.
    """
    from chronoagent.api.health import router as health_router
    from chronoagent.api.routers.dashboard import router as dashboard_router
    from chronoagent.api.routers.escalation import router as escalation_router
    from chronoagent.api.routers.health_scores import router as health_scores_router
    from chronoagent.api.routers.memory import router as memory_router
    from chronoagent.api.routers.metrics import router as metrics_router
    from chronoagent.api.routers.review import router as review_router
    from chronoagent.api.routers.signals import router as signals_router

    resolved_settings = settings or load_settings()

    app = FastAPI(
        title="ChronoAgent",
        description="Temporal health monitoring for LLM multi-agent systems",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    # Production hardening (Phase 9 task 9.1): install the rate-limit
    # middleware before any router so 429 responses never touch handler
    # code.  ``add_middleware`` stores the class + kwargs and Starlette
    # instantiates it when the ASGI stack is built.
    app.add_middleware(
        RateLimitMiddleware,
        config=rate_limit_config or RateLimitConfig(),
        clock=rate_limit_clock,
    )

    app.include_router(health_router)
    app.include_router(dashboard_router)
    app.include_router(escalation_router)
    app.include_router(health_scores_router)
    app.include_router(memory_router)
    app.include_router(metrics_router)
    app.include_router(review_router)
    app.include_router(signals_router)

    return app


app = create_app()
