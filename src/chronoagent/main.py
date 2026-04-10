"""FastAPI application factory.

Lifespan initialises DB / Redis / Chroma connections and tears them down cleanly.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chronoagent.agents.summarizer import ReviewReport
from chronoagent.config import Settings, load_settings
from chronoagent.db.models import Base
from chronoagent.db.session import make_engine, make_session_factory_from_engine
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.observability.logging import configure_logging, get_logger
from chronoagent.pipeline.graph import ReviewPipeline
from chronoagent.scorer.health_scorer import TemporalHealthScorer

logger = get_logger(__name__)


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

    # Database setup — create tables for SQLite (dev/test); use Alembic for prod.
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    app.state.session_factory = make_session_factory_from_engine(engine)

    app.state.pipeline: ReviewPipeline = ReviewPipeline.create()
    app.state.review_store: dict[str, ReviewReport] = {}

    # Messaging bus + health scorer — use LocalBus in dev; swap for RedisBus in prod.
    bus = LocalBus()
    app.state.bus = bus
    app.state.health_scorer = TemporalHealthScorer(bus=bus)

    yield

    app.state.health_scorer.stop()
    logger.info("chronoagent shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional pre-built :class:`Settings` instance.
                  Defaults to :func:`~chronoagent.config.load_settings`.

    Returns:
        Configured :class:`~fastapi.FastAPI` application.
    """
    from chronoagent.api.health import router as health_router
    from chronoagent.api.routers.health_scores import router as health_scores_router
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
    app.include_router(health_router)
    app.include_router(health_scores_router)
    app.include_router(review_router)
    app.include_router(signals_router)

    return app


app = create_app()
