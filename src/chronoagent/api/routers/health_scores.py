"""Health scores router — per-agent temporal health from BOCPD + Chronos.

``GET /api/v1/agents/{agent_id}/health``
    Returns the current health score and BOCPD/Chronos components for one agent.

``GET /api/v1/agents/health``
    Returns the health snapshot for all tracked agents plus a system-level
    aggregate score (mean of all individual scores).

Response shape (single agent)::

    {
        "agent_id": "security_reviewer",
        "health": 0.92,
        "bocpd_score": 0.04,
        "chronos_score": null,
        "components": "bocpd_only"
    }

Response shape (all agents)::

    {
        "system_health": 0.88,
        "agent_count": 3,
        "agents": [
            {"agent_id": "planner", "health": 0.95, "bocpd_score": 0.02, ...},
            ...
        ]
    }
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chronoagent.scorer.health_scorer import HealthUpdate, TemporalHealthScorer

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health-scores"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AgentHealthResponse(BaseModel):
    """Health score for a single agent.

    Attributes:
        agent_id: Agent identifier.
        health: Score in [0, 1].  1 = fully healthy, 0 = anomalous.
        bocpd_score: Raw BOCPD changepoint probability (``None`` if not yet run).
        chronos_score: Raw Chronos anomaly score (``None`` if unavailable).
        components: Which components contributed (``"ensemble"``,
            ``"bocpd_only"``, ``"chronos_only"``, or ``"none"``).
    """

    agent_id: str
    health: float
    bocpd_score: float | None
    chronos_score: float | None
    components: str


class SystemHealthResponse(BaseModel):
    """Aggregate health snapshot for all tracked agents.

    Attributes:
        system_health: Mean health score across all agents (1.0 if none).
        agent_count: Number of agents currently tracked.
        agents: Individual health scores, sorted by ``health`` ascending
            (most at-risk first).
    """

    system_health: float
    agent_count: int
    agents: list[AgentHealthResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _update_to_response(update: HealthUpdate) -> AgentHealthResponse:
    """Convert a ``HealthUpdate`` dataclass to the API response model."""
    have_b = update.bocpd_score is not None
    have_c = update.chronos_score is not None
    if have_b and have_c:
        components = "ensemble"
    elif have_b:
        components = "bocpd_only"
    elif have_c:
        components = "chronos_only"
    else:
        components = "none"
    return AgentHealthResponse(
        agent_id=update.agent_id,
        health=update.health,
        bocpd_score=update.bocpd_score,
        chronos_score=update.chronos_score,
        components=components,
    )


def _get_scorer(request: Request) -> TemporalHealthScorer:
    """Extract the ``TemporalHealthScorer`` from ``app.state``."""
    scorer: TemporalHealthScorer | None = getattr(request.app.state, "health_scorer", None)
    if scorer is None:
        raise HTTPException(
            status_code=503,
            detail="Health scorer not initialised — application still starting up.",
        )
    return scorer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/agents/{agent_id}/health",
    response_model=AgentHealthResponse,
    summary="Get agent health score",
    description="Returns the current temporal health score for a specific agent.",
)
def get_agent_health(agent_id: str, request: Request) -> AgentHealthResponse:
    """Return the current health score for ``agent_id``.

    Args:
        agent_id: Agent identifier (path parameter).
        request: FastAPI request (used to access ``app.state.health_scorer``).

    Returns:
        :class:`AgentHealthResponse` with score and components.

    Raises:
        HTTPException 404: If no health data exists for ``agent_id`` yet.
        HTTPException 503: If the health scorer is not yet initialised.
    """
    if not agent_id:
        raise HTTPException(status_code=422, detail="agent_id must not be empty")

    scorer = _get_scorer(request)
    update = scorer.get_health(agent_id)
    if update is None:
        raise HTTPException(
            status_code=404,
            detail=f"No health data for agent '{agent_id}'. "
            "The agent may not have emitted any signals yet.",
        )

    logger.debug("health_scores.queried", agent_id=agent_id, health=update.health)
    return _update_to_response(update)


@router.get(
    "/agents/health",
    response_model=SystemHealthResponse,
    summary="Get system health snapshot",
    description="Returns health scores for all tracked agents and a system aggregate.",
)
def get_system_health(request: Request) -> SystemHealthResponse:
    """Return health scores for all agents and the system aggregate.

    Args:
        request: FastAPI request (used to access ``app.state.health_scorer``).

    Returns:
        :class:`SystemHealthResponse` with per-agent and aggregate scores.

    Raises:
        HTTPException 503: If the health scorer is not yet initialised.
    """
    scorer = _get_scorer(request)
    all_updates = scorer.get_all_health()

    agents = sorted(
        [_update_to_response(u) for u in all_updates.values()],
        key=lambda r: r.health,
    )
    system_health = sum(a.health for a in agents) / len(agents) if agents else 1.0

    logger.debug("health_scores.system_queried", agent_count=len(agents))
    return SystemHealthResponse(
        system_health=round(system_health, 4),
        agent_count=len(agents),
        agents=agents,
    )
