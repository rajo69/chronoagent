"""Health-check routes.

Two endpoints live here:

* ``GET /health`` is the original cheap liveness probe used by Docker /
  Kubernetes readiness checks and the rate-limit exempt list.  It only
  asserts "the ASGI stack is responding" and returns a trivial payload.
* ``GET /api/v1/health`` is the comprehensive per-component report added
  in Phase 9 task 9.4.  It reads ``app.state.component_status``
  (populated by the graceful-degradation helpers in
  :mod:`chronoagent.main`, task 9.3), probes the active LLM backend
  config, and rolls everything up into an overall
  ``healthy``/``degraded``/``unhealthy`` label with HTTP 200 / 200 / 503.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from chronoagent.config import Settings
from chronoagent.observability.components import ComponentMode, ComponentStatus
from chronoagent.observability.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


OverallStatus = Literal["healthy", "degraded", "unhealthy"]


# ── Response models ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Legacy ``/health`` liveness payload."""

    status: str
    version: str


class ComponentReport(BaseModel):
    """Per-component status line in the comprehensive health payload."""

    name: str
    mode: ComponentMode
    detail: str


class HealthReport(BaseModel):
    """Comprehensive ``/api/v1/health`` payload.

    ``status`` rolls up every entry in ``components`` using the rule
    implemented in :func:`_aggregate_status`: any ``unavailable`` makes
    the service ``unhealthy``, any ``fallback`` without ``unavailable``
    makes it ``degraded``, and an all-``primary`` system is ``healthy``.
    """

    status: OverallStatus
    version: str
    components: dict[str, ComponentReport] = Field(default_factory=dict)


# ── Aggregation + probes ─────────────────────────────────────────────────────


def _aggregate_status(statuses: Iterable[ComponentStatus]) -> OverallStatus:
    """Roll up a collection of component statuses into an overall label.

    The precedence is strict: ``unavailable`` dominates ``fallback``
    which dominates ``primary``.  An empty iterable is treated as
    ``healthy`` (nothing has reported a problem).
    """
    worst: OverallStatus = "healthy"
    for status in statuses:
        if status.mode == "unavailable":
            return "unhealthy"
        if status.mode == "fallback":
            worst = "degraded"
    return worst


def _probe_together_ai(settings: Settings) -> ComponentStatus:
    """Report the Together.ai backend status without making a network call.

    The probe is config-only: hitting the real API on every health check
    would burn paid quota and add latency to ops polling.  When Together
    is not the active LLM backend we still report a ``primary`` entry
    labelled as inactive so the component list stays stable across
    environments.  When it *is* active but ``CHRONO_TOGETHER_API_KEY`` is
    empty, the component flips to ``unavailable`` because the first real
    request from any agent will fail.
    """
    if settings.llm_backend != "together":
        return ComponentStatus(
            name="together_ai",
            mode="primary",
            detail=f"inactive (active backend: {settings.llm_backend})",
        )
    if not settings.together_api_key:
        return ComponentStatus(
            name="together_ai",
            mode="unavailable",
            detail="CHRONO_TOGETHER_API_KEY not set",
        )
    return ComponentStatus(
        name="together_ai",
        mode="primary",
        detail=f"configured (model: {settings.together_model})",
    )


def _probe_ollama(settings: Settings) -> ComponentStatus:
    """Report the Ollama backend status (config-only, same reasoning as Together).

    Callers only invoke this when ``settings.llm_backend == "ollama"``;
    the key is just to confirm a base URL is configured so agents have
    somewhere to reach.  A real network probe would add 100-200 ms to
    every health poll and is better left to a dedicated ``?probe=true``
    query parameter in a future iteration.
    """
    if not settings.ollama_base_url:
        return ComponentStatus(
            name="ollama",
            mode="unavailable",
            detail="CHRONO_OLLAMA_BASE_URL not set",
        )
    return ComponentStatus(
        name="ollama",
        mode="primary",
        detail=f"configured ({settings.ollama_base_url}, model: {settings.ollama_model})",
    )


def _build_report(request: Request) -> HealthReport:
    """Assemble a :class:`HealthReport` from ``request.app.state``.

    Reads the component_status dict populated by
    :func:`chronoagent.main.lifespan` (Phase 9 task 9.3) for the
    infrastructure components, then layers on the ``api`` self-entry and
    fresh LLM backend probes.  Ollama is only included when the active
    backend is Ollama so test / prod environments running Mock or
    Together keep the component list honest.
    """
    from chronoagent import __version__

    state_status: dict[str, ComponentStatus] = getattr(request.app.state, "component_status", {})
    settings: Settings = request.app.state.settings

    components: dict[str, ComponentStatus] = {}
    components["api"] = ComponentStatus(
        name="api",
        mode="primary",
        detail="responding",
    )
    # Pull the 9.3-populated infrastructure statuses in a deterministic
    # order.  Missing keys (e.g. lifespan not yet run in a bare
    # ``create_app`` probe) are skipped so the endpoint still returns a
    # well-formed payload instead of KeyError-ing.
    for key in ("bus", "database", "chromadb", "forecaster"):
        status = state_status.get(key)
        if status is not None:
            components[key] = status

    components["together_ai"] = _probe_together_ai(settings)
    if settings.llm_backend == "ollama":
        components["ollama"] = _probe_ollama(settings)

    overall = _aggregate_status(components.values())

    if overall != "healthy":
        logger.warning(
            "health_report_non_healthy",
            status=overall,
            degraded=[k for k, v in components.items() if v.mode == "fallback"],
            unavailable=[k for k, v in components.items() if v.mode == "unavailable"],
        )

    return HealthReport(
        status=overall,
        version=__version__,
        components={
            name: ComponentReport(name=status.name, mode=status.mode, detail=status.detail)
            for name, status in components.items()
        },
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service liveness status.

    Returns:
        :class:`HealthResponse` with ``status="ok"`` when the service is running.
    """
    from chronoagent import __version__

    return HealthResponse(status="ok", version=__version__)


@router.get("/api/v1/health", response_model=HealthReport, tags=["ops"])
async def api_v1_health(request: Request, response: Response) -> HealthReport:
    """Return a per-component health report.

    Rolls up infrastructure statuses written to ``app.state.component_status``
    during lifespan (Phase 9 task 9.3) together with fresh LLM backend
    config probes into an overall ``healthy`` / ``degraded`` / ``unhealthy``
    label.  HTTP status mirrors the overall label: 200 for healthy and
    degraded (the service is still serving traffic, possibly on a
    fallback), 503 for unhealthy (an expected component is unavailable
    and the next real request will fail).

    Args:
        request: The inbound request; ``request.app.state`` carries the
            component-status dict and the :class:`Settings` instance.
        response: Starlette response object used to flip the status code
            to 503 without reconstructing a ``JSONResponse`` by hand.

    Returns:
        The :class:`HealthReport` payload.
    """
    report = _build_report(request)
    if report.status == "unhealthy":
        response.status_code = 503
    return report
