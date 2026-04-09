"""Health-check router."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service liveness status.

    Returns:
        :class:`HealthResponse` with ``status="ok"`` when the service is running.
    """
    from chronoagent import __version__

    return HealthResponse(status="ok", version=__version__)
