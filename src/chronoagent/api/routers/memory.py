"""Memory integrity and quarantine router (Phase 6 task 6.4).

Exposes three endpoints that wire the
:class:`~chronoagent.memory.integrity.MemoryIntegrityModule` (tasks 6.1/6.2)
and :class:`~chronoagent.memory.quarantine.QuarantineStore` (task 6.3) to HTTP
callers:

``GET /api/v1/memory/integrity``
    Returns the current state of the integrity module (baseline status, retrieval
    statistics, signal weights) plus a summary of the quarantine collection
    (document count and IDs).  Read-only; safe to call repeatedly.

``POST /api/v1/memory/quarantine``
    Accepts a list of document IDs (typically the ``flagged_ids`` field of an
    :class:`~chronoagent.memory.integrity.IntegrityResult` returned by a prior
    integrity check) and an optional human-readable reason, then moves the named
    documents from the active :class:`~chronoagent.memory.store.MemoryStore` into
    the quarantine collection.  Idempotent: IDs already quarantined or absent from
    the active store are silently skipped.

``POST /api/v1/memory/approve``
    Restores previously quarantined documents back to the active store and removes
    them from the quarantine collection.  Idempotent: IDs not currently quarantined
    are silently skipped.

All three endpoints read ``app.state.active_store``, ``app.state.quarantine_store``,
and (for GET) ``app.state.integrity_module``; the lifespan in
:mod:`chronoagent.main` initialises these before the first request arrives.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from chronoagent.memory.integrity import MemoryIntegrityModule
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.memory.store import MemoryStore
from chronoagent.messaging.bus import MessageBus

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["memory"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class IntegrityStatusResponse(BaseModel):
    """Current state of the memory integrity subsystem.

    Attributes:
        baseline_fitted: Whether the IsolationForest baseline has been fitted.
        baseline_size: Number of clean embeddings the baseline was fitted on.
            Zero when no baseline has been fitted.
        pending_refit_count: New clean docs buffered since the last refit;
            resets to zero on each fit or auto-refit.
        total_retrievals: Lifetime count of (doc_id, retrieval) events recorded
            by the integrity module.
        flag_threshold: Aggregate score at which a document is flagged.
        weights: Normalised signal weights, keyed by signal name, summing to 1.0.
        quarantine_count: Number of documents currently held in the quarantine
            collection.
        quarantined_ids: ChromaDB IDs of every document currently quarantined.
    """

    baseline_fitted: bool
    baseline_size: int
    pending_refit_count: int
    total_retrievals: int
    flag_threshold: float
    weights: dict[str, float]
    quarantine_count: int
    quarantined_ids: list[str]


class QuarantineRequest(BaseModel):
    """Request body for POST /api/v1/memory/quarantine.

    Attributes:
        ids: Document IDs to move into quarantine.  Typically the
            ``flagged_ids`` field of an
            :class:`~chronoagent.memory.integrity.IntegrityResult`.
        reason: Optional human-readable reason recorded as metadata on each
            quarantined document.  Forwarded verbatim to
            :meth:`~chronoagent.memory.quarantine.QuarantineStore.quarantine`.
        agent_id: Optional agent identifier.  When provided a
            ``"memory.quarantine"`` event is published on the message bus so
            the :class:`~chronoagent.escalation.escalation_manager.EscalationHandler`
            can auto-escalate on quarantine events.
    """

    ids: list[str]
    reason: str | None = None
    agent_id: str | None = None


class QuarantineResponse(BaseModel):
    """Response body for POST /api/v1/memory/quarantine.

    Attributes:
        quarantined: IDs that were actually moved into quarantine.  A subset of
            the requested IDs; empty when all supplied IDs were already
            quarantined or absent from the active store.
    """

    quarantined: list[str]


class ApproveRequest(BaseModel):
    """Request body for POST /api/v1/memory/approve.

    Attributes:
        ids: Quarantined document IDs to restore to the active store.
    """

    ids: list[str]


class ApproveResponse(BaseModel):
    """Response body for POST /api/v1/memory/approve.

    Attributes:
        approved: IDs that were actually restored to the active store.  Empty
            when none of the supplied IDs were currently quarantined.
    """

    approved: list[str]


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_active_store(request: Request) -> MemoryStore:
    """Inject the shared :class:`~chronoagent.memory.store.MemoryStore`.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`MemoryStore` stored on ``app.state.active_store``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if the store has not been initialised.
    """
    store = getattr(request.app.state, "active_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Active memory store not initialised")
    return store  # type: ignore[no-any-return]


def _get_quarantine_store(request: Request) -> QuarantineStore:
    """Inject the shared :class:`~chronoagent.memory.quarantine.QuarantineStore`.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`QuarantineStore` stored on ``app.state.quarantine_store``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if the store has not been initialised.
    """
    store = getattr(request.app.state, "quarantine_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Quarantine store not initialised")
    return store  # type: ignore[no-any-return]


def _get_bus(request: Request) -> MessageBus | None:
    """Inject the optional shared message bus.

    Returns ``None`` when the bus has not been initialised rather than raising
    so that callers can treat bus publishing as best-effort.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`~chronoagent.messaging.bus.MessageBus` stored on
        ``app.state.bus``, or ``None``.
    """
    return getattr(request.app.state, "bus", None)


def _get_integrity_module(request: Request) -> MemoryIntegrityModule:
    """Inject the shared :class:`~chronoagent.memory.integrity.MemoryIntegrityModule`.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`MemoryIntegrityModule` stored on ``app.state.integrity_module``.

    Raises:
        :class:`~fastapi.HTTPException` 503 if the module has not been initialised.
    """
    module = getattr(request.app.state, "integrity_module", None)
    if module is None:
        raise HTTPException(status_code=503, detail="Integrity module not initialised")
    return module  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/memory/integrity", response_model=IntegrityStatusResponse)
def get_integrity_status(
    quarantine_store: Annotated[QuarantineStore, Depends(_get_quarantine_store)],
    integrity_module: Annotated[MemoryIntegrityModule, Depends(_get_integrity_module)],
) -> IntegrityStatusResponse:
    """Return the current state of the memory integrity subsystem.

    Aggregates read-only properties from the
    :class:`~chronoagent.memory.integrity.MemoryIntegrityModule` (baseline
    status, retrieval statistics, signal weights) and the
    :class:`~chronoagent.memory.quarantine.QuarantineStore` (document count and
    IDs currently held in quarantine).

    Args:
        quarantine_store: Shared quarantine store (injected).
        integrity_module: Shared integrity module (injected).

    Returns:
        :class:`IntegrityStatusResponse` with combined module and quarantine state.
    """
    quarantined_ids = quarantine_store.list_ids()
    logger.debug(
        "memory.integrity.status",
        baseline_fitted=integrity_module.baseline_fitted,
        quarantine_count=len(quarantined_ids),
        total_retrievals=integrity_module.total_retrievals,
    )
    return IntegrityStatusResponse(
        baseline_fitted=integrity_module.baseline_fitted,
        baseline_size=integrity_module.baseline_size,
        pending_refit_count=integrity_module.pending_refit_count,
        total_retrievals=integrity_module.total_retrievals,
        flag_threshold=integrity_module.flag_threshold,
        weights=integrity_module.weights,
        quarantine_count=len(quarantined_ids),
        quarantined_ids=quarantined_ids,
    )


@router.post("/memory/quarantine", response_model=QuarantineResponse)
def quarantine_docs(
    body: QuarantineRequest,
    active_store: Annotated[MemoryStore, Depends(_get_active_store)],
    quarantine_store: Annotated[QuarantineStore, Depends(_get_quarantine_store)],
    bus: Annotated[MessageBus | None, Depends(_get_bus)],
) -> QuarantineResponse:
    """Move documents from the active store into the quarantine collection.

    Designed to accept ``IntegrityResult.flagged_ids`` directly from a
    :meth:`~chronoagent.memory.integrity.MemoryIntegrityModule.check_retrieval`
    call.  Documents moved into quarantine are excluded from future active-store
    retrieval until approved.

    The operation is idempotent: IDs already in quarantine or absent from the
    active store are silently skipped, so callers can replay the same
    ``flagged_ids`` list without side effects.

    When ``body.agent_id`` is provided and ``moved`` is non-empty, a
    ``"memory.quarantine"`` event is published on the message bus so that
    :class:`~chronoagent.escalation.escalation_manager.EscalationHandler` can
    auto-escalate on quarantine events.

    Args:
        body: Request body with the IDs to quarantine, an optional reason, and
            an optional ``agent_id`` for bus publishing.
        active_store: Shared active :class:`~chronoagent.memory.store.MemoryStore`
            (injected).
        quarantine_store: Shared :class:`~chronoagent.memory.quarantine.QuarantineStore`
            (injected).
        bus: Optional shared message bus (injected; ``None`` when not initialised).

    Returns:
        :class:`QuarantineResponse` listing the IDs actually moved into quarantine.
    """
    moved = quarantine_store.quarantine(active_store, body.ids, reason=body.reason)
    logger.info(
        "memory.quarantine",
        requested=len(body.ids),
        quarantined=len(moved),
        reason=body.reason,
        agent_id=body.agent_id,
    )
    if moved and body.agent_id is not None and bus is not None:
        bus.publish(
            "memory.quarantine",
            {"agent_id": body.agent_id, "ids": moved, "reason": body.reason},
        )
    return QuarantineResponse(quarantined=moved)


@router.post("/memory/approve", response_model=ApproveResponse)
def approve_docs(
    body: ApproveRequest,
    active_store: Annotated[MemoryStore, Depends(_get_active_store)],
    quarantine_store: Annotated[QuarantineStore, Depends(_get_quarantine_store)],
) -> ApproveResponse:
    """Restore quarantined documents back to the active store.

    Strips quarantine bookkeeping metadata (``quarantined_at``,
    ``quarantine_reason``) before re-inserting each record, using the original
    embedding so the cosine vector space is unperturbed by the round trip.

    The operation is idempotent: IDs not currently in the quarantine collection
    are silently skipped.

    Args:
        body: Request body with the quarantined document IDs to restore.
        active_store: Shared active :class:`~chronoagent.memory.store.MemoryStore`
            (injected).
        quarantine_store: Shared :class:`~chronoagent.memory.quarantine.QuarantineStore`
            (injected).

    Returns:
        :class:`ApproveResponse` listing the IDs actually restored to the active store.
    """
    restored = quarantine_store.approve(active_store, body.ids)
    logger.info(
        "memory.approve",
        requested=len(body.ids),
        approved=len(restored),
    )
    return ApproveResponse(approved=restored)
