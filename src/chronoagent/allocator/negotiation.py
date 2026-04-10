"""Contract-net negotiation protocol for decentralized task allocation.

Phase 5, task 5.2 of the project plan.

Overview
--------
This module implements the contract-net variant used by the ChronoAgent
task allocator.  When a task needs to be assigned, the allocator
"broadcasts" it to all registered agents; each agent submits a bid
computed as:

    bid(agent, task) = capability(agent, task) * health(agent)

The agent with the **highest** bid wins the task.  Ties are broken
deterministically by the order of
:data:`chronoagent.allocator.capability_weights.AGENT_IDS`.  If no
agent's bid clears the configured :data:`DEFAULT_BID_THRESHOLD`, the
task is escalated to a human rather than silently handed to a degraded
agent.

Design notes
------------
* **Pure function.**  :func:`run_contract_net` depends only on its
  arguments: it performs no I/O, no logging, and no bus calls.  Phase
  5 task 5.3 (:mod:`chronoagent.allocator.task_allocator`) is the piece
  that subscribes to ``health_updates`` and feeds the snapshot in.
  Keeping the negotiation logic pure makes it trivial to test and to
  reuse inside audit / replay tooling.
* **Deterministic.**  The winning agent is a pure function of the
  inputs; tie-breaking uses the canonical ``AGENT_IDS`` order so
  identical snapshots always produce identical allocations.  This is
  important for reproducible experiments and audit logs.
* **Total over AGENT_IDS.**  The function iterates over every agent in
  :data:`AGENT_IDS` in order, even if the ``health_snapshot`` mapping
  omits some of them.  Missing entries fall back to
  ``missing_health_default`` (default ``1.0``, i.e. optimistic
  "healthy until proven otherwise") so the system can allocate work
  before any ``HealthUpdate`` has been observed.  Extra keys in the
  snapshot (e.g. stale or typo'd agent IDs) are ignored, never
  assigned.
* **Escalation is not an error.**  "All bids below threshold" returns
  a :class:`NegotiationResult` with ``escalated=True`` and
  ``assigned_agent=None``.  Callers should treat this as a normal
  outcome and forward it to the escalation layer (Phase 7).
* **Off-diagonal weights > 0.**  The default capability matrix keeps
  every off-diagonal entry strictly inside ``(0, 1)``, so a non-
  specialist always has a chance to cover for a degraded peer.  The
  only way to fail to assign is for **every** bid to fall below
  ``threshold`` simultaneously, which is the Phase 5 exit criterion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from chronoagent.allocator.capability_weights import (
    AGENT_IDS,
    DEFAULT_CAPABILITY_MATRIX,
    CapabilityMatrix,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum winning bid required to assign a task without escalation.
#: With a healthy specialist (capability=1.0, health=1.0) the bid is
#: 1.0, so the task is always assigned on the happy path.  Values below
#: this threshold mean the best candidate has effective capacity below
#: 25%, which is a strong signal that the task should go to a human
#: rather than a compromised or exhausted agent.
DEFAULT_BID_THRESHOLD: Final[float] = 0.25

#: Default fallback when an agent has no ``HealthUpdate`` in the
#: snapshot yet.  Optimistic by design: at boot, specialists should be
#: trusted until the scorer has evidence otherwise.  The task allocator
#: (Phase 5.3) can override this per-call.
DEFAULT_MISSING_HEALTH: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidHealthError(ValueError):
    """Raised when a health value is NaN or outside ``[0, 1]``."""

    def __init__(self, agent_id: str, health: float) -> None:
        super().__init__(
            f"Invalid health for agent_id={agent_id!r}: {health!r}. "
            "Expected a finite float in [0, 1]."
        )
        self.agent_id = agent_id
        self.health = health


class InvalidThresholdError(ValueError):
    """Raised when a negotiation threshold is NaN or outside ``[0, 1]``."""

    def __init__(self, threshold: float) -> None:
        super().__init__(
            f"Invalid bid threshold: {threshold!r}. Expected a finite float in [0, 1]."
        )
        self.threshold = threshold


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bid:
    """A single agent's bid in a contract-net round.

    Attributes
    ----------
    agent_id:
        The bidding agent's canonical ID.  Always one of
        :data:`AGENT_IDS`.
    capability:
        Static proficiency in ``[0, 1]`` drawn from the capability
        matrix for the (agent, task) pair.
    health:
        Live health score in ``[0, 1]`` drawn from the snapshot.
    score:
        The effective bid, ``capability * health``.  Kept as a stored
        field (not a property) so serialising the bid for audit logs
        is a straight dataclass dump.
    """

    agent_id: str
    capability: float
    health: float
    score: float


@dataclass(frozen=True)
class NegotiationResult:
    """Outcome of one contract-net round.

    Attributes
    ----------
    task_id:
        Opaque identifier for the task under negotiation.
    task_type:
        One of :data:`chronoagent.allocator.capability_weights.TASK_TYPES`.
    assigned_agent:
        The winning agent's ID, or ``None`` if the task was escalated.
    escalated:
        True when the maximum bid fell strictly below ``threshold`` and
        the task should be handed off to a human.
    winning_bid:
        The highest-scoring :class:`Bid`, or ``None`` if escalated.
    all_bids:
        Every bid in the round, ordered by :data:`AGENT_IDS`.  Included
        so Phase 5.5's audit record has the full rationale.
    rationale:
        Human-readable explanation of the decision, safe to log.
    threshold:
        The minimum-bid threshold used for this decision.  Recorded so
        audit replays can reproduce the outcome even if the default
        changes later.
    """

    task_id: str
    task_type: str
    assigned_agent: str | None
    escalated: bool
    winning_bid: Bid | None
    all_bids: tuple[Bid, ...]
    rationale: str
    threshold: float


# ---------------------------------------------------------------------------
# Core protocol
# ---------------------------------------------------------------------------


def _coerce_health(agent_id: str, raw: float, *, default: float) -> float:
    """Validate and return a usable health score for *agent_id*.

    Applied to both explicit snapshot entries and the fallback default.
    Rejects NaN and out-of-range values rather than clamping silently;
    a bad health value is a bug in the scorer, not something the
    allocator should paper over.
    """
    value = float(raw) if raw is not None else default
    if math.isnan(value) or not (0.0 <= value <= 1.0):
        raise InvalidHealthError(agent_id, value)
    return value


def run_contract_net(
    task_id: str,
    task_type: str,
    health_snapshot: Mapping[str, float],
    *,
    matrix: CapabilityMatrix = DEFAULT_CAPABILITY_MATRIX,
    threshold: float = DEFAULT_BID_THRESHOLD,
    missing_health_default: float = DEFAULT_MISSING_HEALTH,
) -> NegotiationResult:
    """Run a single contract-net round and return the allocation decision.

    Each agent in :data:`AGENT_IDS` bids
    ``capability(agent, task_type) * health(agent)``.  The highest bid
    wins.  Ties are broken by the canonical ``AGENT_IDS`` order (the
    agent appearing first in ``AGENT_IDS`` wins).  If the maximum bid
    is strictly below ``threshold`` the task is escalated.

    Args:
        task_id: Opaque identifier recorded in the returned result.
        task_type: One of
            :data:`chronoagent.allocator.capability_weights.TASK_TYPES`.
            Raises :class:`UnknownTaskTypeError` (via
            :meth:`CapabilityMatrix.column`) if not registered.
        health_snapshot: Mapping ``{agent_id: health}`` where health is
            a finite float in ``[0, 1]``.  Agents missing from the
            mapping fall back to ``missing_health_default``.  Extra
            keys for unknown agent IDs are ignored.
        matrix: Capability matrix to use.  Defaults to
            :data:`DEFAULT_CAPABILITY_MATRIX`.
        threshold: Minimum winning bid required to assign the task.
            Must be a finite float in ``[0, 1]``.
        missing_health_default: Fallback health used when an agent is
            missing from the snapshot.  Must be a finite float in
            ``[0, 1]``.

    Returns:
        A :class:`NegotiationResult` describing either the winning
        agent or the escalation decision, plus the full bid ledger.

    Raises:
        InvalidThresholdError: If ``threshold`` or
            ``missing_health_default`` is NaN / out of ``[0, 1]``.
        InvalidHealthError: If any health value in the snapshot is
            NaN / out of ``[0, 1]``.
        UnknownTaskTypeError: If ``task_type`` is not registered in
            ``matrix``.
    """
    _validate_threshold(threshold)
    # Validate the default up-front; easier to diagnose than a lazy
    # InvalidHealthError on the first missing agent.
    if math.isnan(missing_health_default) or not (0.0 <= missing_health_default <= 1.0):
        raise InvalidHealthError("<missing_health_default>", missing_health_default)

    # Raises UnknownTaskTypeError if the task type is unknown.  We only
    # need one column since every agent bids on the same task type.
    column = matrix.column(task_type)

    bids: list[Bid] = []
    for agent_id in AGENT_IDS:
        if agent_id in health_snapshot:
            health = _coerce_health(
                agent_id,
                health_snapshot[agent_id],
                default=missing_health_default,
            )
        else:
            health = missing_health_default
        capability = column[agent_id]
        bids.append(
            Bid(
                agent_id=agent_id,
                capability=capability,
                health=health,
                score=capability * health,
            )
        )

    all_bids = tuple(bids)

    # Tie-breaking: iterate in AGENT_IDS order and only replace on
    # strict improvement.  The first agent to reach a given maximum
    # keeps the win, which matches CapabilityMatrix.primary_agent.
    best = all_bids[0]
    for bid in all_bids[1:]:
        if bid.score > best.score:
            best = bid

    if best.score < threshold:
        rationale = (
            f"escalated: max bid {best.score:.4f} by {best.agent_id!r} "
            f"is below threshold {threshold:.4f}"
        )
        return NegotiationResult(
            task_id=task_id,
            task_type=task_type,
            assigned_agent=None,
            escalated=True,
            winning_bid=None,
            all_bids=all_bids,
            rationale=rationale,
            threshold=threshold,
        )

    rationale = (
        f"assigned to {best.agent_id!r}: bid={best.score:.4f} "
        f"(capability={best.capability:.4f} * health={best.health:.4f}) "
        f">= threshold {threshold:.4f}"
    )
    return NegotiationResult(
        task_id=task_id,
        task_type=task_type,
        assigned_agent=best.agent_id,
        escalated=False,
        winning_bid=best,
        all_bids=all_bids,
        rationale=rationale,
        threshold=threshold,
    )


def _validate_threshold(threshold: float) -> None:
    """Reject NaN / out-of-range thresholds early.

    Split out so the error message points at the threshold rather than
    the per-agent health validator.
    """
    if math.isnan(threshold) or not (0.0 <= threshold <= 1.0):
        raise InvalidThresholdError(threshold)


__all__ = [
    "DEFAULT_BID_THRESHOLD",
    "DEFAULT_MISSING_HEALTH",
    "Bid",
    "InvalidHealthError",
    "InvalidThresholdError",
    "NegotiationResult",
    "run_contract_net",
]
