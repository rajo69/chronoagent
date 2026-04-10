"""Decentralized task allocator (Phase 5, task 5.3).

Overview
--------
:class:`DecentralizedTaskAllocator` is the thin stateful shell around
the pure :func:`chronoagent.allocator.negotiation.run_contract_net`
function from Phase 5.2.  Its job is to:

1. Subscribe to the ``health_updates`` channel on a
   :class:`~chronoagent.messaging.bus.MessageBus` and cache the latest
   :class:`~chronoagent.scorer.health_scorer.HealthUpdate` for every
   agent that publishes one.
2. Project that cache into a ``{agent_id: health}`` snapshot on demand.
3. Invoke ``run_contract_net`` for a given ``(task_id, task_type)`` pair
   and return the :class:`NegotiationResult`.

Design notes
------------
* **Wrapper, not logic.**  All allocation logic lives in
  ``negotiation.run_contract_net``.  Keeping this class dumb means the
  audit layer (task 5.5) and round-robin fallback (task 5.6) can wrap
  ``allocate`` without re-touching the negotiation internals.
* **Thread-safe snapshot.**  The health cache is guarded by a
  :class:`threading.Lock` so the bus-handler thread and the allocator
  caller thread can interleave safely.  ``allocate`` copies the
  projection under the lock and releases it before calling the
  (pure, fast) negotiation function.
* **Optimistic start-up.**  When no ``HealthUpdate`` has arrived yet,
  the snapshot is empty.  ``run_contract_net`` falls back to
  ``DEFAULT_MISSING_HEALTH = 1.0`` for missing agents, so the system
  can route tasks the moment it boots, before the health scorer has
  produced anything.  Callers that want to be pessimistic at boot can
  override ``missing_health_default`` at construction time.
* **Dict- and dataclass-tolerant payloads.**  The scorer publishes
  ``vars(HealthUpdate)`` (a dict) onto the bus, but tests often publish
  ``HealthUpdate`` instances directly.  ``_handle_health_update``
  accepts both shapes and logs a warning for anything else.  Malformed
  payloads are dropped rather than raising, because the bus handler
  runs on a background thread and an exception would poison the bus.
* **Idempotent stop().**  ``stop`` unsubscribes the handler from the
  bus and is safe to call multiple times.  Downstream code (FastAPI
  lifespan, tests) can rely on this without tracking extra state.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from chronoagent.allocator.capability_weights import (
    AGENT_IDS,
    DEFAULT_CAPABILITY_MATRIX,
    CapabilityMatrix,
)
from chronoagent.allocator.negotiation import (
    DEFAULT_BID_THRESHOLD,
    DEFAULT_MISSING_HEALTH,
    NegotiationResult,
    run_contract_net,
)
from chronoagent.messaging.bus import MessageBus
from chronoagent.scorer.health_scorer import HEALTH_CHANNEL, HealthUpdate

logger = logging.getLogger(__name__)


class DecentralizedTaskAllocator:
    """Health-aware task router wrapping the contract-net protocol.

    Parameters
    ----------
    bus:
        Message bus to subscribe to.  The allocator listens on
        :data:`chronoagent.scorer.health_scorer.HEALTH_CHANNEL` for
        :class:`HealthUpdate` payloads.
    matrix:
        Capability matrix used by the negotiation protocol.  Defaults
        to :data:`DEFAULT_CAPABILITY_MATRIX`.
    threshold:
        Minimum winning bid required to assign a task without
        escalation.  Defaults to :data:`DEFAULT_BID_THRESHOLD`.
    missing_health_default:
        Health assumed for agents that have not yet published a
        :class:`HealthUpdate`.  Defaults to
        :data:`DEFAULT_MISSING_HEALTH` (``1.0`` — optimistic at boot).
    """

    def __init__(
        self,
        bus: MessageBus,
        *,
        matrix: CapabilityMatrix = DEFAULT_CAPABILITY_MATRIX,
        threshold: float = DEFAULT_BID_THRESHOLD,
        missing_health_default: float = DEFAULT_MISSING_HEALTH,
    ) -> None:
        self._bus = bus
        self._matrix = matrix
        self._threshold = threshold
        self._missing_health_default = missing_health_default
        self._health_cache: dict[str, HealthUpdate] = {}
        self._lock = threading.Lock()
        self._stopped = False
        bus.subscribe(HEALTH_CHANNEL, self._handle_health_update)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> CapabilityMatrix:
        """Return the capability matrix used for negotiation."""
        return self._matrix

    @property
    def threshold(self) -> float:
        """Return the minimum-bid threshold used for negotiation."""
        return self._threshold

    @property
    def missing_health_default(self) -> float:
        """Return the fallback health used for agents with no update yet."""
        return self._missing_health_default

    def get_snapshot(self) -> dict[str, HealthUpdate]:
        """Return a copy of the current ``{agent_id: HealthUpdate}`` cache.

        Only agents that have published at least one health update appear
        in the returned dict.  Missing agents are *not* populated with a
        synthetic update; use :meth:`get_health_snapshot` for the
        projected ``{agent_id: float}`` view if you need a default.
        """
        with self._lock:
            return dict(self._health_cache)

    def get_health_snapshot(self) -> dict[str, float]:
        """Return ``{agent_id: health}`` projected from the cache.

        Only cached agents are included; callers that want a total
        mapping should rely on ``run_contract_net`` to apply the
        ``missing_health_default`` fallback for gaps.
        """
        with self._lock:
            return {agent_id: update.health for agent_id, update in self._health_cache.items()}

    def allocate(self, task_id: str, task_type: str) -> NegotiationResult:
        """Run a contract-net round for ``(task_id, task_type)``.

        Projects the current health cache into a ``{agent_id: float}``
        snapshot and delegates to
        :func:`chronoagent.allocator.negotiation.run_contract_net`.
        The lock is released before calling the negotiation function,
        which is pure and therefore safe to run without the lock held.

        Args:
            task_id: Opaque identifier recorded in the result.
            task_type: One of
                :data:`chronoagent.allocator.capability_weights.TASK_TYPES`.

        Returns:
            The :class:`NegotiationResult` from the negotiation round,
            either with an assigned agent or escalated for human review.
        """
        snapshot = self.get_health_snapshot()
        return run_contract_net(
            task_id=task_id,
            task_type=task_type,
            health_snapshot=snapshot,
            matrix=self._matrix,
            threshold=self._threshold,
            missing_health_default=self._missing_health_default,
        )

    def stop(self) -> None:
        """Unsubscribe from the bus.  Idempotent; safe to call twice."""
        if self._stopped:
            return
        self._bus.unsubscribe(HEALTH_CHANNEL, self._handle_health_update)
        self._stopped = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_health_update(self, _channel: str, message: Any) -> None:
        """Cache an inbound health update for later use by ``allocate``.

        Accepts either a :class:`HealthUpdate` instance or a ``dict``
        with the same fields (the scorer publishes
        ``vars(HealthUpdate)``, which is a dict).  Unknown agent IDs
        are still cached: the negotiation protocol ignores extras, so
        stale entries cost nothing to keep, and dropping them would
        hide misconfiguration.
        """
        update = self._coerce_update(message)
        if update is None:
            return
        with self._lock:
            self._health_cache[update.agent_id] = update
        logger.debug(
            "task_allocator: cached health agent=%s health=%.3f",
            update.agent_id,
            update.health,
        )

    @staticmethod
    def _coerce_update(message: Any) -> HealthUpdate | None:
        """Best-effort conversion of ``message`` into a ``HealthUpdate``.

        Returns ``None`` (and logs a warning) for payloads we can't
        parse.  Swallowing is deliberate: the caller is the bus handler
        thread, and raising would tear down the publisher.
        """
        if isinstance(message, HealthUpdate):
            return message
        if isinstance(message, dict):
            try:
                return HealthUpdate(**message)
            except TypeError:
                logger.warning("task_allocator: dict fields do not match HealthUpdate: %r", message)
                return None
        logger.warning("task_allocator: unexpected health payload type %s", type(message).__name__)
        return None


__all__ = [
    "AGENT_IDS",
    "DecentralizedTaskAllocator",
]
