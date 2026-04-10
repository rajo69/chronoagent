"""TemporalHealthScorer — orchestrates BOCPD + Chronos for per-agent health.

Listens for signal updates on the message bus, maintains a rolling signal
buffer per agent, runs the BOCPD + Chronos ensemble, and publishes
``HealthUpdate`` messages back onto the bus.

Channel conventions
-------------------
- Inbound:  ``"signal_updates"``   payload: ``SignalPayload``
- Outbound: ``"health_updates"``   payload: ``HealthUpdate``
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from chronoagent.messaging.bus import MessageBus
from chronoagent.scorer.bocpd import BOCPD
from chronoagent.scorer.chronos_forecaster import ChronosForecaster
from chronoagent.scorer.ensemble import EnsembleResult, EnsembleScorer

logger = logging.getLogger(__name__)

SIGNAL_CHANNEL = "signal_updates"
HEALTH_CHANNEL = "health_updates"


@dataclass(frozen=True)
class SignalPayload:
    """Inbound signal update from the behavioral monitor.

    Attributes
    ----------
    agent_id:
        Unique identifier for the agent.
    task_id:
        Task identifier associated with the signal.
    value:
        Scalar signal value (e.g. aggregate KL divergence).
    """

    agent_id: str
    task_id: str
    value: float


@dataclass(frozen=True)
class HealthUpdate:
    """Outbound health score published to the message bus.

    Attributes
    ----------
    agent_id:
        Agent this score applies to.
    health:
        Score in [0, 1].  1 = fully healthy, 0 = anomalous.
    bocpd_score:
        Raw BOCPD changepoint probability (``None`` if not run).
    chronos_score:
        Raw Chronos anomaly score (``None`` if unavailable).
    """

    agent_id: str
    health: float
    bocpd_score: float | None
    chronos_score: float | None


@dataclass
class _AgentState:
    """Per-agent mutable state maintained by the scorer."""

    bocpd: BOCPD
    buffer: list[float] = field(default_factory=list)


class TemporalHealthScorer:
    """Subscribes to signal updates and publishes health scores.

    Parameters
    ----------
    bus:
        Message bus instance (``LocalBus`` in dev, ``RedisBus`` in prod).
    buffer_size:
        Maximum number of past signal values kept per agent for Chronos.
    hazard_lambda:
        Expected run length passed to each agent's BOCPD instance.
    w_bocpd:
        Ensemble weight for BOCPD scores.
    w_chronos:
        Ensemble weight for Chronos scores.
    """

    def __init__(
        self,
        bus: MessageBus,
        buffer_size: int = 100,
        hazard_lambda: float = 50.0,
        w_bocpd: float = 0.5,
        w_chronos: float = 0.5,
    ) -> None:
        self._bus = bus
        self._buffer_size = buffer_size
        self._ensemble = EnsembleScorer(w_bocpd=w_bocpd, w_chronos=w_chronos)
        self._chronos = ChronosForecaster()
        self._hazard_lambda = hazard_lambda
        self._agents: dict[str, _AgentState] = {}
        self._lock = threading.Lock()
        # Latest health result per agent — readable via get_health().
        self._health_cache: dict[str, HealthUpdate] = {}
        bus.subscribe(SIGNAL_CHANNEL, self._handle_signal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_health(self, agent_id: str) -> HealthUpdate | None:
        """Return the most recent ``HealthUpdate`` for ``agent_id``, or ``None``."""
        return self._health_cache.get(agent_id)

    def get_all_health(self) -> dict[str, HealthUpdate]:
        """Return a snapshot of health scores for all tracked agents."""
        return dict(self._health_cache)

    def stop(self) -> None:
        """Unsubscribe from the signal channel (idempotent)."""
        self._bus.unsubscribe(SIGNAL_CHANNEL, self._handle_signal)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_signal(self, _channel: str, message: Any) -> None:
        """Process an inbound signal payload and publish a health update."""
        try:
            if isinstance(message, dict):
                payload = SignalPayload(**message)
            elif isinstance(message, SignalPayload):
                payload = message
            else:
                logger.warning("health_scorer: unexpected message type %s", type(message))
                return
        except (TypeError, KeyError):
            logger.warning("health_scorer: malformed signal payload: %s", message)
            return

        with self._lock:
            state = self._agents.setdefault(
                payload.agent_id,
                _AgentState(bocpd=BOCPD(hazard_lambda=self._hazard_lambda)),
            )
            state.buffer.append(payload.value)
            if len(state.buffer) > self._buffer_size:
                state.buffer = state.buffer[-self._buffer_size :]

            bocpd_score = state.bocpd.update(payload.value)
            chronos_score = self._chronos.compute_anomaly_score(state.buffer[:-1], payload.value)

        result: EnsembleResult = self._ensemble.score(bocpd_score, chronos_score)
        update = HealthUpdate(
            agent_id=payload.agent_id,
            health=result.health,
            bocpd_score=result.bocpd_score,
            chronos_score=result.chronos_score,
        )
        self._health_cache[payload.agent_id] = update
        self._bus.publish(HEALTH_CHANNEL, vars(update))
        logger.debug(
            "health_scorer: agent=%s health=%.3f bocpd=%.3f",
            payload.agent_id,
            update.health,
            bocpd_score,
        )
