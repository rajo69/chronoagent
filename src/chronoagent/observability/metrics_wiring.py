"""Bus-to-metrics wiring (Phase 8 task 8.3).

The :class:`~chronoagent.observability.metrics.ChronoAgentMetrics` sink
is intentionally passive.  It exposes update methods but does **not**
subscribe to the message bus on its own, because:

1. It keeps the sink unit-testable without a running bus.
2. It lets a production deployment choose which channels to wire up (a
   Prometheus setup that ships only allocation metrics, for example,
   can skip the health-updates subscriber entirely).
3. It keeps the lifespan in :mod:`chronoagent.main` explicit about what
   side effects the metrics sink produces.

This module bridges the gap: :func:`subscribe_metrics_to_bus` attaches
closure handlers for every bus channel ChronoAgent publishes, and
:func:`unsubscribe_metrics_from_bus` tears them down on shutdown.  The
closures parse the same dict payloads the downstream consumers do, so
the metrics sink stays decoupled from the publisher types.

The subscribers are returned as a :class:`MetricsSubscribers` record so
the teardown call does not need to re-derive the exact callable
identities that were passed to :meth:`MessageBus.subscribe`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chronoagent.messaging.bus import MessageBus, MessageHandler
from chronoagent.observability.logging import get_logger
from chronoagent.observability.metrics import ChronoAgentMetrics
from chronoagent.scorer.health_scorer import (
    HEALTH_CHANNEL,
    HealthUpdate,
)

logger = get_logger(__name__)

ESCALATION_CHANNEL = "escalations"
QUARANTINE_CHANNEL = "memory.quarantine"


@dataclass(frozen=True)
class MetricsSubscribers:
    """Handles returned by :func:`subscribe_metrics_to_bus`.

    Each attribute is the exact closure that was registered on the bus,
    so :func:`unsubscribe_metrics_from_bus` can pass them back to
    :meth:`MessageBus.unsubscribe` without recomputing identity.

    Attributes:
        on_health_update: Handler bound to the ``health_updates`` channel.
        on_escalation: Handler bound to the ``escalations`` channel.
        on_quarantine: Handler bound to the ``memory.quarantine`` channel.
    """

    on_health_update: MessageHandler
    on_escalation: MessageHandler
    on_quarantine: MessageHandler


def _make_health_handler(metrics: ChronoAgentMetrics) -> MessageHandler:
    """Build a bus handler that forwards health updates to the metrics sink."""

    def handler(_channel: str, message: Any) -> None:
        try:
            if isinstance(message, HealthUpdate):
                update = message
            elif isinstance(message, dict):
                update = HealthUpdate(**message)
            else:
                logger.warning(
                    "metrics_wiring.health.unexpected_type",
                    msg_type=type(message).__name__,
                )
                return
        except (TypeError, KeyError) as exc:
            logger.warning("metrics_wiring.health.malformed", error=str(exc))
            return
        metrics.observe_health_update(update)

    return handler


def _make_escalation_handler(metrics: ChronoAgentMetrics) -> MessageHandler:
    """Build a bus handler that increments the escalations counter."""

    def handler(_channel: str, message: Any) -> None:
        if not isinstance(message, dict):
            logger.warning(
                "metrics_wiring.escalation.unexpected_type",
                msg_type=type(message).__name__,
            )
            return
        trigger = str(message.get("trigger", "unknown"))
        metrics.observe_escalation(trigger)

    return handler


def _make_quarantine_handler(metrics: ChronoAgentMetrics) -> MessageHandler:
    """Build a bus handler that counts quarantine events and moved docs."""

    def handler(_channel: str, message: Any) -> None:
        if not isinstance(message, dict):
            logger.warning(
                "metrics_wiring.quarantine.unexpected_type",
                msg_type=type(message).__name__,
            )
            return
        ids_raw = message.get("ids", [])
        try:
            doc_count = len(ids_raw)
        except TypeError:
            doc_count = 0
        metrics.observe_quarantine_event(doc_count)

    return handler


def subscribe_metrics_to_bus(bus: MessageBus, metrics: ChronoAgentMetrics) -> MetricsSubscribers:
    """Attach metrics closures to every ChronoAgent bus channel.

    Subscribes handlers on three channels:

    * ``health_updates`` (from the temporal health scorer): updates
      per-agent gauges and the update counter.
    * ``escalations`` (from the escalation handler): increments the
      escalation counter labelled by trigger.
    * ``memory.quarantine`` (from the memory integrity pipeline):
      increments the quarantine-event and quarantined-doc counters.

    Args:
        bus: The shared :class:`~chronoagent.messaging.bus.MessageBus`
            instance.
        metrics: The :class:`ChronoAgentMetrics` sink that the closures
            should update.

    Returns:
        :class:`MetricsSubscribers` record carrying the exact handlers
        that were registered.  Pass this to
        :func:`unsubscribe_metrics_from_bus` at shutdown.
    """
    on_health = _make_health_handler(metrics)
    on_escalation = _make_escalation_handler(metrics)
    on_quarantine = _make_quarantine_handler(metrics)

    bus.subscribe(HEALTH_CHANNEL, on_health)
    bus.subscribe(ESCALATION_CHANNEL, on_escalation)
    bus.subscribe(QUARANTINE_CHANNEL, on_quarantine)

    return MetricsSubscribers(
        on_health_update=on_health,
        on_escalation=on_escalation,
        on_quarantine=on_quarantine,
    )


def unsubscribe_metrics_from_bus(bus: MessageBus, subscribers: MetricsSubscribers) -> None:
    """Detach the metrics closures previously returned by :func:`subscribe_metrics_to_bus`.

    Safe to call once; calling it twice on the same ``subscribers``
    record delegates to :meth:`MessageBus.unsubscribe`, which is a no-op
    for unknown handlers on ``LocalBus``.

    Args:
        bus: The message bus the closures were attached to.
        subscribers: The record returned by
            :func:`subscribe_metrics_to_bus`.
    """
    bus.unsubscribe(HEALTH_CHANNEL, subscribers.on_health_update)
    bus.unsubscribe(ESCALATION_CHANNEL, subscribers.on_escalation)
    bus.unsubscribe(QUARANTINE_CHANNEL, subscribers.on_quarantine)


__all__ = [
    "ESCALATION_CHANNEL",
    "QUARANTINE_CHANNEL",
    "MetricsSubscribers",
    "subscribe_metrics_to_bus",
    "unsubscribe_metrics_from_bus",
]
