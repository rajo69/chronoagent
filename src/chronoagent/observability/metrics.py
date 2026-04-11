"""Prometheus metrics for ChronoAgent subsystems (Phase 8 task 8.3).

Overview
--------
:class:`ChronoAgentMetrics` bundles gauges, counters, and histograms for
every subsystem worth surfacing to a production-grade monitoring stack:

* **Agent health** (gauges + counter): per-agent health, BOCPD and
  Chronos components, plus a system-wide mean and an update counter.
* **Task allocation** (counter + histogram): allocation outcomes
  labelled by task type and ``assigned``/``escalated``/``fallback``
  plus the winning-bid score distribution.
* **Memory integrity** (gauges + counter + histogram): baseline size,
  pending refit count, quarantine size, integrity-check flag rate, and
  aggregate-score distribution.
* **Escalation queue** (gauge + counter): pending count and total
  events labelled by trigger.
* **Reviews** (counter + histogram): total reviews by overall risk and
  pipeline-duration distribution.

All metrics live in an isolated :class:`prometheus_client.CollectorRegistry`
instance attached to the class.  This avoids clashes with
``prometheus_client``'s global default registry (useful in tests, where
re-instantiating the class must not raise ``Duplicated timeseries``) and
keeps the scraped ``/metrics`` output limited to ChronoAgent signals.

The update methods are passive: they only mutate metric state.
Subscribing them to the message bus or wiring them into hot code paths
is the caller's job (see :mod:`chronoagent.main` for the default
wiring).

Channel convention
------------------
The module does **not** own bus subscriptions.  It exposes plain Python
methods that a caller can bind to bus events, call from a FastAPI
endpoint, or invoke from a test.  This keeps ``metrics.py`` a pure
observability sink with zero cross-module dependencies on messaging.

Content-type
------------
:data:`CONTENT_TYPE_LATEST` re-exports the canonical Prometheus
exposition-format content type so the ``/metrics`` router can set the
HTTP header without importing from ``prometheus_client`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CONTENT_TYPE_LATEST as _PROM_CONTENT_TYPE
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from chronoagent.allocator.negotiation import NegotiationResult
    from chronoagent.memory.integrity import IntegrityResult
    from chronoagent.scorer.health_scorer import HealthUpdate

__all__ = [
    "CONTENT_TYPE_LATEST",
    "ChronoAgentMetrics",
]

#: Canonical Prometheus exposition-format content type.  Re-exported so
#: ``api/routers/metrics.py`` does not need to import ``prometheus_client``.
CONTENT_TYPE_LATEST: str = _PROM_CONTENT_TYPE

#: Histogram buckets for normalised ``[0, 1]`` scores (bid, aggregate).
_SCORE_BUCKETS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)

#: Histogram buckets for latency in seconds, spanning interactive to slow.
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)


class ChronoAgentMetrics:
    """Prometheus metrics sink for ChronoAgent subsystems.

    All metrics are registered against an isolated
    :class:`~prometheus_client.CollectorRegistry` stored on
    :attr:`registry`.  The isolation matters for three reasons:

    1. Tests can instantiate many collectors without hitting
       ``Duplicated timeseries in CollectorRegistry`` errors from the
       global registry.
    2. The ``/metrics`` endpoint scrapes only ChronoAgent metrics,
       keeping the exposition output lean.
    3. Embedding ChronoAgent inside a larger process that already uses
       the default registry does not cause metric-name collisions.

    Parameters
    ----------
    registry:
        Optional :class:`~prometheus_client.CollectorRegistry` to attach
        metrics to.  When ``None`` (the default) a fresh isolated
        registry is created.  Pass an explicit registry when you need
        the metrics to join a pre-existing collection (rare).
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry: CollectorRegistry = registry if registry is not None else CollectorRegistry()

        # ------------------------------------------------------------------
        # Gauges: current-state snapshots
        # ------------------------------------------------------------------
        self.agent_health: Gauge = Gauge(
            "chronoagent_agent_health",
            "Current health score per agent in [0, 1]. 1.0 = fully healthy.",
            labelnames=("agent_id",),
            registry=self.registry,
        )
        self.agent_bocpd_score: Gauge = Gauge(
            "chronoagent_agent_bocpd_score",
            "Raw BOCPD changepoint probability per agent.",
            labelnames=("agent_id",),
            registry=self.registry,
        )
        self.agent_chronos_score: Gauge = Gauge(
            "chronoagent_agent_chronos_score",
            "Raw Chronos anomaly score per agent.",
            labelnames=("agent_id",),
            registry=self.registry,
        )
        self.system_health: Gauge = Gauge(
            "chronoagent_system_health",
            "Mean health score across all tracked agents in [0, 1].",
            registry=self.registry,
        )
        self.pending_escalations: Gauge = Gauge(
            "chronoagent_escalation_queue_pending",
            "Number of escalation records with status='pending'.",
            registry=self.registry,
        )
        self.quarantine_size: Gauge = Gauge(
            "chronoagent_memory_quarantine_size",
            "Number of documents currently in the quarantine collection.",
            registry=self.registry,
        )
        self.memory_baseline_size: Gauge = Gauge(
            "chronoagent_memory_baseline_size",
            "Number of embeddings in the memory integrity IsolationForest baseline.",
            registry=self.registry,
        )
        self.memory_pending_refit: Gauge = Gauge(
            "chronoagent_memory_pending_refit",
            "New clean embeddings buffered since the last baseline refit.",
            registry=self.registry,
        )

        # ------------------------------------------------------------------
        # Counters: monotonic event totals
        # ------------------------------------------------------------------
        self.health_updates_total: Counter = Counter(
            "chronoagent_health_updates",
            "Total health updates published per agent.",
            labelnames=("agent_id",),
            registry=self.registry,
        )
        self.allocations_total: Counter = Counter(
            "chronoagent_allocations",
            "Total task allocation decisions, labelled by task type and outcome.",
            labelnames=("task_type", "outcome"),
            registry=self.registry,
        )
        self.escalations_total: Counter = Counter(
            "chronoagent_escalations",
            "Total escalation events labelled by trigger.",
            labelnames=("trigger",),
            registry=self.registry,
        )
        self.quarantine_events_total: Counter = Counter(
            "chronoagent_quarantine_events",
            "Total quarantine events (groups of documents moved to quarantine).",
            registry=self.registry,
        )
        self.quarantined_docs_total: Counter = Counter(
            "chronoagent_quarantined_docs",
            "Total documents ever moved to quarantine (summed across events).",
            registry=self.registry,
        )
        self.reviews_total: Counter = Counter(
            "chronoagent_reviews",
            "Total reviews completed, labelled by overall risk.",
            labelnames=("risk",),
            registry=self.registry,
        )
        self.integrity_checks_total: Counter = Counter(
            "chronoagent_integrity_checks",
            "Total memory integrity checks, labelled by flagged outcome.",
            labelnames=("flagged",),
            registry=self.registry,
        )

        # ------------------------------------------------------------------
        # Histograms: distributions
        # ------------------------------------------------------------------
        self.allocation_bid_score: Histogram = Histogram(
            "chronoagent_allocation_bid_score",
            "Distribution of winning allocation bid scores, per task type.",
            labelnames=("task_type",),
            buckets=_SCORE_BUCKETS,
            registry=self.registry,
        )
        self.review_duration_seconds: Histogram = Histogram(
            "chronoagent_review_duration_seconds",
            "Distribution of end-to-end review pipeline durations in seconds.",
            labelnames=("risk",),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.integrity_aggregate_score: Histogram = Histogram(
            "chronoagent_integrity_aggregate_score",
            "Distribution of memory integrity max-aggregate scores across checks.",
            buckets=_SCORE_BUCKETS,
            registry=self.registry,
        )

    # ---------------------------------------------------------------------
    # Update methods -- health scorer
    # ---------------------------------------------------------------------

    def observe_health_update(self, update: HealthUpdate) -> None:
        """Record a :class:`HealthUpdate` from the temporal health scorer.

        Sets the per-agent health gauge and both component gauges (when
        present), and increments the per-agent update counter.

        Args:
            update: The :class:`~chronoagent.scorer.health_scorer.HealthUpdate`
                published on the ``health_updates`` bus channel.
        """
        self.agent_health.labels(agent_id=update.agent_id).set(update.health)
        if update.bocpd_score is not None:
            self.agent_bocpd_score.labels(agent_id=update.agent_id).set(update.bocpd_score)
        if update.chronos_score is not None:
            self.agent_chronos_score.labels(agent_id=update.agent_id).set(update.chronos_score)
        self.health_updates_total.labels(agent_id=update.agent_id).inc()

    def set_system_health(self, value: float) -> None:
        """Set the system-wide mean health gauge.

        Args:
            value: Mean health across tracked agents, in ``[0, 1]``.
        """
        self.system_health.set(value)

    # ---------------------------------------------------------------------
    # Update methods -- escalation queue
    # ---------------------------------------------------------------------

    def set_pending_escalations(self, count: int) -> None:
        """Set the pending-escalations gauge.

        Args:
            count: Number of ``EscalationRecord`` rows with
                ``status == "pending"``.
        """
        self.pending_escalations.set(count)

    def observe_escalation(self, trigger: str) -> None:
        """Increment the escalation counter for ``trigger``.

        Args:
            trigger: Escalation trigger string.  Canonical values are
                ``"low_health"`` and ``"quarantine_event"`` but any
                string is accepted for forward compatibility.
        """
        self.escalations_total.labels(trigger=trigger).inc()

    # ---------------------------------------------------------------------
    # Update methods -- memory integrity + quarantine
    # ---------------------------------------------------------------------

    def set_quarantine_size(self, count: int) -> None:
        """Set the quarantine-size gauge.

        Args:
            count: Current document count in the quarantine collection.
        """
        self.quarantine_size.set(count)

    def set_memory_baseline_size(self, count: int) -> None:
        """Set the memory-integrity baseline-size gauge.

        Args:
            count: Number of embeddings in the IsolationForest baseline.
        """
        self.memory_baseline_size.set(count)

    def set_memory_pending_refit(self, count: int) -> None:
        """Set the gauge for clean embeddings buffered for the next refit.

        Args:
            count: Value of
                :attr:`~chronoagent.memory.integrity.MemoryIntegrityModule.pending_refit_count`.
        """
        self.memory_pending_refit.set(count)

    def observe_quarantine_event(self, doc_count: int) -> None:
        """Record a quarantine event that moved ``doc_count`` documents.

        Increments the event counter by one and the document counter by
        ``doc_count``.

        Args:
            doc_count: Number of documents moved to quarantine in this
                event.  Must be non-negative.
        """
        self.quarantine_events_total.inc()
        if doc_count > 0:
            self.quarantined_docs_total.inc(doc_count)

    def observe_integrity_check(self, result: IntegrityResult) -> None:
        """Record a memory integrity check result.

        Increments the ``integrity_checks_total`` counter labelled by
        whether any doc was flagged, and observes the ``max_aggregate``
        score in the aggregate histogram.

        Args:
            result: The
                :class:`~chronoagent.memory.integrity.IntegrityResult`
                returned by
                :meth:`~chronoagent.memory.integrity.MemoryIntegrityModule.check_retrieval`.
        """
        flagged_label = "false" if result.is_clean else "true"
        self.integrity_checks_total.labels(flagged=flagged_label).inc()
        self.integrity_aggregate_score.observe(result.max_aggregate)

    # ---------------------------------------------------------------------
    # Update methods -- task allocator
    # ---------------------------------------------------------------------

    def observe_allocation(self, result: NegotiationResult) -> None:
        """Record a task allocation decision.

        Increments ``allocations_total`` labelled by task type and one
        of ``"assigned"`` (winner picked a valid bid), ``"escalated"``
        (no bid cleared the threshold), or ``"fallback"`` (negotiation
        raised and the round-robin fallback picked an agent).  When a
        winning bid is present its score is observed in
        ``allocation_bid_score``.

        Args:
            result: The :class:`~chronoagent.allocator.negotiation.NegotiationResult`
                returned by
                :meth:`~chronoagent.allocator.task_allocator.DecentralizedTaskAllocator.allocate`.
        """
        if result.escalated:
            outcome = "escalated"
        elif result.winning_bid is None:
            outcome = "fallback"
        else:
            outcome = "assigned"
        self.allocations_total.labels(task_type=result.task_type, outcome=outcome).inc()
        if result.winning_bid is not None:
            self.allocation_bid_score.labels(task_type=result.task_type).observe(
                result.winning_bid.score
            )

    # ---------------------------------------------------------------------
    # Update methods -- review pipeline
    # ---------------------------------------------------------------------

    def observe_review(self, risk: str, duration_seconds: float) -> None:
        """Record a completed review.

        Increments ``reviews_total`` labelled by ``risk`` and observes
        the pipeline duration in ``review_duration_seconds``.

        Args:
            risk: Overall risk label from the
                :class:`~chronoagent.agents.summarizer.ReviewReport`
                (``"none"``, ``"low"``, ``"medium"``, ``"high"``, or
                ``"critical"``).
            duration_seconds: Wall-clock pipeline duration in seconds.
                Must be non-negative.
        """
        self.reviews_total.labels(risk=risk).inc()
        self.review_duration_seconds.labels(risk=risk).observe(duration_seconds)

    # ---------------------------------------------------------------------
    # Exposition
    # ---------------------------------------------------------------------

    def render(self) -> bytes:
        """Return the Prometheus exposition-format payload.

        Returns:
            Bytes suitable for a ``text/plain; version=0.0.4; charset=utf-8``
            HTTP response body.  The encoding is handled by
            :func:`prometheus_client.generate_latest`.
        """
        payload: bytes = generate_latest(self.registry)
        return payload
