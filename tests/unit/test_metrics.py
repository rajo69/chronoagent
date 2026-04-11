"""Unit tests for Phase 8 task 8.3 Prometheus metrics.

Covers:
- ``ChronoAgentMetrics`` sink: registry isolation, health update
  observations, system health gauge, allocation outcome routing,
  escalation counter, quarantine observations, integrity check
  observations, review observations, exposition output.
- ``metrics_wiring``: bus subscribers fan signals into the sink.
- ``GET /metrics`` router: poll-driven gauge refresh, content type,
  503 when sink missing, integration with the full app lifespan.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Generator
from typing import Any

import chromadb
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.allocator.negotiation import Bid, NegotiationResult
from chronoagent.config import Settings
from chronoagent.db.models import Base, EscalationRecord
from chronoagent.memory.integrity import DocSignal, IntegrityResult, MemoryIntegrityModule
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.observability.metrics import (
    CONTENT_TYPE_LATEST,
    ChronoAgentMetrics,
)
from chronoagent.observability.metrics_wiring import (
    ESCALATION_CHANNEL,
    QUARANTINE_CHANNEL,
    MetricsSubscribers,
    subscribe_metrics_to_bus,
    unsubscribe_metrics_from_bus,
)
from chronoagent.scorer.health_scorer import HEALTH_CHANNEL, HealthUpdate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_value(
    metrics: ChronoAgentMetrics, name: str, labels: dict[str, str] | None = None
) -> float | None:
    """Parse the rendered exposition output and return a sample value.

    Args:
        metrics: The metrics sink to render.
        name: Sample name (e.g., ``chronoagent_agent_health``; for
            counters use the ``_total`` suffix that prometheus_client
            appends automatically).
        labels: Optional label dict to match.

    Returns:
        The matching sample's value, or ``None`` if no sample matches.
    """
    text = metrics.render().decode("utf-8")
    labels = labels or {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return None


# ---------------------------------------------------------------------------
# ChronoAgentMetrics sink
# ---------------------------------------------------------------------------


class TestChronoAgentMetricsIsolation:
    def test_default_registry_is_isolated(self) -> None:
        """Constructing two sinks must not raise a duplicate-metric error."""
        m1 = ChronoAgentMetrics()
        m2 = ChronoAgentMetrics()
        assert m1.registry is not m2.registry

    def test_render_returns_exposition_bytes(self) -> None:
        metrics = ChronoAgentMetrics()
        payload = metrics.render()
        assert isinstance(payload, bytes)
        text = payload.decode("utf-8")
        assert "chronoagent_agent_health" in text
        assert "# TYPE chronoagent_system_health gauge" in text

    def test_content_type_is_prometheus_text(self) -> None:
        assert CONTENT_TYPE_LATEST.startswith("text/plain")


class TestObserveHealthUpdate:
    def test_full_update_sets_all_gauges_and_counter(self) -> None:
        metrics = ChronoAgentMetrics()
        update = HealthUpdate(
            agent_id="planner",
            health=0.72,
            bocpd_score=0.11,
            chronos_score=0.34,
        )
        metrics.observe_health_update(update)

        assert _sample_value(metrics, "chronoagent_agent_health", {"agent_id": "planner"}) == 0.72
        assert (
            _sample_value(metrics, "chronoagent_agent_bocpd_score", {"agent_id": "planner"}) == 0.11
        )
        assert (
            _sample_value(metrics, "chronoagent_agent_chronos_score", {"agent_id": "planner"})
            == 0.34
        )
        assert (
            _sample_value(metrics, "chronoagent_health_updates_total", {"agent_id": "planner"})
            == 1.0
        )

    def test_missing_components_are_skipped(self) -> None:
        metrics = ChronoAgentMetrics()
        update = HealthUpdate(
            agent_id="summarizer",
            health=0.5,
            bocpd_score=None,
            chronos_score=None,
        )
        metrics.observe_health_update(update)

        assert _sample_value(metrics, "chronoagent_agent_health", {"agent_id": "summarizer"}) == 0.5
        # Component gauges were never set, so no samples for summarizer exist.
        assert (
            _sample_value(metrics, "chronoagent_agent_bocpd_score", {"agent_id": "summarizer"})
            is None
        )
        assert (
            _sample_value(metrics, "chronoagent_agent_chronos_score", {"agent_id": "summarizer"})
            is None
        )

    def test_counter_monotonic_across_updates(self) -> None:
        metrics = ChronoAgentMetrics()
        for i in range(5):
            metrics.observe_health_update(
                HealthUpdate(
                    agent_id="planner", health=0.5 + i * 0.01, bocpd_score=None, chronos_score=None
                )
            )
        assert (
            _sample_value(metrics, "chronoagent_health_updates_total", {"agent_id": "planner"})
            == 5.0
        )
        # Final gauge reflects only the last value.
        assert _sample_value(
            metrics, "chronoagent_agent_health", {"agent_id": "planner"}
        ) == pytest.approx(0.54)


class TestSystemAndQueueGauges:
    def test_set_system_health(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.set_system_health(0.91)
        assert _sample_value(metrics, "chronoagent_system_health") == pytest.approx(0.91)

    def test_set_pending_escalations(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.set_pending_escalations(7)
        assert _sample_value(metrics, "chronoagent_escalation_queue_pending") == 7.0

    def test_set_quarantine_size(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.set_quarantine_size(3)
        assert _sample_value(metrics, "chronoagent_memory_quarantine_size") == 3.0

    def test_memory_baseline_and_pending_refit(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.set_memory_baseline_size(42)
        metrics.set_memory_pending_refit(5)
        assert _sample_value(metrics, "chronoagent_memory_baseline_size") == 42.0
        assert _sample_value(metrics, "chronoagent_memory_pending_refit") == 5.0


class TestObserveAllocation:
    @staticmethod
    def _result(
        *,
        escalated: bool = False,
        winning_bid: Bid | None = None,
        task_type: str = "security_review",
    ) -> NegotiationResult:
        return NegotiationResult(
            task_id="pr1::security_review",
            task_type=task_type,
            assigned_agent=None if escalated else "security_reviewer",
            escalated=escalated,
            winning_bid=winning_bid,
            all_bids=(),
            rationale="",
            threshold=0.5,
        )

    def test_assigned_outcome_records_winning_bid(self) -> None:
        metrics = ChronoAgentMetrics()
        bid = Bid(agent_id="security_reviewer", capability=1.0, health=0.95, score=0.95)
        metrics.observe_allocation(self._result(winning_bid=bid))

        assert (
            _sample_value(
                metrics,
                "chronoagent_allocations_total",
                {"task_type": "security_review", "outcome": "assigned"},
            )
            == 1.0
        )
        assert (
            _sample_value(
                metrics,
                "chronoagent_allocation_bid_score_count",
                {"task_type": "security_review"},
            )
            == 1.0
        )
        assert _sample_value(
            metrics,
            "chronoagent_allocation_bid_score_sum",
            {"task_type": "security_review"},
        ) == pytest.approx(0.95)

    def test_escalated_outcome_no_bid_observation(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_allocation(self._result(escalated=True))
        assert (
            _sample_value(
                metrics,
                "chronoagent_allocations_total",
                {"task_type": "security_review", "outcome": "escalated"},
            )
            == 1.0
        )
        # Histogram count stays at 0 because no winning bid was observed.
        assert (
            _sample_value(
                metrics,
                "chronoagent_allocation_bid_score_count",
                {"task_type": "security_review"},
            )
            is None
        )

    def test_fallback_outcome_when_winning_bid_is_none_and_not_escalated(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_allocation(self._result(winning_bid=None, escalated=False))
        assert (
            _sample_value(
                metrics,
                "chronoagent_allocations_total",
                {"task_type": "security_review", "outcome": "fallback"},
            )
            == 1.0
        )


class TestObserveEscalationAndQuarantine:
    def test_escalation_counter_by_trigger(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_escalation("low_health")
        metrics.observe_escalation("low_health")
        metrics.observe_escalation("quarantine_event")
        assert (
            _sample_value(metrics, "chronoagent_escalations_total", {"trigger": "low_health"})
            == 2.0
        )
        assert (
            _sample_value(metrics, "chronoagent_escalations_total", {"trigger": "quarantine_event"})
            == 1.0
        )

    def test_quarantine_event_increments_both_counters(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_quarantine_event(3)
        metrics.observe_quarantine_event(2)
        assert _sample_value(metrics, "chronoagent_quarantine_events_total") == 2.0
        assert _sample_value(metrics, "chronoagent_quarantined_docs_total") == 5.0

    def test_zero_doc_event_still_counts(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_quarantine_event(0)
        assert _sample_value(metrics, "chronoagent_quarantine_events_total") == 1.0
        # No doc bump, so the docs counter stays at its 0 default.
        assert _sample_value(metrics, "chronoagent_quarantined_docs_total") == 0.0


class TestObserveIntegrityCheck:
    @staticmethod
    def _result(*, flagged: bool, max_aggregate: float) -> IntegrityResult:
        signal = DocSignal(
            doc_id="d1",
            embedding_outlier=0.0,
            freshness_anomaly=0.0,
            retrieval_frequency=0.0,
            content_embedding_mismatch=0.0,
            aggregate=max_aggregate,
            flagged=flagged,
        )
        return IntegrityResult(
            query="q",
            signals=[signal],
            flagged_ids=["d1"] if flagged else [],
            max_aggregate=max_aggregate,
            timestamp=0.0,
        )

    def test_clean_result_records_false_label(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_integrity_check(self._result(flagged=False, max_aggregate=0.2))
        assert (
            _sample_value(metrics, "chronoagent_integrity_checks_total", {"flagged": "false"})
            == 1.0
        )
        assert _sample_value(metrics, "chronoagent_integrity_aggregate_score_count") == 1.0
        assert _sample_value(metrics, "chronoagent_integrity_aggregate_score_sum") == pytest.approx(
            0.2
        )

    def test_flagged_result_records_true_label(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_integrity_check(self._result(flagged=True, max_aggregate=0.9))
        assert (
            _sample_value(metrics, "chronoagent_integrity_checks_total", {"flagged": "true"}) == 1.0
        )


class TestObserveReview:
    def test_review_counter_and_histogram(self) -> None:
        metrics = ChronoAgentMetrics()
        metrics.observe_review("medium", 0.75)
        metrics.observe_review("medium", 1.25)
        metrics.observe_review("high", 2.5)

        assert _sample_value(metrics, "chronoagent_reviews_total", {"risk": "medium"}) == 2.0
        assert _sample_value(metrics, "chronoagent_reviews_total", {"risk": "high"}) == 1.0
        assert (
            _sample_value(metrics, "chronoagent_review_duration_seconds_count", {"risk": "medium"})
            == 2.0
        )
        assert _sample_value(
            metrics, "chronoagent_review_duration_seconds_sum", {"risk": "medium"}
        ) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Bus wiring
# ---------------------------------------------------------------------------


class TestMetricsWiring:
    def test_subscribe_fans_bus_events_into_sink(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            bus.publish(
                HEALTH_CHANNEL,
                {
                    "agent_id": "style_reviewer",
                    "health": 0.4,
                    "bocpd_score": 0.1,
                    "chronos_score": 0.2,
                },
            )
            bus.publish(ESCALATION_CHANNEL, {"trigger": "low_health", "agent_id": "style_reviewer"})
            bus.publish(QUARANTINE_CHANNEL, {"ids": ["d1", "d2", "d3"], "agent_id": "memory"})

            assert (
                _sample_value(metrics, "chronoagent_agent_health", {"agent_id": "style_reviewer"})
                == 0.4
            )
            assert (
                _sample_value(metrics, "chronoagent_escalations_total", {"trigger": "low_health"})
                == 1.0
            )
            assert _sample_value(metrics, "chronoagent_quarantine_events_total") == 1.0
            assert _sample_value(metrics, "chronoagent_quarantined_docs_total") == 3.0
        finally:
            unsubscribe_metrics_from_bus(bus, subs)

    def test_unsubscribe_stops_updates(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        unsubscribe_metrics_from_bus(bus, subs)

        bus.publish(
            HEALTH_CHANNEL,
            {"agent_id": "planner", "health": 0.9, "bocpd_score": None, "chronos_score": None},
        )
        # Counter stays at its default (no sample emitted) because the
        # handler was removed before the publish.
        assert (
            _sample_value(metrics, "chronoagent_health_updates_total", {"agent_id": "planner"})
            is None
        )

    def test_malformed_health_payload_is_dropped(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            bus.publish(HEALTH_CHANNEL, {"agent_id": "planner"})  # missing required fields
            bus.publish(HEALTH_CHANNEL, "not a dict")
            # Nothing raised, nothing recorded.
            assert (
                _sample_value(metrics, "chronoagent_health_updates_total", {"agent_id": "planner"})
                is None
            )
        finally:
            unsubscribe_metrics_from_bus(bus, subs)

    def test_health_handler_accepts_dataclass_payload(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            bus.publish(
                HEALTH_CHANNEL,
                HealthUpdate(agent_id="planner", health=0.88, bocpd_score=None, chronos_score=None),
            )
            assert (
                _sample_value(metrics, "chronoagent_agent_health", {"agent_id": "planner"}) == 0.88
            )
        finally:
            unsubscribe_metrics_from_bus(bus, subs)

    def test_escalation_non_dict_payload_is_dropped(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            bus.publish(ESCALATION_CHANNEL, "not a dict")
            assert (
                _sample_value(metrics, "chronoagent_escalations_total", {"trigger": "unknown"})
                is None
            )
        finally:
            unsubscribe_metrics_from_bus(bus, subs)

    def test_quarantine_non_dict_payload_is_dropped(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            bus.publish(QUARANTINE_CHANNEL, ["d1", "d2"])
            # Counter was never incremented, so it stays at its 0.0 default.
            assert _sample_value(metrics, "chronoagent_quarantine_events_total") == 0.0
        finally:
            unsubscribe_metrics_from_bus(bus, subs)

    def test_quarantine_ids_without_len_falls_back_to_zero(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            bus.publish(QUARANTINE_CHANNEL, {"ids": 5})  # int has no len
            assert _sample_value(metrics, "chronoagent_quarantine_events_total") == 1.0
            assert _sample_value(metrics, "chronoagent_quarantined_docs_total") == 0.0
        finally:
            unsubscribe_metrics_from_bus(bus, subs)

    def test_subscribers_record_carries_exact_callables(self) -> None:
        bus = LocalBus()
        metrics = ChronoAgentMetrics()
        subs = subscribe_metrics_to_bus(bus, metrics)
        try:
            assert isinstance(subs, MetricsSubscribers)
            assert callable(subs.on_health_update)
            assert callable(subs.on_escalation)
            assert callable(subs.on_quarantine)
        finally:
            unsubscribe_metrics_from_bus(bus, subs)


# ---------------------------------------------------------------------------
# /metrics router integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """TestClient with the lifespan-installed metrics sink and a StaticPool DB.

    The default lifespan creates an in-memory SQLite engine with the
    default (non-shared) pool, which gives every new session a fresh
    empty database.  For tests that need to see rows across connections
    we override ``app.state.session_factory`` with a StaticPool-backed
    engine, mirroring ``test_dashboard_router.client_app``.
    """
    from chronoagent.main import create_app

    settings = Settings(database_url="sqlite:///:memory:", env="test", llm_backend="mock")
    app = create_app(settings=settings)

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    with TestClient(app) as tc:
        app.state.session_factory = factory
        yield tc


class TestMetricsRoute:
    def test_get_metrics_returns_exposition(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "chronoagent_agent_health" in body
        assert "chronoagent_system_health" in body
        assert "chronoagent_escalation_queue_pending" in body

    def test_metrics_reflects_pending_escalation(self, client: TestClient) -> None:
        """After inserting a pending escalation, the scrape should see 1."""
        app = client.app
        eid = uuid.uuid4().hex
        with app.state.session_factory() as session:  # type: ignore[attr-defined]
            session.add(
                EscalationRecord(
                    id=eid,
                    agent_id="planner",
                    trigger="low_health",
                    status="pending",
                    context={"health_score": 0.1},
                    resolution_notes=None,
                    created_at=datetime.datetime.now(datetime.UTC),
                    resolved_at=None,
                )
            )
            session.commit()

        body = client.get("/metrics").text
        assert "chronoagent_escalation_queue_pending 1.0" in body

    def test_metrics_reflects_bus_driven_health_updates(self, client: TestClient) -> None:
        """Publishing on the bus must surface in the next scrape."""
        app = client.app
        bus = app.state.bus  # type: ignore[attr-defined]
        bus.publish(
            HEALTH_CHANNEL,
            {"agent_id": "planner", "health": 0.42, "bocpd_score": 0.2, "chronos_score": 0.3},
        )

        body = client.get("/metrics").text
        assert 'chronoagent_agent_health{agent_id="planner"} 0.42' in body
        # System health gauge is refreshed on scrape from the scorer cache.
        assert "chronoagent_system_health" in body

    def test_metrics_503_when_sink_missing(self) -> None:
        """A bare app without the lifespan-installed sink returns 503."""
        from fastapi import FastAPI

        from chronoagent.api.routers.metrics import router as metrics_router

        app = FastAPI()
        app.include_router(metrics_router)
        with TestClient(app) as tc:
            resp = tc.get("/metrics")
        assert resp.status_code == 503

    def test_metrics_refresh_tolerates_missing_state(self) -> None:
        """Partial state must not crash the scrape; missing deps are skipped."""
        from fastapi import FastAPI

        from chronoagent.api.routers.metrics import router as metrics_router

        app = FastAPI()
        app.include_router(metrics_router)
        app.state.metrics = ChronoAgentMetrics()
        # Intentionally leave scorer/quarantine/integrity/session_factory unset.
        with TestClient(app) as tc:
            resp = tc.get("/metrics")
        assert resp.status_code == 200
        assert "chronoagent_agent_health" in resp.text

    def test_metrics_refresh_populates_gauges_from_state(self) -> None:
        """Happy-path refresh: scorer + quarantine + integrity + DB all present."""
        from fastapi import FastAPI

        from chronoagent.api.routers.metrics import router as metrics_router
        from chronoagent.scorer.health_scorer import TemporalHealthScorer

        suffix = uuid.uuid4().hex
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        factory: sessionmaker[Session] = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus)
        chroma = chromadb.EphemeralClient()
        qstore = QuarantineStore(chroma.get_or_create_collection(f"q_{suffix}"))
        integrity = MemoryIntegrityModule(MockBackend())

        app = FastAPI()
        app.include_router(metrics_router)
        app.state.metrics = ChronoAgentMetrics()
        app.state.health_scorer = scorer
        app.state.quarantine_store = qstore
        app.state.integrity_module = integrity
        app.state.session_factory = factory

        # Pre-seed a pending escalation row.
        with factory() as session:
            session.add(
                EscalationRecord(
                    id=uuid.uuid4().hex,
                    agent_id="planner",
                    trigger="low_health",
                    status="pending",
                    context={"health_score": 0.1},
                    resolution_notes=None,
                    created_at=datetime.datetime.now(datetime.UTC),
                    resolved_at=None,
                )
            )
            session.commit()

        with TestClient(app) as tc:
            body = tc.get("/metrics").text

        # Empty scorer cache -> system_health defaults to 1.0; pending
        # escalations is 1, memory baseline/pending_refit/quarantine are 0.
        assert "chronoagent_system_health 1.0" in body
        assert "chronoagent_escalation_queue_pending 1.0" in body
        assert "chronoagent_memory_quarantine_size 0.0" in body
        assert "chronoagent_memory_baseline_size 0.0" in body
        assert "chronoagent_memory_pending_refit 0.0" in body

        scorer.stop()

    def test_system_health_defaults_to_one_with_no_agents(self, client: TestClient) -> None:
        """An empty scorer must leave system health at 1.0 (healthy default)."""
        body = client.get("/metrics").text
        assert "chronoagent_system_health 1.0" in body


# ---------------------------------------------------------------------------
# Type signature smoke tests
# ---------------------------------------------------------------------------


class TestModuleApiSurface:
    def test_exports(self) -> None:
        import chronoagent.observability.metrics as mod

        assert hasattr(mod, "ChronoAgentMetrics")
        assert hasattr(mod, "CONTENT_TYPE_LATEST")

    def test_wiring_exports(self) -> None:
        import chronoagent.observability.metrics_wiring as wiring

        assert wiring.HEALTH_CHANNEL == "health_updates"
        assert wiring.ESCALATION_CHANNEL == "escalations"
        assert wiring.QUARANTINE_CHANNEL == "memory.quarantine"

    def test_render_is_idempotent(self) -> None:
        metrics = ChronoAgentMetrics()
        a: bytes = metrics.render()
        b: bytes = metrics.render()
        assert a == b

    def test_unused_labels_observe_is_safe(self) -> None:
        """Calling observe_allocation with a novel task_type creates a fresh series."""
        metrics = ChronoAgentMetrics()
        bid = Bid(agent_id="security_reviewer", capability=1.0, health=0.8, score=0.8)
        result: NegotiationResult = NegotiationResult(
            task_id="x",
            task_type="security_review",
            assigned_agent="security_reviewer",
            escalated=False,
            winning_bid=bid,
            all_bids=(),
            rationale="",
            threshold=0.5,
        )
        metrics.observe_allocation(result)
        metrics.observe_allocation(result)
        assert (
            _sample_value(
                metrics,
                "chronoagent_allocations_total",
                {"task_type": "security_review", "outcome": "assigned"},
            )
            == 2.0
        )


# Explicit import silencer for mypy's --strict unused-import check on Any.
_ = Any
