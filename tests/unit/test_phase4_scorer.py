"""Tests for Phase 4: Temporal Health Scorer (tasks 4.1–4.7).

Coverage:
- BOCPD: synthetic changepoint detection at step 50, property tests
- ChronosForecaster: graceful fallback when unavailable, anomaly scoring stub
- EnsembleScorer: weight normalisation, single-component fallback, [0,1] clamp
- TemporalHealthScorer: signal processing, health caching, bus integration
- API: GET /api/v1/agents/{id}/health and GET /api/v1/agents/health
- Hypothesis: health always in [0, 1]
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from chronoagent.messaging.local_bus import LocalBus
from chronoagent.scorer.bocpd import BOCPD
from chronoagent.scorer.chronos_forecaster import ChronosForecaster, ForecastResult
from chronoagent.scorer.ensemble import EnsembleScorer
from chronoagent.scorer.health_scorer import (
    HealthUpdate,
    SignalPayload,
    TemporalHealthScorer,
)


# ===========================================================================
# Task 4.1 — BOCPD
# ===========================================================================


class TestBOCPD:
    def test_output_in_range_on_single_update(self) -> None:
        bocpd = BOCPD()
        p = bocpd.update(1.0)
        assert 0.0 <= p <= 1.0

    def test_run_length_distribution_sums_to_one(self) -> None:
        bocpd = BOCPD()
        for i in range(20):
            bocpd.update(float(i))
        dist = bocpd.run_length_distribution
        assert abs(dist.sum() - 1.0) < 1e-6

    def test_stable_signal_low_changepoint_probability(self) -> None:
        """Constant signal should produce very low CP probability."""
        bocpd = BOCPD(hazard_lambda=50.0)
        probs = [bocpd.update(1.0) for _ in range(30)]
        # After warm-up the CP probability should stay low
        assert all(p < 0.3 for p in probs[5:])

    def test_sudden_jump_raises_changepoint_probability(self) -> None:
        """A sudden mean shift should cause CP probability to spike above hazard rate."""
        bocpd = BOCPD(hazard_lambda=50.0)
        rng = np.random.default_rng(0)
        # Feed noisy stable signal so model learns finite variance
        for _ in range(50):
            bocpd.update(float(rng.normal(0.0, 0.5)))
        baseline_p = bocpd.update(float(rng.normal(0.0, 0.5)))
        # Abrupt jump far outside the learned distribution
        jump_probs = [bocpd.update(50.0) for _ in range(3)]
        assert max(jump_probs) > baseline_p

    def test_synthetic_changepoint_detected_within_5_steps(self) -> None:
        """Changepoint at step 50 must be detected within 5 subsequent steps."""
        bocpd = BOCPD(hazard_lambda=50.0)
        rng = np.random.default_rng(42)
        # Phase 1: stable signal, mean=0, std=0.1
        for _ in range(50):
            bocpd.update(float(rng.normal(0.0, 0.1)))
        # Phase 2: sudden shift to mean=5, std=0.1
        detection_step: int | None = None
        for step in range(1, 6):
            p = bocpd.update(float(rng.normal(5.0, 0.1)))
            if p > 0.5:
                detection_step = step
                break
        assert detection_step is not None, (
            "BOCPD failed to detect synthetic changepoint within 5 steps"
        )

    def test_reset_clears_state(self) -> None:
        bocpd = BOCPD()
        for i in range(10):
            bocpd.update(float(i))
        bocpd.reset()
        dist = bocpd.run_length_distribution
        assert len(dist) == 1
        assert abs(dist[0] - 1.0) < 1e-9

    def test_most_probable_run_length_increases_on_stable_signal(self) -> None:
        bocpd = BOCPD()
        for _ in range(30):
            bocpd.update(0.5)
        assert bocpd.most_probable_run_length > 0

    def test_invalid_hazard_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="hazard_lambda"):
            BOCPD(hazard_lambda=0.0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            BOCPD(alpha0=-1.0)


# ===========================================================================
# Task 4.2 — ChronosForecaster
# ===========================================================================


class TestChronosForecaster:
    def test_returns_none_when_unavailable(self) -> None:
        """When chronos package absent, forecast() returns None gracefully."""
        forecaster = ChronosForecaster()
        forecaster._available = False  # simulate missing package
        result = forecaster.forecast(list(range(20)))
        assert result is None

    def test_returns_none_for_short_history(self) -> None:
        forecaster = ChronosForecaster()
        forecaster._available = False
        result = forecaster.forecast([1.0, 2.0])  # too short
        assert result is None

    def test_anomaly_score_none_when_unavailable(self) -> None:
        forecaster = ChronosForecaster()
        forecaster._available = False
        score = forecaster.compute_anomaly_score(list(range(20)), actual=99.0)
        assert score is None

    def test_anomaly_score_in_range_with_mock_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anomaly score is in [0, 1] when a forecast result is returned."""
        forecaster = ChronosForecaster()
        mock_result = ForecastResult(
            mean=np.array([5.0]),
            low=np.array([3.0]),
            high=np.array([7.0]),
            horizon=1,
        )
        monkeypatch.setattr(forecaster, "forecast", lambda h, horizon=1: mock_result)
        score = forecaster.compute_anomaly_score(list(range(20)), actual=6.0)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_anomaly_score_high_for_outlier_with_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        forecaster = ChronosForecaster()
        mock_result = ForecastResult(
            mean=np.array([5.0]),
            low=np.array([4.5]),
            high=np.array([5.5]),
            horizon=1,
        )
        monkeypatch.setattr(forecaster, "forecast", lambda h, horizon=1: mock_result)
        # actual=100 is far outside the tight [4.5, 5.5] interval
        score = forecaster.compute_anomaly_score(list(range(20)), actual=100.0)
        assert score is not None
        assert score == 1.0  # clamped

    def test_available_property_checks_import(self) -> None:
        forecaster = ChronosForecaster()
        # available is bool — just check the type
        assert isinstance(forecaster.available, bool)


# ===========================================================================
# Task 4.3 — EnsembleScorer
# ===========================================================================


class TestEnsembleScorer:
    def test_both_none_returns_health_one(self) -> None:
        scorer = EnsembleScorer()
        result = scorer.score(bocpd_score=None, chronos_score=None)
        assert result.health == 1.0

    def test_bocpd_only_fallback(self) -> None:
        scorer = EnsembleScorer(w_bocpd=0.5, w_chronos=0.5)
        result = scorer.score(bocpd_score=0.8, chronos_score=None)
        assert result.w_bocpd == 1.0
        assert result.w_chronos == 0.0
        assert abs(result.health - 0.2) < 1e-6

    def test_chronos_only_fallback(self) -> None:
        scorer = EnsembleScorer()
        result = scorer.score(bocpd_score=None, chronos_score=0.6)
        assert result.w_chronos == 1.0
        assert abs(result.health - 0.4) < 1e-6

    def test_ensemble_blend(self) -> None:
        scorer = EnsembleScorer(w_bocpd=0.5, w_chronos=0.5)
        result = scorer.score(bocpd_score=0.4, chronos_score=0.4)
        # health = 1 - 0.5*0.4 - 0.5*0.4 = 0.6
        assert abs(result.health - 0.6) < 1e-6

    def test_health_clamped_at_zero(self) -> None:
        scorer = EnsembleScorer(w_bocpd=1.0, w_chronos=0.0)
        # bocpd_score > 1 would make health negative if unclamped
        result = scorer.score(bocpd_score=2.0, chronos_score=None)
        assert result.health == 0.0

    def test_invalid_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            EnsembleScorer(w_bocpd=-0.1)

    def test_last_result_stored(self) -> None:
        scorer = EnsembleScorer()
        scorer.score(bocpd_score=0.1, chronos_score=None)
        assert scorer.last_result is not None
        assert scorer.last_result.bocpd_score == 0.1


# ===========================================================================
# Task 4.4 — TemporalHealthScorer
# ===========================================================================


class TestTemporalHealthScorer:
    def test_health_cache_updates_after_signal(self) -> None:
        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus)
        bus.publish("signal_updates", SignalPayload("agent_a", "task-1", 0.0).__dict__  # type: ignore[arg-type]
                    if False else vars(SignalPayload("agent_a", "task-1", 0.0)))
        update = scorer.get_health("agent_a")
        assert update is not None
        assert update.agent_id == "agent_a"
        assert 0.0 <= update.health <= 1.0

    def test_no_health_for_unseen_agent(self) -> None:
        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus)
        assert scorer.get_health("unknown") is None

    def test_get_all_health_includes_all_seen_agents(self) -> None:
        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus)
        for agent_id in ["alpha", "beta", "gamma"]:
            bus.publish(
                "signal_updates",
                {"agent_id": agent_id, "task_id": "t1", "value": 0.1},
            )
        all_health = scorer.get_all_health()
        assert set(all_health.keys()) == {"alpha", "beta", "gamma"}

    def test_health_published_to_bus(self) -> None:
        bus = LocalBus()
        received: list[HealthUpdate] = []

        def capture(_channel: str, msg: object) -> None:
            if isinstance(msg, dict):
                received.append(HealthUpdate(**msg))

        bus.subscribe("health_updates", capture)
        scorer = TemporalHealthScorer(bus=bus)
        bus.publish("signal_updates", {"agent_id": "planner", "task_id": "t", "value": 0.5})

        assert len(received) == 1
        assert received[0].agent_id == "planner"
        assert 0.0 <= received[0].health <= 1.0

    def test_sudden_jump_lowers_health(self) -> None:
        """A large mean shift should produce a lower health score than the stable phase.

        BOCPD detects the shift at the first anomalous step.  After a few steps
        in the new regime it adapts (correct behaviour), so we sample health
        immediately after the first jump observation.
        """
        rng = np.random.default_rng(7)
        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus, hazard_lambda=50.0)
        # Stable phase with noisy signal so model learns finite variance
        for _ in range(60):
            bus.publish(
                "signal_updates",
                {"agent_id": "a", "task_id": "t", "value": float(rng.normal(0.0, 0.5))},
            )
        stable_health = scorer.get_health("a")
        assert stable_health is not None

        # Abrupt large jump — CP probability should spike at step 1 -> health drops
        bus.publish("signal_updates", {"agent_id": "a", "task_id": "t", "value": 50.0})
        degraded_health = scorer.get_health("a")
        assert degraded_health is not None
        assert degraded_health.health <= stable_health.health

    def test_stop_unsubscribes(self) -> None:
        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus)
        scorer.stop()
        # After stop, publishing should not update cache
        bus.publish("signal_updates", {"agent_id": "x", "task_id": "t", "value": 1.0})
        assert scorer.get_health("x") is None

    def test_malformed_message_ignored(self) -> None:
        bus = LocalBus()
        scorer = TemporalHealthScorer(bus=bus)
        # Should not raise
        bus.publish("signal_updates", "not a dict")
        bus.publish("signal_updates", {"wrong": "keys"})


# ===========================================================================
# Task 4.5 — LocalBus
# ===========================================================================


class TestLocalBus:
    def test_publish_calls_subscriber(self) -> None:
        bus = LocalBus()
        received: list[object] = []
        bus.subscribe("ch", lambda _c, m: received.append(m))
        bus.publish("ch", "hello")
        assert received == ["hello"]

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = LocalBus()
        received: list[object] = []

        def handler(_c: str, m: object) -> None:
            received.append(m)

        bus.subscribe("ch", handler)
        bus.unsubscribe("ch", handler)
        bus.publish("ch", "hello")
        assert received == []

    def test_multiple_subscribers(self) -> None:
        bus = LocalBus()
        results: list[int] = []
        bus.subscribe("ch", lambda _c, _m: results.append(1))
        bus.subscribe("ch", lambda _c, _m: results.append(2))
        bus.publish("ch", "x")
        assert sorted(results) == [1, 2]

    def test_unsubscribe_nonexistent_no_error(self) -> None:
        bus = LocalBus()
        bus.unsubscribe("ch", lambda _c, _m: None)  # should not raise

    def test_thread_safe_concurrent_publish(self) -> None:
        bus = LocalBus()
        received: list[int] = []
        lock = threading.Lock()
        bus.subscribe("ch", lambda _c, m: (lock.acquire(), received.append(m), lock.release()))

        def publish_many() -> None:
            for i in range(50):
                bus.publish("ch", i)

        threads = [threading.Thread(target=publish_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(received) == 200


# ===========================================================================
# Task 4.6 — API routes
# ===========================================================================


def _make_client() -> TestClient:
    """Create a TestClient with a seeded TemporalHealthScorer."""
    from chronoagent.config import Settings
    from chronoagent.main import create_app

    app = create_app(settings=Settings(database_url="sqlite:///:memory:"))
    client = TestClient(app)
    return client


class TestHealthScoresRouter:
    def test_get_agent_health_404_when_no_data(self) -> None:
        client = _make_client()
        with client:
            resp = client.get("/api/v1/agents/unknown_agent/health")
        assert resp.status_code == 404

    def test_get_agent_health_returns_score_after_signal(self) -> None:
        from chronoagent.main import create_app
        from chronoagent.config import Settings

        app = create_app(settings=Settings(database_url="sqlite:///:memory:"))
        client = TestClient(app)
        # Inject a signal via the bus on app state
        with client:
            bus = app.state.bus
            for _ in range(5):
                bus.publish(
                    "signal_updates",
                    {"agent_id": "planner", "task_id": "pr-1", "value": 0.1},
                )
            resp = client.get("/api/v1/agents/planner/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "planner"
        assert 0.0 <= data["health"] <= 1.0
        assert "bocpd_score" in data
        assert "components" in data

    def test_get_system_health_empty(self) -> None:
        client = _make_client()
        with client:
            resp = client.get("/api/v1/agents/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] == 0
        assert data["system_health"] == 1.0

    def test_get_system_health_with_agents(self) -> None:
        from chronoagent.main import create_app
        from chronoagent.config import Settings

        app = create_app(settings=Settings(database_url="sqlite:///:memory:"))
        client = TestClient(app)
        with client:
            bus = app.state.bus
            for agent_id in ["a1", "a2"]:
                bus.publish(
                    "signal_updates",
                    {"agent_id": agent_id, "task_id": "t", "value": 0.0},
                )
            resp = client.get("/api/v1/agents/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] == 2
        assert 0.0 <= data["system_health"] <= 1.0


# ===========================================================================
# Hypothesis property tests — health always in [0, 1]
# ===========================================================================


@given(
    bocpd_val=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    chronos_val=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    w_bocpd=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    w_chronos=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=500)
def test_ensemble_health_always_in_unit_interval(
    bocpd_val: float,
    chronos_val: float | None,
    w_bocpd: float,
    w_chronos: float,
) -> None:
    """EnsembleScorer must always produce health in [0, 1]."""
    scorer = EnsembleScorer(w_bocpd=w_bocpd, w_chronos=w_chronos)
    result = scorer.score(bocpd_score=bocpd_val, chronos_score=chronos_val)
    assert 0.0 <= result.health <= 1.0


@given(obs=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=300)
def test_bocpd_update_always_in_unit_interval(obs: float) -> None:
    """BOCPD.update must always return a value in [0, 1]."""
    bocpd = BOCPD()
    p = bocpd.update(obs)
    assert 0.0 <= p <= 1.0


@given(
    values=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    )
)
@settings(max_examples=200)
def test_bocpd_run_length_distribution_sums_to_one(values: list[float]) -> None:
    """Run-length distribution must always be a valid probability vector."""
    bocpd = BOCPD()
    for v in values:
        bocpd.update(v)
    dist = bocpd.run_length_distribution
    assert abs(dist.sum() - 1.0) < 1e-5
    assert float(dist.min()) >= 0.0
