"""Tests for the comprehensive ``/api/v1/health`` endpoint (Phase 9 task 9.4).

The endpoint rolls up ``app.state.component_status`` (populated by the
graceful-degradation helpers in :mod:`chronoagent.main`, task 9.3) with fresh
LLM backend probes into an overall healthy / degraded / unhealthy label.
These tests cover:

* the aggregation precedence rule in :func:`_aggregate_status`
* each LLM backend probe branch (inactive, active-no-key, active-configured)
* the full wired-up endpoint under a real :class:`TestClient`, including the
  monkeypatched-make_engine fallback path from 9.3 that surfaces as
  ``database: fallback`` -> overall ``degraded`` -> HTTP 200
* the 503 contract when a component is ``unavailable``
* regression: the legacy ``/health`` liveness probe still responds
* rate-limit exemption: ``/api/v1/health`` bypasses the rate limiter
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from chronoagent import main as main_module
from chronoagent.api.health import (
    ComponentReport,
    HealthReport,
    _aggregate_status,
    _probe_ollama,
    _probe_together_ai,
)
from chronoagent.api.middleware import RateLimitConfig
from chronoagent.config import Settings
from chronoagent.main import create_app
from chronoagent.observability.components import ComponentStatus

# ── Aggregation rule ─────────────────────────────────────────────────────────


class TestAggregateStatus:
    """``_aggregate_status`` obeys the strict precedence rule."""

    def test_empty_iterable_is_healthy(self) -> None:
        assert _aggregate_status([]) == "healthy"

    def test_all_primary_is_healthy(self) -> None:
        statuses = [
            ComponentStatus(name="a", mode="primary", detail="ok"),
            ComponentStatus(name="b", mode="primary", detail="ok"),
        ]
        assert _aggregate_status(statuses) == "healthy"

    def test_any_fallback_downgrades_to_degraded(self) -> None:
        statuses = [
            ComponentStatus(name="a", mode="primary", detail="ok"),
            ComponentStatus(name="b", mode="fallback", detail="backup"),
            ComponentStatus(name="c", mode="primary", detail="ok"),
        ]
        assert _aggregate_status(statuses) == "degraded"

    def test_any_unavailable_dominates_fallback(self) -> None:
        statuses = [
            ComponentStatus(name="a", mode="fallback", detail="backup"),
            ComponentStatus(name="b", mode="unavailable", detail="gone"),
        ]
        assert _aggregate_status(statuses) == "unhealthy"

    def test_unavailable_short_circuits(self) -> None:
        """The unavailable entry returns immediately without scanning the rest."""
        seen: list[str] = []

        def tracking_iter() -> Any:
            for s in [
                ComponentStatus(name="a", mode="unavailable", detail="gone"),
                ComponentStatus(name="b", mode="primary", detail="ok"),
            ]:
                seen.append(s.name)
                yield s

        result = _aggregate_status(tracking_iter())
        assert result == "unhealthy"
        assert seen == ["a"]


# ── Together.ai probe ────────────────────────────────────────────────────────


class TestProbeTogetherAi:
    """``_probe_together_ai`` is config-only and covers all three branches."""

    def test_inactive_backend_returns_primary(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        status = _probe_together_ai(settings)
        assert status.name == "together_ai"
        assert status.mode == "primary"
        assert "inactive" in status.detail
        assert "mock" in status.detail

    def test_active_with_missing_key_is_unavailable(self) -> None:
        settings = Settings(env="prod", llm_backend="together", together_api_key="")
        status = _probe_together_ai(settings)
        assert status.mode == "unavailable"
        assert "CHRONO_TOGETHER_API_KEY" in status.detail

    def test_active_with_key_is_primary(self) -> None:
        settings = Settings(
            env="prod",
            llm_backend="together",
            together_api_key="sk-fake",
            together_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        status = _probe_together_ai(settings)
        assert status.mode == "primary"
        assert "configured" in status.detail
        assert "Mixtral" in status.detail


# ── Ollama probe ─────────────────────────────────────────────────────────────


class TestProbeOllama:
    """``_probe_ollama`` is config-only and only called when Ollama is active."""

    def test_configured_base_url_is_primary(self) -> None:
        settings = Settings(
            env="dev",
            llm_backend="ollama",
            ollama_base_url="http://localhost:11434",
            ollama_model="phi3:mini",
        )
        status = _probe_ollama(settings)
        assert status.name == "ollama"
        assert status.mode == "primary"
        assert "http://localhost:11434" in status.detail
        assert "phi3:mini" in status.detail

    def test_empty_base_url_is_unavailable(self) -> None:
        settings = Settings(env="dev", llm_backend="ollama", ollama_base_url="")
        status = _probe_ollama(settings)
        assert status.mode == "unavailable"
        assert "CHRONO_OLLAMA_BASE_URL" in status.detail


# ── _build_report unit coverage ──────────────────────────────────────────────


class TestBuildReportHelpers:
    """``_build_report`` composes all the pieces with the right keys."""

    def test_includes_api_entry_always(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            components = resp.json()["components"]
            assert "api" in components
            assert components["api"]["mode"] == "primary"
            assert components["api"]["detail"] == "responding"

    def test_excludes_ollama_when_backend_not_ollama(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert "ollama" not in resp.json()["components"]

    def test_includes_ollama_when_backend_is_ollama(self) -> None:
        settings = Settings(
            env="dev",
            llm_backend="ollama",
            ollama_base_url="http://localhost:11434",
        )
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            components = resp.json()["components"]
            assert "ollama" in components
            assert components["ollama"]["mode"] == "primary"


# ── End-to-end happy / degraded / unhealthy ──────────────────────────────────


@pytest.fixture()
def force_chronos_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``ChronosForecaster`` so :func:`_probe_forecaster` reports primary.

    The `chronos-forecasting` package is not installed in dev/CI, so the
    forecaster legitimately reports ``fallback`` in the default test env.
    Tests that want to assert the *healthy* path force the probe to see an
    available forecaster here.
    """

    class FakeForecaster:
        @property
        def available(self) -> bool:
            return True

    monkeypatch.setattr(main_module, "ChronosForecaster", FakeForecaster)


class TestEndpointHealthy:
    """Fully-primary environment rolls up as healthy with all six components."""

    def test_returns_200(self, force_chronos_available: None) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_status_is_healthy(self, force_chronos_available: None) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            assert payload["status"] == "healthy"

    def test_six_components_present(self, force_chronos_available: None) -> None:
        """Test env should have api + bus + database + chromadb + forecaster + together_ai."""
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            assert set(payload["components"].keys()) == {
                "api",
                "bus",
                "database",
                "chromadb",
                "forecaster",
                "together_ai",
            }

    def test_version_field_populated(self, force_chronos_available: None) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            assert payload["version"]

    def test_schema_round_trips(self, force_chronos_available: None) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            report = HealthReport.model_validate(payload)
            assert isinstance(report.components["api"], ComponentReport)


class TestEndpointDegradedOnForecaster:
    """The default dev/test environment surfaces the BOCPD-only fallback.

    The ``chronos-forecasting`` package is intentionally absent from CI so
    the health endpoint reports ``forecaster: fallback`` and the service
    rolls up as ``degraded`` with HTTP 200.  This test locks in that
    surface area so operators can tell at a glance when they are running
    with the reduced-capability ensemble.
    """

    def test_default_env_is_degraded_with_forecaster_fallback(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["status"] == "degraded"
            assert payload["components"]["forecaster"]["mode"] == "fallback"
            assert "BOCPD-only" in payload["components"]["forecaster"]["detail"]


class TestEndpointDegraded:
    """Monkeypatching ``make_engine`` to raise drives the database fallback."""

    def test_status_is_degraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_make_engine(_settings: Settings) -> Any:
            raise RuntimeError("simulated postgres outage")

        monkeypatch.setattr(main_module, "make_engine", broken_make_engine)

        settings = Settings(
            env="test",
            llm_backend="mock",
            database_url="postgresql://nowhere/db",
        )
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            # Degraded systems still serve 200; only 'unhealthy' triggers 503.
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["status"] == "degraded"
            assert payload["components"]["database"]["mode"] == "fallback"

    def test_degraded_mentions_runtime_error_in_detail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def broken_make_engine(_settings: Settings) -> Any:
            raise RuntimeError("simulated outage")

        monkeypatch.setattr(main_module, "make_engine", broken_make_engine)

        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            assert "RuntimeError" in payload["components"]["database"]["detail"]


class TestEndpointUnhealthy:
    """Together.ai active + missing key makes the service unhealthy (503)."""

    def test_returns_503_when_together_key_missing(self) -> None:
        settings = Settings(
            env="test",
            llm_backend="together",
            together_api_key="",
        )
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 503
            payload = resp.json()
            assert payload["status"] == "unhealthy"
            assert payload["components"]["together_ai"]["mode"] == "unavailable"

    def test_unavailable_dominates_even_with_other_primaries(self) -> None:
        """Other components are primary but the unavailable one wins."""
        settings = Settings(env="test", llm_backend="together", together_api_key="")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            assert payload["components"]["bus"]["mode"] == "primary"
            assert payload["components"]["database"]["mode"] == "primary"
            assert payload["status"] == "unhealthy"


# ── Regression + rate-limit exemption ────────────────────────────────────────


class TestLegacyHealthEndpoint:
    """The original ``/health`` liveness probe must not regress."""

    def test_still_returns_ok_payload(self) -> None:
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["status"] == "ok"
            assert payload["version"]


class TestRateLimitExemption:
    """``/api/v1/health`` is exempt from the rate limiter."""

    def test_default_exempt_list_includes_api_v1_health(self) -> None:
        """Regression on :class:`RateLimitConfig` defaults."""
        config = RateLimitConfig()
        assert "/api/v1/health" in config.exempt_paths

    def test_many_requests_never_429(self) -> None:
        """Far more requests than the GET budget (60/min) must all pass."""
        settings = Settings(env="test", llm_backend="mock")
        app = create_app(settings=settings)
        with TestClient(app) as client:
            for _ in range(120):
                resp = client.get("/api/v1/health")
                assert resp.status_code != 429
