"""Unit tests for FastAPI app factory, health endpoint, and logging configuration."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from chronoagent.config import Settings
from chronoagent.main import create_app
from chronoagent.observability.logging import configure_logging, get_logger


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def test_settings() -> Settings:
    """Return a Settings instance configured for the test environment."""
    return Settings(env="test", llm_backend="mock")


@pytest.fixture()
def client(test_settings: Settings) -> TestClient:
    """Return a synchronous TestClient for the app under test."""
    app = create_app(settings=test_settings)
    return TestClient(app)


# ── Health endpoint ────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """GET /health returns expected payload."""

    def test_status_ok(self, client: TestClient) -> None:
        """Health endpoint returns HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_field(self, client: TestClient) -> None:
        """Response body contains status='ok'."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_version_field(self, client: TestClient) -> None:
        """Response body contains a non-empty version string."""
        response = client.get("/health")
        assert response.json()["version"] != ""

    def test_content_type_json(self, client: TestClient) -> None:
        """Response content-type is application/json."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ── App factory ───────────────────────────────────────────────────────────────


class TestCreateApp:
    """create_app builds a valid FastAPI application."""

    def test_returns_fastapi_app(self, test_settings: Settings) -> None:
        """create_app returns a FastAPI instance."""
        from fastapi import FastAPI

        app = create_app(settings=test_settings)
        assert isinstance(app, FastAPI)

    def test_settings_stored_on_state(self, test_settings: Settings) -> None:
        """Settings are accessible via app.state.settings."""
        app = create_app(settings=test_settings)
        assert app.state.settings is test_settings

    def test_default_settings_when_none(self) -> None:
        """create_app works without explicit settings."""
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)


# ── Logging ───────────────────────────────────────────────────────────────────


class TestConfigureLogging:
    """configure_logging sets up structlog correctly."""

    def test_dev_mode_does_not_raise(self) -> None:
        """configure_logging('dev') completes without error."""
        configure_logging("dev")

    def test_prod_mode_does_not_raise(self) -> None:
        """configure_logging('prod') completes without error."""
        configure_logging("prod")

    def test_test_mode_does_not_raise(self) -> None:
        """configure_logging('test') completes without error."""
        configure_logging("test")

    def test_root_logger_has_handler(self) -> None:
        """After configuration the root logger has at least one handler."""
        configure_logging("dev")
        assert len(logging.getLogger().handlers) >= 1


class TestGetLogger:
    """get_logger returns a bound structlog logger."""

    def test_returns_logger(self) -> None:
        """get_logger returns a non-None logger."""
        configure_logging("dev")
        logger = get_logger(__name__)
        assert logger is not None

    def test_logger_has_info_method(self) -> None:
        """Returned logger exposes an info method."""
        configure_logging("dev")
        logger = get_logger(__name__)
        assert callable(getattr(logger, "info", None))
