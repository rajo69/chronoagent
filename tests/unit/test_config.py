"""Unit tests for chronoagent.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chronoagent.config import Settings, load_settings, load_yaml_config


class TestSettingsDefaults:
    """Settings loads correct defaults without any env vars set."""

    def test_env_default(self) -> None:
        """Default env is dev."""
        s = Settings()
        assert s.env == "dev"

    def test_llm_backend_default(self) -> None:
        """Default LLM backend is together."""
        s = Settings()
        assert s.llm_backend == "together"

    def test_together_model_default(self) -> None:
        """Default Together model is Mixtral."""
        s = Settings()
        assert "Mixtral" in s.together_model

    def test_redis_url_default(self) -> None:
        """Redis URL defaults to localhost."""
        s = Settings()
        assert "localhost" in s.redis_url

    def test_database_url_default(self) -> None:
        """Database URL defaults to SQLite."""
        s = Settings()
        assert "sqlite" in s.database_url

    def test_forecaster_enabled_default(self) -> None:
        """Forecaster is enabled by default."""
        s = Settings()
        assert s.forecaster_enabled is True

    def test_health_score_window_default(self) -> None:
        """Health score window defaults to 50."""
        s = Settings()
        assert s.health_score_window == 50

    def test_escalation_threshold_default(self) -> None:
        """Escalation threshold defaults to 0.3."""
        s = Settings()
        assert s.escalation_threshold == pytest.approx(0.3)

    def test_escalation_cooldown_default(self) -> None:
        """Escalation cooldown defaults to 300 seconds."""
        s = Settings()
        assert s.escalation_cooldown == 300


class TestSettingsEnvOverride:
    """Environment variables override defaults."""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHRONO_ENV overrides the default."""
        monkeypatch.setenv("CHRONO_ENV", "prod")
        s = Settings()
        assert s.env == "prod"

    def test_backend_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHRONO_LLM_BACKEND overrides the default."""
        monkeypatch.setenv("CHRONO_LLM_BACKEND", "mock")
        s = Settings()
        assert s.llm_backend == "mock"

    def test_together_api_key_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHRONO_TOGETHER_API_KEY is loaded from env."""
        monkeypatch.setenv("CHRONO_TOGETHER_API_KEY", "sk-test-key")
        s = Settings()
        assert s.together_api_key == "sk-test-key"


class TestSettingsValidation:
    """Invalid values raise ValidationError."""

    def test_invalid_env(self) -> None:
        """Unknown env value raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(env="staging")  # type: ignore[arg-type]

    def test_invalid_backend(self) -> None:
        """Unknown backend raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(llm_backend="openai")  # type: ignore[arg-type]

    def test_weight_out_of_range(self) -> None:
        """Ensemble weight > 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(ensemble_weights_bocpd=1.5)

    def test_threshold_out_of_range(self) -> None:
        """Escalation threshold > 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(escalation_threshold=2.0)


class TestLoadYamlConfig:
    """load_yaml_config reads YAML files correctly."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent path returns empty dict."""
        result = load_yaml_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_reads_values(self, tmp_path: Path) -> None:
        """Values are parsed from a valid YAML file."""
        cfg = tmp_path / "test.yaml"
        cfg.write_text("env: test\nhealth_score_window: 99\n")
        result = load_yaml_config(cfg)
        assert result["env"] == "test"
        assert result["health_score_window"] == 99


class TestLoadSettings:
    """load_settings merges YAML overlays with env vars."""

    def test_load_settings_defaults(self) -> None:
        """load_settings returns defaults when no YAML is provided."""
        s = load_settings()
        assert s.env in ("dev", "prod", "test")

    def test_load_settings_with_yaml(self, tmp_path: Path) -> None:
        """YAML overlay values are applied."""
        cfg = tmp_path / "overlay.yaml"
        cfg.write_text("env: test\nhealth_score_window: 25\n")
        s = load_settings(yaml_path=cfg)
        assert s.env == "test"
        assert s.health_score_window == 25
