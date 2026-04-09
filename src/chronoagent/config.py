"""Application configuration loaded from environment variables and YAML overlays.

Hierarchy (lowest → highest priority): defaults → base.yaml → env-specific yaml → env vars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ChronoAgent runtime configuration.

    All fields can be overridden via ``CHRONO_*`` environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="CHRONO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    env: Literal["dev", "prod", "test"] = "dev"

    # ── LLM backend ───────────────────────────────────────────────────────────
    llm_backend: Literal["together", "mock", "ollama"] = "together"
    together_api_key: str = Field(default="", repr=False)
    together_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # ── Infrastructure ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./chronoagent.db"
    chroma_persist_dir: str = "./chroma_data"

    # ── Forecaster ────────────────────────────────────────────────────────────
    forecaster_enabled: bool = True
    forecaster_horizon: int = 10
    bocpd_hazard_rate: float = 0.01
    ensemble_weights_bocpd: float = 0.4
    ensemble_weights_chronos: float = 0.6

    # ── Health scoring ────────────────────────────────────────────────────────
    health_score_window: int = 50

    # ── Escalation ────────────────────────────────────────────────────────────
    escalation_threshold: float = 0.3
    escalation_cooldown: int = 300  # seconds

    @field_validator("ensemble_weights_bocpd", "ensemble_weights_chronos")
    @classmethod
    def _weight_in_range(cls, v: float) -> float:
        """Ensure ensemble weights are in [0, 1]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Ensemble weight must be in [0, 1], got {v}")
        return v

    @field_validator("escalation_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        """Ensure escalation threshold is in [0, 1]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Escalation threshold must be in [0, 1], got {v}")
        return v


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a flat dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of config values (may be nested).
    """
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return dict(data)


def _flatten(nested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into ``CHRONO_``-prefixed keys.

    Args:
        nested: Possibly nested configuration dictionary.
        prefix: Key prefix accumulated during recursion.

    Returns:
        Flat dictionary with ``CHRONO_``-prefixed keys suitable for
        :class:`Settings` initialisation.
    """
    out: dict[str, Any] = {}
    for k, v in nested.items():
        full_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{full_key}_"))
        else:
            out[f"CHRONO_{full_key.upper()}"] = v
    return out


def load_settings(yaml_path: Path | None = None) -> Settings:
    """Build a :class:`Settings` instance with optional YAML overlay.

    Merge order: defaults < base.yaml < ``yaml_path`` < environment variables.

    Args:
        yaml_path: Optional path to a YAML config overlay (e.g. ``dev.yaml``).

    Returns:
        Fully resolved :class:`Settings` instance.
    """
    overrides: dict[str, Any] = {}

    base_yaml = Path("configs/base.yaml")
    if base_yaml.exists():
        overrides.update(_flatten(load_yaml_config(base_yaml)))

    if yaml_path is not None:
        overrides.update(_flatten(load_yaml_config(yaml_path)))

    return Settings(**{k.removeprefix("CHRONO_").lower(): v for k, v in overrides.items()})
