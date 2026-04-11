"""Experiment configuration schema (Phase 10 task 10.1).

Phase 10 of the project plan ships six experiment YAMLs (main result,
generalization, three ablations, one baseline) plus a runner that consumes
them.  This module defines the typed pydantic models that every experiment
YAML loads through, so the runner (task 10.6) and analysis layer (10.7-10.8)
have a single source of truth for what an experiment is and what its valid
parameter ranges are.

The schema is intentionally narrow.  It captures only the parameters the
PLAN.md task 10.1 wording enumerates:

* :class:`ExperimentConfig` -- top-level container with reproducibility +
  per-run knobs (``name``, ``seed``, ``num_runs``, ``num_prs``).
* :class:`AttackConfig` -- which memory-poisoning attack to apply, against
  which agent collection, at which step, with which strategy variant.
* :class:`AblationConfig` -- four boolean toggles for forecaster, BOCPD,
  health scoring, and memory integrity, used to switch parts of the system
  off for ablation studies.
* :class:`SystemConfig` -- runtime knobs the experiment harness needs to
  override per-run (LLM backend, BOCPD hazard, escalation thresholds).

YAML loading goes through :meth:`ExperimentConfig.from_yaml`, which uses
``yaml.safe_load`` and surfaces ``pydantic.ValidationError`` for any
malformed file.  The model uses ``model_config = ConfigDict(extra="forbid")``
so a typo in a YAML key is a hard failure rather than silent shadowing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Attack-type vocabulary.  ``"none"`` is the no-attack baseline used by the
# clean-system experiment configurations.
AttackType = Literal["minja", "agentpoison", "none"]

# Target collection for the attack.  ``"both"`` poisons every agent
# collection in the experiment; the per-agent values poison just that
# agent's memory.
AttackTarget = Literal["security_reviewer", "summarizer", "both"]

# Active LLM backend for the experiment.  ``"mock"`` is the deterministic
# default that lets every experiment YAML reproduce on a clean machine
# without API keys.
LLMBackend = Literal["mock", "together", "ollama"]


class AttackConfig(BaseModel):
    """Memory-poisoning attack parameters for one experiment.

    Attributes:
        type: Which attack to run.  ``"minja"`` and ``"agentpoison"`` map
            onto the implementations in :mod:`chronoagent.memory.poisoning`.
            ``"none"`` runs the system clean and is used by ablation
            controls and the no-attack baselines.
        target: Which agent collection(s) to poison.  Use ``"both"`` to
            poison every agent in the experiment.
        injection_step: Zero-indexed step at which the attack is injected.
            For a typical experiment with calibration + poisoned phase the
            value is the calibration length (e.g. ``10`` for a 25-step run
            with 10 calibration steps).
        n_poison_docs: Number of malicious documents the attack injects
            into the target collection.  Mirrors the ``n_poison`` argument
            of :meth:`chronoagent.memory.poisoning.MINJAStyleAttack.inject`.
        strategy: Free-form short identifier for the attack variant
            (e.g. ``"low_noise"``, ``"high_noise"``, ``"trigger_phrase"``).
            The runner records it on the result row so analysis tables
            can group runs by strategy without re-deriving it from
            ``n_poison_docs``.
    """

    model_config = ConfigDict(extra="forbid")

    type: AttackType
    target: AttackTarget = "both"
    injection_step: int = Field(ge=0)
    n_poison_docs: int = Field(default=10, ge=0)
    strategy: str = "default"

    @field_validator("strategy")
    @classmethod
    def _strategy_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy must be a non-empty string")
        return value


class AblationConfig(BaseModel):
    """Toggle individual subsystems on or off for ablation studies.

    Each flag corresponds to one of the four ChronoAgent subsystems whose
    contribution we ablate in Section 5 of the paper:

    * ``forecaster`` -- :class:`~chronoagent.scorer.chronos_forecaster.ChronosForecaster`
      ensemble component.  When ``False`` the health scorer falls back to
      BOCPD-only mode (the same path Phase 9 task 9.3 surfaces as
      ``forecaster: fallback`` in the comprehensive health endpoint).
    * ``bocpd`` -- :class:`~chronoagent.scorer.bocpd.BOCPD` changepoint
      detector.  When ``False`` the ensemble runs Chronos-only.
    * ``health`` -- the entire :class:`~chronoagent.scorer.health_scorer.TemporalHealthScorer`.
      When ``False`` the allocator runs without per-agent health gating
      and degrades to round-robin on every task.
    * ``integrity`` -- :class:`~chronoagent.memory.integrity.MemoryIntegrityModule`.
      When ``False`` no documents are quarantined regardless of attack
      severity.

    Defaults are the full system (every flag ``True``).  Ablation YAMLs
    override one flag at a time so the contribution of each subsystem is
    measurable in isolation.
    """

    model_config = ConfigDict(extra="forbid")

    forecaster: bool = True
    bocpd: bool = True
    health: bool = True
    integrity: bool = True


class SystemConfig(BaseModel):
    """Runtime knobs the experiment harness overrides per run.

    Kept deliberately narrow: anything that lives in
    :class:`chronoagent.config.Settings` and is unaffected by an
    experiment (database URL, log format, etc.) is left to the
    application defaults.  Only the knobs that change between
    experiments live here.

    Attributes:
        llm_backend: Which language-model backend to drive the agents
            with.  ``"mock"`` is the deterministic default.
        bocpd_hazard_lambda: BOCPD expected run length.  Lower values
            make the detector more sensitive to changepoints.
        health_threshold: Health score below which the allocator
            escalates a task instead of assigning it.
        integrity_threshold: Aggregate integrity score above which a
            document is flagged for quarantine.
    """

    model_config = ConfigDict(extra="forbid")

    llm_backend: LLMBackend = "mock"
    bocpd_hazard_lambda: float = Field(default=50.0, gt=0.0)
    health_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    integrity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class ExperimentConfig(BaseModel):
    """Top-level configuration for one Phase 10 experiment.

    Every experiment YAML in ``configs/experiments/`` loads into this
    model via :meth:`from_yaml`.  The runner (task 10.6) consumes the
    parsed model directly: there is no intermediate dict-of-dicts layer
    so a typo in a YAML key fails fast at load time rather than at
    measurement time three minutes into a run.

    Attributes:
        name: Short experiment identifier used for output filenames and
            log/event names.  Snake_case ASCII only, enforced by a
            validator so result files cannot contain whitespace or
            shell metacharacters.
        seed: Master seed for the experiment.  The runner derives
            per-run seeds via ``seed + run_index`` so a single
            ``ExperimentConfig`` can drive ``num_runs`` reproducible
            runs without re-loading the YAML.
        num_runs: Number of repeated runs at different per-run seeds
            for statistical aggregation (mean / std / 95% CI).  PLAN.md
            uses ``5`` as the default so an experiment is feasible on
            CPU-only hardware.
        num_prs: Number of synthetic PRs each run processes.  Maps to
            the ``n_steps`` parameter on the existing Phase 1
            ``SignalValidationRunner`` and to the post-injection step
            count for Phase 10 runners.
        attack: Memory-poisoning attack to apply this experiment.
        ablation: Subsystem toggles for ablation studies.
        system: Per-experiment runtime overrides.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = Field(ge=0)
    num_runs: int = Field(default=5, ge=1)
    num_prs: int = Field(default=25, ge=1)
    attack: AttackConfig
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    @field_validator("name")
    @classmethod
    def _name_is_safe_identifier(cls, value: str) -> str:
        """Reject names that would be unsafe to splice into a filename."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be empty")
        # Allow ASCII letters, digits, underscore, and hyphen.  Anything
        # else is a path-traversal or shell-metacharacter risk when the
        # runner uses the name as a filename component.
        for ch in stripped:
            if not (ch.isalnum() or ch in "_-"):
                raise ValueError(
                    f"name must be alphanumeric / underscore / hyphen only, got {value!r}"
                )
        return stripped

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an experiment YAML and validate it against the schema.

        Args:
            path: Filesystem path to a YAML file containing a single
                top-level mapping with the ``ExperimentConfig`` fields.

        Returns:
            A fully validated :class:`ExperimentConfig` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the YAML root is not a mapping.
            pydantic.ValidationError: If any field fails validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"experiment config not found: {path}")
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"experiment config root must be a mapping, got {type(raw).__name__}")
        return cls.model_validate(raw)

    def per_run_seed(self, run_index: int) -> int:
        """Derive a per-run seed from the master seed.

        Args:
            run_index: Zero-indexed run number in ``range(num_runs)``.

        Returns:
            ``self.seed + run_index``.  The runner uses this so each
            run inside one experiment is independently reproducible
            but the whole experiment can be replayed by re-loading
            the YAML.
        """
        if not 0 <= run_index < self.num_runs:
            raise ValueError(f"run_index {run_index} out of range for num_runs={self.num_runs}")
        return self.seed + run_index
