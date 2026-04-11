"""Unit tests for ``chronoagent.experiments.config_schema`` (Phase 10 task 10.1).

These tests cover four things:

* Defaults: every field with a default value comes through unchanged when
  the YAML omits it.
* Validation rules: each ``Field(...)`` constraint and each
  ``@field_validator`` rejects the malformed input it is meant to reject.
* ``ExperimentConfig.from_yaml``: round-trip from a YAML string on disk
  via :func:`pathlib.Path` and surfaces the right errors for missing
  files, non-mapping roots, and pydantic validation failures.
* ``per_run_seed``: derives ``seed + run_index`` and refuses out-of-range
  indices.

The tests do not import or run the experiment runner; the schema is the
contract that 10.6 will consume, and we lock it in here so the runner
work can rely on stable field names and validators.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from chronoagent.experiments.config_schema import (
    AblationConfig,
    AttackConfig,
    ExperimentConfig,
    SystemConfig,
)

# ── AttackConfig ─────────────────────────────────────────────────────────────


class TestAttackConfig:
    """Per-field validation on the attack-config sub-model."""

    def test_minimal_construction(self) -> None:
        cfg = AttackConfig(type="minja", injection_step=10)
        assert cfg.type == "minja"
        assert cfg.target == "both"
        assert cfg.injection_step == 10
        assert cfg.n_poison_docs == 10
        assert cfg.strategy == "default"

    @pytest.mark.parametrize("attack_type", ["minja", "agentpoison", "none"])
    def test_all_attack_types_accepted(self, attack_type: str) -> None:
        cfg = AttackConfig(type=attack_type, injection_step=0)  # type: ignore[arg-type]
        assert cfg.type == attack_type

    def test_unknown_attack_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AttackConfig(type="bogus", injection_step=0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("target", ["security_reviewer", "summarizer", "both"])
    def test_all_targets_accepted(self, target: str) -> None:
        cfg = AttackConfig(type="minja", injection_step=5, target=target)  # type: ignore[arg-type]
        assert cfg.target == target

    def test_unknown_target_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AttackConfig(type="minja", injection_step=5, target="planner")  # type: ignore[arg-type]

    def test_negative_injection_step_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AttackConfig(type="minja", injection_step=-1)

    def test_negative_n_poison_docs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AttackConfig(type="minja", injection_step=0, n_poison_docs=-1)

    def test_empty_strategy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AttackConfig(type="minja", injection_step=0, strategy="   ")

    def test_strategy_with_surrounding_whitespace_accepted_verbatim(self) -> None:
        """Whitespace-only is rejected, but a non-empty strategy with surrounding
        whitespace passes through unchanged.  The validator only enforces
        non-emptiness; it does not silently mutate user input."""
        cfg = AttackConfig(type="minja", injection_step=0, strategy="  high_noise  ")
        assert cfg.strategy == "  high_noise  "

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AttackConfig(  # type: ignore[call-arg]
                type="minja",
                injection_step=0,
                bogus="value",
            )


# ── AblationConfig ───────────────────────────────────────────────────────────


class TestAblationConfig:
    """Default ablation has every subsystem on; flags are independent."""

    def test_defaults_full_system(self) -> None:
        cfg = AblationConfig()
        assert cfg.forecaster is True
        assert cfg.bocpd is True
        assert cfg.health is True
        assert cfg.integrity is True

    def test_individual_flags_can_be_disabled(self) -> None:
        cfg = AblationConfig(forecaster=False)
        assert cfg.forecaster is False
        assert cfg.bocpd is True

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AblationConfig(retrieval=False)  # type: ignore[call-arg]


# ── SystemConfig ─────────────────────────────────────────────────────────────


class TestSystemConfig:
    """SystemConfig defaults are usable on a clean machine."""

    def test_defaults_are_mock_backend(self) -> None:
        cfg = SystemConfig()
        assert cfg.llm_backend == "mock"
        assert cfg.bocpd_hazard_lambda == 50.0
        assert cfg.health_threshold == 0.3
        assert cfg.integrity_threshold == 0.6

    @pytest.mark.parametrize("backend", ["mock", "together", "ollama"])
    def test_all_backends_accepted(self, backend: str) -> None:
        cfg = SystemConfig(llm_backend=backend)  # type: ignore[arg-type]
        assert cfg.llm_backend == backend

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(llm_backend="claude")  # type: ignore[arg-type]

    def test_zero_hazard_lambda_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(bocpd_hazard_lambda=0.0)

    def test_health_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(health_threshold=1.5)

    def test_integrity_threshold_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(integrity_threshold=-0.1)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(quarantine_threshold=0.5)  # type: ignore[call-arg]


# ── ExperimentConfig ─────────────────────────────────────────────────────────


def _minimal_payload() -> dict[str, object]:
    """Return the smallest valid mapping that can build an ExperimentConfig."""
    return {
        "name": "main_experiment",
        "seed": 42,
        "attack": {"type": "minja", "injection_step": 10},
    }


class TestExperimentConfig:
    """Top-level config: defaults, name validation, and nested submodels."""

    def test_minimal_construction(self) -> None:
        cfg = ExperimentConfig(**_minimal_payload())  # type: ignore[arg-type]
        assert cfg.name == "main_experiment"
        assert cfg.seed == 42
        assert cfg.num_runs == 5
        assert cfg.num_prs == 25
        assert cfg.attack.type == "minja"
        assert cfg.ablation.forecaster is True
        assert cfg.system.llm_backend == "mock"

    def test_full_construction_overrides_every_default(self) -> None:
        cfg = ExperimentConfig(
            name="ablation_no_bocpd",
            seed=7,
            num_runs=3,
            num_prs=50,
            attack=AttackConfig(
                type="agentpoison",
                target="security_reviewer",
                injection_step=15,
                n_poison_docs=20,
                strategy="trigger_phrase",
            ),
            ablation=AblationConfig(bocpd=False),
            system=SystemConfig(
                llm_backend="together",
                bocpd_hazard_lambda=100.0,
                health_threshold=0.5,
                integrity_threshold=0.75,
            ),
        )
        assert cfg.attack.target == "security_reviewer"
        assert cfg.ablation.bocpd is False
        assert cfg.system.llm_backend == "together"

    @pytest.mark.parametrize("bad_name", ["", "   ", "with space", "../escape", "weird!"])
    def test_unsafe_names_rejected(self, bad_name: str) -> None:
        payload = _minimal_payload()
        payload["name"] = bad_name
        with pytest.raises(ValidationError):
            ExperimentConfig(**payload)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "good_name",
        ["main", "main_experiment", "ablation-no-bocpd", "v2_run"],
    )
    def test_safe_names_accepted(self, good_name: str) -> None:
        payload = _minimal_payload()
        payload["name"] = good_name
        cfg = ExperimentConfig(**payload)  # type: ignore[arg-type]
        assert cfg.name == good_name

    def test_negative_seed_rejected(self) -> None:
        payload = _minimal_payload()
        payload["seed"] = -1
        with pytest.raises(ValidationError):
            ExperimentConfig(**payload)  # type: ignore[arg-type]

    def test_zero_num_runs_rejected(self) -> None:
        payload = _minimal_payload()
        payload["num_runs"] = 0
        with pytest.raises(ValidationError):
            ExperimentConfig(**payload)  # type: ignore[arg-type]

    def test_zero_num_prs_rejected(self) -> None:
        payload = _minimal_payload()
        payload["num_prs"] = 0
        with pytest.raises(ValidationError):
            ExperimentConfig(**payload)  # type: ignore[arg-type]

    def test_extra_top_level_field_rejected(self) -> None:
        payload = _minimal_payload()
        payload["bogus"] = "value"
        with pytest.raises(ValidationError):
            ExperimentConfig(**payload)  # type: ignore[arg-type]


# ── from_yaml ────────────────────────────────────────────────────────────────


class TestFromYaml:
    """``ExperimentConfig.from_yaml`` parses files end-to-end."""

    def test_round_trip_minimal(self, tmp_path: Path) -> None:
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(_minimal_payload()), encoding="utf-8")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.name == "main_experiment"
        assert cfg.attack.type == "minja"

    def test_round_trip_full(self, tmp_path: Path) -> None:
        payload = {
            "name": "main_experiment",
            "seed": 42,
            "num_runs": 5,
            "num_prs": 25,
            "attack": {
                "type": "minja",
                "target": "summarizer",
                "injection_step": 10,
                "n_poison_docs": 15,
                "strategy": "low_noise",
            },
            "ablation": {
                "forecaster": True,
                "bocpd": True,
                "health": True,
                "integrity": False,
            },
            "system": {
                "llm_backend": "mock",
                "bocpd_hazard_lambda": 25.0,
                "health_threshold": 0.4,
                "integrity_threshold": 0.7,
            },
        }
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.attack.target == "summarizer"
        assert cfg.ablation.integrity is False
        assert cfg.system.bocpd_hazard_lambda == 25.0

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml(tmp_path / "does_not_exist.yaml")

    def test_non_mapping_root_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "exp.yaml"
        path.write_text("- list\n- of\n- items\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            ExperimentConfig.from_yaml(path)

    def test_typo_surfaces_validation_error(self, tmp_path: Path) -> None:
        """A YAML key typo must hard-fail at load time, not silently shadow."""
        payload = _minimal_payload()
        payload["nam"] = payload.pop("name")  # typo
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValidationError):
            ExperimentConfig.from_yaml(path)

    def test_accepts_str_path_argument(self, tmp_path: Path) -> None:
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(_minimal_payload()), encoding="utf-8")
        cfg = ExperimentConfig.from_yaml(str(path))
        assert cfg.name == "main_experiment"


# ── per_run_seed ─────────────────────────────────────────────────────────────


class TestPerRunSeed:
    """``per_run_seed`` derives reproducible per-run seeds."""

    def test_run_zero_returns_master_seed(self) -> None:
        cfg = ExperimentConfig(**_minimal_payload())  # type: ignore[arg-type]
        assert cfg.per_run_seed(0) == 42

    def test_increments_by_run_index(self) -> None:
        cfg = ExperimentConfig(**_minimal_payload())  # type: ignore[arg-type]
        assert cfg.per_run_seed(4) == 46

    def test_negative_run_index_rejected(self) -> None:
        cfg = ExperimentConfig(**_minimal_payload())  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            cfg.per_run_seed(-1)

    def test_run_index_beyond_num_runs_rejected(self) -> None:
        cfg = ExperimentConfig(**_minimal_payload())  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            cfg.per_run_seed(cfg.num_runs)
