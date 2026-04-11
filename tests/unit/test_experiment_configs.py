"""Unit tests for the Phase 10 task 10.5 experiment YAML configs.

These tests lock in the six experiment YAMLs under ``configs/experiments/``
that the Phase 10 runner (task 10.6) will consume. The point of this file
is not to re-test :mod:`chronoagent.experiments.config_schema` (task 10.1
already covers that), but to guarantee that:

1. Every YAML round-trips through :meth:`ExperimentConfig.from_yaml` with
   no validation errors, so the runner can load them blindly.
2. The ablation blocks realise the paper's "one-knob delta per row" table
   structure: each ``ablation_no_*`` YAML flips exactly one
   :class:`AblationConfig` flag to ``False`` with every other flag still
   ``True``. The main + agentpoison experiments have every flag ``True``,
   and ``baseline_sentinel`` has every flag ``False`` to mark the
   "full ChronoAgent stack off, dispatch the reactive comparator"
   hand-off that the runner (10.6) picks up from the name.
3. The attack blocks pin the attack type the paper table in PLAN.md
   Phase 10 promises: main + every ablation + the Sentinel baseline use
   MINJA; the generalisation experiment uses AgentPoison.
4. The shared reproducibility knobs (``seed``, ``num_runs``, ``num_prs``,
   ``attack.injection_step``, ``attack.n_poison_docs``,
   ``system.llm_backend``) are identical across all six configs so the
   ablation table deltas are attributable only to the ablation block.

Every assertion here fires at ``from_yaml`` time: if any YAML is
rewritten in a way that breaks the delta contract, the test will fail
with a human-readable diff.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chronoagent.experiments.config_schema import (
    AblationConfig,
    ExperimentConfig,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"

# The six YAMLs PLAN.md task 10.5 enumerates. ``signal_validation.yaml``
# is intentionally excluded: that file is the Phase 1 runner input and
# does NOT round-trip through the 10.1 schema.
EXPERIMENT_NAMES = (
    "main_experiment",
    "agentpoison_experiment",
    "ablation_no_forecaster",
    "ablation_no_bocpd",
    "ablation_no_health_scores",
    "baseline_sentinel",
)

# Expected ablation block for every experiment. Derived from PLAN.md's
# Phase 10 task 10.5 table. If the table grows a new config this map
# must grow with it.
EXPECTED_ABLATIONS: dict[str, dict[str, bool]] = {
    "main_experiment": {
        "forecaster": True,
        "bocpd": True,
        "health": True,
        "integrity": True,
    },
    "agentpoison_experiment": {
        "forecaster": True,
        "bocpd": True,
        "health": True,
        "integrity": True,
    },
    "ablation_no_forecaster": {
        "forecaster": False,
        "bocpd": True,
        "health": True,
        "integrity": True,
    },
    "ablation_no_bocpd": {
        "forecaster": True,
        "bocpd": False,
        "health": True,
        "integrity": True,
    },
    "ablation_no_health_scores": {
        "forecaster": True,
        "bocpd": True,
        "health": False,
        "integrity": True,
    },
    "baseline_sentinel": {
        "forecaster": False,
        "bocpd": False,
        "health": False,
        "integrity": False,
    },
}

# Expected attack type for each experiment.
EXPECTED_ATTACK_TYPES: dict[str, str] = {
    "main_experiment": "minja",
    "agentpoison_experiment": "agentpoison",
    "ablation_no_forecaster": "minja",
    "ablation_no_bocpd": "minja",
    "ablation_no_health_scores": "minja",
    "baseline_sentinel": "minja",
}

# Shared reproducibility knobs every config must carry verbatim so the
# paper's ablation table deltas are attributable ONLY to the ablation block.
SHARED_SEED = 42
SHARED_NUM_RUNS = 5
SHARED_NUM_PRS = 25
SHARED_INJECTION_STEP = 10
SHARED_N_POISON_DOCS = 10
SHARED_LLM_BACKEND = "mock"

# The three ablation YAMLs the paper's per-subsystem contribution table
# will cite. Each MUST flip exactly one ``AblationConfig`` flag.
ABLATION_ONE_KNOB_NAMES = (
    "ablation_no_forecaster",
    "ablation_no_bocpd",
    "ablation_no_health_scores",
)


def _load(name: str) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(CONFIG_DIR / f"{name}.yaml")


# ---------------------------------------------------------------------------
# Presence + round-trip
# ---------------------------------------------------------------------------


class TestConfigFilesExist:
    """Every PLAN.md-listed YAML is actually on disk."""

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_yaml_file_exists(self, name: str) -> None:
        path = CONFIG_DIR / f"{name}.yaml"
        assert path.is_file(), f"missing experiment config: {path}"

    def test_exactly_six_experiment_yamls_in_directory(self) -> None:
        # ``signal_validation.yaml`` is a Phase 1 runner config and is
        # intentionally excluded. Any other unexpected YAML in this
        # directory should trip the test so new configs get a deliberate
        # review pass.
        yaml_files = {p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "signal_validation"}
        assert yaml_files == set(EXPERIMENT_NAMES)


class TestRoundTripThroughSchema:
    """Every YAML validates against the 10.1 pydantic schema."""

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_from_yaml_succeeds(self, name: str) -> None:
        cfg = _load(name)
        assert isinstance(cfg, ExperimentConfig)

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_name_matches_filename(self, name: str) -> None:
        cfg = _load(name)
        assert cfg.name == name

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_model_dump_round_trip(self, name: str) -> None:
        """``model_validate(model_dump())`` must be a no-op."""
        cfg = _load(name)
        reparsed = ExperimentConfig.model_validate(cfg.model_dump())
        assert reparsed == cfg


# ---------------------------------------------------------------------------
# Nested block coverage
# ---------------------------------------------------------------------------


class TestEveryNestedBlockExercised:
    """Each YAML must populate ``attack``, ``ablation``, and ``system``.

    The 10.5 requirement is that every config exercises every nested
    block so the runner test harness has a realistic fixture regardless
    of which experiment is selected.
    """

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_attack_block_populated(self, name: str) -> None:
        cfg = _load(name)
        assert cfg.attack.type in ("minja", "agentpoison", "none")
        assert cfg.attack.target == "both"
        assert cfg.attack.injection_step >= 0
        assert cfg.attack.n_poison_docs >= 0
        assert cfg.attack.strategy

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_ablation_block_is_an_ablation_config(self, name: str) -> None:
        cfg = _load(name)
        assert isinstance(cfg.ablation, AblationConfig)

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_system_block_populated(self, name: str) -> None:
        cfg = _load(name)
        assert cfg.system.llm_backend in ("mock", "together", "ollama")
        assert cfg.system.bocpd_hazard_lambda > 0.0
        assert 0.0 <= cfg.system.health_threshold <= 1.0
        assert 0.0 <= cfg.system.integrity_threshold <= 1.0


# ---------------------------------------------------------------------------
# Shared reproducibility knobs
# ---------------------------------------------------------------------------


class TestSharedReproducibilityKnobs:
    """The six configs must agree on every knob OUTSIDE the ablation block.

    This is what makes the paper's ablation table honest: if seed /
    num_runs / injection_step / n_poison_docs / llm_backend drift between
    configs, the table's deltas are no longer attributable to the one
    knob the ablation row names.
    """

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_seed(self, name: str) -> None:
        assert _load(name).seed == SHARED_SEED

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_num_runs(self, name: str) -> None:
        assert _load(name).num_runs == SHARED_NUM_RUNS

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_num_prs(self, name: str) -> None:
        assert _load(name).num_prs == SHARED_NUM_PRS

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_injection_step(self, name: str) -> None:
        assert _load(name).attack.injection_step == SHARED_INJECTION_STEP

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_n_poison_docs(self, name: str) -> None:
        assert _load(name).attack.n_poison_docs == SHARED_N_POISON_DOCS

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_llm_backend(self, name: str) -> None:
        assert _load(name).system.llm_backend == SHARED_LLM_BACKEND


# ---------------------------------------------------------------------------
# Attack type per PLAN.md table
# ---------------------------------------------------------------------------


class TestAttackTypePerPlanTable:
    """Lock in the Attack column of PLAN.md's task 10.5 table."""

    @pytest.mark.parametrize(
        "name,expected_type",
        sorted(EXPECTED_ATTACK_TYPES.items()),
    )
    def test_attack_type_matches_plan(self, name: str, expected_type: str) -> None:
        assert _load(name).attack.type == expected_type


# ---------------------------------------------------------------------------
# Ablation deltas: the headline invariant of 10.5
# ---------------------------------------------------------------------------


class TestAblationDeltasPerPlanTable:
    """The ablation block exact-matches PLAN.md's 10.5 table row-by-row."""

    @pytest.mark.parametrize(
        "name,expected",
        sorted(EXPECTED_ABLATIONS.items()),
    )
    def test_ablation_exact_match(self, name: str, expected: dict[str, bool]) -> None:
        cfg = _load(name)
        actual = {
            "forecaster": cfg.ablation.forecaster,
            "bocpd": cfg.ablation.bocpd,
            "health": cfg.ablation.health,
            "integrity": cfg.ablation.integrity,
        }
        assert actual == expected

    def test_main_experiment_full_system(self) -> None:
        """Control row: every subsystem ON."""
        ab = _load("main_experiment").ablation
        assert ab.forecaster is True
        assert ab.bocpd is True
        assert ab.health is True
        assert ab.integrity is True

    def test_agentpoison_experiment_full_system(self) -> None:
        """Generalisation row: same full stack as main_experiment."""
        ab = _load("agentpoison_experiment").ablation
        assert ab.forecaster is True
        assert ab.bocpd is True
        assert ab.health is True
        assert ab.integrity is True

    @pytest.mark.parametrize("name", ABLATION_ONE_KNOB_NAMES)
    def test_one_knob_delta_vs_main(self, name: str) -> None:
        """Exactly one ablation flag differs from main_experiment.

        This is the paper-facing invariant: every ablation row in the
        table must flip ONE and only ONE knob so the contribution of
        each subsystem is measurable in isolation.
        """
        main_ab = _load("main_experiment").ablation
        ablated_ab = _load(name).ablation
        differing_flags = [
            flag
            for flag in ("forecaster", "bocpd", "health", "integrity")
            if getattr(main_ab, flag) != getattr(ablated_ab, flag)
        ]
        assert differing_flags, f"{name} is identical to main_experiment"
        assert len(differing_flags) == 1, (
            f"{name} flips {len(differing_flags)} flags ({differing_flags}); expected exactly 1"
        )

    def test_ablation_no_forecaster_has_forecaster_off(self) -> None:
        ab = _load("ablation_no_forecaster").ablation
        assert ab.forecaster is False
        assert ab.bocpd is True
        assert ab.health is True
        assert ab.integrity is True

    def test_ablation_no_bocpd_has_bocpd_off(self) -> None:
        ab = _load("ablation_no_bocpd").ablation
        assert ab.forecaster is True
        assert ab.bocpd is False
        assert ab.health is True
        assert ab.integrity is True

    def test_ablation_no_health_scores_has_health_off(self) -> None:
        ab = _load("ablation_no_health_scores").ablation
        assert ab.forecaster is True
        assert ab.bocpd is True
        assert ab.health is False
        assert ab.integrity is True

    def test_baseline_sentinel_has_every_flag_off(self) -> None:
        """Sentinel is NOT a one-knob delta row.

        It flips every ablation flag to ``False`` to mark "full
        ChronoAgent stack off, runner dispatches the reactive
        comparator". The runner (10.6) takes the additional hint from
        the name field.
        """
        ab = _load("baseline_sentinel").ablation
        assert ab.forecaster is False
        assert ab.bocpd is False
        assert ab.health is False
        assert ab.integrity is False


# ---------------------------------------------------------------------------
# per_run_seed compatibility
# ---------------------------------------------------------------------------


class TestPerRunSeedOverShippedConfigs:
    """Every config supports the 5-run aggregation the runner expects."""

    @pytest.mark.parametrize("name", EXPERIMENT_NAMES)
    def test_five_distinct_per_run_seeds(self, name: str) -> None:
        cfg = _load(name)
        seeds = [cfg.per_run_seed(i) for i in range(cfg.num_runs)]
        assert seeds == [SHARED_SEED + i for i in range(SHARED_NUM_RUNS)]
        assert len(set(seeds)) == SHARED_NUM_RUNS
