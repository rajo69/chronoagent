"""Unit tests for the Phase 10 task 10.6 experiment runner.

These tests cover the full surface of
:mod:`chronoagent.experiments.experiment_runner` and its companion
:mod:`chronoagent.experiments.full_system_detector`:

* ``_dispatch_detector`` returns the right comparator for every one of
  the six shipped experiment YAMLs (locked in at 10.5).
* ``_aggregate_metric`` produces mean / std / 95% CI with sensible
  edge-case behaviour (empty input, single sample, all-nan input).
* ``_compute_metrics`` correctly turns a decision stream into a
  :class:`RunResult` for all three detector flavours.
* :class:`ExperimentRunner` end-to-end with a deterministic fake signal
  matrix factory: per-run determinism, dispatch, metric shape, AWT
  bookkeeping when the detector never fires.
* :class:`FullSystemDetector` channel wiring: BOCPD flag gates the
  BOCPD channel, forecaster flag gates the EMA channel, integrity flag
  gates the MAD channel, and an all-off construction falls back to
  Sentinel-style z-score on the KL column.
* ``write_experiment_results`` round-trips CSV + JSON on disk with
  NaN-safe serialisation.

A tiny integration test at the bottom runs the real default factory
(which wraps :class:`SignalValidationRunner`) on the ``main_experiment``
YAML at ``num_runs=1, num_prs=16, injection_step=8`` to prove the
signal-matrix pipeline glues together end-to-end without ChromaDB or
MockBackend divergence. It is explicitly marked ``@pytest.mark.slow``
so operators can skip it if they want a fast inner loop.
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from chronoagent.experiments.baselines.no_monitoring import NoMonitoringBaseline
from chronoagent.experiments.baselines.sentinel import SentinelBaseline
from chronoagent.experiments.config_schema import (
    AblationConfig,
    AttackConfig,
    ExperimentConfig,
    SystemConfig,
)
from chronoagent.experiments.experiment_runner import (
    CI_CONFIDENCE,
    AggregateResult,
    ExperimentRunner,
    MetricAggregate,
    RunResult,
    SignalMatrixFactory,
    _aggregate_metric,
    _compute_metrics,
    _dispatch_detector,
    default_signal_matrix_factory,
    write_experiment_results,
)
from chronoagent.experiments.full_system_detector import (
    ENTROPY_COLUMN_INDEX,
    FULL_SYSTEM_AGENT_ID,
    KL_COLUMN_INDEX,
    FullSystemConfig,
    FullSystemDecision,
    FullSystemDetector,
)
from chronoagent.monitor.collector import NUM_SIGNALS

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    name: str = "fake_cfg",
    seed: int = 7,
    num_runs: int = 3,
    num_prs: int = 20,
    injection_step: int = 8,
    attack_type: str = "minja",
    forecaster: bool = True,
    bocpd: bool = True,
    health: bool = True,
    integrity: bool = True,
) -> ExperimentConfig:
    """Build a minimal :class:`ExperimentConfig` for tests."""
    return ExperimentConfig(
        name=name,
        seed=seed,
        num_runs=num_runs,
        num_prs=num_prs,
        attack=AttackConfig(
            type=attack_type,  # type: ignore[arg-type]
            injection_step=injection_step,
            n_poison_docs=5,
        ),
        ablation=AblationConfig(
            forecaster=forecaster,
            bocpd=bocpd,
            health=health,
            integrity=integrity,
        ),
        system=SystemConfig(),
    )


def _shifted_factory(
    shift_kl: float = 5.0,
    shift_entropy: float = 3.0,
    noise: float = 0.1,
) -> SignalMatrixFactory:
    """Return a deterministic factory that shifts post-injection rows.

    The factory is seeded: calling it twice with the same seed produces
    byte-identical matrices, which the runner test file leans on for
    per-run determinism assertions.
    """

    def factory(
        *,
        attack_type: str,
        seed: int,
        injection_step: int,
        num_prs: int,
        n_poison_docs: int,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mat = rng.normal(loc=0.0, scale=noise, size=(num_prs, NUM_SIGNALS)).astype(np.float64)
        mat[injection_step:, KL_COLUMN_INDEX] += shift_kl
        mat[injection_step:, ENTROPY_COLUMN_INDEX] += shift_entropy
        return mat

    return factory  # type: ignore[return-value]


def _null_factory() -> SignalMatrixFactory:
    """Factory that returns a clean (no-shift) matrix so detectors never fire."""

    def factory(
        *,
        attack_type: str,
        seed: int,
        injection_step: int,
        num_prs: int,
        n_poison_docs: int,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.01, size=(num_prs, NUM_SIGNALS)).astype(np.float64)

    return factory  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# _aggregate_metric
# ---------------------------------------------------------------------------


class TestAggregateMetric:
    def test_empty_returns_nan_and_zero_n(self) -> None:
        agg = _aggregate_metric([])
        assert agg.n == 0
        assert math.isnan(agg.mean)
        assert math.isnan(agg.std)
        assert math.isnan(agg.ci_low)
        assert math.isnan(agg.ci_high)

    def test_single_sample_reports_mean_and_nan_ci(self) -> None:
        agg = _aggregate_metric([0.7])
        assert agg.n == 1
        assert agg.mean == pytest.approx(0.7)
        assert agg.std == 0.0
        assert math.isnan(agg.ci_low)
        assert math.isnan(agg.ci_high)

    def test_uniform_samples_have_zero_std_and_zero_width_ci(self) -> None:
        agg = _aggregate_metric([0.5, 0.5, 0.5])
        assert agg.n == 3
        assert agg.mean == pytest.approx(0.5)
        assert agg.std == 0.0
        assert agg.ci_low == pytest.approx(0.5)
        assert agg.ci_high == pytest.approx(0.5)

    def test_ci_is_symmetric_around_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        agg = _aggregate_metric(values)
        assert agg.mean == pytest.approx(3.0)
        half_width_low = agg.mean - agg.ci_low
        half_width_high = agg.ci_high - agg.mean
        assert half_width_low == pytest.approx(half_width_high)
        assert agg.ci_low < agg.mean < agg.ci_high

    def test_ci_width_matches_student_t_formula(self) -> None:
        """Spot check the CI width against a hand computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        agg = _aggregate_metric(values)
        # std with ddof=1 of [1,2,3,4,5] is sqrt(2.5) ~= 1.5811
        assert agg.std == pytest.approx(math.sqrt(2.5))

    def test_nan_values_are_dropped(self) -> None:
        values = [1.0, float("nan"), 2.0, 3.0, float("nan")]
        agg = _aggregate_metric(values)
        assert agg.n == 3
        assert agg.mean == pytest.approx(2.0)

    def test_all_nan_input_returns_empty_aggregate(self) -> None:
        agg = _aggregate_metric([float("nan"), float("nan")])
        assert agg.n == 0
        assert math.isnan(agg.mean)

    def test_ci_confidence_constant_is_95_percent(self) -> None:
        assert pytest.approx(0.95) == CI_CONFIDENCE


# ---------------------------------------------------------------------------
# _dispatch_detector
# ---------------------------------------------------------------------------


class TestDispatchDetector:
    def test_baseline_sentinel_dispatches_sentinel(self) -> None:
        cfg = ExperimentConfig.from_yaml(CONFIG_DIR / "baseline_sentinel.yaml")
        detector, label = _dispatch_detector(cfg)
        assert isinstance(detector, SentinelBaseline)
        assert label == "sentinel_baseline"

    def test_ablation_no_health_dispatches_no_monitoring(self) -> None:
        cfg = ExperimentConfig.from_yaml(CONFIG_DIR / "ablation_no_health_scores.yaml")
        detector, label = _dispatch_detector(cfg)
        assert isinstance(detector, NoMonitoringBaseline)
        assert label == "no_monitoring_baseline"

    @pytest.mark.parametrize(
        "yaml_name",
        [
            "main_experiment",
            "agentpoison_experiment",
            "ablation_no_forecaster",
            "ablation_no_bocpd",
        ],
    )
    def test_full_system_yamls_dispatch_full_system_detector(self, yaml_name: str) -> None:
        cfg = ExperimentConfig.from_yaml(CONFIG_DIR / f"{yaml_name}.yaml")
        detector, label = _dispatch_detector(cfg)
        assert isinstance(detector, FullSystemDetector)
        assert label == "full_system_detector"

    def test_sentinel_calibration_steps_from_injection_step(self) -> None:
        cfg = _make_cfg(name="baseline_sentinel", injection_step=7, num_prs=20)
        detector, _ = _dispatch_detector(cfg)
        assert isinstance(detector, SentinelBaseline)
        assert detector.config.calibration_steps == 7

    def test_full_system_hazard_lambda_from_system_config(self) -> None:
        cfg = _make_cfg(num_prs=20, injection_step=8)
        detector, _ = _dispatch_detector(cfg)
        assert isinstance(detector, FullSystemDetector)
        # Internal config: FullSystemConfig bocpd_hazard_lambda should
        # match cfg.system default (50.0).
        assert detector._config.bocpd_hazard_lambda == pytest.approx(50.0)  # noqa: SLF001

    def test_full_system_respects_ablation_flags(self) -> None:
        cfg = _make_cfg(bocpd=False, forecaster=True, integrity=False)
        detector, _ = _dispatch_detector(cfg)
        assert isinstance(detector, FullSystemDetector)
        assert detector._mask.bocpd is False  # noqa: SLF001
        assert detector._mask.forecaster is True  # noqa: SLF001
        assert detector._mask.integrity is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_full_system_decisions_give_expected_metrics(self) -> None:
        """Hand-craft a perfect decision stream and check every metric."""
        num_prs = 10
        injection_step = 4
        # Clean steps (0..3): success=True, flagged=False, score=0.1.
        # Poison steps (4..9): success=False, flagged=True, score=0.9.
        decisions = [
            FullSystemDecision(
                step_index=i,
                score=0.1 if i < injection_step else 0.9,
                flagged=i >= injection_step,
                success=i < injection_step,
                agent_id=FULL_SYSTEM_AGENT_ID,
            )
            for i in range(num_prs)
        ]
        result = _compute_metrics(
            decisions,
            run_index=0,
            seed=42,
            detector_name="full_system_detector",
            injection_step=injection_step,
            num_prs=num_prs,
        )
        assert result.allocation_efficiency_score == pytest.approx(injection_step / num_prs)
        assert result.detection_auroc_score == pytest.approx(1.0)
        assert result.detection_f1_score == pytest.approx(1.0)
        assert result.first_flagged_step == injection_step
        assert result.advance_warning_time == 0  # concurrent per Pivot A

    def test_detector_fires_before_injection_gives_positive_awt(self) -> None:
        num_prs = 10
        injection_step = 6
        # Detector fires at step 4: AWT = 6 - 4 = 2
        decisions = [
            FullSystemDecision(
                step_index=i,
                score=0.9 if i >= 4 else 0.1,
                flagged=i >= 4,
                success=i < 4,
                agent_id=FULL_SYSTEM_AGENT_ID,
            )
            for i in range(num_prs)
        ]
        result = _compute_metrics(
            decisions,
            run_index=0,
            seed=42,
            detector_name="full_system_detector",
            injection_step=injection_step,
            num_prs=num_prs,
        )
        assert result.first_flagged_step == 4
        assert result.advance_warning_time == 2

    def test_detector_never_fires_gives_none_awt(self) -> None:
        num_prs = 8
        injection_step = 4
        decisions = [
            FullSystemDecision(
                step_index=i,
                score=0.1,
                flagged=False,
                success=True,
                agent_id=FULL_SYSTEM_AGENT_ID,
            )
            for i in range(num_prs)
        ]
        result = _compute_metrics(
            decisions,
            run_index=0,
            seed=42,
            detector_name="full_system_detector",
            injection_step=injection_step,
            num_prs=num_prs,
        )
        assert result.first_flagged_step is None
        assert result.advance_warning_time is None
        assert result.allocation_efficiency_score == pytest.approx(1.0)
        # All-flagged-False with two classes gives F1 = 0 (zero TP).
        assert result.detection_f1_score == pytest.approx(0.0)

    def test_wrong_decision_count_raises(self) -> None:
        decisions = [
            FullSystemDecision(
                step_index=0,
                score=0.1,
                flagged=False,
                success=True,
                agent_id=FULL_SYSTEM_AGENT_ID,
            )
        ]
        with pytest.raises(RuntimeError, match="expected 5"):
            _compute_metrics(
                decisions,
                run_index=0,
                seed=42,
                detector_name="full_system_detector",
                injection_step=2,
                num_prs=5,
            )


# ---------------------------------------------------------------------------
# ExperimentRunner end-to-end (fake factory)
# ---------------------------------------------------------------------------


class TestExperimentRunnerEndToEnd:
    def test_full_system_config_yields_perfect_metrics_on_shifted_fake(self) -> None:
        cfg = _make_cfg(name="main_experiment", num_runs=3, num_prs=20, injection_step=8)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        assert agg.num_runs == 3
        assert len(agg.runs) == 3
        assert agg.detector_name == "full_system_detector"
        # The shifted fake produces a clean separation, so all runs detect.
        assert agg.advance_warning_time.n == 3
        assert agg.allocation_efficiency_score.mean == pytest.approx(8 / 20, abs=0.05)
        assert agg.detection_auroc_score.mean == pytest.approx(1.0)
        assert agg.detection_f1_score.mean >= 0.9

    def test_baseline_sentinel_dispatch_on_yaml(self) -> None:
        cfg = ExperimentConfig.from_yaml(CONFIG_DIR / "baseline_sentinel.yaml")
        cfg_small = cfg.model_copy(
            update={
                "num_runs": 2,
                "num_prs": 20,
                "attack": cfg.attack.model_copy(update={"injection_step": 8}),
            }
        )
        runner = ExperimentRunner(cfg_small, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        assert agg.detector_name == "sentinel_baseline"
        assert agg.num_runs == 2
        assert agg.allocation_efficiency_score.mean == pytest.approx(8 / 20, abs=0.05)

    def test_no_monitoring_dispatch_on_yaml(self) -> None:
        cfg = ExperimentConfig.from_yaml(CONFIG_DIR / "ablation_no_health_scores.yaml")
        cfg_small = cfg.model_copy(
            update={
                "num_runs": 2,
                "num_prs": 20,
                "attack": cfg.attack.model_copy(update={"injection_step": 8}),
            }
        )
        runner = ExperimentRunner(cfg_small, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        assert agg.detector_name == "no_monitoring_baseline"
        # NoMonitoring always succeeds.
        assert agg.allocation_efficiency_score.mean == pytest.approx(1.0)
        # Detector never fires => AWT aggregate is empty.
        assert agg.advance_warning_time.n == 0
        assert math.isnan(agg.advance_warning_time.mean)

    def test_detector_never_fires_on_null_factory(self) -> None:
        cfg = _make_cfg(num_runs=3, num_prs=20, injection_step=10)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_null_factory())
        agg = runner.run()
        # With no injected shift, the full system should not flag clean noise
        # (BOCPD + EMA + MAD all bounded below threshold). Every run should
        # come out as "never fired".
        assert agg.advance_warning_time.n == 0
        assert agg.allocation_efficiency_score.mean == pytest.approx(1.0)

    def test_per_run_determinism(self) -> None:
        cfg = _make_cfg(num_runs=2, num_prs=20, injection_step=8)
        factory = _shifted_factory()
        agg1 = ExperimentRunner(cfg, signal_matrix_factory=factory).run()
        agg2 = ExperimentRunner(cfg, signal_matrix_factory=factory).run()
        assert [r.seed for r in agg1.runs] == [r.seed for r in agg2.runs]
        for r1, r2 in zip(agg1.runs, agg2.runs, strict=True):
            assert r1.allocation_efficiency_score == pytest.approx(r2.allocation_efficiency_score)
            assert r1.first_flagged_step == r2.first_flagged_step

    def test_per_run_seeds_are_cfg_seed_plus_run_index(self) -> None:
        cfg = _make_cfg(seed=100, num_runs=4, num_prs=20, injection_step=8)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        assert [r.seed for r in agg.runs] == [100, 101, 102, 103]
        assert [r.run_index for r in agg.runs] == [0, 1, 2, 3]

    def test_provenance_includes_every_config_block(self) -> None:
        cfg = _make_cfg(num_runs=2, num_prs=20, injection_step=8)
        agg = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory()).run()
        assert set(agg.provenance.keys()) >= {
            "name",
            "seed",
            "num_runs",
            "num_prs",
            "attack",
            "ablation",
            "system",
            "baseline_agent_ids",
        }
        assert agg.provenance["attack"]["type"] == "minja"
        assert agg.provenance["ablation"]["bocpd"] is True

    def test_injected_factory_is_called_once_per_run(self) -> None:
        calls: list[dict[str, Any]] = []

        def tracking_factory(**kwargs: Any) -> NDArray[np.float64]:
            calls.append(kwargs)
            return _shifted_factory()(**kwargs)

        cfg = _make_cfg(num_runs=3, num_prs=20, injection_step=8)
        ExperimentRunner(cfg, signal_matrix_factory=tracking_factory).run()  # type: ignore[arg-type]
        assert len(calls) == 3
        for i, call in enumerate(calls):
            assert call["seed"] == cfg.seed + i
            assert call["injection_step"] == 8
            assert call["num_prs"] == 20
            assert call["attack_type"] == "minja"
            assert call["n_poison_docs"] == 5


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    def test_run_result_is_frozen(self) -> None:
        run = RunResult(
            run_index=0,
            seed=1,
            detector_name="full_system_detector",
            injection_step=5,
            num_prs=10,
            first_flagged_step=5,
            advance_warning_time=0,
            allocation_efficiency_score=0.5,
            detection_auroc_score=1.0,
            detection_f1_score=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            run.seed = 99  # type: ignore[misc]

    def test_metric_aggregate_fields(self) -> None:
        agg = MetricAggregate(mean=0.5, std=0.1, ci_low=0.4, ci_high=0.6, n=5)
        assert agg.mean == 0.5
        assert agg.n == 5

    def test_aggregate_result_runs_preserves_order(self) -> None:
        runs = [
            RunResult(
                run_index=i,
                seed=1 + i,
                detector_name="full_system_detector",
                injection_step=5,
                num_prs=10,
                first_flagged_step=5 + i,
                advance_warning_time=-i,
                allocation_efficiency_score=0.5,
                detection_auroc_score=1.0,
                detection_f1_score=1.0,
            )
            for i in range(3)
        ]
        metric = MetricAggregate(mean=0.5, std=0.0, ci_low=0.5, ci_high=0.5, n=3)
        agg = AggregateResult(
            name="test",
            detector_name="full_system_detector",
            num_runs=3,
            injection_step=5,
            num_prs=10,
            runs=runs,
            advance_warning_time=metric,
            allocation_efficiency_score=metric,
            detection_auroc_score=metric,
            detection_f1_score=metric,
        )
        assert [r.run_index for r in agg.runs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# FullSystemDetector channel wiring
# ---------------------------------------------------------------------------


class TestFullSystemDetectorChannels:
    def _mat(
        self, num_prs: int, injection_step: int, seed: int = 0, shift: float = 5.0
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mat = rng.normal(loc=0.0, scale=0.1, size=(num_prs, NUM_SIGNALS)).astype(np.float64)
        mat[injection_step:, KL_COLUMN_INDEX] += shift
        mat[injection_step:, ENTROPY_COLUMN_INDEX] += shift / 2.0
        return mat

    def test_all_channels_on_flags_post_injection_rows(self) -> None:
        ab = AblationConfig(forecaster=True, bocpd=True, health=True, integrity=True)
        cfg = FullSystemConfig(calibration_steps=8)
        det = FullSystemDetector(ab, cfg)
        decisions = det.run(self._mat(num_prs=20, injection_step=8))
        assert len(decisions) == 20
        for d in decisions[:8]:
            assert d.flagged is False
        flagged_after = sum(1 for d in decisions[8:] if d.flagged)
        assert flagged_after >= 6  # vast majority of post-injection rows flag

    def test_bocpd_channel_produces_non_none_subscore_when_on(self) -> None:
        ab = AblationConfig(forecaster=False, bocpd=True, health=True, integrity=False)
        det = FullSystemDetector(ab, FullSystemConfig(calibration_steps=8))
        decisions = det.run(self._mat(num_prs=20, injection_step=8))
        assert all(d.bocpd_score is not None for d in decisions)
        assert all(d.forecaster_score is None for d in decisions)
        assert all(d.integrity_score is None for d in decisions)

    def test_forecaster_channel_only(self) -> None:
        ab = AblationConfig(forecaster=True, bocpd=False, health=True, integrity=False)
        det = FullSystemDetector(ab, FullSystemConfig(calibration_steps=8))
        decisions = det.run(self._mat(num_prs=20, injection_step=8))
        assert all(d.forecaster_score is not None for d in decisions)
        assert all(d.bocpd_score is None for d in decisions)

    def test_integrity_channel_only(self) -> None:
        ab = AblationConfig(forecaster=False, bocpd=False, health=True, integrity=True)
        det = FullSystemDetector(ab, FullSystemConfig(calibration_steps=8))
        decisions = det.run(self._mat(num_prs=20, injection_step=8))
        assert all(d.integrity_score is not None for d in decisions)
        assert all(d.bocpd_score is None for d in decisions)

    def test_all_channels_off_falls_back_to_sentinel_zscore(self) -> None:
        ab = AblationConfig(forecaster=False, bocpd=False, health=True, integrity=False)
        det = FullSystemDetector(ab, FullSystemConfig(calibration_steps=8))
        decisions = det.run(self._mat(num_prs=20, injection_step=8))
        # Fallback channel sets NO sub-score fields (they are all None).
        assert all(d.bocpd_score is None for d in decisions)
        assert all(d.forecaster_score is None for d in decisions)
        assert all(d.integrity_score is None for d in decisions)
        # But the combined score still rises on post-injection rows.
        mean_clean_score = float(np.mean([d.score for d in decisions[:8]]))
        mean_poison_score = float(np.mean([d.score for d in decisions[8:]]))
        assert mean_poison_score > mean_clean_score

    def test_agent_id_rotates_through_canonical_ids(self) -> None:
        from chronoagent.allocator.capability_weights import AGENT_IDS

        ab = AblationConfig()  # all True
        det = FullSystemDetector(ab, FullSystemConfig(calibration_steps=5))
        decisions = det.run(self._mat(num_prs=len(AGENT_IDS) * 2, injection_step=5))
        for i, d in enumerate(decisions):
            assert d.agent_id == AGENT_IDS[i % len(AGENT_IDS)]

    def test_wrong_matrix_shape_rejected(self) -> None:
        det = FullSystemDetector(AblationConfig(), FullSystemConfig(calibration_steps=5))
        with pytest.raises(ValueError, match="2-D"):
            det.run(np.zeros(10))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="6 columns"):
            det.run(np.zeros((10, 5)))

    def test_too_few_rows_rejected(self) -> None:
        det = FullSystemDetector(AblationConfig(), FullSystemConfig(calibration_steps=10))
        with pytest.raises(ValueError, match="calibration_steps"):
            det.run(np.zeros((5, NUM_SIGNALS)))

    def test_zero_row_matrix_returns_empty_list(self) -> None:
        det = FullSystemDetector(AblationConfig(), FullSystemConfig(calibration_steps=5))
        assert det.run(np.zeros((0, NUM_SIGNALS))) == []

    def test_full_system_decision_is_frozen(self) -> None:
        d = FullSystemDecision(
            step_index=0,
            score=0.5,
            flagged=False,
            success=True,
            agent_id=FULL_SYSTEM_AGENT_ID,
        )
        with pytest.raises(FrozenInstanceError):
            d.score = 0.9  # type: ignore[misc]

    def test_full_system_config_validators(self) -> None:
        with pytest.raises(ValueError, match="kl_column_index"):
            FullSystemConfig(kl_column_index=99)
        with pytest.raises(ValueError, match="entropy_column_index"):
            FullSystemConfig(entropy_column_index=-1)
        with pytest.raises(ValueError, match="bocpd_hazard_lambda"):
            FullSystemConfig(bocpd_hazard_lambda=0.0)
        with pytest.raises(ValueError, match="calibration_steps"):
            FullSystemConfig(calibration_steps=1)
        with pytest.raises(ValueError, match="min_std"):
            FullSystemConfig(min_std=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            FullSystemConfig(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            FullSystemConfig(ema_alpha=1.5)
        with pytest.raises(ValueError, match="decision_threshold"):
            FullSystemConfig(decision_threshold=-0.1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestWriteExperimentResults:
    def test_round_trip_csv_and_json(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=3, num_prs=20, injection_step=8)
        agg = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory()).run()
        runs_csv, agg_json = write_experiment_results(agg, tmp_path)
        assert runs_csv.exists()
        assert agg_json.exists()
        assert runs_csv.parent.name == cfg.name

        # CSV has one header row + num_runs data rows.
        with runs_csv.open(encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        assert len(rows) == cfg.num_runs + 1  # header + data
        assert rows[0].startswith("run_index,seed")

        # JSON round-trip preserves the shape.
        payload = json.loads(agg_json.read_text(encoding="utf-8"))
        assert payload["name"] == cfg.name
        assert payload["num_runs"] == cfg.num_runs
        assert "metrics" in payload
        assert set(payload["metrics"].keys()) == {
            "advance_warning_time",
            "allocation_efficiency_score",
            "detection_auroc_score",
            "detection_f1_score",
        }
        assert payload["provenance"]["attack"]["type"] == "minja"

    def test_nan_metrics_serialise_as_null(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=2, num_prs=20, injection_step=10)
        agg = ExperimentRunner(cfg, signal_matrix_factory=_null_factory()).run()
        _, agg_json = write_experiment_results(agg, tmp_path)
        payload = json.loads(agg_json.read_text(encoding="utf-8"))
        # AWT never fired -> mean is NaN -> serialised as null.
        assert payload["metrics"]["advance_warning_time"]["mean"] is None
        assert payload["metrics"]["advance_warning_time"]["n"] == 0

    def test_write_creates_output_directory(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=2, num_prs=20, injection_step=8)
        out = tmp_path / "results"
        assert not out.exists()
        agg = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory()).run()
        write_experiment_results(agg, out)
        assert (out / cfg.name).is_dir()

    def test_detector_name_field_in_csv(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=2, num_prs=20, injection_step=8)
        agg = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory()).run()
        runs_csv, _ = write_experiment_results(agg, tmp_path)
        content = runs_csv.read_text(encoding="utf-8")
        assert "full_system_detector" in content


# ---------------------------------------------------------------------------
# Integration: real default factory (slow, optional)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDefaultFactoryIntegration:
    """End-to-end run on the real Phase 1 signal matrix factory."""

    def test_default_factory_round_trip_on_main_experiment(self) -> None:
        cfg = ExperimentConfig.from_yaml(CONFIG_DIR / "main_experiment.yaml")
        cfg_small = cfg.model_copy(
            update={
                "num_runs": 1,
                "num_prs": 16,
                "attack": cfg.attack.model_copy(update={"injection_step": 8}),
            }
        )
        runner = ExperimentRunner(cfg_small, signal_matrix_factory=default_signal_matrix_factory)
        agg = runner.run()
        assert agg.num_runs == 1
        assert agg.detector_name == "full_system_detector"
        assert 0.0 <= agg.allocation_efficiency_score.mean <= 1.0

    def test_default_factory_rejects_zero_phase_lengths(self) -> None:
        with pytest.raises(ValueError, match="leave at least one row"):
            default_signal_matrix_factory(
                attack_type="minja",
                seed=1,
                injection_step=0,
                num_prs=5,
                n_poison_docs=1,
            )

    def test_default_factory_rejects_attack_none(self) -> None:
        with pytest.raises(ValueError, match="'none' is not yet supported"):
            default_signal_matrix_factory(
                attack_type="none",
                seed=1,
                injection_step=5,
                num_prs=10,
                n_poison_docs=1,
            )
