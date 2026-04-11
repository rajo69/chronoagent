"""Unit tests for ``chronoagent.experiments.baselines.sentinel`` (task 10.3).

Coverage plan:

* :class:`SentinelConfig` -- defaults and each ``__post_init__`` validator.
* :class:`SentinelDecision` -- frozen, default agent_id, field round-trip.
* :class:`SentinelBaseline.calibrate` -- wrong-shape rejection,
  too-few-rows rejection, min_std floor handles constant columns, extra
  rows beyond ``calibration_steps`` are ignored.
* :class:`SentinelBaseline.decide` -- rejects before calibration, accepts
  both :class:`StepSignals` and numpy-vector input, wrong-shape rejection,
  z-score math, boundary (``z == threshold`` does NOT flag), success is
  the logical negation of flagged.
* :class:`SentinelBaseline.run` -- on an all-clean random matrix, no row
  flags (the variance identity bounds calibration-window max |z| by
  sqrt(N-1) = 3.0 with the default config); on a shifted post-calibration
  block, every shifted row flags; step_index is monotonic.
* **Metric integration smoke test** -- feed decisions into the 10.2
  metric functions (``allocation_efficiency``, ``detection_auroc``,
  ``detection_f1``) and assert the baseline achieves AUROC = 1.0 on a
  well-separated two-block matrix.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from chronoagent.experiments.baselines.sentinel import (
    SENTINEL_AGENT_ID,
    SentinelBaseline,
    SentinelConfig,
    SentinelDecision,
)
from chronoagent.experiments.metrics import (
    allocation_efficiency,
    detection_auroc,
    detection_f1,
)
from chronoagent.monitor.collector import NUM_SIGNALS, StepSignals

# ── SentinelConfig ───────────────────────────────────────────────────────────


class TestSentinelConfig:
    """Defaults and validator rejection cases for :class:`SentinelConfig`."""

    def test_default_values(self) -> None:
        cfg = SentinelConfig()
        assert cfg.calibration_steps == 10
        assert cfg.z_threshold == 3.0
        assert cfg.min_std == 1e-6

    def test_frozen(self) -> None:
        cfg = SentinelConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.calibration_steps = 20  # type: ignore[misc]

    def test_calibration_steps_must_be_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="calibration_steps must be >= 2"):
            SentinelConfig(calibration_steps=1)

    def test_calibration_steps_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="calibration_steps must be >= 2"):
            SentinelConfig(calibration_steps=0)

    def test_z_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="z_threshold must be > 0"):
            SentinelConfig(z_threshold=0.0)

    def test_negative_z_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="z_threshold must be > 0"):
            SentinelConfig(z_threshold=-1.0)

    def test_min_std_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min_std must be > 0"):
            SentinelConfig(min_std=0.0)


# ── SentinelDecision ─────────────────────────────────────────────────────────


class TestSentinelDecision:
    """Shape, defaults, and immutability of the per-step decision record."""

    def test_construct_with_explicit_fields(self) -> None:
        decision = SentinelDecision(step_index=3, score=1.5, flagged=False, success=True)
        assert decision.step_index == 3
        assert decision.score == 1.5
        assert decision.flagged is False
        assert decision.success is True
        assert decision.agent_id == SENTINEL_AGENT_ID

    def test_agent_id_defaults_to_sentinel_baseline(self) -> None:
        decision = SentinelDecision(step_index=0, score=0.0, flagged=False, success=True)
        assert decision.agent_id == "sentinel_baseline"

    def test_frozen(self) -> None:
        decision = SentinelDecision(step_index=0, score=0.0, flagged=False, success=True)
        with pytest.raises(FrozenInstanceError):
            decision.score = 99.0  # type: ignore[misc]

    def test_success_and_flagged_independently_settable(self) -> None:
        # The class does not enforce success = not flagged; the
        # SentinelBaseline DOES, and a separate test covers that
        # invariant at the call site.
        d = SentinelDecision(step_index=0, score=5.0, flagged=True, success=True)
        assert d.flagged is True
        assert d.success is True


# ── SentinelBaseline.calibrate ───────────────────────────────────────────────


class TestSentinelBaselineCalibrate:
    """Shape / min-size validation and the min_std floor."""

    def _clean_matrix(self, n: int = 15, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(loc=100.0, scale=5.0, size=(n, NUM_SIGNALS))

    def test_calibration_marks_baseline(self) -> None:
        baseline = SentinelBaseline()
        assert baseline.is_calibrated is False
        baseline.calibrate(self._clean_matrix())
        assert baseline.is_calibrated is True

    def test_wrong_rank_rejected(self) -> None:
        baseline = SentinelBaseline()
        with pytest.raises(ValueError, match="clean_matrix must have shape"):
            baseline.calibrate(np.zeros(10))  # 1-D not 2-D

    def test_wrong_column_count_rejected(self) -> None:
        baseline = SentinelBaseline()
        with pytest.raises(ValueError, match=f"shape \\(N, {NUM_SIGNALS}\\)"):
            baseline.calibrate(np.zeros((15, NUM_SIGNALS + 1)))

    def test_too_few_rows_rejected(self) -> None:
        baseline = SentinelBaseline()
        with pytest.raises(ValueError, match="need at least 10"):
            baseline.calibrate(np.zeros((5, NUM_SIGNALS)))

    def test_exactly_calibration_steps_rows_accepted(self) -> None:
        baseline = SentinelBaseline()
        baseline.calibrate(self._clean_matrix(n=10))  # exactly calibration_steps
        assert baseline.is_calibrated is True

    def test_extra_rows_beyond_window_ignored(self) -> None:
        """Rows beyond calibration_steps must not influence the baseline."""
        m = self._clean_matrix(n=10, seed=1)
        extended = np.vstack([m, np.full((5, NUM_SIGNALS), 99999.0)])
        b1 = SentinelBaseline()
        b2 = SentinelBaseline()
        b1.calibrate(m)
        b2.calibrate(extended)
        assert b1._baseline_mean is not None
        assert b2._baseline_mean is not None
        np.testing.assert_array_equal(b1._baseline_mean, b2._baseline_mean)
        assert b1._baseline_std is not None
        assert b2._baseline_std is not None
        np.testing.assert_array_equal(b1._baseline_std, b2._baseline_std)

    def test_min_std_floor_handles_constant_column(self) -> None:
        """Constant columns get floored std so z-score math never divides by zero."""
        m = np.ones((10, NUM_SIGNALS), dtype=np.float64)
        baseline = SentinelBaseline(SentinelConfig(min_std=1e-3))
        baseline.calibrate(m)
        assert baseline._baseline_std is not None
        assert np.all(baseline._baseline_std == 1e-3)

    def test_calibrate_is_idempotent(self) -> None:
        """A second call overwrites the baseline; no residue from the first."""
        baseline = SentinelBaseline()
        m1 = self._clean_matrix(n=10, seed=1) * 10.0  # large values
        m2 = self._clean_matrix(n=10, seed=2) * 0.01  # tiny values
        baseline.calibrate(m1)
        mean1 = baseline._baseline_mean.copy()  # type: ignore[union-attr]
        baseline.calibrate(m2)
        mean2 = baseline._baseline_mean  # type: ignore[union-attr]
        assert mean2 is not None
        assert not np.allclose(mean1, mean2)


# ── SentinelBaseline.decide ──────────────────────────────────────────────────


class TestSentinelBaselineDecide:
    """Score computation, input shapes, and the un-calibrated guard."""

    def _ready_baseline(self, seed: int = 0) -> SentinelBaseline:
        baseline = SentinelBaseline()
        rng = np.random.default_rng(seed)
        baseline.calibrate(rng.normal(loc=100.0, scale=5.0, size=(15, NUM_SIGNALS)))
        return baseline

    def test_decide_before_calibrate_raises(self) -> None:
        baseline = SentinelBaseline()
        with pytest.raises(RuntimeError, match="before calibrate"):
            baseline.decide(np.zeros(NUM_SIGNALS), step_index=0)

    def test_decide_accepts_stepsignals(self) -> None:
        baseline = self._ready_baseline()
        signals = StepSignals(
            total_latency_ms=100.0,
            retrieval_count=3,
            token_count=50,
            kl_divergence=0.1,
            tool_calls=2,
            memory_query_entropy=0.5,
        )
        decision = baseline.decide(signals, step_index=12)
        assert decision.step_index == 12
        assert isinstance(decision.score, float)
        assert decision.agent_id == SENTINEL_AGENT_ID

    def test_decide_accepts_numpy_vector(self) -> None:
        baseline = self._ready_baseline()
        decision = baseline.decide(np.full(NUM_SIGNALS, 100.0), step_index=0)
        assert decision.step_index == 0

    def test_wrong_vector_shape_rejected(self) -> None:
        baseline = self._ready_baseline()
        with pytest.raises(ValueError, match=f"\\({NUM_SIGNALS},\\)"):
            baseline.decide(np.zeros(NUM_SIGNALS + 1), step_index=0)

    def test_zero_deviation_gives_zero_score(self) -> None:
        """A step exactly at the baseline mean has score 0."""
        baseline = SentinelBaseline()
        m = np.full((10, NUM_SIGNALS), 50.0)
        m[:, 0] = np.arange(10, dtype=np.float64)  # introduce variance
        baseline.calibrate(m)
        assert baseline._baseline_mean is not None
        decision = baseline.decide(baseline._baseline_mean, step_index=0)
        assert decision.score == pytest.approx(0.0)
        assert decision.flagged is False
        assert decision.success is True

    def test_boundary_exactly_threshold_does_not_flag(self) -> None:
        """``score == z_threshold`` stays unflagged (strict ``>`` semantics)."""
        baseline = SentinelBaseline(SentinelConfig(z_threshold=2.0))
        # Build a 10-row clean matrix so baseline std is exactly 1 on col 0.
        m = np.zeros((10, NUM_SIGNALS))
        # Values [-0.5, 0.5] repeated: mean 0, sample std sqrt(1/9 * 10 * 0.25)...
        # Simpler: construct with a known std and craft a signal at exactly 2σ.
        m[:, 0] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        baseline.calibrate(m)
        assert baseline._baseline_mean is not None
        assert baseline._baseline_std is not None
        vec = baseline._baseline_mean.copy()
        vec[0] = baseline._baseline_mean[0] + 2.0 * baseline._baseline_std[0]
        decision = baseline.decide(vec, step_index=0)
        assert decision.score == pytest.approx(2.0)
        assert decision.flagged is False  # strict >, boundary passes
        assert decision.success is True

    def test_success_is_negation_of_flagged(self) -> None:
        baseline = self._ready_baseline()
        # Far-from-mean vector flags.
        assert baseline._baseline_mean is not None
        vec = baseline._baseline_mean + 100.0
        d = baseline.decide(vec, step_index=0)
        assert d.flagged is True
        assert d.success is False

    def test_score_is_max_across_signals(self) -> None:
        baseline = SentinelBaseline()
        m = np.zeros((10, NUM_SIGNALS))
        m[:, 0] = np.arange(10, dtype=np.float64)  # std on col 0
        m[:, 1] = np.arange(10, dtype=np.float64) * 2  # larger std on col 1
        baseline.calibrate(m)
        # Craft a vector at +5σ on col 0 and +1σ on col 1.
        assert baseline._baseline_mean is not None
        assert baseline._baseline_std is not None
        vec = baseline._baseline_mean.copy()
        vec[0] = baseline._baseline_mean[0] + 5.0 * baseline._baseline_std[0]
        vec[1] = baseline._baseline_mean[1] + 1.0 * baseline._baseline_std[1]
        d = baseline.decide(vec, step_index=0)
        assert d.score == pytest.approx(5.0)


# ── SentinelBaseline.run ─────────────────────────────────────────────────────


class TestSentinelBaselineRun:
    """End-to-end: calibrate + decide over a full matrix."""

    def test_returns_one_decision_per_row(self) -> None:
        rng = np.random.default_rng(42)
        m = rng.normal(loc=100.0, scale=5.0, size=(20, NUM_SIGNALS))
        decisions = SentinelBaseline().run(m)
        assert len(decisions) == 20

    def test_step_indices_are_monotonic(self) -> None:
        rng = np.random.default_rng(42)
        m = rng.normal(loc=100.0, scale=5.0, size=(20, NUM_SIGNALS))
        decisions = SentinelBaseline().run(m)
        assert [d.step_index for d in decisions] == list(range(20))

    def test_clean_random_run_never_flags_calibration_window(self) -> None:
        """Variance identity bounds calibration-window max |z| by sqrt(N-1)."""
        rng = np.random.default_rng(123)
        m = rng.normal(loc=100.0, scale=5.0, size=(30, NUM_SIGNALS))
        decisions = SentinelBaseline().run(m)
        # With calibration_steps=10, the bound is sqrt(9) = 3.0 and the
        # threshold is strict >, so NO calibration-window row flags.
        for d in decisions[:10]:
            assert d.flagged is False, f"Calibration row {d.step_index} flagged: score={d.score}"

    def test_shifted_post_calibration_block_flags(self) -> None:
        """A large post-calibration shift must flag every shifted row."""
        rng = np.random.default_rng(7)
        clean = rng.normal(loc=100.0, scale=1.0, size=(10, NUM_SIGNALS))
        shifted = rng.normal(loc=200.0, scale=1.0, size=(10, NUM_SIGNALS))
        m = np.vstack([clean, shifted])
        decisions = SentinelBaseline().run(m)
        assert all(d.flagged is False for d in decisions[:10])
        assert all(d.flagged is True for d in decisions[10:])

    def test_success_mirrors_flagged_across_full_run(self) -> None:
        rng = np.random.default_rng(11)
        clean = rng.normal(loc=100.0, scale=1.0, size=(10, NUM_SIGNALS))
        shifted = rng.normal(loc=500.0, scale=1.0, size=(10, NUM_SIGNALS))
        m = np.vstack([clean, shifted])
        decisions = SentinelBaseline().run(m)
        for d in decisions:
            assert d.success is (not d.flagged)

    def test_constant_signal_run_is_stable(self) -> None:
        """Constant-valued run never flags (every row sits at the baseline mean)."""
        m = np.full((20, NUM_SIGNALS), 42.0, dtype=np.float64)
        decisions = SentinelBaseline().run(m)
        assert all(d.flagged is False for d in decisions)
        assert all(d.success is True for d in decisions)


# ── Integration with Phase 10.2 metrics ──────────────────────────────────────


class TestSentinelMetricsIntegration:
    """The baseline's decision stream must plug into the 10.2 metric fns."""

    def _two_block_matrix(self) -> np.ndarray:
        """10 clean rows + 10 strongly-shifted rows (deterministic)."""
        rng = np.random.default_rng(99)
        clean = rng.normal(loc=100.0, scale=1.0, size=(10, NUM_SIGNALS))
        shifted = rng.normal(loc=300.0, scale=1.0, size=(10, NUM_SIGNALS))
        return np.vstack([clean, shifted])

    def test_allocation_efficiency_accepts_decision_mapping(self) -> None:
        """SentinelDecision is not a Mapping, so the runner projects to dicts."""
        m = self._two_block_matrix()
        decisions = SentinelBaseline().run(m)
        rows = [
            {"step_index": d.step_index, "success": d.success, "agent_id": d.agent_id}
            for d in decisions
        ]
        eff = allocation_efficiency(rows)
        # Clean 10 all success, shifted 10 all flagged (success=False) -> 0.5.
        assert eff == pytest.approx(0.5)

    def test_detection_auroc_is_one_on_well_separated_blocks(self) -> None:
        """Paper-quality assertion: a clean|shift matrix -> AUROC = 1.0."""
        m = self._two_block_matrix()
        decisions = SentinelBaseline().run(m)
        y_true = np.array([0] * 10 + [1] * 10)
        y_scores = np.array([d.score for d in decisions])
        assert detection_auroc(y_true, y_scores) == 1.0

    def test_detection_f1_is_one_on_well_separated_blocks(self) -> None:
        """Same separation -> every shifted row flagged, F1 = 1.0."""
        m = self._two_block_matrix()
        decisions = SentinelBaseline().run(m)
        y_true = np.array([0] * 10 + [1] * 10)
        y_pred = np.array([int(d.flagged) for d in decisions])
        assert detection_f1(y_true, y_pred) == 1.0

    def test_first_flagged_step_is_injection_step(self) -> None:
        """The runner will derive detection_step = first flagged index."""
        m = self._two_block_matrix()
        decisions = SentinelBaseline().run(m)
        first_flag = next((d.step_index for d in decisions if d.flagged), -1)
        assert first_flag == 10  # row index of the shifted block
