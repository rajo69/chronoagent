"""Unit tests for kl_divergence.py — edge cases and known distributions (tasks 3.2, 3.7)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from chronoagent.monitor.kl_divergence import (
    KLCalibrator,
    _fit_diagonal_gaussian,
    _kl_diagonal_gaussians,
    kl_gaussians_scipy,
)

# ---------------------------------------------------------------------------
# _fit_diagonal_gaussian
# ---------------------------------------------------------------------------


class TestFitDiagonalGaussian:
    def test_mean_correct_1d(self) -> None:
        emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        mean, var = _fit_diagonal_gaussian(emb)
        assert mean == pytest.approx([2.0, 3.0])

    def test_var_regularised(self) -> None:
        """Variance must be strictly positive after regularisation."""
        emb = np.ones((5, 3), dtype=np.float64)  # zero empirical variance
        _, var = _fit_diagonal_gaussian(emb)
        assert (var > 0).all()

    def test_single_sample_var_regularised(self) -> None:
        """Single embedding row — variance is 0 + reg."""
        emb = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        _, var = _fit_diagonal_gaussian(emb)
        assert (var > 0).all()

    def test_output_shapes(self) -> None:
        n, d = 10, 8
        emb = np.random.default_rng(0).standard_normal((n, d))
        mean, var = _fit_diagonal_gaussian(emb)
        assert mean.shape == (d,)
        assert var.shape == (d,)

    def test_var_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((20, 4))
        mean, var = _fit_diagonal_gaussian(emb)
        expected_var = emb.var(axis=0) + 1e-6
        assert var == pytest.approx(expected_var)


# ---------------------------------------------------------------------------
# _kl_diagonal_gaussians — analytic formula
# ---------------------------------------------------------------------------


class TestKLDiagonalGaussians:
    def test_identical_distributions_kl_is_zero(self) -> None:
        """KL(p ‖ p) = 0 for identical distributions."""
        mean = np.array([1.0, -1.0, 0.0])
        var = np.array([2.0, 0.5, 1.0])
        kl = _kl_diagonal_gaussians(mean, var, mean, var)
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_kl_non_negative(self) -> None:
        """KL divergence must always be ≥ 0."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            d = rng.integers(1, 10)
            mean_q = rng.standard_normal(d)
            var_q = rng.random(d) + 0.01
            mean_p = rng.standard_normal(d)
            var_p = rng.random(d) + 0.01
            kl = _kl_diagonal_gaussians(mean_q, var_q, mean_p, var_p)
            assert kl >= 0.0, f"KL was negative: {kl}"

    def test_known_1d_value(self) -> None:
        """KL(N(0,1) ‖ N(0,2)) = (1 + 0)/2 + log(sqrt(2)) - 0.5 = 0.5 * (log2 + 1/2 - 1)."""
        # KL(N(μ_q, σ_q²) ‖ N(μ_p, σ_p²)) = 0.5*(log(σ_p²/σ_q²) + σ_q²/σ_p² + (μ_q-μ_p)²/σ_p² - 1)
        # q = N(0,1): mean=0, var=1; p = N(0,2): mean=0, var=2
        # KL = 0.5*(log(2/1) + 1/2 + 0 - 1) = 0.5*(ln2 - 0.5)
        mean_q = np.array([0.0])
        var_q = np.array([1.0])
        mean_p = np.array([0.0])
        var_p = np.array([2.0])
        expected = 0.5 * (np.log(2.0) + 0.5 - 1.0)
        kl = _kl_diagonal_gaussians(mean_q, var_q, mean_p, var_p)
        assert kl == pytest.approx(expected, rel=1e-6)

    def test_mean_shift_increases_kl(self) -> None:
        """Shifting the mean of q away from p should increase KL."""
        var = np.array([1.0])
        kl_close = _kl_diagonal_gaussians(np.array([0.1]), var, np.array([0.0]), var)
        kl_far = _kl_diagonal_gaussians(np.array([5.0]), var, np.array([0.0]), var)
        assert kl_far > kl_close

    def test_multidimensional_sum(self) -> None:
        """KL for D-dim diagonal Gaussian = sum of per-dim KL."""
        rng = np.random.default_rng(99)
        d = 5
        mean_q = rng.standard_normal(d)
        var_q = rng.random(d) + 0.1
        mean_p = rng.standard_normal(d)
        var_p = rng.random(d) + 0.1

        kl_total = _kl_diagonal_gaussians(mean_q, var_q, mean_p, var_p)
        kl_sum = sum(
            float(
                _kl_diagonal_gaussians(
                    mean_q[i : i + 1], var_q[i : i + 1], mean_p[i : i + 1], var_p[i : i + 1]
                )
            )
            for i in range(d)
        )
        assert kl_total == pytest.approx(kl_sum, rel=1e-6)


# ---------------------------------------------------------------------------
# kl_gaussians_scipy — numeric validation of analytic formula
# ---------------------------------------------------------------------------


class TestKLGaussiansScipy:
    def test_matches_analytic_single_dim(self) -> None:
        """Scipy grid approximation should agree with analytic formula to ~1%."""
        mean_q = np.array([0.0])
        std_q = np.array([1.0])
        mean_p = np.array([2.0])
        std_p = np.array([1.5])

        scipy_val = kl_gaussians_scipy(mean_q, std_q, mean_p, std_p, n_grid=2000)

        var_q = std_q**2
        var_p = std_p**2
        analytic_val = _kl_diagonal_gaussians(mean_q, var_q, mean_p, var_p)

        assert scipy_val == pytest.approx(analytic_val, rel=0.01)

    def test_matches_analytic_multidim(self) -> None:
        """Multi-dim scipy approximation should match analytic to ~2%."""
        rng = np.random.default_rng(13)
        d = 3
        mean_q = rng.standard_normal(d)
        std_q = rng.random(d) + 0.5
        mean_p = rng.standard_normal(d)
        std_p = rng.random(d) + 0.5

        scipy_val = kl_gaussians_scipy(mean_q, std_q, mean_p, std_p, n_grid=1000)
        analytic_val = _kl_diagonal_gaussians(mean_q, std_q**2, mean_p, std_p**2)

        assert scipy_val == pytest.approx(analytic_val, rel=0.02)

    def test_identical_distributions_near_zero(self) -> None:
        mean = np.array([1.0, -1.0])
        std = np.array([0.5, 2.0])
        scipy_val = kl_gaussians_scipy(mean, std, mean, std)
        assert scipy_val == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# KLCalibrator
# ---------------------------------------------------------------------------


class TestKLCalibratorInit:
    def test_default_not_calibrated(self) -> None:
        cal = KLCalibrator()
        assert not cal.is_calibrated
        assert cal.n_steps_collected == 0

    def test_n_calibration_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_calibration"):
            KLCalibrator(n_calibration=0)

    def test_negative_reg_raises(self) -> None:
        with pytest.raises(ValueError, match="reg"):
            KLCalibrator(reg=-1.0)

    def test_zero_reg_allowed(self) -> None:
        """reg=0 is allowed — the code handles it via clipping."""
        cal = KLCalibrator(reg=0.0)
        assert cal is not None

    def test_repr_before_calibration(self) -> None:
        cal = KLCalibrator(n_calibration=5)
        r = repr(cal)
        assert "0/5" in r

    def test_repr_after_calibration(self) -> None:
        cal = KLCalibrator(n_calibration=1)
        cal.update(np.ones((3, 4), dtype=np.float64))
        assert "calibrated" in repr(cal)


class TestKLCalibratorUpdate:
    def _clean_embeddings(self, n: int = 3, d: int = 4, seed: int = 0) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((n, d)) for _ in range(20)]

    def test_calibrated_after_n_steps(self) -> None:
        cal = KLCalibrator(n_calibration=5)
        batches = self._clean_embeddings()
        for b in batches[:5]:
            cal.update(b)
        assert cal.is_calibrated

    def test_not_calibrated_before_n_steps(self) -> None:
        cal = KLCalibrator(n_calibration=5)
        batches = self._clean_embeddings()
        for b in batches[:4]:
            cal.update(b)
        assert not cal.is_calibrated

    def test_update_ignored_after_calibration(self) -> None:
        cal = KLCalibrator(n_calibration=3)
        batches = self._clean_embeddings()
        for b in batches[:3]:
            cal.update(b)
        mean_after_calib = cal.baseline_mean.copy()  # type: ignore[union-attr]
        # Add 10 more different batches
        for b in batches[3:13]:
            cal.update(b)
        assert cal.baseline_mean == pytest.approx(mean_after_calib)

    def test_update_rejects_1d_array(self) -> None:
        cal = KLCalibrator()
        with pytest.raises(ValueError, match="2-D"):
            cal.update(np.array([1.0, 2.0, 3.0]))

    def test_update_rejects_empty_array(self) -> None:
        cal = KLCalibrator()
        with pytest.raises(ValueError, match="2-D"):
            cal.update(np.empty((0, 4)))

    def test_n_steps_collected_increments(self) -> None:
        cal = KLCalibrator(n_calibration=10)
        batches = self._clean_embeddings()
        for i, b in enumerate(batches[:5]):
            cal.update(b)
            assert cal.n_steps_collected == i + 1

    def test_baseline_mean_and_var_properties(self) -> None:
        cal = KLCalibrator(n_calibration=3)
        batches = self._clean_embeddings(n=4, d=6)
        for b in batches[:3]:
            cal.update(b)
        assert cal.baseline_mean is not None
        assert cal.baseline_var is not None
        assert cal.baseline_mean.shape == (6,)
        assert cal.baseline_var.shape == (6,)
        assert (cal.baseline_var > 0).all()


class TestKLCalibratorComputeKL:
    def _calibrate(self, cal: KLCalibrator, d: int = 4, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(cal.n_calibration):
            cal.update(rng.standard_normal((5, d)))

    def test_returns_zero_before_calibration(self) -> None:
        cal = KLCalibrator(n_calibration=5)
        emb = np.ones((3, 4), dtype=np.float64)
        assert cal.compute_kl(emb) == 0.0

    def test_returns_zero_for_empty_embeddings_after_calib(self) -> None:
        cal = KLCalibrator(n_calibration=3)
        self._calibrate(cal)
        assert cal.compute_kl(np.empty((0, 4))) == 0.0

    def test_returns_zero_for_1d_after_calib(self) -> None:
        cal = KLCalibrator(n_calibration=3)
        self._calibrate(cal)
        assert cal.compute_kl(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0

    def test_clean_distribution_low_kl(self) -> None:
        """Embeddings from the same distribution as the baseline → low KL."""
        rng = np.random.default_rng(5)
        d = 8
        cal = KLCalibrator(n_calibration=10)
        for _ in range(10):
            cal.update(rng.standard_normal((5, d)))
        # Draw from the same distribution
        clean = rng.standard_normal((20, d))
        kl = cal.compute_kl(clean)
        assert kl >= 0.0
        assert kl < 5.0  # loose upper bound for same-distribution sample

    def test_shifted_distribution_higher_kl(self) -> None:
        """Mean-shifted embeddings should produce higher KL than clean ones."""
        rng = np.random.default_rng(42)
        d = 8
        cal = KLCalibrator(n_calibration=10)
        for _ in range(10):
            cal.update(rng.standard_normal((5, d)))

        clean = rng.standard_normal((20, d))  # same distribution
        shifted = rng.standard_normal((20, d)) + 10.0  # very different mean

        kl_clean = cal.compute_kl(clean)
        kl_shifted = cal.compute_kl(shifted)
        assert kl_shifted > kl_clean

    def test_kl_non_negative(self) -> None:
        """KL must always be ≥ 0."""
        rng = np.random.default_rng(77)
        d = 4
        cal = KLCalibrator(n_calibration=5)
        self._calibrate(cal, d=d)
        for _ in range(30):
            emb = rng.standard_normal((rng.integers(1, 10), d))
            assert cal.compute_kl(emb) >= 0.0

    def test_zero_variance_embeddings_handled(self) -> None:
        """All-identical embeddings (zero variance) should not crash."""
        cal = KLCalibrator(n_calibration=3)
        d = 4
        # Calibrate with diverse data
        rng = np.random.default_rng(1)
        for _ in range(3):
            cal.update(rng.standard_normal((5, d)))

        # Compute KL for constant embeddings (zero empirical variance)
        const_emb = np.ones((5, d), dtype=np.float64)
        kl = cal.compute_kl(const_emb)
        assert kl >= 0.0
        assert np.isfinite(kl)

    def test_single_sample_embedding_handled(self) -> None:
        """Single embedding (1, D) should not crash."""
        cal = KLCalibrator(n_calibration=3)
        d = 4
        rng = np.random.default_rng(2)
        for _ in range(3):
            cal.update(rng.standard_normal((5, d)))

        single = rng.standard_normal((1, d))
        kl = cal.compute_kl(single)
        assert kl >= 0.0
        assert np.isfinite(kl)

    def test_n_calibration_1_works(self) -> None:
        """KLCalibrator should work with n_calibration=1."""
        cal = KLCalibrator(n_calibration=1)
        emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        cal.update(emb)
        assert cal.is_calibrated
        kl = cal.compute_kl(np.array([[5.0, 6.0]], dtype=np.float64))
        assert kl >= 0.0


# ---------------------------------------------------------------------------
# Hypothesis property tests (task 3.7)
# ---------------------------------------------------------------------------


_positive_var = arrays(
    dtype=np.float64,
    shape=st.integers(1, 8),
    elements=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False, allow_infinity=False),
)
_mean = arrays(
    dtype=np.float64,
    shape=st.integers(1, 8),
    elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)


class TestKLDivergencePropertyTests:
    @given(
        mean_q=arrays(
            np.float64,
            st.integers(1, 6),
            elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        ),
        log_var_q=arrays(
            np.float64,
            st.integers(1, 6),
            elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
        ),
        mean_p=arrays(
            np.float64,
            st.integers(1, 6),
            elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        ),
        log_var_p=arrays(
            np.float64,
            st.integers(1, 6),
            elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=200)
    def test_kl_always_non_negative(
        self,
        mean_q: np.ndarray,
        log_var_q: np.ndarray,
        mean_p: np.ndarray,
        log_var_p: np.ndarray,
    ) -> None:
        """KL divergence must always be ≥ 0 for any valid inputs."""
        d = min(mean_q.shape[0], log_var_q.shape[0], mean_p.shape[0], log_var_p.shape[0])
        var_q = np.exp(log_var_q[:d]) + 1e-6  # ensure > 0
        var_p = np.exp(log_var_p[:d]) + 1e-6
        kl = _kl_diagonal_gaussians(mean_q[:d], var_q, mean_p[:d], var_p)
        assert kl >= 0.0, f"KL was negative: {kl}"
        assert np.isfinite(kl)

    @given(
        embeddings=arrays(
            np.float64,
            st.tuples(st.integers(1, 10), st.integers(1, 8)),
            elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100)
    def test_calibrator_kl_non_negative(self, embeddings: np.ndarray) -> None:
        """KLCalibrator.compute_kl always returns a non-negative finite value."""
        cal = KLCalibrator(n_calibration=1)
        d = embeddings.shape[1]
        # Calibrate with clean data (identity-ish)
        clean = np.eye(min(d, 3), d, dtype=np.float64) + 0.01
        cal.update(clean)
        kl = cal.compute_kl(embeddings)
        assert kl >= 0.0
        assert np.isfinite(kl)
