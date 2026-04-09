"""KL-divergence calibrator for retrieval embedding distributions (task 1.3).

Fits a diagonal multivariate Gaussian on clean retrieval embeddings during a
calibration phase, then computes per-step KL divergence from that baseline.

KL divergence increases when retrieved documents shift away from the clean
distribution — the expected effect of memory-poisoning attacks.

Analytic formula for diagonal Gaussians q = N(μ_q, diag(σ_q²)) vs
p = N(μ_p, diag(σ_p²)):

    KL(q ‖ p) = ½ Σ_i [log(σ_p_i² / σ_q_i²)
                         + (σ_q_i² + (μ_q_i − μ_p_i)²) / σ_p_i²
                         − 1]

This equals ``scipy.stats.entropy(q_grid, p_grid)`` evaluated on a fine
uniform grid in the limit of grid resolution → ∞.  The analytic form is used
here for numerical stability and O(d) efficiency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_kl  # KL(p‖q) = entropy(p, q)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

#: Small value added to variance estimates to prevent division by zero.
_REG: float = 1e-6

#: Number of grid points used by :func:`kl_gaussians_scipy` for validation.
_GRID_POINTS: int = 500


def _fit_diagonal_gaussian(
    embeddings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (mean, variance) for a diagonal Gaussian fit to *embeddings*.

    Args:
        embeddings: Shape ``(N, D)`` float64 array, N samples, D dimensions.

    Returns:
        Tuple of ``(mean, var)`` each of shape ``(D,)``.  Variance is
        regularised by adding :data:`_REG` to prevent zero-variance dims.
    """
    mean: NDArray[np.float64] = embeddings.mean(axis=0)
    var: NDArray[np.float64] = embeddings.var(axis=0) + _REG
    return mean, var


def _kl_diagonal_gaussians(
    mean_q: NDArray[np.float64],
    var_q: NDArray[np.float64],
    mean_p: NDArray[np.float64],
    var_p: NDArray[np.float64],
) -> float:
    """Analytic KL(q ‖ p) for diagonal multivariate Gaussians.

    Guaranteed non-negative by numerical clipping of per-dimension terms.

    Args:
        mean_q: Shape ``(D,)`` — mean of query distribution q.
        var_q:  Shape ``(D,)`` — per-dim variance of q (must be > 0).
        mean_p: Shape ``(D,)`` — mean of baseline distribution p.
        var_p:  Shape ``(D,)`` — per-dim variance of p (must be > 0).

    Returns:
        Scalar KL divergence ≥ 0.
    """
    kl_terms: NDArray[np.float64] = 0.5 * (
        np.log(var_p / var_q)
        + (var_q + (mean_q - mean_p) ** 2) / var_p
        - 1.0
    )
    # Clip individual terms to 0 to absorb floating-point rounding below zero.
    return float(np.sum(np.maximum(kl_terms, 0.0)))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def kl_gaussians_scipy(
    mean_q: NDArray[np.float64],
    std_q: NDArray[np.float64],
    mean_p: NDArray[np.float64],
    std_p: NDArray[np.float64],
    *,
    n_grid: int = _GRID_POINTS,
) -> float:
    """Compute KL(q ‖ p) via ``scipy.stats.entropy`` on a discretised grid.

    This is a reference implementation used for unit-testing the analytic
    :func:`_kl_diagonal_gaussians`.  It works per-dimension, summing the
    result across all dimensions.

    Args:
        mean_q: Shape ``(D,)`` — mean of query distribution.
        std_q:  Shape ``(D,)`` — standard deviation of query distribution.
        mean_p: Shape ``(D,)`` — mean of baseline distribution.
        std_p:  Shape ``(D,)`` — standard deviation of baseline distribution.
        n_grid: Number of grid points per dimension (default 500).

    Returns:
        Approximate KL divergence ≥ 0.  Converges to the analytic value as
        *n_grid* → ∞.
    """
    from scipy.stats import norm  # local import — only used for validation

    total: float = 0.0
    d: int = mean_q.shape[0]
    for i in range(d):
        lo = min(mean_q[i] - 4.0 * std_q[i], mean_p[i] - 4.0 * std_p[i])
        hi = max(mean_q[i] + 4.0 * std_q[i], mean_p[i] + 4.0 * std_p[i])
        grid = np.linspace(lo, hi, n_grid)
        dx = grid[1] - grid[0]
        q_pdf = norm.pdf(grid, loc=mean_q[i], scale=std_q[i]) * dx
        p_pdf = norm.pdf(grid, loc=mean_p[i], scale=std_p[i]) * dx
        total += float(scipy_kl(q_pdf, p_pdf))
    return total


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class KLCalibrator:
    """Fits a Gaussian baseline on clean embeddings, then computes per-step KL.

    Usage::

        calibrator = KLCalibrator(n_calibration=20)

        # --- calibration phase (clean run) ---
        for embeddings in clean_step_embeddings:
            calibrator.update(embeddings)          # shape (k, D)
            if calibrator.is_calibrated:
                break

        # --- monitoring phase ---
        for embeddings in run_embeddings:
            kl = calibrator.compute_kl(embeddings)  # float ≥ 0
            signals.kl_divergence = kl

    Attributes:
        n_calibration: Number of steps collected before calibration is locked.
        is_calibrated: ``True`` once the baseline Gaussian has been fitted.
    """

    def __init__(
        self,
        n_calibration: int = 20,
        reg: float = _REG,
    ) -> None:
        """Initialise the calibrator.

        Args:
            n_calibration: Minimum number of clean steps required to fit the
                baseline.  Defaults to 20 (matches PLAN.md signal reference).
            reg: Regularisation constant added to per-dimension variance to
                prevent numerical instability.  Defaults to 1e-6.
        """
        if n_calibration < 1:
            raise ValueError(f"n_calibration must be ≥ 1, got {n_calibration}")
        if reg < 0:
            raise ValueError(f"reg must be ≥ 0, got {reg}")

        self.n_calibration: int = n_calibration
        self._reg: float = reg
        self._buffer: list[NDArray[np.float64]] = []
        self._baseline_mean: NDArray[np.float64] | None = None
        self._baseline_var: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """``True`` once the baseline Gaussian has been fitted."""
        return self._baseline_mean is not None

    @property
    def n_steps_collected(self) -> int:
        """Number of calibration steps accumulated so far."""
        return len(self._buffer)

    @property
    def baseline_mean(self) -> NDArray[np.float64] | None:
        """Baseline Gaussian mean (shape ``(D,)``), or ``None`` if not yet calibrated."""
        return self._baseline_mean

    @property
    def baseline_var(self) -> NDArray[np.float64] | None:
        """Baseline Gaussian variance (shape ``(D,)``), or ``None`` if not yet calibrated."""
        return self._baseline_var

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def update(self, embeddings: NDArray[np.float64]) -> None:
        """Add retrieval embeddings from one step to the calibration buffer.

        Once :attr:`n_calibration` steps have been collected the baseline
        Gaussian is fitted and :attr:`is_calibrated` becomes ``True``.
        Subsequent calls are ignored.

        Args:
            embeddings: Shape ``(k, D)`` float64 array — the ``k`` retrieved
                document embeddings for a single pipeline step.

        Raises:
            ValueError: If *embeddings* has fewer than 2 dimensions or is
                empty.
        """
        if self.is_calibrated:
            return

        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] == 0:
            raise ValueError(
                f"embeddings must be a non-empty 2-D array, got shape {emb.shape}"
            )

        self._buffer.append(emb)

        if len(self._buffer) >= self.n_calibration:
            self._fit_baseline()

    def _fit_baseline(self) -> None:
        """Fit the baseline diagonal Gaussian from the calibration buffer."""
        all_embeddings: NDArray[np.float64] = np.concatenate(self._buffer, axis=0)
        self._baseline_mean, self._baseline_var = _fit_diagonal_gaussian(
            all_embeddings
        )
        # Ensure regularisation is applied with instance-level reg.
        self._baseline_var = self._baseline_var - _REG + self._reg

    # ------------------------------------------------------------------
    # Per-step KL computation
    # ------------------------------------------------------------------

    def compute_kl(self, embeddings: NDArray[np.float64]) -> float:
        """Compute KL divergence of *embeddings* vs the calibrated baseline.

        Returns ``0.0`` before calibration is complete or when *embeddings*
        is empty.

        Args:
            embeddings: Shape ``(k, D)`` float64 array — retrieved document
                embeddings for the current step.

        Returns:
            KL divergence ≥ 0.  Higher values indicate greater distributional
            shift from the clean baseline.
        """
        if not self.is_calibrated:
            return 0.0

        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] == 0:
            return 0.0

        step_mean, step_var = _fit_diagonal_gaussian(emb)
        # Apply instance-level regularisation.
        step_var = step_var - _REG + self._reg

        assert self._baseline_mean is not None  # type narrowing
        assert self._baseline_var is not None

        return _kl_diagonal_gaussians(
            step_mean,
            step_var,
            self._baseline_mean,
            self._baseline_var,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_calibrated:
            status = "calibrated"
        else:
            status = f"{self.n_steps_collected}/{self.n_calibration} steps"
        return f"KLCalibrator(n_calibration={self.n_calibration}, status={status!r})"
