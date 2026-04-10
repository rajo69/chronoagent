"""Online Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

Reference: Adams, R. P. & MacKay, D. J. C. (2007). Bayesian Online Changepoint
           Detection. arXiv:0710.3742.

Implements streaming BOCPD using a Normal-Inverse-Chi-Squared conjugate prior.
Each ``update`` call returns P(changepoint at t | x_{1:t}) in [0, 1].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln  # type: ignore[import-untyped]


class _StudentTPredictor:
    """Conjugate NIX predictive distribution tracking one row per run length.

    Maintains sufficient statistics (mu, kappa, alpha, beta) for every
    active run-length hypothesis.  After each observation the arrays grow
    by one element (for the newly started run).
    """

    def __init__(
        self,
        mu0: float,
        kappa0: float,
        alpha0: float,
        beta0: float,
    ) -> None:
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        # Start with a single hypothesis: run length 0, prior params.
        self._mu = np.array([mu0])
        self._kappa = np.array([kappa0])
        self._alpha = np.array([alpha0])
        self._beta = np.array([beta0])

    def pdf(self, x: float) -> NDArray[np.float64]:
        """Return predictive Student-t pdf P(x | run_length) for every run.

        Parameters
        ----------
        x:
            Scalar observation value.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_runs,) with pdf values (non-negative).
        """
        nu: NDArray[np.float64] = 2.0 * self._alpha
        sigma2: NDArray[np.float64] = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa)
        t: NDArray[np.float64] = (x - self._mu) / np.sqrt(sigma2)
        log_pdf: NDArray[np.float64] = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * sigma2)
            - ((nu + 1) / 2) * np.log1p(t**2 / nu)
        )
        return np.exp(log_pdf)  # type: ignore[no-any-return]

    def update(self, x: float) -> None:
        """Grow the sufficient-statistics arrays with a new observation.

        After this call the arrays have one extra element at index 0
        corresponding to a freshly reset run (prior params).
        """
        kappa_new: NDArray[np.float64] = self._kappa + 1.0
        mu_new: NDArray[np.float64] = (self._kappa * self._mu + x) / kappa_new
        alpha_new: NDArray[np.float64] = self._alpha + 0.5
        beta_new: NDArray[np.float64] = (
            self._beta + 0.5 * self._kappa * (x - self._mu) ** 2 / kappa_new
        )

        # Prepend a fresh run using the prior.
        self._mu = np.concatenate([[self.mu0], mu_new])
        self._kappa = np.concatenate([[self.kappa0], kappa_new])
        self._alpha = np.concatenate([[self.alpha0], alpha_new])
        self._beta = np.concatenate([[self.beta0], beta_new])

    def reset(self) -> None:
        """Reinitialise to a single run-length-0 hypothesis."""
        self._mu = np.array([self.mu0])
        self._kappa = np.array([self.kappa0])
        self._alpha = np.array([self.alpha0])
        self._beta = np.array([self.beta0])


class BOCPD:
    """Streaming Bayesian Online Changepoint Detector.

    Ingests one scalar observation at a time.  Each ``update`` call returns
    the posterior changepoint probability P(r_t = 0 | x_{1:t}) in [0, 1].

    Parameters
    ----------
    hazard_lambda:
        Expected run length in number of steps (geometric hazard H = 1/lambda).
        Longer lambda = expects fewer changepoints.
    mu0:
        Prior mean for the Normal-Inverse-Chi-Squared model.
    kappa0:
        Prior pseudo-count for the mean.
    alpha0:
        Prior shape for the variance (must be > 0).
    beta0:
        Prior scale for the variance (must be > 0).
    """

    def __init__(
        self,
        hazard_lambda: float = 50.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        if hazard_lambda <= 0:
            raise ValueError(f"hazard_lambda must be > 0, got {hazard_lambda}")
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be > 0")

        self._hazard: float = 1.0 / hazard_lambda
        self._predictor = _StudentTPredictor(mu0, kappa0, alpha0, beta0)
        # Log-posterior over run lengths.  Starts with P(r_0 = 0) = 1.
        self._log_R: NDArray[np.float64] = np.array([0.0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obs: float) -> float:
        """Incorporate a new observation and return the changepoint probability.

        Parameters
        ----------
        obs:
            Latest scalar signal value (e.g. KL divergence at this step).

        Returns
        -------
        float
            Changepoint signal in [0, 1].  Computed as
            ``H * P(x | prior) / P(x | x_{1:t-1})``, clamped to [0, 1].
            This is near zero during stable operation and spikes toward 1.0
            when the observation is far outside the learned distribution but
            plausible under the prior (i.e., the regime has shifted).
        """
        pred_probs: NDArray[np.float64] = self._predictor.pdf(obs)
        r_dist: NDArray[np.float64] = np.exp(self._log_R)

        hazard = self._hazard

        # Growth: existing run continues (multiply by 1 - hazard).
        growth: NDArray[np.float64] = r_dist * pred_probs * (1.0 - hazard)
        # Changepoint: any run resets (mass funnelled to r = 0).
        cp_mass: float = float(np.sum(r_dist * pred_probs * hazard))

        # New distribution: [P(r=0), P(r=1), P(r=2), ...]
        new_r: NDArray[np.float64] = np.concatenate([[cp_mass], growth])

        # evidence = P(x_t | x_{1:t-1}) = total probability before normalisation.
        evidence: float = float(new_r.sum())
        if evidence > 0.0:
            new_r /= evidence
        else:
            # Numerical underflow — fall back to uniform (rare).
            new_r = np.ones_like(new_r) / len(new_r)

        self._log_R = np.log(np.maximum(new_r, 1e-300))

        # Grow the predictor's sufficient statistics for the next step.
        self._predictor.update(obs)

        # Changepoint signal: hazard * P(x | prior) / P(x | x_{1:t-1}).
        # After a regime shift the long-run models are all surprised (evidence
        # is dominated by r_dist[0] * pred_probs[0]) so this ratio → 1.  During
        # stable operation pred_probs[long_run] >> pred_probs[0] so the ratio
        # is well below hazard.  Bounded to [0, 1] by construction since
        # evidence >= r_dist[0] * pred_probs[0] = hazard * pred_probs[0].
        prior_pred: float = float(pred_probs[0])
        if evidence > 0.0:
            return float(min(1.0, hazard * prior_pred / evidence))
        return 1.0

    def reset(self) -> None:
        """Reset the detector to its initial state (discard all observations)."""
        self._predictor.reset()
        self._log_R = np.array([0.0])

    @property
    def run_length_distribution(self) -> NDArray[np.float64]:
        """Current posterior over run lengths as a probability vector (copy)."""
        return np.exp(self._log_R).copy()

    @property
    def most_probable_run_length(self) -> int:
        """MAP run length estimate at the current timestep."""
        return int(np.argmax(self._log_R))
