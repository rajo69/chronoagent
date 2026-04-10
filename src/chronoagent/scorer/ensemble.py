"""Ensemble scorer combining BOCPD and Chronos anomaly scores.

Formula: health = 1 - clip(w_bocpd * bocpd_score + w_chronos * chronos_score, 0, 1)

When one component is unavailable the remaining component's weight is
promoted to 1.0 automatically (graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnsembleResult:
    """Output of the ensemble scorer for a single timestep.

    Attributes
    ----------
    health:
        Per-agent health score in [0, 1].  1 = fully healthy, 0 = anomalous.
    bocpd_score:
        Raw BOCPD changepoint probability (or ``None`` if unavailable).
    chronos_score:
        Raw Chronos anomaly score (or ``None`` if unavailable).
    w_bocpd:
        Effective BOCPD weight used (after fallback adjustment).
    w_chronos:
        Effective Chronos weight used (after fallback adjustment).
    """

    health: float
    bocpd_score: float | None
    chronos_score: float | None
    w_bocpd: float
    w_chronos: float


@dataclass
class EnsembleScorer:
    """Combines BOCPD and Chronos scores into a single health value.

    Parameters
    ----------
    w_bocpd:
        Nominal weight for BOCPD changepoint probability [0, 1].
    w_chronos:
        Nominal weight for Chronos anomaly score [0, 1].

    Both weights default to 0.5 (equal blend).  They are re-normalised
    when only one component is present.
    """

    w_bocpd: float = 0.5
    w_chronos: float = 0.5
    # Store for inspection / testing.
    last_result: EnsembleResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.w_bocpd <= 1.0):
            raise ValueError(f"w_bocpd must be in [0, 1], got {self.w_bocpd}")
        if not (0.0 <= self.w_chronos <= 1.0):
            raise ValueError(f"w_chronos must be in [0, 1], got {self.w_chronos}")

    def score(
        self,
        bocpd_score: float | None,
        chronos_score: float | None,
    ) -> EnsembleResult:
        """Compute a health score from component scores.

        Parameters
        ----------
        bocpd_score:
            Changepoint probability from BOCPD in [0, 1], or ``None``.
        chronos_score:
            Anomaly score from Chronos in [0, 1], or ``None``.

        Returns
        -------
        EnsembleResult
            Health score in [0, 1] along with component details.
        """
        w_b, w_c = self._effective_weights(bocpd_score, chronos_score)
        anomaly = 0.0
        if bocpd_score is not None:
            anomaly += w_b * bocpd_score
        if chronos_score is not None:
            anomaly += w_c * chronos_score

        health = float(max(0.0, min(1.0, 1.0 - anomaly)))

        result = EnsembleResult(
            health=health,
            bocpd_score=bocpd_score,
            chronos_score=chronos_score,
            w_bocpd=w_b,
            w_chronos=w_c,
        )
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_weights(
        self,
        bocpd_score: float | None,
        chronos_score: float | None,
    ) -> tuple[float, float]:
        """Return (w_bocpd, w_chronos) adjusted for absent components.

        If both are absent, return (0, 0) — health defaults to 1.0.
        If only one is present, its weight is 1.0.
        If both present, re-normalise the configured weights.
        """
        have_b = bocpd_score is not None
        have_c = chronos_score is not None

        if not have_b and not have_c:
            return 0.0, 0.0

        if have_b and not have_c:
            return 1.0, 0.0

        if have_c and not have_b:
            return 0.0, 1.0

        total = self.w_bocpd + self.w_chronos
        if total <= 0.0:
            return 0.5, 0.5
        return self.w_bocpd / total, self.w_chronos / total
