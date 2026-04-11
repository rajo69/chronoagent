"""Reactive Sentinel baseline for the Phase 10 experiment suite (task 10.3).

Phase 10 compares the full ChronoAgent stack (Chronos forecaster + BOCPD +
temporal health scores + memory integrity) against simpler baselines so the
paper tables in task 10.8 can isolate what each component contributes. The
Sentinel baseline stands in for the "classical" reactive anomaly-detection
approach: it inspects every completed step post-hoc, compares the observed
behavioural signals against a static per-signal baseline learned from the
first N clean steps, and flags any step whose maximum absolute z-score
exceeds a configurable threshold.

Design choices locked in
------------------------
* **Purely reactive, single-step, no memory.** The decision for step *t*
  depends only on step *t*'s signals and the clean-window calibration
  statistics. There is no BOCPD-style run-length posterior, no Chronos
  forecast residual, no rolling window, no health score. Matches PLAN.md
  task 10.3 wording: "Reactive baseline: execution trace matching, no
  temporal forecasting, no health scores".
* **Decision interface matches the runner / metrics contract.** Each step
  produces a :class:`SentinelDecision` carrying a ``success: bool`` (feeds
  :func:`~chronoagent.experiments.metrics.allocation_efficiency`), a
  ``score: float`` (max |z| across monitored signals, feeds
  :func:`~chronoagent.experiments.metrics.detection_auroc`), and a
  ``flagged: bool`` (feeds
  :func:`~chronoagent.experiments.metrics.detection_f1`). The runner
  (task 10.6) will use the index of the first flagged decision as the
  detection step for
  :func:`~chronoagent.experiments.metrics.advance_warning_time`.
* **Calibration-window steps never flag themselves.** With the default
  configuration (``calibration_steps=10``, ``z_threshold=3.0``), the
  variance identity bounds each calibration-window z-score by
  ``sqrt(N - 1) = 3.0``, and the ``>`` comparison rejects the boundary.
  The tests assert this empirically with random noise so a future change
  to the calibration math cannot silently break the invariant.
* **Deterministic.** No randomness anywhere; the same input matrix always
  produces the same decision list. The runner gets reproducibility for
  free and does not need to thread a seed through the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from chronoagent.monitor.collector import NUM_SIGNALS, StepSignals

__all__ = [
    "SENTINEL_AGENT_ID",
    "SentinelBaseline",
    "SentinelConfig",
    "SentinelDecision",
]

#: Agent identifier recorded on every :class:`SentinelDecision` so that
#: downstream aggregation (task 10.6 runner, task 10.8 tables) can
#: disambiguate baseline rows from full-system rows without re-threading
#: a label through every function.
SENTINEL_AGENT_ID: str = "sentinel_baseline"


@dataclass(frozen=True)
class SentinelConfig:
    """Parameters controlling the reactive Sentinel detector.

    Attributes:
        calibration_steps: Number of leading rows used to learn the
            baseline mean and standard deviation for every monitored
            signal. Must be ``>= 2`` so the sample standard deviation
            (``ddof=1``) is defined. Default of 10 mirrors the Phase 1
            KL calibrator.
        z_threshold: Strict upper bound on the per-step max |z|. Steps
            whose max |z| is strictly greater than this value are
            flagged. Must be ``> 0``.
        min_std: Floor applied to the per-signal standard deviation
            during calibration so constant-signal columns (several of
            the MockBackend signals are literal constants) cannot
            divide by zero. Must be strictly positive.
    """

    calibration_steps: int = 10
    z_threshold: float = 3.0
    min_std: float = 1e-6

    def __post_init__(self) -> None:
        if self.calibration_steps < 2:
            raise ValueError(f"calibration_steps must be >= 2, got {self.calibration_steps}")
        if self.z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0, got {self.z_threshold}")
        if self.min_std <= 0:
            raise ValueError(f"min_std must be > 0, got {self.min_std}")


@dataclass(frozen=True)
class SentinelDecision:
    """Per-step outcome of the reactive Sentinel detector.

    Attributes:
        step_index: 0-based step index within the run.
        score: Max absolute z-score across monitored signals at this
            step. Feeds :func:`detection_auroc` as the per-step detector
            score.
        flagged: ``True`` iff ``score > SentinelConfig.z_threshold``.
            Feeds :func:`detection_f1` as the binary prediction.
        success: ``True`` iff the step was not flagged, so the Sentinel
            gate let the task through. Feeds :func:`allocation_efficiency`
            via the mapping-style ``{"success": <bool>, ...}`` entry.
        agent_id: Always :data:`SENTINEL_AGENT_ID`. Stored on every
            decision so aggregated allocator-audit CSVs can
            disambiguate baseline vs full-system rows on a single
            column.
    """

    step_index: int
    score: float
    flagged: bool
    success: bool
    agent_id: str = SENTINEL_AGENT_ID


class SentinelBaseline:
    """Reactive signature-style baseline over a ``(T, NUM_SIGNALS)`` matrix.

    Workflow:

    1. :meth:`calibrate` learns per-signal mean and ``ddof=1`` std from a
       clean window (typically the first
       :attr:`SentinelConfig.calibration_steps` rows of a run matrix).
    2. :meth:`decide` takes one :class:`StepSignals` or 1-D float vector
       and returns a :class:`SentinelDecision`.
    3. :meth:`run` is the convenience entry point the Phase 10 runner
       (task 10.6) calls: it calibrates on the first
       :attr:`SentinelConfig.calibration_steps` rows and scores every
       row in the full matrix, including the calibration window (which
       never flags by construction).

    A single instance can be re-used across runs that share a config;
    :meth:`calibrate` is idempotent and overwrites the stored baseline
    on every call.
    """

    def __init__(self, config: SentinelConfig | None = None) -> None:
        self._config = config or SentinelConfig()
        self._baseline_mean: NDArray[np.float64] | None = None
        self._baseline_std: NDArray[np.float64] | None = None

    @property
    def config(self) -> SentinelConfig:
        """The frozen :class:`SentinelConfig` in force for this instance."""
        return self._config

    @property
    def is_calibrated(self) -> bool:
        """``True`` after :meth:`calibrate` has populated the baseline."""
        return self._baseline_mean is not None and self._baseline_std is not None

    def calibrate(self, clean_matrix: NDArray[np.float64]) -> None:
        """Learn per-signal mean and std from a clean signal matrix.

        Only the first :attr:`SentinelConfig.calibration_steps` rows are
        used; any extra rows are ignored. Columns must match the
        :data:`~chronoagent.monitor.collector.SIGNAL_LABELS` order and
        there must be at least ``calibration_steps`` rows.

        Args:
            clean_matrix: Shape ``(N, NUM_SIGNALS)`` array, ``N >= calibration_steps``.

        Raises:
            ValueError: On wrong rank, wrong column count, or too few rows.
        """
        if clean_matrix.ndim != 2 or clean_matrix.shape[1] != NUM_SIGNALS:
            raise ValueError(
                f"clean_matrix must have shape (N, {NUM_SIGNALS}), got {clean_matrix.shape}"
            )
        if clean_matrix.shape[0] < self._config.calibration_steps:
            raise ValueError(
                f"clean_matrix has {clean_matrix.shape[0]} rows, "
                f"need at least {self._config.calibration_steps}"
            )
        window = clean_matrix[: self._config.calibration_steps].astype(np.float64)
        mean = window.mean(axis=0)
        std = window.std(axis=0, ddof=1)
        std = np.maximum(std, self._config.min_std)
        self._baseline_mean = mean
        self._baseline_std = std

    def decide(
        self,
        signals: StepSignals | NDArray[np.float64],
        step_index: int,
    ) -> SentinelDecision:
        """Score one step and return a :class:`SentinelDecision`.

        Args:
            signals: Either a :class:`StepSignals` dataclass or a 1-D
                ``float64`` array of length :data:`NUM_SIGNALS`.
            step_index: 0-based step index within the run; recorded on
                the decision but not otherwise used.

        Returns:
            :class:`SentinelDecision` with ``score``, ``flagged``, and
            ``success`` populated.

        Raises:
            RuntimeError: If :meth:`calibrate` has not yet been called.
            ValueError: If *signals* is a numpy array of the wrong shape.
        """
        if not self.is_calibrated:
            raise RuntimeError("SentinelBaseline.decide called before calibrate()")
        if isinstance(signals, StepSignals):
            vec = signals.to_array()
        else:
            vec = np.asarray(signals, dtype=np.float64)
        if vec.shape != (NUM_SIGNALS,):
            raise ValueError(f"signals vector must have shape ({NUM_SIGNALS},), got {vec.shape}")
        # is_calibrated guarantees both arrays are non-None, but mypy
        # needs the explicit cast.
        mean = cast("NDArray[np.float64]", self._baseline_mean)
        std = cast("NDArray[np.float64]", self._baseline_std)
        z = np.abs((vec - mean) / std)
        score = float(z.max())
        flagged = score > self._config.z_threshold
        return SentinelDecision(
            step_index=int(step_index),
            score=score,
            flagged=flagged,
            success=not flagged,
        )

    def run(self, signal_matrix: NDArray[np.float64]) -> list[SentinelDecision]:
        """Calibrate then score an entire run matrix.

        Calibration uses the first :attr:`SentinelConfig.calibration_steps`
        rows; scoring covers every row (including the calibration
        window, which cannot flag by construction with the default
        configuration).

        Args:
            signal_matrix: Shape ``(T, NUM_SIGNALS)`` full-run matrix.

        Returns:
            List of :class:`SentinelDecision`, one per row, in step order.
        """
        self.calibrate(signal_matrix)
        return [self.decide(signal_matrix[i], step_index=i) for i in range(signal_matrix.shape[0])]
