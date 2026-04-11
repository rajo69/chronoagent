"""Offline replay detector for the full ChronoAgent monitoring stack.

This module is the Phase 10 task 10.6 comparator for the four "full system"
experiment configs (``main_experiment``, ``agentpoison_experiment``, and the
three ``ablation_no_*`` rows). It consumes a ``(T, NUM_SIGNALS)`` signal
matrix produced by the Phase 1 signal-validation runner and emits one
decision per row in the same shape as
:class:`~chronoagent.experiments.baselines.sentinel.SentinelDecision` /
:class:`~chronoagent.experiments.baselines.no_monitoring.NoMonitoringDecision`,
so the Phase 10 runner can drive any comparator through a single call site
and the 10.2 metric functions can consume every detector's output unchanged.

The detector is a deliberately honest offline replay of ChronoAgent's
real monitoring components, toggled by the three anomaly-facing
:class:`~chronoagent.experiments.config_schema.AblationConfig` flags:

* ``bocpd`` runs the project's real :class:`chronoagent.scorer.bocpd.BOCPD`
  implementation offline over the ``kl_divergence`` signal column (index
  3 of :data:`~chronoagent.monitor.collector.SIGNAL_LABELS`). This is the
  same BOCPD the online temporal health scorer uses, just fed from a
  pre-recorded matrix instead of the live bus, so the ablation row
  genuinely measures the contribution of the changepoint detector.

* ``forecaster`` adds a lightweight exponential-moving-average (EMA)
  residual channel on the same KL column. This is a placeholder stand-in
  for the Chronos-2 forecaster which is optional and not always installed
  (see the Phase 9.3 graceful-degradation path). The stand-in is not the
  full Chronos model, but it gives the runner a NON-TRIVIAL per-row
  score that still depends on the forecaster flag so the ablation row is
  not silently identical to the full-system row. The docstring and the
  module-level :data:`FORECASTER_IS_PLACEHOLDER` flag make this
  explicit; follow-up work to wire the real Chronos model through
  :class:`~chronoagent.scorer.chronos_forecaster.ChronosForecaster` is
  tracked for Phase 10.7 / 10.8.

* ``integrity`` adds a MAD-outlier score on the
  ``memory_query_entropy`` signal column (index 5). Again an offline
  stand-in for :class:`~chronoagent.memory.integrity.MemoryIntegrityModule`
  which is online and bus-driven, but matches the memory-integrity
  module's design intent (flag steps whose memory-query distribution
  looks anomalous vs the clean window).

The sub-scores of every enabled channel are averaged, and a step is
flagged iff the combined score exceeds :data:`DECISION_THRESHOLD`. The
combined score is always in ``[0, 1]`` by construction (each sub-score
is bounded). ``success = not flagged`` matches the Sentinel baseline
convention so the 10.2 ``allocation_efficiency`` function works on the
decision stream unchanged.

If every anomaly channel is disabled (``bocpd=False`` AND
``forecaster=False`` AND ``integrity=False``), the detector falls back
to Sentinel-style z-score thresholding on the KL column so the runner
never has to emit an "undefined" decision stream. In practice this only
happens for the ``baseline_sentinel`` config, which the runner dispatches
through the real SentinelBaseline class before reaching this fallback.

The ``ablation.health`` flag is handled by the Phase 10 runner itself
(it dispatches to :class:`~chronoagent.experiments.baselines.no_monitoring.NoMonitoringBaseline`
when health is off), so this module does NOT read that flag. It is
documented here to keep the ablation semantics in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chronoagent.allocator.capability_weights import AGENT_IDS
from chronoagent.monitor.collector import NUM_SIGNALS
from chronoagent.scorer.bocpd import BOCPD

if TYPE_CHECKING:
    from chronoagent.experiments.config_schema import AblationConfig

# Column index of the ``kl_divergence`` signal in the ``(T, NUM_SIGNALS)``
# matrix produced by the Phase 1 signal-validation runner. Matches
# :data:`chronoagent.monitor.collector.SIGNAL_LABELS`.
KL_COLUMN_INDEX: int = 3

# Column index of the ``memory_query_entropy`` signal. Used by the
# integrity sub-score when ``ablation.integrity`` is enabled.
ENTROPY_COLUMN_INDEX: int = 5

# Baseline-as-a-whole label for allocator-audit disambiguation. Every
# ``FullSystemDecision.agent_id`` carries one of the canonical
# :data:`~chronoagent.allocator.capability_weights.AGENT_IDS` (the agent
# the allocator would have dispatched the step to) rather than this
# label; the module constant exists so aggregated result CSVs can tag
# rows that came from the full-system detector without colliding with
# the real agent ids.
FULL_SYSTEM_AGENT_ID: str = "full_system_detector"

# Flag indicating whether the ``forecaster`` sub-score is a placeholder
# (EMA residual) or a real Chronos call. v1 is always a placeholder;
# 10.7 / 10.8 follow-up work will replace the EMA path with a real
# Chronos inference.
FORECASTER_IS_PLACEHOLDER: bool = True


@dataclass(frozen=True)
class FullSystemConfig:
    """Hyperparameters for the offline full-system detector.

    Attributes:
        kl_column_index: Column index of the KL-divergence signal. Locked
            at ``KL_COLUMN_INDEX = 3`` by default; exposed on the config
            so tests can point at a different column if a future signal
            order shuffle happens.
        entropy_column_index: Column index of the memory-query-entropy
            signal. Used by the integrity sub-score.
        bocpd_hazard_lambda: Expected run length for the BOCPD prior.
            Mirrors :attr:`~chronoagent.experiments.config_schema.SystemConfig.bocpd_hazard_lambda`
            which defaults to 50.0.
        calibration_steps: Number of clean-phase rows used to fit the
            Sentinel z-score fallback and the EMA forecaster baseline.
            When the runner constructs the detector this is set from
            ``cfg.attack.injection_step`` so the detector's calibration
            window matches the experiment's clean phase.
        min_std: Std floor used by the Sentinel z-score fallback and
            the MAD outlier score, identical to the Sentinel baseline's
            handling of constant columns.
        ema_alpha: EMA smoothing factor for the forecaster stand-in. A
            low value (e.g. 0.2) gives a slowly-moving baseline so the
            residual spikes on regime shifts; a high value tracks the
            live signal too tightly.
        decision_threshold: Combined-score threshold for the
            ``flagged`` decision. Strict ``>`` so boundary ties pass,
            mirroring the Sentinel baseline convention.
    """

    kl_column_index: int = KL_COLUMN_INDEX
    entropy_column_index: int = ENTROPY_COLUMN_INDEX
    bocpd_hazard_lambda: float = 50.0
    calibration_steps: int = 10
    min_std: float = 1e-6
    ema_alpha: float = 0.2
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.kl_column_index < 0 or self.kl_column_index >= NUM_SIGNALS:
            raise ValueError(
                f"kl_column_index must be in [0, {NUM_SIGNALS}), got {self.kl_column_index}"
            )
        if self.entropy_column_index < 0 or self.entropy_column_index >= NUM_SIGNALS:
            raise ValueError(
                f"entropy_column_index must be in [0, {NUM_SIGNALS}), "
                f"got {self.entropy_column_index}"
            )
        if self.bocpd_hazard_lambda <= 0:
            raise ValueError(f"bocpd_hazard_lambda must be > 0, got {self.bocpd_hazard_lambda}")
        if self.calibration_steps < 2:
            raise ValueError(f"calibration_steps must be >= 2, got {self.calibration_steps}")
        if self.min_std <= 0:
            raise ValueError(f"min_std must be > 0, got {self.min_std}")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError(f"decision_threshold must be in [0, 1], got {self.decision_threshold}")


@dataclass(frozen=True)
class FullSystemDecision:
    """One per-step decision from the offline full-system detector.

    The shape mirrors :class:`~chronoagent.experiments.baselines.sentinel.SentinelDecision`
    so the 10.2 metric functions and the Phase 10 runner can treat
    decisions from every comparator uniformly.

    Attributes:
        step_index: Zero-indexed row of the signal matrix this decision
            was computed from.
        score: Combined anomaly score in ``[0, 1]`` (strict bounds).
        flagged: ``True`` iff ``score > config.decision_threshold``.
        success: ``not flagged``. Pinned as a separate field because
            ``allocation_efficiency`` reads ``.success`` directly and
            some future experiment may want to decouple the two.
        agent_id: Canonical agent id the allocator would dispatch this
            step to, NOT the baseline-wide label. Cycles through
            :data:`~chronoagent.allocator.capability_weights.AGENT_IDS`
            in round-robin order so aggregated audit CSVs have the same
            per-agent breakdown as the Sentinel and NoMonitoring runs.
        bocpd_score: BOCPD changepoint probability in ``[0, 1]`` if
            BOCPD is enabled for this run, else ``None``. Exposed for
            audit / analysis; not read by the metric functions.
        forecaster_score: EMA residual magnitude in ``[0, 1]`` if the
            forecaster channel is enabled, else ``None``.
        integrity_score: MAD outlier score in ``[0, 1]`` if integrity
            is enabled, else ``None``.
    """

    step_index: int
    score: float
    flagged: bool
    success: bool
    agent_id: str
    bocpd_score: float | None = None
    forecaster_score: float | None = None
    integrity_score: float | None = None


@dataclass(frozen=True)
class _ChannelMask:
    """Which sub-score channels are active for this detector instance."""

    bocpd: bool
    forecaster: bool
    integrity: bool

    @property
    def any_enabled(self) -> bool:
        return self.bocpd or self.forecaster or self.integrity


@dataclass
class _SentinelFallback:
    """Sentinel-style z-score thresholding on the KL column.

    Used as the scoring channel when every anomaly flag is off. Not a
    class on the public API because it is strictly an internal fallback;
    the real Sentinel baseline (task 10.3) is dispatched directly by the
    runner when ``cfg.name == "baseline_sentinel"``.
    """

    min_std: float
    mean_: float = field(default=0.0, init=False)
    std_: float = field(default=1.0, init=False)

    def calibrate(self, clean_slice: NDArray[np.float64]) -> None:
        self.mean_ = float(np.mean(clean_slice))
        raw_std = float(np.std(clean_slice, ddof=1)) if clean_slice.size >= 2 else 0.0
        self.std_ = max(raw_std, self.min_std)

    def score(self, obs: float) -> float:
        z = abs((obs - self.mean_) / self.std_)
        # Squash into [0, 1] via a logistic-like mapping so the fallback
        # channel contributes on the same scale as the BOCPD / EMA /
        # integrity channels (which are already bounded in [0, 1]).
        # ``1 - exp(-z / 3)`` hits 0.5 at z ~= 2, 0.95 at z ~= 9.
        return float(1.0 - np.exp(-z / 3.0))


class FullSystemDetector:
    """Offline replay of the ChronoAgent full-system detection stack.

    The detector's public API mirrors
    :class:`~chronoagent.experiments.baselines.sentinel.SentinelBaseline`:
    construct once per run, call :meth:`run` with the ``(T, NUM_SIGNALS)``
    signal matrix, and receive a ``list[FullSystemDecision]`` in step
    order.

    Args:
        ablation: The Phase 10 ablation configuration driving which
            sub-score channels are active.
        config: Optional detector hyperparameters. Defaults to
            :class:`FullSystemConfig` with Sentinel-compatible
            calibration_steps / min_std defaults.

    Design notes:

    * The detector does NOT read ``ablation.health``. The Phase 10
      runner dispatches to :class:`NoMonitoringBaseline` when health is
      off; this class is only constructed when the runner has already
      decided a full-system replay is appropriate.
    * The BOCPD channel runs the real project BOCPD implementation
      offline; the forecaster channel is an honest EMA-residual
      placeholder until Chronos wiring lands in 10.7 / 10.8.
    * Calibration is done in-line inside :meth:`run` (it needs the
      clean slice of the input matrix to fit the Sentinel fallback and
      the EMA baseline). Unlike Sentinel, the detector does NOT raise
      ``RuntimeError`` on an un-calibrated :meth:`score` call because
      the public API is ``run`` only; there is no single-step
      :meth:`decide` entry point.
    """

    def __init__(
        self,
        ablation: AblationConfig,
        config: FullSystemConfig | None = None,
    ) -> None:
        self._config: FullSystemConfig = config if config is not None else FullSystemConfig()
        self._mask = _ChannelMask(
            bocpd=ablation.bocpd,
            forecaster=ablation.forecaster,
            integrity=ablation.integrity,
        )
        self._fallback: _SentinelFallback | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, signal_matrix: NDArray[np.float64]) -> list[FullSystemDecision]:
        """Score every row of ``signal_matrix`` and return decisions.

        Args:
            signal_matrix: Shape ``(T, NUM_SIGNALS)`` float64 array
                produced by the Phase 10 runner's signal matrix
                factory (usually a wrapped
                :class:`~chronoagent.experiments.runner.SignalValidationRunner`).

        Returns:
            One :class:`FullSystemDecision` per row in step order.

        Raises:
            ValueError: If the matrix shape is wrong or there are
                fewer rows than ``config.calibration_steps``.
        """
        self._validate_shape(signal_matrix)
        n_rows = int(signal_matrix.shape[0])
        if n_rows == 0:
            return []
        if n_rows < self._config.calibration_steps:
            raise ValueError(
                f"signal_matrix has {n_rows} rows, need at least "
                f"calibration_steps={self._config.calibration_steps}"
            )

        kl_col = signal_matrix[:, self._config.kl_column_index].astype(np.float64)
        entropy_col = signal_matrix[:, self._config.entropy_column_index].astype(np.float64)
        clean_kl = kl_col[: self._config.calibration_steps]
        clean_entropy = entropy_col[: self._config.calibration_steps]

        bocpd_scores = self._bocpd_channel(kl_col) if self._mask.bocpd else None
        forecaster_scores = (
            self._forecaster_channel(kl_col, clean_kl) if self._mask.forecaster else None
        )
        integrity_scores = (
            self._integrity_channel(entropy_col, clean_entropy) if self._mask.integrity else None
        )
        fallback_scores: NDArray[np.float64] | None = None
        if not self._mask.any_enabled:
            fallback_scores = self._fallback_channel(kl_col, clean_kl)

        decisions: list[FullSystemDecision] = []
        for step_index in range(n_rows):
            sub_scores: list[float] = []
            b = None
            f = None
            i = None
            if bocpd_scores is not None:
                b = float(bocpd_scores[step_index])
                sub_scores.append(b)
            if forecaster_scores is not None:
                f = float(forecaster_scores[step_index])
                sub_scores.append(f)
            if integrity_scores is not None:
                i = float(integrity_scores[step_index])
                sub_scores.append(i)
            if fallback_scores is not None:
                sub_scores.append(float(fallback_scores[step_index]))

            combined = float(np.mean(sub_scores)) if sub_scores else 0.0
            flagged = combined > self._config.decision_threshold
            agent_id = AGENT_IDS[step_index % len(AGENT_IDS)]
            decisions.append(
                FullSystemDecision(
                    step_index=step_index,
                    score=combined,
                    flagged=flagged,
                    success=not flagged,
                    agent_id=agent_id,
                    bocpd_score=b,
                    forecaster_score=f,
                    integrity_score=i,
                )
            )
        return decisions

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    def _bocpd_channel(self, kl_col: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run the project BOCPD implementation offline on the KL column."""
        bocpd = BOCPD(hazard_lambda=self._config.bocpd_hazard_lambda)
        scores = np.empty(kl_col.shape[0], dtype=np.float64)
        for idx, obs in enumerate(kl_col):
            scores[idx] = bocpd.update(float(obs))
        # BOCPD returns values in [0, 1] already; no extra clamping.
        return scores

    def _forecaster_channel(
        self,
        kl_col: NDArray[np.float64],
        clean_slice: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """EMA residual stand-in for the Chronos forecaster.

        v1 computes the EMA baseline from the clean calibration window
        and emits, for every subsequent row, a bounded magnitude of
        ``|obs - ema|`` normalised by the clean slice's std. The EMA is
        then updated with the new observation so the baseline drifts
        with the signal. The magnitude is squashed into ``[0, 1]`` the
        same way the Sentinel fallback does so the forecaster channel
        contributes on the same scale as BOCPD.
        """
        alpha = self._config.ema_alpha
        std = float(np.std(clean_slice, ddof=1)) if clean_slice.size >= 2 else 0.0
        std = max(std, self._config.min_std)
        ema = float(np.mean(clean_slice)) if clean_slice.size else 0.0
        scores = np.empty(kl_col.shape[0], dtype=np.float64)
        for idx, obs in enumerate(kl_col):
            residual = abs(float(obs) - ema) / std
            scores[idx] = float(1.0 - np.exp(-residual / 3.0))
            ema = (1.0 - alpha) * ema + alpha * float(obs)
        return scores

    def _integrity_channel(
        self,
        entropy_col: NDArray[np.float64],
        clean_slice: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """MAD-outlier score on the memory-query-entropy column."""
        median = float(np.median(clean_slice)) if clean_slice.size else 0.0
        abs_dev = np.abs(clean_slice - median)
        mad = float(np.median(abs_dev)) if abs_dev.size else 0.0
        # Scale factor 1.4826 makes MAD a consistent estimator for
        # Gaussian std; clamp to min_std to avoid division blow-ups on
        # constant columns.
        sigma = max(1.4826 * mad, self._config.min_std)
        raw = np.abs(entropy_col - median) / sigma
        # Squash with the same logistic mapping as the fallback /
        # forecaster channels so every sub-score shares a scale.
        return 1.0 - np.exp(-raw / 3.0)

    def _fallback_channel(
        self,
        kl_col: NDArray[np.float64],
        clean_slice: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sentinel-style z-score thresholding on the KL column."""
        self._fallback = _SentinelFallback(min_std=self._config.min_std)
        self._fallback.calibrate(clean_slice)
        return np.array([self._fallback.score(float(obs)) for obs in kl_col], dtype=np.float64)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_shape(self, signal_matrix: NDArray[np.float64]) -> None:
        if signal_matrix.ndim != 2:
            raise ValueError(f"signal_matrix must be 2-D, got shape {signal_matrix.shape}")
        if signal_matrix.shape[1] != NUM_SIGNALS:
            raise ValueError(
                f"signal_matrix must have {NUM_SIGNALS} columns, got shape {signal_matrix.shape}"
            )
