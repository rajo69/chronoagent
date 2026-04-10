"""Lazy-loaded Chronos-2-Small wrapper for temporal forecasting.

Chronos-2-Small (46M params, Apache 2.0) is loaded on first use.
If the ``chronos-forecasting`` package is absent the forecaster returns
``None`` gracefully so the ensemble falls back to BOCPD-only mode.

Reference checkpoint: ``amazon/chronos-t5-small``
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum history length before forecasting is meaningful.
_MIN_HISTORY = 10
# Default prediction horizon in steps.
_DEFAULT_HORIZON = 10


@dataclass(frozen=True)
class ForecastResult:
    """Result of a Chronos forecast call.

    Attributes
    ----------
    mean:
        Mean predicted values for the next ``horizon`` steps.
    low:
        Lower bound of the 80% prediction interval.
    high:
        Upper bound of the 80% prediction interval.
    horizon:
        Number of steps forecast ahead.
    """

    mean: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    horizon: int


class ChronosForecaster:
    """Thin wrapper around ``ChronosPipeline`` with lazy loading.

    Parameters
    ----------
    model_id:
        HuggingFace checkpoint name.  Defaults to ``amazon/chronos-t5-small``.
    device:
        Torch device string.  Defaults to ``cpu`` for safety.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-small",
        device: str = "cpu",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._pipeline: Any | None = None
        self._available: bool | None = None  # None = not yet probed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the chronos-forecasting package is installed and loadable."""
        if self._available is None:
            self._available = importlib.util.find_spec("chronos") is not None
        return self._available

    def forecast(
        self,
        history: list[float] | NDArray[np.float64],
        horizon: int = _DEFAULT_HORIZON,
    ) -> ForecastResult | None:
        """Forecast the next ``horizon`` steps given a signal history.

        Parameters
        ----------
        history:
            Observed signal values in chronological order.
        horizon:
            Number of future steps to predict.

        Returns
        -------
        ForecastResult or None
            ``None`` if chronos is unavailable or history is too short.
        """
        hist_arr = np.asarray(history, dtype=np.float32)
        if len(hist_arr) < _MIN_HISTORY:
            logger.debug(
                "chronos_forecaster: history length %d < minimum %d, skipping",
                len(hist_arr),
                _MIN_HISTORY,
            )
            return None

        if not self.available:
            return None

        pipeline = self._get_pipeline()
        if pipeline is None:
            return None

        try:
            import torch  # type: ignore[import-untyped]

            context = torch.tensor(hist_arr).unsqueeze(0)  # (1, T)
            quantile_levels = [0.1, 0.5, 0.9]
            forecast_obj = pipeline.predict_quantiles(
                context,
                prediction_length=horizon,
                quantile_levels=quantile_levels,
            )
            # forecast_obj shape: (1, num_quantiles, horizon)
            low: NDArray[np.float64] = forecast_obj[0, 0].numpy().astype(np.float64)
            mean: NDArray[np.float64] = forecast_obj[0, 1].numpy().astype(np.float64)
            high: NDArray[np.float64] = forecast_obj[0, 2].numpy().astype(np.float64)
            return ForecastResult(mean=mean, low=low, high=high, horizon=horizon)
        except Exception:  # noqa: BLE001
            logger.exception("chronos_forecaster: forecast failed")
            self._available = False
            return None

    def compute_anomaly_score(
        self,
        history: list[float] | NDArray[np.float64],
        actual: float,
    ) -> float | None:
        """Score how anomalous ``actual`` is relative to the 1-step forecast.

        Uses the 80% prediction interval: a value inside the interval scores
        near 0, a value far outside scores near 1.

        Parameters
        ----------
        history:
            Signal history *not* including ``actual``.
        actual:
            The observed value to evaluate.

        Returns
        -------
        float or None
            Anomaly score in [0, 1], or ``None`` if forecasting unavailable.
        """
        result = self.forecast(history, horizon=1)
        if result is None:
            return None

        lo = float(result.low[0])
        hi = float(result.high[0])
        predicted = float(result.mean[0])

        interval_width = max(hi - lo, 1e-8)
        deviation = abs(actual - predicted)
        # Normalised deviation clipped to [0, 1].
        score = min(deviation / interval_width, 1.0)
        return float(score)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Any | None:
        """Return the loaded ChronosPipeline, loading it on first call."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # type: ignore[import-untyped]
            from chronos import ChronosPipeline  # type: ignore[import-untyped]

            self._pipeline = ChronosPipeline.from_pretrained(
                self._model_id,
                device_map=self._device,
                torch_dtype=torch.float32,
            )
            logger.info("chronos_forecaster: loaded %s", self._model_id)
        except Exception:  # noqa: BLE001
            logger.warning("chronos_forecaster: could not load %s, disabling", self._model_id)
            self._available = False
            return None
        return self._pipeline
