"""Phase 10 metric functions for the research experiment suite (task 10.2).

Four paper-facing metrics that quantify attack detection and allocation
quality over the experiment harness outputs:

* :func:`advance_warning_time` -- steps between injection and detection.
  Per Pivot A (see ``docs/phase1_decision.md``), concurrent detection counts
  as ``AWT = 0`` instead of NO-GO, so the return value is intentionally a
  signed int: positive means early warning, zero means concurrent, negative
  means late detection.
* :func:`allocation_efficiency` -- cumulative success rate of allocator
  decisions across a run. Accepts either a plain ``Sequence[bool]`` or a
  sequence of mapping-style audit records carrying a ``"success"`` key, so
  the caller can pass the audit trail rows from the task allocator without
  projecting them first.
* :func:`detection_auroc` -- ROC-AUC for the attack detector's score stream.
* :func:`detection_f1` -- F1 score for the attack detector's binary
  predictions.

AUROC and F1 both delegate to scikit-learn and translate undefined inputs
(empty arrays, single-class labels) into ``float('nan')`` rather than
raising, so downstream aggregation across runs can use :func:`numpy.nanmean`
without wrapping every call in ``try``/``except``. AUROC for a perfectly
ranked label vector returns 1.0; AUROC for a perfectly inverted ranking
returns 0.0; AUROC for uninformative scores tends to 0.5 in expectation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, roc_auc_score

__all__ = [
    "AllocationResult",
    "advance_warning_time",
    "allocation_efficiency",
    "detection_auroc",
    "detection_f1",
]

#: Allowed element type for :func:`allocation_efficiency`. Either a raw
#: boolean (``True`` = success) or a mapping with a boolean ``"success"``
#: key (matches the shape of the allocator's audit-trail rows).
AllocationResult = bool | Mapping[str, Any]


def advance_warning_time(injection_step: int, detection_step: int) -> int:
    """Compute Advance Warning Time (AWT) in steps.

    AWT is the number of steps by which detection lands before injection.
    Per Pivot A, detection that coincides with the injection step counts as
    ``AWT = 0`` (concurrent detection is still a valid outcome), not as a
    miss. A negative AWT means the detector was late.

    Args:
        injection_step: 0-based step index at which the attack was injected.
            Must be non-negative.
        detection_step: 0-based step index at which the detector flagged
            the attack. Must be non-negative; callers should filter
            runs where the detector never fired before computing AWT.

    Returns:
        ``injection_step - detection_step``. Positive = early warning,
        ``0`` = concurrent detection, negative = late detection.

    Raises:
        ValueError: If either argument is negative.
    """
    if injection_step < 0:
        raise ValueError(f"injection_step must be non-negative, got {injection_step}")
    if detection_step < 0:
        raise ValueError(f"detection_step must be non-negative, got {detection_step}")
    return int(injection_step) - int(detection_step)


def allocation_efficiency(results: Sequence[AllocationResult]) -> float:
    """Cumulative success rate of allocator decisions.

    Each element of *results* represents one allocator decision. A decision
    is either a raw ``bool`` (``True`` means the task was successfully
    assigned to a healthy agent) or a mapping with a ``"success"`` boolean
    key (so the caller can pass through audit-trail rows from the task
    allocator without projecting them).

    Args:
        results: Sequence of allocation outcomes.

    Returns:
        Fraction of successful allocations in ``[0.0, 1.0]``. An empty
        sequence returns ``0.0`` (vacuously no successes).

    Raises:
        KeyError: If a mapping element is missing the ``"success"`` key.
        TypeError: If an element is neither ``bool`` nor ``Mapping``.
    """
    n = len(results)
    if n == 0:
        return 0.0
    successes = 0
    for item in results:
        # Note: ``bool`` must be checked before ``Mapping`` because
        # ``isinstance(True, Mapping)`` is False but we want an explicit
        # ordering for clarity.
        if isinstance(item, bool):
            if item:
                successes += 1
        elif isinstance(item, Mapping):
            if "success" not in item:
                raise KeyError(f"allocation result mapping missing 'success' key: {item!r}")
            if bool(item["success"]):
                successes += 1
        else:
            raise TypeError(
                "allocation results must be bool or Mapping with a "
                f"'success' key, got {type(item).__name__}"
            )
    return successes / n


def detection_auroc(y_true: ArrayLike, y_scores: ArrayLike) -> float:
    """Compute ROC-AUC for the attack detector's score stream.

    Thin wrapper over :func:`sklearn.metrics.roc_auc_score` that translates
    undefined inputs into ``float('nan')`` instead of raising, so
    per-run results can be aggregated with :func:`numpy.nanmean`.

    Undefined cases:

    * Either input is empty.
    * ``y_true`` contains only a single class.

    Args:
        y_true: Ground-truth binary labels (``1`` = attacked step,
            ``0`` = clean step).
        y_scores: Detector output scores -- higher means more confident
            attack. Length must match ``y_true``.

    Returns:
        ROC-AUC in ``[0.0, 1.0]``, or ``float('nan')`` if undefined.
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)
    if y_true_arr.size == 0 or y_scores_arr.size == 0:
        return float("nan")
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(roc_auc_score(y_true_arr, y_scores_arr))


def detection_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute F1 score for the attack detector's binary predictions.

    Thin wrapper over :func:`sklearn.metrics.f1_score` with
    ``zero_division=0.0`` so all-zero prediction/label combinations return
    ``0.0`` instead of emitting an ``UndefinedMetricWarning``. Empty arrays
    return ``float('nan')`` so aggregators can use :func:`numpy.nanmean`.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Detector's binary predictions. Length must match ``y_true``.

    Returns:
        F1 in ``[0.0, 1.0]``, or ``float('nan')`` for empty inputs.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(f1_score(y_true_arr, y_pred_arr, zero_division=0.0))
