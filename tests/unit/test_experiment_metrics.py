"""Unit tests for ``chronoagent.experiments.metrics`` (Phase 10 task 10.2).

Coverage plan per function:

* :func:`~chronoagent.experiments.metrics.advance_warning_time` -- happy
  paths for early / concurrent / late detection, invariants
  (``AWT`` is the ground-truth difference), and ``ValueError`` for negative
  step indices.
* :func:`~chronoagent.experiments.metrics.allocation_efficiency` -- bool
  input, mapping input with a ``"success"`` key, mixed sequences, empty
  sequence returns ``0.0``, bad shapes raise ``KeyError`` / ``TypeError``.
* :func:`~chronoagent.experiments.metrics.detection_auroc` -- perfect
  ranking = ``1.0`` (paper-quality assertion), perfectly inverted ranking
  = ``0.0``, uninformative all-same scores stay bounded, empty arrays and
  single-class labels return ``nan``.
* :func:`~chronoagent.experiments.metrics.detection_f1` -- exact-match
  predictions = ``1.0`` (paper-quality assertion), all-wrong predictions
  = ``0.0``, empty arrays return ``nan``, all-zero / all-one edge cases
  via ``zero_division=0.0`` silence warnings and return ``0.0``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chronoagent.experiments.metrics import (
    AllocationResult,
    advance_warning_time,
    allocation_efficiency,
    detection_auroc,
    detection_f1,
)

# ── advance_warning_time ─────────────────────────────────────────────────────


class TestAdvanceWarningTime:
    """Happy paths, Pivot-A concurrent semantics, and input validation."""

    def test_early_detection_returns_positive(self) -> None:
        # Detection at step 3, injection at step 10 -> 7 steps of warning.
        assert advance_warning_time(injection_step=10, detection_step=3) == 7

    def test_concurrent_detection_returns_zero(self) -> None:
        # Pivot A: concurrent detection is AWT = 0, not a miss.
        assert advance_warning_time(injection_step=5, detection_step=5) == 0

    def test_late_detection_returns_negative(self) -> None:
        assert advance_warning_time(injection_step=3, detection_step=10) == -7

    def test_zero_injection_zero_detection(self) -> None:
        assert advance_warning_time(0, 0) == 0

    @pytest.mark.parametrize(
        ("injection", "detection", "expected"),
        [
            (100, 0, 100),
            (50, 25, 25),
            (12, 11, 1),
            (7, 7, 0),
            (4, 9, -5),
        ],
    )
    def test_parametrized_subtraction_invariant(
        self, injection: int, detection: int, expected: int
    ) -> None:
        assert advance_warning_time(injection, detection) == expected

    def test_return_type_is_int_not_numpy(self) -> None:
        # Numpy int inputs must still produce a plain Python int so
        # downstream JSON serialization (paper output) does not trip.
        result = advance_warning_time(np.int64(10), np.int64(3))  # type: ignore[arg-type]
        assert isinstance(result, int)
        assert not isinstance(result, np.integer)

    def test_negative_injection_rejected(self) -> None:
        with pytest.raises(ValueError, match="injection_step must be non-negative"):
            advance_warning_time(injection_step=-1, detection_step=0)

    def test_negative_detection_rejected(self) -> None:
        with pytest.raises(ValueError, match="detection_step must be non-negative"):
            advance_warning_time(injection_step=0, detection_step=-1)


# ── allocation_efficiency ────────────────────────────────────────────────────


class TestAllocationEfficiencyBoolInput:
    """Passing a plain ``Sequence[bool]`` is the simplest harness path."""

    def test_all_successes_is_one(self) -> None:
        assert allocation_efficiency([True, True, True]) == 1.0

    def test_all_failures_is_zero(self) -> None:
        assert allocation_efficiency([False, False, False]) == 0.0

    def test_half_and_half(self) -> None:
        assert allocation_efficiency([True, False, True, False]) == 0.5

    def test_two_thirds(self) -> None:
        assert allocation_efficiency([True, True, False]) == pytest.approx(2 / 3)

    def test_single_success(self) -> None:
        assert allocation_efficiency([True]) == 1.0

    def test_single_failure(self) -> None:
        assert allocation_efficiency([False]) == 0.0

    def test_empty_sequence_returns_zero(self) -> None:
        # Empty is vacuously 0.0 so downstream cumulative tracking
        # never has to branch on len().
        assert allocation_efficiency([]) == 0.0


class TestAllocationEfficiencyMappingInput:
    """Mapping-style audit-trail rows with a ``"success"`` key."""

    def test_mapping_all_success(self) -> None:
        rows: list[AllocationResult] = [
            {"task_id": "t1", "success": True},
            {"task_id": "t2", "success": True},
        ]
        assert allocation_efficiency(rows) == 1.0

    def test_mapping_mixed(self) -> None:
        rows: list[AllocationResult] = [
            {"success": True},
            {"success": False},
            {"success": True},
            {"success": False},
        ]
        assert allocation_efficiency(rows) == 0.5

    def test_mapping_with_extra_fields_ignored(self) -> None:
        rows: list[AllocationResult] = [
            {"task_id": "x", "agent_id": "a1", "bid": 0.9, "success": True},
            {"task_id": "y", "agent_id": "a2", "bid": 0.1, "success": False},
        ]
        assert allocation_efficiency(rows) == 0.5

    def test_mapping_missing_success_key_raises(self) -> None:
        with pytest.raises(KeyError, match="missing 'success' key"):
            allocation_efficiency([{"task_id": "t1"}])  # type: ignore[list-item]

    def test_mapping_with_truthy_non_bool_success_counts(self) -> None:
        # bool(item["success"]) handles sklearn/pydantic rows that
        # serialise booleans as 1/0 ints. Not a guarantee, but a useful
        # ergonomic we lock in.
        rows: list[AllocationResult] = [{"success": 1}, {"success": 0}]  # type: ignore[dict-item]
        assert allocation_efficiency(rows) == 0.5


class TestAllocationEfficiencyMixedAndErrors:
    """Mixed sequences and bad element types."""

    def test_mixed_bool_and_mapping(self) -> None:
        rows: list[AllocationResult] = [
            True,
            {"success": False},
            True,
            {"success": True},
        ]
        assert allocation_efficiency(rows) == 0.75

    def test_non_bool_non_mapping_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="bool or Mapping"):
            allocation_efficiency(["success", "fail"])  # type: ignore[list-item]

    def test_none_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="bool or Mapping"):
            allocation_efficiency([None])  # type: ignore[list-item]


# ── detection_auroc ──────────────────────────────────────────────────────────


class TestDetectionAuroc:
    """ROC-AUC wrapper: paper-quality assertions and nan propagation."""

    def test_perfect_ranking_is_one(self) -> None:
        # Paper-quality assertion: when scores sort y_true perfectly,
        # AUROC must be 1.0 exactly.
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert detection_auroc(y_true, y_scores) == 1.0

    def test_inverted_ranking_is_zero(self) -> None:
        # Perfectly backwards ranking -> AUROC = 0.0.
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert detection_auroc(y_true, y_scores) == 0.0

    def test_uninformative_scores_near_half(self) -> None:
        # All-same scores: every pair is a tie, AUROC = 0.5 by sklearn
        # convention.
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert detection_auroc(y_true, y_scores) == pytest.approx(0.5)

    def test_partial_overlap_between_zero_and_one(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        result = detection_auroc(y_true, y_scores)
        assert 0.0 < result < 1.0

    def test_accepts_python_lists(self) -> None:
        assert detection_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_empty_y_true_returns_nan(self) -> None:
        assert math.isnan(detection_auroc(np.array([]), np.array([])))

    def test_empty_y_scores_returns_nan(self) -> None:
        assert math.isnan(detection_auroc(np.array([0, 1]), np.array([])))

    def test_single_class_all_zero_returns_nan(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        assert math.isnan(detection_auroc(y_true, y_scores))

    def test_single_class_all_one_returns_nan(self) -> None:
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        assert math.isnan(detection_auroc(y_true, y_scores))

    def test_all_zero_scores_with_two_classes_still_defined(self) -> None:
        # Not undefined: y_true has two classes, scores are ties, AUROC
        # is 0.5 via sklearn's tie-breaking convention.
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.zeros(4)
        result = detection_auroc(y_true, y_scores)
        assert result == pytest.approx(0.5)

    def test_return_type_is_python_float(self) -> None:
        result = detection_auroc([0, 1], [0.0, 1.0])
        assert type(result) is float

    def test_nan_return_type_is_python_float(self) -> None:
        # NaN from our code should still be a plain float, not np.nan's
        # numpy scalar -- important for JSON serialisation.
        result = detection_auroc([], [])
        assert type(result) is float
        assert math.isnan(result)


# ── detection_f1 ─────────────────────────────────────────────────────────────


class TestDetectionF1:
    """F1 wrapper: paper-quality assertions, edge cases, zero-division."""

    def test_perfect_predictions_is_one(self) -> None:
        # Paper-quality assertion: exact-match predictions -> F1 = 1.0.
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        assert detection_f1(y_true, y_pred) == 1.0

    def test_all_wrong_predictions_is_zero(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert detection_f1(y_true, y_pred) == 0.0

    def test_half_correct_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        # TP=1, FP=1, FN=1 -> precision=0.5, recall=0.5, F1=0.5.
        assert detection_f1(y_true, y_pred) == pytest.approx(0.5)

    def test_accepts_python_lists(self) -> None:
        assert detection_f1([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0

    def test_empty_y_true_returns_nan(self) -> None:
        assert math.isnan(detection_f1(np.array([]), np.array([])))

    def test_empty_y_pred_returns_nan(self) -> None:
        assert math.isnan(detection_f1(np.array([0, 1]), np.array([])))

    def test_all_zero_true_and_pred_returns_zero_silently(self) -> None:
        # zero_division=0.0 returns 0.0 without warning.
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert detection_f1(y_true, y_pred) == 0.0

    def test_all_zero_true_with_false_positives_is_zero(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        # Precision = 0/3 -> 0 (via zero_division fallback), recall undefined
        # -> 0, F1 = 0.
        assert detection_f1(y_true, y_pred) == 0.0

    def test_single_class_one_exact_match(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        assert detection_f1(y_true, y_pred) == 1.0

    def test_no_warnings_on_zero_division(self) -> None:
        import warnings as _warnings

        from sklearn.exceptions import UndefinedMetricWarning

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UndefinedMetricWarning)
            # Should not raise: the wrapper suppresses the warning class
            # and passes ``zero_division=0.0`` to sklearn.
            detection_f1(np.array([0, 0]), np.array([0, 0]))

    def test_return_type_is_python_float(self) -> None:
        result = detection_f1([0, 1], [0, 1])
        assert type(result) is float

    def test_nan_return_type_is_python_float(self) -> None:
        result = detection_f1([], [])
        assert type(result) is float
        assert math.isnan(result)
