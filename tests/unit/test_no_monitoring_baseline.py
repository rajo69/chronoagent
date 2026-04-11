"""Unit tests for ``chronoagent.experiments.baselines.no_monitoring`` (task 10.4).

Coverage plan:

* :class:`NoMonitoringDecision` -- frozen, field round-trip, explicit
  construction.
* :class:`NoMonitoringBaseline.decide` -- round-robin cycling matches
  ``AGENT_IDS`` order, wraps at ``len(AGENT_IDS)``, is deterministic,
  rejects negative indices, and every decision reports ``success=True``.
* :class:`NoMonitoringBaseline.run` -- wrong-shape rejection, zero-row
  matrix, typical 20-row matrix, content of the matrix is ignored
  (shifted block produces the same agent sequence as a clean block).
* **Metric integration smoke test** -- feed decisions into the 10.2
  metric functions and assert: (a) ``allocation_efficiency`` = 1.0
  on any non-empty run, (b) ``detection_auroc`` = nan on a single-class
  label stream (no baseline flags anything), (c) ``detection_f1`` = 0.0
  on all-zero predictions vs mixed labels (sklearn's zero_division=0.0
  path), and (d) AUROC = nan when the all-zero prediction stream is
  paired with a single-class label vector.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from chronoagent.allocator.capability_weights import AGENT_IDS
from chronoagent.experiments.baselines.no_monitoring import (
    NO_MONITORING_AGENT_ID,
    NoMonitoringBaseline,
    NoMonitoringDecision,
)
from chronoagent.experiments.metrics import (
    allocation_efficiency,
    detection_auroc,
    detection_f1,
)
from chronoagent.monitor.collector import NUM_SIGNALS

# ── NoMonitoringDecision ─────────────────────────────────────────────────────


class TestNoMonitoringDecision:
    """Field round-trip and frozen semantics for :class:`NoMonitoringDecision`."""

    def test_explicit_fields_round_trip(self) -> None:
        d = NoMonitoringDecision(step_index=3, success=True, agent_id="planner")
        assert d.step_index == 3
        assert d.success is True
        assert d.agent_id == "planner"

    def test_frozen(self) -> None:
        d = NoMonitoringDecision(step_index=0, success=True, agent_id="planner")
        with pytest.raises(FrozenInstanceError):
            d.step_index = 42  # type: ignore[misc]

    def test_success_false_is_representable_at_the_dataclass_level(self) -> None:
        """The dataclass does not enforce success=True; the baseline does.

        Keeping the dataclass permissive lets tests and the runner
        construct hand-crafted decisions for projection / aggregation
        checks without fighting the type.
        """
        d = NoMonitoringDecision(step_index=0, success=False, agent_id="planner")
        assert d.success is False


# ── NoMonitoringBaseline.decide ──────────────────────────────────────────────


class TestNoMonitoringBaselineDecide:
    """Round-robin cycling, determinism, and input validation."""

    def test_step_zero_returns_first_agent(self) -> None:
        d = NoMonitoringBaseline().decide(0)
        assert d.step_index == 0
        assert d.agent_id == AGENT_IDS[0]
        assert d.success is True

    def test_step_one_returns_second_agent(self) -> None:
        d = NoMonitoringBaseline().decide(1)
        assert d.agent_id == AGENT_IDS[1]

    def test_cycle_wraps_at_length(self) -> None:
        baseline = NoMonitoringBaseline()
        n = len(AGENT_IDS)
        assert baseline.decide(n).agent_id == AGENT_IDS[0]
        assert baseline.decide(n + 1).agent_id == AGENT_IDS[1]
        assert baseline.decide(2 * n - 1).agent_id == AGENT_IDS[n - 1]

    def test_cycle_follows_canonical_agent_order(self) -> None:
        """The full first cycle must match AGENT_IDS element-for-element."""
        baseline = NoMonitoringBaseline()
        for i, agent_id in enumerate(AGENT_IDS):
            assert baseline.decide(i).agent_id == agent_id

    def test_every_decision_succeeds(self) -> None:
        baseline = NoMonitoringBaseline()
        for i in range(50):
            assert baseline.decide(i).success is True

    def test_decisions_are_deterministic(self) -> None:
        a = NoMonitoringBaseline()
        b = NoMonitoringBaseline()
        for i in range(20):
            assert a.decide(i) == b.decide(i)

    def test_negative_step_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="step_index must be non-negative"):
            NoMonitoringBaseline().decide(-1)

    def test_step_index_is_recorded_verbatim(self) -> None:
        d = NoMonitoringBaseline().decide(17)
        assert d.step_index == 17


# ── NoMonitoringBaseline.run ─────────────────────────────────────────────────


class TestNoMonitoringBaselineRun:
    """End-to-end: one decision per row, shape validation, content ignored."""

    def test_returns_one_decision_per_row(self) -> None:
        m = np.zeros((20, NUM_SIGNALS), dtype=np.float64)
        decisions = NoMonitoringBaseline().run(m)
        assert len(decisions) == 20

    def test_step_indices_are_monotonic(self) -> None:
        m = np.zeros((25, NUM_SIGNALS), dtype=np.float64)
        decisions = NoMonitoringBaseline().run(m)
        assert [d.step_index for d in decisions] == list(range(25))

    def test_agent_ids_cycle_through_canonical_order(self) -> None:
        m = np.zeros((len(AGENT_IDS) * 3, NUM_SIGNALS), dtype=np.float64)
        decisions = NoMonitoringBaseline().run(m)
        expected = list(AGENT_IDS) * 3
        assert [d.agent_id for d in decisions] == expected

    def test_all_decisions_are_success(self) -> None:
        m = np.zeros((30, NUM_SIGNALS), dtype=np.float64)
        decisions = NoMonitoringBaseline().run(m)
        assert all(d.success is True for d in decisions)

    def test_zero_row_matrix_returns_empty_list(self) -> None:
        m = np.zeros((0, NUM_SIGNALS), dtype=np.float64)
        decisions = NoMonitoringBaseline().run(m)
        assert decisions == []

    def test_signal_matrix_content_is_ignored(self) -> None:
        """Shifted and clean blocks produce identical decision lists.

        The baseline does no detection, so the actual signal values
        never influence the output. This locks in that invariant.
        """
        clean = np.zeros((10, NUM_SIGNALS), dtype=np.float64)
        shifted = np.full((10, NUM_SIGNALS), 1e6, dtype=np.float64)
        nan_filled = np.full((10, NUM_SIGNALS), np.nan, dtype=np.float64)
        baseline = NoMonitoringBaseline()
        assert baseline.run(clean) == baseline.run(shifted) == baseline.run(nan_filled)

    def test_wrong_rank_rejected(self) -> None:
        m = np.zeros(NUM_SIGNALS, dtype=np.float64)
        with pytest.raises(ValueError, match="must have shape"):
            NoMonitoringBaseline().run(m)

    def test_wrong_column_count_rejected(self) -> None:
        m = np.zeros((10, NUM_SIGNALS + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="must have shape"):
            NoMonitoringBaseline().run(m)

    def test_deterministic_across_calls(self) -> None:
        m = np.zeros((15, NUM_SIGNALS), dtype=np.float64)
        a = NoMonitoringBaseline().run(m)
        b = NoMonitoringBaseline().run(m)
        assert a == b


# ── Module-level constant ────────────────────────────────────────────────────


class TestNoMonitoringAgentIdConstant:
    """The module-level label stays stable and distinct from real agents."""

    def test_value_is_no_monitoring_baseline(self) -> None:
        assert NO_MONITORING_AGENT_ID == "no_monitoring_baseline"

    def test_distinct_from_canonical_agent_ids(self) -> None:
        """The baseline label must not collide with a real agent id."""
        assert NO_MONITORING_AGENT_ID not in AGENT_IDS


# ── Integration with Phase 10.2 metrics ──────────────────────────────────────


class TestNoMonitoringMetricsIntegration:
    """Decision stream plugs into the 10.2 metric functions uniformly."""

    def _matrix(self, rows: int = 20) -> np.ndarray:
        return np.zeros((rows, NUM_SIGNALS), dtype=np.float64)

    def test_allocation_efficiency_is_one_on_any_nonempty_run(self) -> None:
        """Every step succeeds by construction, so efficiency = 1.0 exactly."""
        decisions = NoMonitoringBaseline().run(self._matrix(20))
        rows = [
            {"step_index": d.step_index, "success": d.success, "agent_id": d.agent_id}
            for d in decisions
        ]
        assert allocation_efficiency(rows) == 1.0

    def test_allocation_efficiency_handles_length_one_run(self) -> None:
        decisions = NoMonitoringBaseline().run(self._matrix(1))
        rows = [{"success": d.success} for d in decisions]
        assert allocation_efficiency(rows) == 1.0

    def test_detection_auroc_is_nan_for_single_class_labels(self) -> None:
        """No monitoring means no attack labels and no detector scores.

        The runner will pass all-zero labels (no attack ever detected)
        and all-zero scores (nothing ever flagged) to AUROC. The 10.2
        short-circuit returns nan on single-class labels, which is the
        honest signal: a detector that does nothing has no AUROC.
        """
        n = 20
        y_true = np.zeros(n, dtype=int)
        y_scores = np.zeros(n, dtype=float)
        auroc = detection_auroc(y_true, y_scores)
        assert math.isnan(auroc)

    def test_detection_f1_is_zero_on_all_zero_preds_vs_mixed_labels(self) -> None:
        """When the runner evaluates the baseline against a ground truth
        attack-window, the baseline's all-zero predictions yield F1 = 0.0
        under sklearn's ``zero_division=0.0`` path. This is the honest
        signal the paper tables cite.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
        y_pred = np.zeros_like(y_true)
        assert detection_f1(y_true, y_pred) == 0.0

    def test_detection_f1_is_zero_when_labels_and_preds_all_zero(self) -> None:
        """Both streams empty of attacks -> F1 = 0.0 silently (no warning)."""
        y_true = np.zeros(10, dtype=int)
        y_pred = np.zeros(10, dtype=int)
        assert detection_f1(y_true, y_pred) == 0.0

    def test_decisions_disambiguate_by_agent_id(self) -> None:
        """Each decision's agent_id is a real canonical agent, not the
        baseline label: the runner should see the specific dispatch
        target on every row so allocator-efficiency analyses work on
        the same column as the full-system runs.
        """
        decisions = NoMonitoringBaseline().run(self._matrix(len(AGENT_IDS)))
        observed = {d.agent_id for d in decisions}
        assert observed == set(AGENT_IDS)
        assert NO_MONITORING_AGENT_ID not in observed
