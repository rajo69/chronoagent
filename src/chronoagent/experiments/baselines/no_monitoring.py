"""No-monitoring round-robin baseline for the Phase 10 experiment suite (task 10.4).

The Phase 10 paper's ablation table isolates what *any* monitoring at all
contributes over a dumb comparator: round-robin allocation with no
behavioural signals, no integrity checks, no temporal health scoring,
and no detection. This module ships that comparator so the runner
(task 10.6) can drive it with the same per-step decision interface as
the reactive Sentinel baseline (task 10.3) and the full ChronoAgent
stack.

Design choices locked in
------------------------
* **Round-robin over the canonical agent order.** The baseline cycles
  through :data:`chronoagent.allocator.capability_weights.AGENT_IDS`
  (``planner`` -> ``security_reviewer`` -> ``style_reviewer`` ->
  ``summarizer``) deterministically. Seeds are irrelevant because there
  is no randomness anywhere; two calls with the same ``signal_matrix``
  return byte-identical decision lists.
* **Every step succeeds by construction.** Without a detector, the
  baseline never flags a step, so :attr:`NoMonitoringDecision.success`
  is ``True`` on every row. The metric projection in task 10.2 reads
  :func:`~chronoagent.experiments.metrics.allocation_efficiency` = 1.0,
  and :func:`~chronoagent.experiments.metrics.detection_auroc` /
  :func:`~chronoagent.experiments.metrics.detection_f1` fall through
  their single-class short-circuits to ``float('nan')``. That is the
  honest signal: a detector that does nothing has no AUROC or F1 to
  report, and should contribute nothing to downstream ``np.nanmean``
  aggregation.
* **Decision interface matches the runner contract.** Each step
  produces a :class:`NoMonitoringDecision` carrying the same core fields
  that :class:`~chronoagent.experiments.baselines.sentinel.SentinelDecision`
  exposes (``step_index``, ``success``, ``agent_id``) plus no detector
  score field -- because there is no detector. The runner projects to a
  ``{"success": ..., "agent_id": ...}`` dict when handing rows to the
  metric functions, exactly as it does for the Sentinel baseline.
* **Signal matrix is accepted but ignored.** :meth:`NoMonitoringBaseline.run`
  takes a ``(T, NUM_SIGNALS)`` matrix so the runner can drive every
  baseline with a uniform call site. The matrix is used only to count
  rows and to validate shape; its contents are deliberately untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chronoagent.allocator.capability_weights import AGENT_IDS
from chronoagent.monitor.collector import NUM_SIGNALS

__all__ = [
    "NO_MONITORING_AGENT_ID",
    "NoMonitoringBaseline",
    "NoMonitoringDecision",
]

#: Agent identifier used when the baseline is referenced as a logical
#: detector (e.g. aggregated result rows) rather than as an allocator.
#: Individual :class:`NoMonitoringDecision` rows carry the *specific*
#: round-robin agent (one of :data:`AGENT_IDS`) in their ``agent_id``
#: field so allocator-efficiency analyses see real agent labels; this
#: module-level constant is the label for the baseline-as-a-whole.
NO_MONITORING_AGENT_ID: str = "no_monitoring_baseline"


@dataclass(frozen=True)
class NoMonitoringDecision:
    """Per-step outcome of the round-robin no-monitoring baseline.

    Attributes:
        step_index: 0-based step index within the run.
        success: Always ``True``. Without a detector there is nothing to
            flag, so every allocation is reported as successful. Feeds
            :func:`~chronoagent.experiments.metrics.allocation_efficiency`
            via the mapping-style ``{"success": True, ...}`` entry.
        agent_id: The agent the baseline dispatched to at this step.
            Cycles through :data:`AGENT_IDS` in order:
            step 0 -> ``AGENT_IDS[0]``, step 1 -> ``AGENT_IDS[1]``, etc.,
            wrapping back to index 0 after ``len(AGENT_IDS)`` steps.
    """

    step_index: int
    success: bool
    agent_id: str


class NoMonitoringBaseline:
    """Round-robin allocator with no monitoring at all.

    Workflow:

    1. :meth:`decide` is a pure function of the step index: it returns
       a :class:`NoMonitoringDecision` with ``success=True`` and
       ``agent_id = AGENT_IDS[step_index % len(AGENT_IDS)]``.
    2. :meth:`run` is the convenience entry point the Phase 10 runner
       (task 10.6) calls: it validates the shape of a
       ``(T, NUM_SIGNALS)`` matrix and returns one decision per row.

    The class holds no state, so a single instance can be reused across
    runs and configurations without reset. It exists as a class (rather
    than a pair of free functions) for symmetry with
    :class:`~chronoagent.experiments.baselines.sentinel.SentinelBaseline`
    so the runner can instantiate either baseline through a uniform
    ``Baseline().run(signal_matrix)`` call site.
    """

    def decide(self, step_index: int) -> NoMonitoringDecision:
        """Return the round-robin decision for a single step.

        Args:
            step_index: 0-based step index within the run. Must be
                non-negative.

        Returns:
            :class:`NoMonitoringDecision` with ``success=True`` and
            the round-robin agent for this step.

        Raises:
            ValueError: If *step_index* is negative.
        """
        if step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {step_index}")
        agent_id = AGENT_IDS[step_index % len(AGENT_IDS)]
        return NoMonitoringDecision(
            step_index=int(step_index),
            success=True,
            agent_id=agent_id,
        )

    def run(self, signal_matrix: NDArray[np.float64]) -> list[NoMonitoringDecision]:
        """Produce one decision per row of a full-run signal matrix.

        The matrix contents are ignored; only its shape is used, so the
        method is deterministic in the row count alone. A zero-row
        matrix returns an empty list (vacuously valid).

        Args:
            signal_matrix: Shape ``(T, NUM_SIGNALS)`` full-run matrix.
                Contents are not inspected.

        Returns:
            List of :class:`NoMonitoringDecision`, one per row, in step
            order, cycling through :data:`AGENT_IDS`.

        Raises:
            ValueError: On wrong rank or wrong column count.
        """
        if signal_matrix.ndim != 2 or signal_matrix.shape[1] != NUM_SIGNALS:
            raise ValueError(
                f"signal_matrix must have shape (T, {NUM_SIGNALS}), got {signal_matrix.shape}"
            )
        return [self.decide(i) for i in range(signal_matrix.shape[0])]
