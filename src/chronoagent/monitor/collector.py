"""BehavioralCollector: records per-step agent signals for experiment analysis.

Signals are collected at each processing step (one step = one PR processed by
the agent pair).  After a run, :meth:`BehavioralCollector.get_signal_matrix`
returns a ``(T, 6)`` NumPy array ready for statistical analysis.

Signal ordering (column index):
    0 — total_latency_ms
    1 — retrieval_count
    2 — token_count
    3 — kl_divergence
    4 — tool_calls
    5 — memory_query_entropy
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Signal schema
# ---------------------------------------------------------------------------

#: Number of behavioural signals tracked per step.
NUM_SIGNALS: int = 6

#: Human-readable column labels matching the ``(T, 6)`` matrix.
SIGNAL_LABELS: list[str] = [
    "total_latency_ms",
    "retrieval_count",
    "token_count",
    "kl_divergence",
    "tool_calls",
    "memory_query_entropy",
]


@dataclass
class StepSignals:
    """Behavioural signals recorded for a single agent-pipeline step.

    One step corresponds to the full processing of one unit of work (e.g. one
    synthetic PR) through the agent pair.

    Attributes:
        total_latency_ms: End-to-end wall-clock time for the step in ms.
            Includes retrieval + LLM inference for all agents in the pair.
        retrieval_count: Total number of documents returned from ChromaDB
            across all retrieval calls in the step.
        token_count: Approximate token count of the LLM input prompt(s).
            Proxy for prompt complexity; estimated by whitespace-split length.
        kl_divergence: KL divergence of the retrieval embedding distribution
            from the clean baseline.  Set to ``0.0`` until
            :mod:`~chronoagent.monitor.kl_divergence` is wired in (task 1.3).
        tool_calls: Number of discrete tool / retrieval calls issued during
            the step.  Equals the number of ChromaDB queries.
        memory_query_entropy: Shannon entropy of the top-k similarity score
            distribution from ChromaDB.  Set to ``0.0`` until
            :mod:`~chronoagent.monitor.entropy` is wired in (task 1.4).
    """

    total_latency_ms: float = 0.0
    retrieval_count: int = 0
    token_count: int = 0
    kl_divergence: float = 0.0
    tool_calls: int = 0
    memory_query_entropy: float = 0.0

    def to_array(self) -> NDArray[np.float64]:
        """Return signals as a 1-D float64 array of length :data:`NUM_SIGNALS`.

        Returns:
            Shape ``(6,)`` array in column order defined by :data:`SIGNAL_LABELS`.
        """
        return np.array(
            [
                self.total_latency_ms,
                float(self.retrieval_count),
                float(self.token_count),
                self.kl_divergence,
                float(self.tool_calls),
                self.memory_query_entropy,
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class BehavioralCollector:
    """Collects :class:`StepSignals` across an experiment run.

    Usage::

        collector = BehavioralCollector()
        for pr in prs:
            collector.start_step()
            review = reviewer.review(pr)
            summary = summarizer.summarize(pr, review)
            signals = build_signals(review, summary)
            collector.end_step(signals)

        matrix = collector.get_signal_matrix()  # shape (T, 6)

    Attributes:
        steps: Ordered list of completed :class:`StepSignals` records.
    """

    def __init__(self) -> None:
        self.steps: list[StepSignals] = []
        self._step_start: float | None = None
        self._current: StepSignals | None = None

    def start_step(self) -> None:
        """Mark the beginning of a new pipeline step.

        Records the wall-clock start time.  Must be paired with
        :meth:`end_step`.
        """
        self._step_start = time.perf_counter()
        self._current = StepSignals()

    def end_step(self, signals: StepSignals) -> None:
        """Finalise the current step with observed signals and store it.

        If ``signals.total_latency_ms`` is zero (the default), the elapsed
        time since :meth:`start_step` is used instead.

        Args:
            signals: Populated :class:`StepSignals` for this step.

        Raises:
            RuntimeError: If called without a preceding :meth:`start_step`.
        """
        if self._step_start is None:
            raise RuntimeError("end_step() called without a matching start_step()")

        elapsed_ms = (time.perf_counter() - self._step_start) * 1_000
        if signals.total_latency_ms == 0.0:
            signals.total_latency_ms = elapsed_ms

        self.steps.append(signals)
        self._step_start = None
        self._current = None

    def get_signal_matrix(self) -> NDArray[np.float64]:
        """Return all recorded steps as a ``(T, 6)`` float64 matrix.

        Rows are steps in recording order; columns follow :data:`SIGNAL_LABELS`.

        Returns:
            Array of shape ``(T, 6)`` where *T* is the number of completed
            steps.  Returns shape ``(0, 6)`` if no steps have been recorded.
        """
        if not self.steps:
            return np.empty((0, NUM_SIGNALS), dtype=np.float64)
        return np.stack([s.to_array() for s in self.steps], axis=0)

    def reset(self) -> None:
        """Clear all recorded steps and any in-progress step state."""
        self.steps = []
        self._step_start = None
        self._current = None

    def __len__(self) -> int:
        """Return the number of completed steps recorded so far."""
        return len(self.steps)
