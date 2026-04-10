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

Baseline calibration
--------------------
The first *n_calibration* steps are treated as a clean baseline.  Once that
many steps have accumulated, :attr:`BehavioralCollector.is_calibrated` becomes
``True`` and :attr:`BehavioralCollector.baseline_stats` exposes the per-signal
mean and standard deviation of the calibration window.

Rolling window statistics
-------------------------
:meth:`BehavioralCollector.rolling_stats` returns mean / std / count for the
last *window* steps regardless of calibration state.

Persistence
-----------
:meth:`BehavioralCollector.persist_step` writes one :class:`StepSignals`
record to the database via an open SQLAlchemy :class:`~sqlalchemy.orm.Session`.
The caller is responsible for committing the session.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from chronoagent.db.models import AgentSignalRecord


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

#: Small regularisation constant added to std during calibration.
_STD_REG: float = 1e-8


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
# Rolling / baseline stats helper
# ---------------------------------------------------------------------------


@dataclass
class SignalStats:
    """Per-signal descriptive statistics over a window of steps.

    Attributes:
        mean: Per-signal mean; shape ``(NUM_SIGNALS,)``.
        std: Per-signal standard deviation (regularised); shape ``(NUM_SIGNALS,)``.
        count: Number of steps included in the window.
    """

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: int

    def __post_init__(self) -> None:
        assert self.mean.shape == (NUM_SIGNALS,)
        assert self.std.shape == (NUM_SIGNALS,)
        assert self.count > 0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class BehavioralCollector:
    """Collects :class:`StepSignals` across an experiment run.

    In addition to in-memory storage, the collector supports:

    * **Baseline calibration** — after the first *n_calibration* steps,
      :attr:`is_calibrated` becomes ``True`` and :attr:`baseline_stats`
      exposes per-signal mean/std.
    * **Rolling window statistics** — :meth:`rolling_stats` computes mean/std
      over the last *window* steps on demand.
    * **Database persistence** — :meth:`persist_step` writes a single
      :class:`StepSignals` to the DB via a caller-managed SQLAlchemy session.

    Usage::

        collector = BehavioralCollector(n_calibration=20)
        for pr in prs:
            collector.start_step()
            review = reviewer.review(pr)
            summary = summarizer.summarize(pr, review)
            signals = build_signals(review, summary)
            collector.end_step(signals)

        matrix = collector.get_signal_matrix()  # shape (T, 6)
        stats  = collector.rolling_stats(window=10)

    Attributes:
        n_calibration: Number of clean steps used to build the baseline.
        steps: Ordered list of completed :class:`StepSignals` records.
    """

    def __init__(self, n_calibration: int = 50) -> None:
        """Initialise the collector.

        Args:
            n_calibration: Number of steps to collect before locking the
                baseline.  Must be ≥ 1.  Defaults to 50.

        Raises:
            ValueError: If *n_calibration* < 1.
        """
        if n_calibration < 1:
            raise ValueError(f"n_calibration must be ≥ 1, got {n_calibration}")

        self.n_calibration: int = n_calibration
        self.steps: list[StepSignals] = []
        self._step_start: float | None = None
        self._current: StepSignals | None = None
        self._baseline: SignalStats | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """``True`` once the baseline has been fitted from *n_calibration* steps."""
        return self._baseline is not None

    @property
    def baseline_stats(self) -> SignalStats | None:
        """Per-signal mean/std of the calibration window, or ``None`` if not yet calibrated."""
        return self._baseline

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

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

        Triggers baseline calibration once *n_calibration* steps have been
        collected.

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

        self._try_calibrate()

    # ------------------------------------------------------------------
    # Baseline calibration
    # ------------------------------------------------------------------

    def _try_calibrate(self) -> None:
        """Fit baseline stats once *n_calibration* steps have been accumulated."""
        if self.is_calibrated or len(self.steps) < self.n_calibration:
            return

        calib = np.stack([s.to_array() for s in self.steps[: self.n_calibration]], axis=0)
        self._baseline = SignalStats(
            mean=calib.mean(axis=0),
            std=calib.std(axis=0) + _STD_REG,
            count=self.n_calibration,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

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

    def rolling_stats(self, window: int = 50) -> SignalStats | None:
        """Compute mean / std / count over the last *window* steps.

        Args:
            window: Number of most-recent steps to include.  Must be ≥ 1.

        Returns:
            :class:`SignalStats` for the window, or ``None`` if no steps have
            been recorded.

        Raises:
            ValueError: If *window* < 1.
        """
        if window < 1:
            raise ValueError(f"window must be ≥ 1, got {window}")
        if not self.steps:
            return None
        recent = self.steps[-window:]
        mat = np.stack([s.to_array() for s in recent], axis=0)
        return SignalStats(
            mean=mat.mean(axis=0),
            std=mat.std(axis=0) + _STD_REG,
            count=len(recent),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_step(
        self,
        session: Session,
        signals: StepSignals,
        *,
        agent_id: str,
        task_id: str | None = None,
    ) -> AgentSignalRecord:
        """Persist one :class:`StepSignals` record to the database.

        Creates an :class:`~chronoagent.db.models.AgentSignalRecord`, adds it
        to *session*, and returns it.  The caller is responsible for calling
        ``session.commit()`` (or using a context manager that auto-commits).

        Args:
            session: An open SQLAlchemy :class:`~sqlalchemy.orm.Session`.
            signals: The step signals to persist.
            agent_id: Identifier for the agent (e.g. ``"security_reviewer"``).
            task_id: Optional work-unit identifier (e.g. PR id).

        Returns:
            The freshly created (unsaved) :class:`~chronoagent.db.models.AgentSignalRecord`.
        """
        from chronoagent.db.models import AgentSignalRecord  # lazy — avoids circular dep

        record = AgentSignalRecord(
            agent_id=agent_id,
            task_id=task_id,
            timestamp=datetime.datetime.now(datetime.UTC),
            total_latency_ms=signals.total_latency_ms,
            retrieval_count=signals.retrieval_count,
            token_count=signals.token_count,
            kl_divergence=signals.kl_divergence,
            tool_calls=signals.tool_calls,
            memory_query_entropy=signals.memory_query_entropy,
        )
        session.add(record)
        return record

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded steps, calibration state, and any in-progress step."""
        self.steps = []
        self._step_start = None
        self._current = None
        self._baseline = None

    def __len__(self) -> int:
        """Return the number of completed steps recorded so far."""
        return len(self.steps)
