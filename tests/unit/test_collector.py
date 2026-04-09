"""Unit tests for BehavioralCollector and StepSignals (task 1.2)."""

from __future__ import annotations

import pytest

from chronoagent.monitor.collector import (
    NUM_SIGNALS,
    SIGNAL_LABELS,
    BehavioralCollector,
    StepSignals,
)


# ---------------------------------------------------------------------------
# StepSignals
# ---------------------------------------------------------------------------


class TestStepSignals:
    def test_defaults_are_zero(self) -> None:
        s = StepSignals()
        assert s.total_latency_ms == 0.0
        assert s.retrieval_count == 0
        assert s.token_count == 0
        assert s.kl_divergence == 0.0
        assert s.tool_calls == 0
        assert s.memory_query_entropy == 0.0

    def test_to_array_shape_and_dtype(self) -> None:
        import numpy as np

        s = StepSignals(
            total_latency_ms=10.5,
            retrieval_count=3,
            token_count=120,
            kl_divergence=0.42,
            tool_calls=2,
            memory_query_entropy=0.81,
        )
        arr = s.to_array()
        assert arr.shape == (NUM_SIGNALS,)
        assert arr.dtype == np.float64

    def test_to_array_values(self) -> None:
        import numpy as np

        s = StepSignals(
            total_latency_ms=5.0,
            retrieval_count=4,
            token_count=200,
            kl_divergence=1.2,
            tool_calls=3,
            memory_query_entropy=0.55,
        )
        arr = s.to_array()
        expected = [5.0, 4.0, 200.0, 1.2, 3.0, 0.55]
        assert list(arr) == pytest.approx(expected)

    def test_signal_labels_length(self) -> None:
        assert len(SIGNAL_LABELS) == NUM_SIGNALS


# ---------------------------------------------------------------------------
# BehavioralCollector
# ---------------------------------------------------------------------------


class TestBehavioralCollector:
    def _make_signals(self, idx: int = 0) -> StepSignals:
        return StepSignals(
            total_latency_ms=float(10 + idx),
            retrieval_count=idx + 1,
            token_count=100 + idx * 10,
            kl_divergence=0.1 * idx,
            tool_calls=idx + 1,
            memory_query_entropy=0.5,
        )

    def test_empty_matrix_shape(self) -> None:
        import numpy as np

        c = BehavioralCollector()
        mat = c.get_signal_matrix()
        assert mat.shape == (0, NUM_SIGNALS)
        assert mat.dtype == np.float64

    def test_len_starts_at_zero(self) -> None:
        assert len(BehavioralCollector()) == 0

    def test_single_step_records_correctly(self) -> None:
        c = BehavioralCollector()
        s = self._make_signals(0)
        c.start_step()
        c.end_step(s)
        assert len(c) == 1
        mat = c.get_signal_matrix()
        assert mat.shape == (1, NUM_SIGNALS)

    def test_multiple_steps(self) -> None:
        c = BehavioralCollector()
        n = 5
        for i in range(n):
            c.start_step()
            c.end_step(self._make_signals(i))
        assert len(c) == n
        mat = c.get_signal_matrix()
        assert mat.shape == (n, NUM_SIGNALS)

    def test_matrix_values_match_signals(self) -> None:
        import numpy as np

        c = BehavioralCollector()
        s = StepSignals(
            total_latency_ms=99.0,
            retrieval_count=7,
            token_count=333,
            kl_divergence=2.5,
            tool_calls=4,
            memory_query_entropy=0.77,
        )
        c.start_step()
        c.end_step(s)
        mat = c.get_signal_matrix()
        assert mat[0, 1] == pytest.approx(7.0)   # retrieval_count
        assert mat[0, 2] == pytest.approx(333.0)  # token_count
        assert mat[0, 3] == pytest.approx(2.5)    # kl_divergence
        assert mat[0, 4] == pytest.approx(4.0)    # tool_calls
        assert mat[0, 5] == pytest.approx(0.77)   # memory_query_entropy

    def test_latency_filled_from_elapsed_when_zero(self) -> None:
        """If signals.total_latency_ms is 0, elapsed wall-clock time is used."""
        c = BehavioralCollector()
        c.start_step()
        s = StepSignals()  # total_latency_ms defaults to 0.0
        c.end_step(s)
        # The collector should have filled in a positive elapsed time.
        mat = c.get_signal_matrix()
        assert mat[0, 0] > 0.0

    def test_explicit_latency_preserved(self) -> None:
        """If signals.total_latency_ms is set, the collector must not overwrite it."""
        c = BehavioralCollector()
        c.start_step()
        c.end_step(StepSignals(total_latency_ms=42.0))
        mat = c.get_signal_matrix()
        assert mat[0, 0] == pytest.approx(42.0)

    def test_end_step_without_start_raises(self) -> None:
        c = BehavioralCollector()
        with pytest.raises(RuntimeError, match="start_step"):
            c.end_step(StepSignals())

    def test_reset_clears_state(self) -> None:
        c = BehavioralCollector()
        for i in range(3):
            c.start_step()
            c.end_step(self._make_signals(i))
        c.reset()
        assert len(c) == 0
        assert c.get_signal_matrix().shape == (0, NUM_SIGNALS)

    def test_column_ordering_matches_labels(self) -> None:
        """Column order must match SIGNAL_LABELS."""
        import numpy as np

        c = BehavioralCollector()
        s = StepSignals(
            total_latency_ms=1.0,
            retrieval_count=2,
            token_count=3,
            kl_divergence=4.0,
            tool_calls=5,
            memory_query_entropy=6.0,
        )
        c.start_step()
        c.end_step(s)
        mat = c.get_signal_matrix()
        expected_row = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert mat[0] == pytest.approx(expected_row)
