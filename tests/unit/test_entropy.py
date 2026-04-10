"""Unit tests for entropy.py — edge cases and normalization (task 3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from chronoagent.monitor.entropy import retrieval_entropy, step_entropy


# ---------------------------------------------------------------------------
# retrieval_entropy — edge cases
# ---------------------------------------------------------------------------


class TestRetrievalEntropyEdgeCases:
    def test_empty_array_returns_zero(self) -> None:
        result = retrieval_entropy(np.array([], dtype=np.float64))
        assert result == 0.0

    def test_single_element_returns_zero(self) -> None:
        """k=1 → log(1) = 0 max entropy → return 0."""
        result = retrieval_entropy(np.array([0.9]))
        assert result == 0.0

    def test_all_zeros_returns_zero(self) -> None:
        """All-zero scores → no probability mass → entropy 0."""
        result = retrieval_entropy(np.array([0.0, 0.0, 0.0]))
        assert result == 0.0

    def test_all_negative_returns_zero(self) -> None:
        """Negative scores treated as zero mass."""
        result = retrieval_entropy(np.array([-1.0, -2.0, -3.0]))
        assert result == 0.0

    def test_single_nonzero_is_delta_returns_zero(self) -> None:
        """Only one document has weight → delta distribution → entropy 0."""
        result = retrieval_entropy(np.array([1.0, 0.0, 0.0]))
        assert result == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# retrieval_entropy — normalization [0, 1]
# ---------------------------------------------------------------------------


class TestRetrievalEntropyNormalization:
    def test_uniform_returns_one(self) -> None:
        """Uniform distribution = maximum entropy = 1.0."""
        scores = np.ones(5)
        result = retrieval_entropy(scores)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_uniform_two_elements(self) -> None:
        result = retrieval_entropy(np.array([1.0, 1.0]))
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_uniform_large_k(self) -> None:
        k = 100
        result = retrieval_entropy(np.ones(k))
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_result_in_unit_interval(self) -> None:
        """Result must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            k = rng.integers(2, 20)
            scores = rng.random(k)
            result = retrieval_entropy(scores)
            assert 0.0 <= result <= 1.0

    def test_partial_spread_between_bounds(self) -> None:
        """Non-uniform, non-delta → in (0, 1)."""
        result = retrieval_entropy(np.array([0.7, 0.2, 0.1]))
        assert 0.0 < result < 1.0

    def test_more_uniform_higher_entropy(self) -> None:
        """More uniform distribution → higher entropy."""
        concentrated = retrieval_entropy(np.array([0.9, 0.05, 0.05]))
        spread = retrieval_entropy(np.array([0.4, 0.35, 0.25]))
        assert spread > concentrated


# ---------------------------------------------------------------------------
# retrieval_entropy — specific known values
# ---------------------------------------------------------------------------


class TestRetrievalEntropyValues:
    def test_two_equal_scores(self) -> None:
        """H([0.5, 0.5]) / log(2) = 1.0."""
        result = retrieval_entropy(np.array([1.0, 1.0]))
        assert result == pytest.approx(1.0)

    def test_three_equal_scores(self) -> None:
        """H([1/3,1/3,1/3]) / log(3) = 1.0."""
        result = retrieval_entropy(np.array([1.0, 1.0, 1.0]))
        assert result == pytest.approx(1.0)

    def test_scale_invariant(self) -> None:
        """Multiplying scores by a constant should not change entropy."""
        scores = np.array([0.7, 0.2, 0.1])
        assert retrieval_entropy(scores) == pytest.approx(retrieval_entropy(scores * 5.0))

    def test_negative_scores_treated_as_zero(self) -> None:
        """Negative scores should be treated as zero, not weighted negatively."""
        without_neg = retrieval_entropy(np.array([1.0, 0.0, 0.0]))
        with_neg = retrieval_entropy(np.array([1.0, -0.5, -0.5]))
        assert with_neg == pytest.approx(without_neg, abs=1e-10)

    def test_one_hot_delta(self) -> None:
        """Extreme delta: one score >> rest."""
        scores = np.array([1000.0, 0.001, 0.001])
        result = retrieval_entropy(scores)
        # Should be very close to 0 (near-delta)
        assert result < 0.01

    def test_known_entropy_3_elements(self) -> None:
        """Manual check: p = [0.5, 0.25, 0.25], H = log(2)+0.5*log(4) = 3*log(2)/2."""
        scores = np.array([0.5, 0.25, 0.25])
        p = scores / scores.sum()
        h = -np.sum(p * np.log(p))
        h_max = np.log(3.0)
        expected = h / h_max
        result = retrieval_entropy(scores)
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# step_entropy
# ---------------------------------------------------------------------------


class TestStepEntropy:
    def test_empty_batch_list_returns_zero(self) -> None:
        assert step_entropy([]) == 0.0

    def test_single_batch_equals_retrieval_entropy(self) -> None:
        scores = np.array([0.7, 0.2, 0.1])
        expected = retrieval_entropy(scores)
        result = step_entropy([scores])
        assert result == pytest.approx(expected)

    def test_multiple_batches_mean(self) -> None:
        """step_entropy should be the mean of per-batch entropies."""
        b1 = np.array([1.0, 1.0, 1.0])  # entropy = 1.0
        b2 = np.array([1.0, 0.0, 0.0])  # entropy = 0.0
        result = step_entropy([b1, b2])
        assert result == pytest.approx(0.5)

    def test_result_in_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        batches = [rng.random(rng.integers(2, 10)) for _ in range(10)]
        result = step_entropy(batches)
        assert 0.0 <= result <= 1.0

    def test_all_uniform_batches_returns_one(self) -> None:
        batches = [np.ones(5), np.ones(3), np.ones(10)]
        assert step_entropy(batches) == pytest.approx(1.0)

    def test_all_delta_batches_returns_zero(self) -> None:
        batches = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ]
        result = step_entropy(batches)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mixed_batches_between_bounds(self) -> None:
        b1 = np.ones(4)  # max entropy
        b2 = np.array([1.0, 0.0, 0.0, 0.0])  # zero entropy
        result = step_entropy([b1, b2])
        assert 0.0 < result < 1.0

    def test_three_batches_average(self) -> None:
        b1 = np.array([1.0, 1.0])  # entropy = 1.0
        b2 = np.array([1.0, 0.5])  # 2 elements, not uniform
        b3 = np.array([1.0, 1.0, 1.0])  # entropy = 1.0
        e2 = retrieval_entropy(b2)
        expected = (1.0 + e2 + 1.0) / 3.0
        assert step_entropy([b1, b2, b3]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Hypothesis property tests (task 3.7)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


class TestEntropyPropertyTests:
    @given(
        scores=arrays(
            dtype=np.float64,
            shape=st.integers(0, 20),
            elements=st.floats(
                min_value=-10.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    @settings(max_examples=300)
    def test_retrieval_entropy_always_in_unit_interval(self, scores: np.ndarray) -> None:
        """retrieval_entropy must always return a value in [0, 1]."""
        result = retrieval_entropy(scores)
        assert 0.0 <= result <= 1.0, f"entropy {result} out of [0,1] for scores {scores}"
        assert np.isfinite(result)

    @given(
        batches=st.lists(
            arrays(
                dtype=np.float64,
                shape=st.integers(0, 15),
                elements=st.floats(
                    min_value=-10.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=200)
    def test_step_entropy_always_in_unit_interval(self, batches: list[np.ndarray]) -> None:
        """step_entropy must always return a value in [0, 1]."""
        result = step_entropy(batches)
        assert 0.0 <= result <= 1.0, f"step_entropy {result} out of [0,1]"
        assert np.isfinite(result)
