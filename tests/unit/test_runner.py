"""Tests for SignalValidationRunner (task 1.6)."""

from __future__ import annotations

import numpy as np
import pytest

from chronoagent.experiments.runner import (
    ExperimentResult,
    SignalValidationRunner,
    _make_synthetic_prs,
    cohens_d,
)
from chronoagent.monitor.collector import SIGNAL_LABELS

# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


def test_cohens_d_identical_means() -> None:
    """Cohen's d is 0 when both samples have the same mean and variance."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert cohens_d(a, a.copy()) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_large_effect() -> None:
    """Well-separated distributions should yield d > 0.8."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 30).astype(np.float64)
    b = rng.normal(5.0, 1.0, 30).astype(np.float64)
    d = cohens_d(a, b)
    assert d > 0.8


def test_cohens_d_zero_variance() -> None:
    """Returns 0 when pooled std is zero (both arrays identical and constant)."""
    a = np.full(10, 3.0, dtype=np.float64)
    b = np.full(10, 3.0, dtype=np.float64)
    assert cohens_d(a, b) == 0.0


def test_cohens_d_too_small() -> None:
    """Returns 0 for samples shorter than 2."""
    a = np.array([1.0])
    b = np.array([2.0, 3.0])
    assert cohens_d(a, b) == 0.0


def test_cohens_d_symmetric() -> None:
    """Cohen's d is symmetric: d(a,b) == d(b,a)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30).astype(np.float64)
    b = rng.normal(2, 1, 30).astype(np.float64)
    assert cohens_d(a, b) == pytest.approx(cohens_d(b, a), rel=1e-9)


# ---------------------------------------------------------------------------
# _make_synthetic_prs
# ---------------------------------------------------------------------------


def test_make_synthetic_prs_count() -> None:
    """Generates exactly n PRs."""
    prs = _make_synthetic_prs(15, seed=42)
    assert len(prs) == 15


def test_make_synthetic_prs_deterministic() -> None:
    """Same seed → identical PR IDs and titles."""
    prs_a = _make_synthetic_prs(10, seed=7)
    prs_b = _make_synthetic_prs(10, seed=7)
    assert [p.pr_id for p in prs_a] == [p.pr_id for p in prs_b]
    assert [p.title for p in prs_a] == [p.title for p in prs_b]


def test_make_synthetic_prs_unique_ids() -> None:
    """All PR IDs within a batch are unique."""
    prs = _make_synthetic_prs(20, seed=0)
    ids = [p.pr_id for p in prs]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# SignalValidationRunner.create + run (MINJA)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minja_result() -> ExperimentResult:
    """Run a short MINJA experiment (10 steps) — module-scoped for speed."""
    runner = SignalValidationRunner.create(
        attack="minja",
        n_steps=10,
        n_poison_docs=5,
        n_calibration=5,
        seed=42,
        pr_seed=0,
    )
    return runner.run()


def test_minja_result_type(minja_result: ExperimentResult) -> None:
    assert isinstance(minja_result, ExperimentResult)


def test_minja_step_counts(minja_result: ExperimentResult) -> None:
    assert minja_result.n_clean_steps == 10
    assert minja_result.n_poisoned_steps == 10


def test_minja_poison_docs(minja_result: ExperimentResult) -> None:
    # Each collection gets n_poison_docs injected → 2 collections × 5 = 10
    assert minja_result.n_poison_docs == 10


def test_minja_signal_stats_count(minja_result: ExperimentResult) -> None:
    assert len(minja_result.signal_stats) == 6


def test_minja_signal_labels(minja_result: ExperimentResult) -> None:
    assert [s.label for s in minja_result.signal_stats] == SIGNAL_LABELS


def test_minja_stats_non_negative_std(minja_result: ExperimentResult) -> None:
    for s in minja_result.signal_stats:
        assert s.clean_std >= 0.0
        assert s.poisoned_std >= 0.0


def test_minja_cohens_d_non_negative(minja_result: ExperimentResult) -> None:
    for s in minja_result.signal_stats:
        assert s.cohens_d >= 0.0


def test_minja_matrix_shapes(minja_result: ExperimentResult) -> None:
    assert minja_result.clean_matrix.shape == (10, 6)
    assert minja_result.poisoned_matrix.shape == (10, 6)


def test_minja_go_no_go_is_string(minja_result: ExperimentResult) -> None:
    assert minja_result.go_no_go in ("GO", "NO-GO")


def test_minja_summary_contains_decision(minja_result: ExperimentResult) -> None:
    summary = minja_result.summary()
    assert minja_result.go_no_go in summary
    assert any(
        token in summary
        for token in ("Cohen", "cohen", "d>", "d >", "GO")
    )


def test_minja_large_effect_consistent(minja_result: ExperimentResult) -> None:
    """n_large_effects matches manual count."""
    expected = sum(1 for s in minja_result.signal_stats if s.large_effect)
    assert minja_result.n_large_effects == expected


# ---------------------------------------------------------------------------
# SignalValidationRunner.create + run (AgentPoison)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agentpoison_result() -> ExperimentResult:
    runner = SignalValidationRunner.create(
        attack="agentpoison",
        n_steps=10,
        n_poison_docs=5,
        n_calibration=5,
        seed=99,
        pr_seed=10,
    )
    return runner.run()


def test_agentpoison_step_counts(agentpoison_result: ExperimentResult) -> None:
    assert agentpoison_result.n_clean_steps == 10
    assert agentpoison_result.n_poisoned_steps == 10


def test_agentpoison_attack_type(agentpoison_result: ExperimentResult) -> None:
    assert agentpoison_result.attack_type == "AGENTPOISONStyleAttack"


def test_agentpoison_cohens_d_non_negative(agentpoison_result: ExperimentResult) -> None:
    for s in agentpoison_result.signal_stats:
        assert s.cohens_d >= 0.0


# ---------------------------------------------------------------------------
# Invalid attack name
# ---------------------------------------------------------------------------


def test_create_invalid_attack() -> None:
    with pytest.raises(ValueError, match="Unknown attack type"):
        SignalValidationRunner.create(attack="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExperimentResult.summary format
# ---------------------------------------------------------------------------


def test_summary_format(minja_result: ExperimentResult) -> None:
    """Summary string contains all 6 signal labels."""
    summary = minja_result.summary()
    for label in SIGNAL_LABELS:
        assert label in summary
