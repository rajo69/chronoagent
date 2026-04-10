"""Tests for Phase 5 task 5.1: capability weights matrix.

Coverage:
- Default 4 agents x 4 task types matrix is well-formed.
- Diagonal specialists score 1.0; off-diagonal entries are strictly
  inside (0, 1) so the contract-net protocol can always redistribute
  work to a healthy peer (Phase 5 exit criterion).
- Lookup helpers (proficiency, row, column, primary_agent) return the
  expected values and raise typed errors on bad input.
- The matrix is immutable: as_dict returns a deep copy and the
  underlying mappings are read-only.
- Construction validates row/column completeness and value bounds.
- Hypothesis property: every (agent, task) lookup is in [0, 1].
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chronoagent.allocator.capability_weights import (
    AGENT_IDS,
    DEFAULT_CAPABILITY_MATRIX,
    TASK_TYPES,
    CapabilityMatrix,
    UnknownAgentError,
    UnknownTaskTypeError,
)

# ===========================================================================
# Module constants
# ===========================================================================


class TestConstants:
    def test_agent_ids_match_pipeline_agents(self) -> None:
        # These IDs must match the agent_id defaults in
        # chronoagent.agents.* so allocation lines up with health updates.
        assert AGENT_IDS == (
            "planner",
            "security_reviewer",
            "style_reviewer",
            "summarizer",
        )

    def test_task_types_match_registry(self) -> None:
        from chronoagent.agents.registry import AgentRegistry

        registry_tasks = set(AgentRegistry().supported_task_types())
        assert set(TASK_TYPES) == registry_tasks

    def test_four_by_four(self) -> None:
        assert len(AGENT_IDS) == 4
        assert len(TASK_TYPES) == 4


# ===========================================================================
# Default matrix shape & values
# ===========================================================================


class TestDefaultMatrix:
    def test_singleton_is_capability_matrix(self) -> None:
        assert isinstance(DEFAULT_CAPABILITY_MATRIX, CapabilityMatrix)

    def test_all_values_in_unit_interval(self) -> None:
        for agent_id in AGENT_IDS:
            for task_type in TASK_TYPES:
                value = DEFAULT_CAPABILITY_MATRIX.proficiency(agent_id, task_type)
                assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize(
        ("agent_id", "task_type"),
        [
            ("planner", "plan"),
            ("security_reviewer", "security_review"),
            ("style_reviewer", "style_review"),
            ("summarizer", "summarize"),
        ],
    )
    def test_diagonal_specialists_are_perfect(self, agent_id: str, task_type: str) -> None:
        assert DEFAULT_CAPABILITY_MATRIX.proficiency(agent_id, task_type) == 1.0

    def test_off_diagonal_strictly_positive(self) -> None:
        # Phase 5 exit criterion: a degraded specialist must be able to
        # shed work to a peer with nonzero capability.  No off-diagonal
        # cell may be exactly zero.
        for agent_id in AGENT_IDS:
            for task_type in TASK_TYPES:
                if (agent_id, task_type) in {
                    ("planner", "plan"),
                    ("security_reviewer", "security_review"),
                    ("style_reviewer", "style_review"),
                    ("summarizer", "summarize"),
                }:
                    continue
                value = DEFAULT_CAPABILITY_MATRIX.proficiency(agent_id, task_type)
                assert 0.0 < value < 1.0

    def test_security_style_review_overlap_high(self) -> None:
        # Security and style reviewers share a "code review" affinity;
        # their cross weights should beat plan/summarize cross weights.
        sec_on_style = DEFAULT_CAPABILITY_MATRIX.proficiency("security_reviewer", "style_review")
        sec_on_plan = DEFAULT_CAPABILITY_MATRIX.proficiency("security_reviewer", "plan")
        sec_on_sum = DEFAULT_CAPABILITY_MATRIX.proficiency("security_reviewer", "summarize")
        assert sec_on_style > sec_on_plan
        assert sec_on_style > sec_on_sum

        style_on_sec = DEFAULT_CAPABILITY_MATRIX.proficiency("style_reviewer", "security_review")
        style_on_plan = DEFAULT_CAPABILITY_MATRIX.proficiency("style_reviewer", "plan")
        style_on_sum = DEFAULT_CAPABILITY_MATRIX.proficiency("style_reviewer", "summarize")
        assert style_on_sec > style_on_plan
        assert style_on_sec > style_on_sum


# ===========================================================================
# Lookup helpers
# ===========================================================================


class TestProficiency:
    def test_unknown_agent_raises(self) -> None:
        with pytest.raises(UnknownAgentError):
            DEFAULT_CAPABILITY_MATRIX.proficiency("ghost_agent", "plan")

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(UnknownTaskTypeError):
            DEFAULT_CAPABILITY_MATRIX.proficiency("planner", "tea_break")


class TestRow:
    def test_returns_full_row(self) -> None:
        row = DEFAULT_CAPABILITY_MATRIX.row("planner")
        assert set(row.keys()) == set(TASK_TYPES)
        assert row["plan"] == 1.0

    def test_row_is_a_copy(self) -> None:
        row = DEFAULT_CAPABILITY_MATRIX.row("planner")
        row["plan"] = 0.0
        # Mutation must not leak into the canonical matrix.
        assert DEFAULT_CAPABILITY_MATRIX.proficiency("planner", "plan") == 1.0

    def test_unknown_agent_raises(self) -> None:
        with pytest.raises(UnknownAgentError):
            DEFAULT_CAPABILITY_MATRIX.row("ghost_agent")


class TestColumn:
    def test_returns_all_agents_in_canonical_order(self) -> None:
        column = DEFAULT_CAPABILITY_MATRIX.column("security_review")
        assert list(column.keys()) == list(AGENT_IDS)
        assert column["security_reviewer"] == 1.0

    def test_column_is_a_copy(self) -> None:
        column = DEFAULT_CAPABILITY_MATRIX.column("plan")
        column["planner"] = 0.0
        assert DEFAULT_CAPABILITY_MATRIX.proficiency("planner", "plan") == 1.0

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(UnknownTaskTypeError):
            DEFAULT_CAPABILITY_MATRIX.column("tea_break")


class TestPrimaryAgent:
    @pytest.mark.parametrize(
        ("task_type", "expected"),
        [
            ("plan", "planner"),
            ("security_review", "security_reviewer"),
            ("style_review", "style_reviewer"),
            ("summarize", "summarizer"),
        ],
    )
    def test_diagonal_specialists_are_primary(self, task_type: str, expected: str) -> None:
        assert DEFAULT_CAPABILITY_MATRIX.primary_agent(task_type) == expected

    def test_ties_broken_by_agent_id_order(self) -> None:
        # Build a matrix where two agents tie for "summarize"; the
        # earlier one in AGENT_IDS must win for determinism.
        weights: dict[str, dict[str, float]] = {
            "planner": {
                "plan": 1.0,
                "security_review": 0.5,
                "style_review": 0.5,
                "summarize": 0.9,
            },
            "security_reviewer": {
                "plan": 0.5,
                "security_review": 1.0,
                "style_review": 0.5,
                "summarize": 0.9,
            },
            "style_reviewer": {
                "plan": 0.5,
                "security_review": 0.5,
                "style_review": 1.0,
                "summarize": 0.5,
            },
            "summarizer": {
                "plan": 0.5,
                "security_review": 0.5,
                "style_review": 0.5,
                "summarize": 0.5,
            },
        }
        matrix = CapabilityMatrix(weights=weights)
        # planner appears before security_reviewer in AGENT_IDS, so it
        # should win the tie at 0.9.
        assert matrix.primary_agent("summarize") == "planner"

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(UnknownTaskTypeError):
            DEFAULT_CAPABILITY_MATRIX.primary_agent("tea_break")


# ===========================================================================
# Snapshot helper
# ===========================================================================


class TestAsDict:
    def test_round_trip_shape(self) -> None:
        snapshot = DEFAULT_CAPABILITY_MATRIX.as_dict()
        assert set(snapshot.keys()) == set(AGENT_IDS)
        for row in snapshot.values():
            assert set(row.keys()) == set(TASK_TYPES)

    def test_returns_deep_copy(self) -> None:
        snapshot = DEFAULT_CAPABILITY_MATRIX.as_dict()
        snapshot["planner"]["plan"] = 0.0
        assert DEFAULT_CAPABILITY_MATRIX.proficiency("planner", "plan") == 1.0


# ===========================================================================
# Construction validation
# ===========================================================================


def _full_valid_weights() -> dict[str, dict[str, float]]:
    return {agent_id: {task: 0.5 for task in TASK_TYPES} for agent_id in AGENT_IDS}


class TestValidation:
    def test_accepts_well_formed_custom_matrix(self) -> None:
        matrix = CapabilityMatrix(weights=_full_valid_weights())
        for agent_id in AGENT_IDS:
            for task_type in TASK_TYPES:
                assert matrix.proficiency(agent_id, task_type) == 0.5

    def test_rejects_missing_agent_row(self) -> None:
        weights = _full_valid_weights()
        del weights["planner"]
        with pytest.raises(ValueError, match="missing"):
            CapabilityMatrix(weights=weights)

    def test_rejects_extra_agent_row(self) -> None:
        weights = _full_valid_weights()
        weights["ghost_agent"] = {task: 0.5 for task in TASK_TYPES}
        with pytest.raises(ValueError, match="extra"):
            CapabilityMatrix(weights=weights)

    def test_rejects_missing_task_in_row(self) -> None:
        weights = _full_valid_weights()
        del weights["planner"]["plan"]
        with pytest.raises(ValueError, match="missing"):
            CapabilityMatrix(weights=weights)

    def test_rejects_extra_task_in_row(self) -> None:
        weights = _full_valid_weights()
        weights["planner"]["tea_break"] = 0.5
        with pytest.raises(ValueError, match="extra"):
            CapabilityMatrix(weights=weights)

    def test_rejects_negative_value(self) -> None:
        weights = _full_valid_weights()
        weights["planner"]["plan"] = -0.1
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            CapabilityMatrix(weights=weights)

    def test_rejects_value_above_one(self) -> None:
        weights = _full_valid_weights()
        weights["planner"]["plan"] = 1.5
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            CapabilityMatrix(weights=weights)

    def test_rejects_nan_value(self) -> None:
        weights = _full_valid_weights()
        weights["planner"]["plan"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            CapabilityMatrix(weights=weights)

    def test_rejects_non_numeric_value(self) -> None:
        weights: dict[str, dict[str, float]] = _full_valid_weights()
        # mypy: deliberately wrong type to exercise the type guard.
        weights["planner"]["plan"] = "high"  # type: ignore[assignment]
        with pytest.raises(TypeError, match="must be a number"):
            CapabilityMatrix(weights=weights)


# ===========================================================================
# Hypothesis property test
# ===========================================================================


@given(
    agent_id=st.sampled_from(AGENT_IDS),
    task_type=st.sampled_from(TASK_TYPES),
)
@settings(max_examples=50, deadline=None)
def test_proficiency_always_in_unit_interval(agent_id: str, task_type: str) -> None:
    value = DEFAULT_CAPABILITY_MATRIX.proficiency(agent_id, task_type)
    assert 0.0 <= value <= 1.0
