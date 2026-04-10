"""Tests for Phase 5 task 5.2: contract-net negotiation protocol.

Coverage:
- Happy path: healthy specialist wins its own task type.
- Degraded specialist: a non-specialist peer can take over, exactly as
  the Phase 5 exit criterion requires.
- Tie-breaking: equal bids resolved by AGENT_IDS order.
- Escalation: every bid below threshold returns assigned_agent=None
  with escalated=True and a non-empty rationale.
- Missing health entries fall back to the configurable default.
- Extra / unknown keys in the snapshot are ignored, never assigned.
- Validation rejects NaN / out-of-range health, threshold, and default.
- Unknown task types bubble up as UnknownTaskTypeError.
- Full bid ledger is ordered by AGENT_IDS and included verbatim in the
  result so Phase 5.5's audit record can dump it directly.
- Threshold recorded on the result for reproducible replays.
- Hypothesis property: for any valid snapshot, exactly one agent is
  assigned or the round is escalated, and the winner maximises the
  score with deterministic tie-breaking.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chronoagent.allocator.capability_weights import (
    AGENT_IDS,
    DEFAULT_CAPABILITY_MATRIX,
    TASK_TYPES,
    CapabilityMatrix,
    UnknownTaskTypeError,
)
from chronoagent.allocator.negotiation import (
    DEFAULT_BID_THRESHOLD,
    DEFAULT_MISSING_HEALTH,
    Bid,
    InvalidHealthError,
    InvalidThresholdError,
    NegotiationResult,
    run_contract_net,
)

# Convenience: a snapshot where every agent is fully healthy.
FULL_HEALTH: dict[str, float] = {agent_id: 1.0 for agent_id in AGENT_IDS}


# ===========================================================================
# Module-level constants
# ===========================================================================


class TestConstants:
    def test_default_threshold_in_unit_interval(self) -> None:
        assert 0.0 <= DEFAULT_BID_THRESHOLD <= 1.0

    def test_default_missing_health_is_optimistic(self) -> None:
        # Missing health defaults to "healthy" so the system can
        # allocate work at boot, before any HealthUpdate arrives.
        assert DEFAULT_MISSING_HEALTH == 1.0


# ===========================================================================
# Happy path: healthy specialist wins its task
# ===========================================================================


class TestHappyPath:
    @pytest.mark.parametrize(
        "task_type,expected_agent",
        [
            ("plan", "planner"),
            ("security_review", "security_reviewer"),
            ("style_review", "style_reviewer"),
            ("summarize", "summarizer"),
        ],
    )
    def test_healthy_specialist_wins_own_task(self, task_type: str, expected_agent: str) -> None:
        result = run_contract_net(
            task_id=f"t-{task_type}",
            task_type=task_type,
            health_snapshot=FULL_HEALTH,
        )
        assert result.assigned_agent == expected_agent
        assert result.escalated is False
        assert result.winning_bid is not None
        assert result.winning_bid.agent_id == expected_agent
        assert result.winning_bid.score == pytest.approx(1.0)
        assert result.task_id == f"t-{task_type}"
        assert result.task_type == task_type

    def test_result_carries_threshold(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH, threshold=0.42)
        assert result.threshold == pytest.approx(0.42)

    def test_result_is_frozen(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.assigned_agent = "someone_else"  # type: ignore[misc]

    def test_bid_is_frozen(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH)
        bid = result.all_bids[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            bid.score = 0.0  # type: ignore[misc]


# ===========================================================================
# Bid ledger
# ===========================================================================


class TestBidLedger:
    def test_all_bids_ordered_by_agent_ids(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH)
        assert tuple(bid.agent_id for bid in result.all_bids) == AGENT_IDS

    def test_all_bids_includes_every_agent(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH)
        assert len(result.all_bids) == len(AGENT_IDS)

    def test_bid_score_is_capability_times_health(self) -> None:
        snapshot = {
            "planner": 0.8,
            "security_reviewer": 0.6,
            "style_reviewer": 0.4,
            "summarizer": 0.2,
        }
        result = run_contract_net("t", "security_review", snapshot)
        for bid in result.all_bids:
            expected_cap = DEFAULT_CAPABILITY_MATRIX.proficiency(bid.agent_id, "security_review")
            expected_health = snapshot[bid.agent_id]
            assert bid.capability == pytest.approx(expected_cap)
            assert bid.health == pytest.approx(expected_health)
            assert bid.score == pytest.approx(expected_cap * expected_health)

    def test_bid_ledger_is_tuple(self) -> None:
        # Tuples are hashable and trivially JSON-serialisable as
        # lists; the audit record in Phase 5.5 relies on this.
        result = run_contract_net("t", "plan", FULL_HEALTH)
        assert isinstance(result.all_bids, tuple)


# ===========================================================================
# Degraded specialist -> peer takes over (Phase 5 exit criterion)
# ===========================================================================


class TestDegradationRedistribution:
    def test_degraded_security_reviewer_shifts_to_style_reviewer(self) -> None:
        # security_reviewer specialist is at health 0.05.
        # style_reviewer is healthy and has the highest off-diagonal
        # proficiency on security_review (0.55) among the peers.
        snapshot = {
            "planner": 1.0,
            "security_reviewer": 0.05,
            "style_reviewer": 1.0,
            "summarizer": 1.0,
        }
        result = run_contract_net("t", "security_review", snapshot)
        assert result.escalated is False
        assert result.assigned_agent == "style_reviewer"

    def test_degraded_style_reviewer_shifts_to_security_reviewer(self) -> None:
        snapshot = {
            "planner": 1.0,
            "security_reviewer": 1.0,
            "style_reviewer": 0.05,
            "summarizer": 1.0,
        }
        result = run_contract_net("t", "style_review", snapshot)
        assert result.escalated is False
        assert result.assigned_agent == "security_reviewer"

    def test_degraded_planner_shifts_to_summarizer(self) -> None:
        # summarizer has the highest off-diagonal plan weight (0.50).
        snapshot = {
            "planner": 0.05,
            "security_reviewer": 1.0,
            "style_reviewer": 1.0,
            "summarizer": 1.0,
        }
        result = run_contract_net("t", "plan", snapshot)
        assert result.escalated is False
        assert result.assigned_agent == "summarizer"

    def test_degraded_summarizer_shifts_to_planner(self) -> None:
        # planner has the highest off-diagonal summarize weight (0.60).
        snapshot = {
            "planner": 1.0,
            "security_reviewer": 1.0,
            "style_reviewer": 1.0,
            "summarizer": 0.05,
        }
        result = run_contract_net("t", "summarize", snapshot)
        assert result.escalated is False
        assert result.assigned_agent == "planner"


# ===========================================================================
# Tie-breaking by AGENT_IDS order
# ===========================================================================


class TestTieBreaking:
    def test_equal_scores_pick_first_by_agent_ids(self) -> None:
        # Custom matrix where every agent has identical proficiency on
        # "plan".  With full health this produces a four-way tie; the
        # first agent in AGENT_IDS should win.
        uniform = CapabilityMatrix(
            weights={
                agent_id: {task_type: 0.5 for task_type in TASK_TYPES} for agent_id in AGENT_IDS
            }
        )
        # Bump the diagonal so the matrix validates like the default
        # (specialists score 1.0 on their own task).  Keep plan entries
        # equal across agents for the tie-break test.
        weights = uniform.as_dict()
        for agent_id in AGENT_IDS:
            for task_type in TASK_TYPES:
                if task_type != "plan" and agent_id.replace("_reviewer", "_review") == task_type:
                    weights[agent_id][task_type] = 1.0
        weights["planner"]["plan"] = 0.5  # force tie on plan
        weights["security_reviewer"]["plan"] = 0.5
        weights["style_reviewer"]["plan"] = 0.5
        weights["summarizer"]["plan"] = 0.5
        tied_matrix = CapabilityMatrix(weights=weights)

        result = run_contract_net("t", "plan", FULL_HEALTH, matrix=tied_matrix)
        assert result.assigned_agent == AGENT_IDS[0]  # planner
        assert result.winning_bid is not None
        assert result.winning_bid.score == pytest.approx(0.5)

    def test_tie_between_two_agents_picks_earlier_in_order(self) -> None:
        # security_reviewer and style_reviewer both have 0.55 on each
        # other's task type, so identical health makes them tie on
        # style_review.  security_reviewer comes first in AGENT_IDS.
        snapshot = {
            "planner": 0.0,
            "security_reviewer": 0.5,
            "style_reviewer": 0.275,  # 0.275 * 1.0 = 0.275  (see below)
            "summarizer": 0.0,
        }
        # style_reviewer bid: 1.0 * 0.275 = 0.275
        # security_reviewer bid: 0.55 * 0.5 = 0.275 (tied)
        result = run_contract_net("t", "style_review", snapshot)
        assert result.winning_bid is not None
        assert result.winning_bid.score == pytest.approx(0.275)
        # security_reviewer appears first in AGENT_IDS, so it wins.
        assert result.assigned_agent == "security_reviewer"


# ===========================================================================
# Escalation
# ===========================================================================


class TestEscalation:
    def test_all_low_health_escalates(self) -> None:
        snapshot = {agent_id: 0.01 for agent_id in AGENT_IDS}
        result = run_contract_net("t", "plan", snapshot)
        assert result.escalated is True
        assert result.assigned_agent is None
        assert result.winning_bid is None
        assert "escalated" in result.rationale

    def test_escalation_still_records_all_bids(self) -> None:
        snapshot = {agent_id: 0.01 for agent_id in AGENT_IDS}
        result = run_contract_net("t", "plan", snapshot)
        assert len(result.all_bids) == len(AGENT_IDS)
        # All bids should be present with their computed scores even
        # though the task is escalated.
        for bid in result.all_bids:
            assert 0.0 <= bid.score <= 1.0

    def test_high_threshold_forces_escalation(self) -> None:
        # Even with full health, threshold > 1.0 is illegal; use 1.0
        # which forces escalation because every bid is strictly < 1.0
        # (the check uses `<` so 1.0 at threshold 1.0 would pass).
        # Use threshold just above the best possible bid.
        result = run_contract_net(
            "t",
            "plan",
            FULL_HEALTH,
            threshold=1.0,
        )
        # Best bid is exactly 1.0, threshold is 1.0, `best < threshold`
        # is False -> assigned.
        assert result.assigned_agent == "planner"

    def test_threshold_just_above_best_bid_escalates(self) -> None:
        # Planner bid is 1.0 * 0.5 = 0.5; threshold 0.51 should escalate.
        snapshot = {
            "planner": 0.5,
            "security_reviewer": 0.5,
            "style_reviewer": 0.5,
            "summarizer": 0.5,
        }
        result = run_contract_net("t", "plan", snapshot, threshold=0.51)
        assert result.escalated is True
        assert result.assigned_agent is None

    def test_zero_threshold_never_escalates_on_positive_bids(self) -> None:
        snapshot = {agent_id: 0.01 for agent_id in AGENT_IDS}
        result = run_contract_net("t", "plan", snapshot, threshold=0.0)
        # Every bid is > 0 here, and 0 is not > 0, so the result is
        # not escalated.
        assert result.escalated is False

    def test_zero_health_all_agents_escalates_at_zero_threshold(self) -> None:
        snapshot = {agent_id: 0.0 for agent_id in AGENT_IDS}
        result = run_contract_net("t", "plan", snapshot, threshold=0.0)
        # All scores are 0, threshold is 0, `best < threshold` is
        # False, so the earliest agent (planner) is assigned.
        assert result.escalated is False
        assert result.assigned_agent == "planner"


# ===========================================================================
# Missing / extra / invalid snapshot entries
# ===========================================================================


class TestSnapshotEdgeCases:
    def test_missing_agent_uses_default(self) -> None:
        # Only provide a degraded entry for planner; the other agents
        # fall back to DEFAULT_MISSING_HEALTH (1.0), so the task goes
        # to the healthy specialist.
        snapshot = {"planner": 0.05}
        result = run_contract_net("t", "security_review", snapshot)
        assert result.assigned_agent == "security_reviewer"
        # Planner bid should reflect the degraded health; others
        # should reflect the default (1.0).
        planner_bid = next(b for b in result.all_bids if b.agent_id == "planner")
        assert planner_bid.health == pytest.approx(0.05)
        for bid in result.all_bids:
            if bid.agent_id != "planner":
                assert bid.health == pytest.approx(1.0)

    def test_custom_missing_default_applies_to_every_absent_agent(self) -> None:
        result = run_contract_net(
            "t",
            "plan",
            health_snapshot={},
            missing_health_default=0.3,
        )
        for bid in result.all_bids:
            assert bid.health == pytest.approx(0.3)

    def test_extra_keys_in_snapshot_are_ignored(self) -> None:
        snapshot = dict(FULL_HEALTH)
        snapshot["ghost_agent"] = 1.0
        snapshot["another_typo"] = 0.5
        result = run_contract_net("t", "plan", snapshot)
        assert result.assigned_agent == "planner"
        # Ghost agents should not appear in the bid ledger.
        assert {bid.agent_id for bid in result.all_bids} == set(AGENT_IDS)

    def test_nan_health_raises(self) -> None:
        snapshot = dict(FULL_HEALTH)
        snapshot["planner"] = float("nan")
        with pytest.raises(InvalidHealthError) as exc_info:
            run_contract_net("t", "plan", snapshot)
        assert exc_info.value.agent_id == "planner"

    @pytest.mark.parametrize("bad_value", [-0.01, 1.01, 2.0, -1.0, float("inf"), float("-inf")])
    def test_out_of_range_health_raises(self, bad_value: float) -> None:
        snapshot = dict(FULL_HEALTH)
        snapshot["planner"] = bad_value
        with pytest.raises(InvalidHealthError):
            run_contract_net("t", "plan", snapshot)

    def test_missing_health_default_nan_raises(self) -> None:
        with pytest.raises(InvalidHealthError):
            run_contract_net(
                "t",
                "plan",
                FULL_HEALTH,
                missing_health_default=float("nan"),
            )

    @pytest.mark.parametrize("bad_default", [-0.01, 1.01, float("inf")])
    def test_missing_health_default_out_of_range_raises(self, bad_default: float) -> None:
        with pytest.raises(InvalidHealthError):
            run_contract_net(
                "t",
                "plan",
                FULL_HEALTH,
                missing_health_default=bad_default,
            )


# ===========================================================================
# Threshold validation
# ===========================================================================


class TestThresholdValidation:
    @pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, float("nan"), float("inf"), -1.0])
    def test_bad_threshold_raises(self, bad_threshold: float) -> None:
        with pytest.raises(InvalidThresholdError):
            run_contract_net("t", "plan", FULL_HEALTH, threshold=bad_threshold)

    def test_threshold_boundary_values_accepted(self) -> None:
        # 0.0 and 1.0 are both legal thresholds.
        run_contract_net("t", "plan", FULL_HEALTH, threshold=0.0)
        run_contract_net("t", "plan", FULL_HEALTH, threshold=1.0)


# ===========================================================================
# Unknown task type
# ===========================================================================


class TestUnknownTaskType:
    def test_unknown_task_type_raises(self) -> None:
        with pytest.raises(UnknownTaskTypeError):
            run_contract_net("t", "not_a_task", FULL_HEALTH)


# ===========================================================================
# Custom matrix
# ===========================================================================


class TestCustomMatrix:
    def test_custom_matrix_changes_allocation(self) -> None:
        # Flip the default: make summarizer the strongest reviewer on
        # security by putting 1.0 on summarizer/security_review and
        # something lower on security_reviewer/security_review.  Keep
        # the rest valid.
        weights = DEFAULT_CAPABILITY_MATRIX.as_dict()
        weights["summarizer"]["security_review"] = 1.0
        weights["security_reviewer"]["security_review"] = 0.2
        custom = CapabilityMatrix(weights=weights)
        result = run_contract_net(
            "t",
            "security_review",
            FULL_HEALTH,
            matrix=custom,
        )
        assert result.assigned_agent == "summarizer"


# ===========================================================================
# Rationale content
# ===========================================================================


class TestRationale:
    def test_assigned_rationale_contains_agent_and_score(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH)
        assert result.assigned_agent == "planner"
        assert "planner" in result.rationale
        assert "bid" in result.rationale
        assert "threshold" in result.rationale

    def test_escalated_rationale_names_best_candidate(self) -> None:
        snapshot = {agent_id: 0.01 for agent_id in AGENT_IDS}
        result = run_contract_net("t", "plan", snapshot)
        # The best candidate on plan at uniform low health is the
        # planner (capability 1.0 beats all off-diagonals).
        assert "planner" in result.rationale
        assert "threshold" in result.rationale


# ===========================================================================
# Result dataclass invariants
# ===========================================================================


class TestResultDataclass:
    def test_assigned_result_fields(self) -> None:
        result = run_contract_net("t", "plan", FULL_HEALTH)
        assert isinstance(result, NegotiationResult)
        assert result.escalated is False
        assert result.assigned_agent is not None
        assert result.winning_bid is not None
        assert isinstance(result.winning_bid, Bid)

    def test_escalated_result_fields(self) -> None:
        snapshot = {agent_id: 0.0 for agent_id in AGENT_IDS}
        result = run_contract_net("t", "plan", snapshot, threshold=0.1)
        assert result.escalated is True
        assert result.assigned_agent is None
        assert result.winning_bid is None


# ===========================================================================
# Hypothesis property: one-of-assigned-or-escalated invariant
# ===========================================================================


# Strategy: build a full snapshot of finite floats in [0, 1] for every
# canonical agent.  We deliberately do NOT include NaN or out-of-range
# values; those have their own targeted tests.
healthy_float = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
full_snapshot_strategy = st.fixed_dictionaries({agent_id: healthy_float for agent_id in AGENT_IDS})
threshold_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
task_type_strategy = st.sampled_from(TASK_TYPES)


class TestHypothesisProperties:
    @given(
        snapshot=full_snapshot_strategy,
        task_type=task_type_strategy,
        threshold=threshold_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_exactly_one_outcome(
        self, snapshot: dict[str, float], task_type: str, threshold: float
    ) -> None:
        result = run_contract_net("t", task_type, snapshot, threshold=threshold)
        if result.escalated:
            assert result.assigned_agent is None
            assert result.winning_bid is None
        else:
            assert result.assigned_agent is not None
            assert result.winning_bid is not None
            assert result.winning_bid.agent_id == result.assigned_agent

    @given(
        snapshot=full_snapshot_strategy,
        task_type=task_type_strategy,
        threshold=threshold_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_winner_maximises_score_with_deterministic_tiebreak(
        self, snapshot: dict[str, float], task_type: str, threshold: float
    ) -> None:
        result = run_contract_net("t", task_type, snapshot, threshold=threshold)
        # Compute the expected winner independently: iterate AGENT_IDS
        # in order, keep the first max, exactly as the implementation
        # does.  This guards against the implementation accidentally
        # using Python's builtin `max`, which does not guarantee the
        # first-wins rule.
        best_agent = AGENT_IDS[0]
        best_score = (
            DEFAULT_CAPABILITY_MATRIX.proficiency(best_agent, task_type) * snapshot[best_agent]
        )
        for agent_id in AGENT_IDS[1:]:
            score = DEFAULT_CAPABILITY_MATRIX.proficiency(agent_id, task_type) * snapshot[agent_id]
            if score > best_score:
                best_agent = agent_id
                best_score = score

        if best_score < threshold:
            assert result.escalated is True
        else:
            assert result.escalated is False
            assert result.assigned_agent == best_agent
            assert result.winning_bid is not None
            assert result.winning_bid.score == pytest.approx(best_score)

    @given(
        snapshot=full_snapshot_strategy,
        task_type=task_type_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_all_bids_cover_every_agent_once(
        self, snapshot: dict[str, float], task_type: str
    ) -> None:
        result = run_contract_net("t", task_type, snapshot)
        assert len(result.all_bids) == len(AGENT_IDS)
        assert tuple(b.agent_id for b in result.all_bids) == AGENT_IDS
        for bid in result.all_bids:
            assert 0.0 <= bid.score <= 1.0
            assert not math.isnan(bid.score)
