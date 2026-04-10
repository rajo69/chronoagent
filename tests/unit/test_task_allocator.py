"""Tests for Phase 5 task 5.3: ``DecentralizedTaskAllocator``.

Coverage:
- Construction subscribes to the ``health_updates`` channel on the bus.
- Dict-shaped payloads (as published by the health scorer) are cached.
- :class:`HealthUpdate` dataclass payloads are also accepted.
- Unexpected payload types and malformed dicts are logged and dropped,
  not raised (bus-handler context must not tear down the publisher).
- ``get_snapshot`` returns an independent copy (no aliasing).
- ``get_health_snapshot`` projects the cache into the float mapping
  consumed by :func:`run_contract_net`.
- Boot behaviour: with an empty cache, the healthy specialist still
  wins because ``missing_health_default`` defaults to ``1.0``.
- Boot behaviour with a pessimistic default: empty cache escalates.
- Happy path: full-health updates route tasks to specialists.
- Degraded specialist: live health update redistributes its task to a
  non-specialist peer.
- All-low-health: allocator escalates rather than picking a zombie.
- Subsequent health updates overwrite older ones (last write wins).
- Snapshot is captured at allocate-time, not at construction time.
- ``stop`` unsubscribes and is idempotent.
- Custom matrix / threshold / missing_health_default are honoured.
- Thread-safety smoke test: concurrent publishers plus concurrent
  callers produce coherent results.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from chronoagent.allocator import (
    AGENT_IDS,
    DEFAULT_BID_THRESHOLD,
    DEFAULT_CAPABILITY_MATRIX,
    DEFAULT_MISSING_HEALTH,
    CapabilityMatrix,
    DecentralizedTaskAllocator,
    NegotiationResult,
)
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.scorer.health_scorer import HEALTH_CHANNEL, HealthUpdate


def _full_health_update(agent_id: str, health: float = 1.0) -> HealthUpdate:
    return HealthUpdate(
        agent_id=agent_id,
        health=health,
        bocpd_score=0.0,
        chronos_score=0.0,
    )


def _publish_dict(bus: LocalBus, agent_id: str, health: float) -> None:
    # Matches what TemporalHealthScorer._handle_signal publishes:
    # ``bus.publish(HEALTH_CHANNEL, vars(update))``.
    bus.publish(
        HEALTH_CHANNEL,
        {
            "agent_id": agent_id,
            "health": health,
            "bocpd_score": 0.0,
            "chronos_score": None,
        },
    )


# ===========================================================================
# Construction / subscription
# ===========================================================================


class TestConstruction:
    def test_subscribes_on_health_channel(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        # Publishing something on the channel must reach the allocator.
        _publish_dict(bus, "planner", 0.9)
        snapshot = allocator.get_health_snapshot()
        assert snapshot == {"planner": 0.9}

    def test_default_config_matches_negotiation_defaults(self) -> None:
        allocator = DecentralizedTaskAllocator(LocalBus())
        assert allocator.matrix is DEFAULT_CAPABILITY_MATRIX
        assert allocator.threshold == DEFAULT_BID_THRESHOLD
        assert allocator.missing_health_default == DEFAULT_MISSING_HEALTH

    def test_empty_snapshot_at_boot(self) -> None:
        allocator = DecentralizedTaskAllocator(LocalBus())
        assert allocator.get_snapshot() == {}
        assert allocator.get_health_snapshot() == {}


# ===========================================================================
# Payload handling
# ===========================================================================


class TestPayloadHandling:
    def test_accepts_dict_payload(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "planner", 0.8)
        cached = allocator.get_snapshot()
        assert set(cached) == {"planner"}
        assert cached["planner"].health == 0.8
        assert cached["planner"].bocpd_score == 0.0
        assert cached["planner"].chronos_score is None

    def test_accepts_health_update_instance(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        bus.publish(HEALTH_CHANNEL, _full_health_update("security_reviewer", 0.4))
        cached = allocator.get_snapshot()
        assert cached["security_reviewer"].health == 0.4

    def test_malformed_dict_is_logged_and_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        with caplog.at_level(logging.WARNING, logger="chronoagent.allocator.task_allocator"):
            bus.publish(HEALTH_CHANNEL, {"oops": "no agent_id"})
        assert allocator.get_snapshot() == {}
        assert any("do not match HealthUpdate" in rec.message for rec in caplog.records)

    def test_unknown_payload_type_is_logged_and_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        with caplog.at_level(logging.WARNING, logger="chronoagent.allocator.task_allocator"):
            bus.publish(HEALTH_CHANNEL, 12345)
        assert allocator.get_snapshot() == {}
        assert any("unexpected health payload type" in rec.message for rec in caplog.records)

    def test_unknown_agent_ids_still_cached(self) -> None:
        """Extras are tolerated: negotiation ignores them so they cost nothing."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "rogue_agent", 0.1)
        assert "rogue_agent" in allocator.get_health_snapshot()
        # And the unknown agent does not poison allocation.
        result = allocator.allocate("t1", "plan")
        assert result.assigned_agent == "planner"


# ===========================================================================
# Snapshot isolation
# ===========================================================================


class TestSnapshotIsolation:
    def test_get_snapshot_returns_copy(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "planner", 0.9)
        snap = allocator.get_snapshot()
        snap["planner"] = _full_health_update("planner", 0.0)
        snap["injected"] = _full_health_update("injected", 0.0)
        # Internal cache untouched.
        again = allocator.get_snapshot()
        assert again["planner"].health == 0.9
        assert "injected" not in again

    def test_get_health_snapshot_returns_copy(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "planner", 0.7)
        projection = allocator.get_health_snapshot()
        projection["planner"] = 0.0
        projection["extra"] = 0.0
        assert allocator.get_health_snapshot() == {"planner": 0.7}


# ===========================================================================
# allocate()
# ===========================================================================


class TestAllocate:
    def test_boot_optimistic_default_assigns_specialist(self) -> None:
        # Empty cache, default missing_health_default=1.0 -> specialist wins.
        allocator = DecentralizedTaskAllocator(LocalBus())
        result = allocator.allocate("t1", "plan")
        assert isinstance(result, NegotiationResult)
        assert result.assigned_agent == "planner"
        assert result.escalated is False

    def test_boot_pessimistic_default_escalates(self) -> None:
        # With missing_health_default=0.0 every bid is 0 and we escalate.
        allocator = DecentralizedTaskAllocator(LocalBus(), missing_health_default=0.0)
        result = allocator.allocate("t1", "plan")
        assert result.assigned_agent is None
        assert result.escalated is True

    def test_full_health_routes_to_specialist(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 1.0)
        for task_type, expected in (
            ("plan", "planner"),
            ("security_review", "security_reviewer"),
            ("style_review", "style_reviewer"),
            ("summarize", "summarizer"),
        ):
            result = allocator.allocate(f"t-{task_type}", task_type)
            assert result.assigned_agent == expected
            assert result.escalated is False
            assert result.winning_bid is not None
            assert result.winning_bid.score == pytest.approx(1.0)

    def test_degraded_specialist_redirects_to_peer(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        # Security reviewer is the specialist for security_review, but
        # it is degraded to ~0.  The next-best under the default matrix
        # is style_reviewer (0.55).
        _publish_dict(bus, "planner", 1.0)
        _publish_dict(bus, "security_reviewer", 0.01)
        _publish_dict(bus, "style_reviewer", 1.0)
        _publish_dict(bus, "summarizer", 1.0)
        result = allocator.allocate("t-sec", "security_review")
        assert result.assigned_agent == "style_reviewer"
        assert result.escalated is False

    def test_all_low_health_escalates(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 0.05)
        result = allocator.allocate("t-hot", "plan")
        assert result.assigned_agent is None
        assert result.escalated is True
        assert result.winning_bid is None
        assert "below threshold" in result.rationale

    def test_last_write_wins(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "planner", 0.2)
        _publish_dict(bus, "planner", 0.95)
        assert allocator.get_health_snapshot()["planner"] == 0.95

    def test_snapshot_captured_at_allocate_time(self) -> None:
        """Health updates after allocate() must not retroactively mutate
        the previous result, and updates before the next allocate() must
        affect the next call."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "security_reviewer", 1.0)
        first = allocator.allocate("t1", "security_review")
        assert first.assigned_agent == "security_reviewer"

        _publish_dict(bus, "security_reviewer", 0.01)
        _publish_dict(bus, "style_reviewer", 1.0)
        _publish_dict(bus, "planner", 1.0)
        _publish_dict(bus, "summarizer", 1.0)
        second = allocator.allocate("t2", "security_review")
        assert second.assigned_agent == "style_reviewer"
        # The first result is a frozen dataclass and is unchanged.
        assert first.assigned_agent == "security_reviewer"


# ===========================================================================
# Lifecycle
# ===========================================================================


class TestLifecycle:
    def test_stop_unsubscribes(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_dict(bus, "planner", 0.5)
        assert allocator.get_health_snapshot() == {"planner": 0.5}
        allocator.stop()
        # After stop, further publishes should not update the cache.
        _publish_dict(bus, "planner", 0.1)
        assert allocator.get_health_snapshot() == {"planner": 0.5}

    def test_stop_is_idempotent(self) -> None:
        allocator = DecentralizedTaskAllocator(LocalBus())
        allocator.stop()
        allocator.stop()  # must not raise


# ===========================================================================
# Custom config
# ===========================================================================


class TestCustomConfig:
    def test_custom_matrix_affects_allocation(self) -> None:
        bus = LocalBus()
        # Identity-ish matrix where planner is the ONLY agent that can
        # handle style_review at any reasonable weight.
        weights: dict[str, dict[str, float]] = {
            "planner": {"plan": 1.0, "security_review": 0.5, "style_review": 0.9, "summarize": 0.5},
            "security_reviewer": {
                "plan": 0.1,
                "security_review": 1.0,
                "style_review": 0.1,
                "summarize": 0.1,
            },
            "style_reviewer": {
                "plan": 0.1,
                "security_review": 0.1,
                "style_review": 0.2,
                "summarize": 0.1,
            },
            "summarizer": {
                "plan": 0.1,
                "security_review": 0.1,
                "style_review": 0.1,
                "summarize": 1.0,
            },
        }
        matrix = CapabilityMatrix(weights=weights)
        allocator = DecentralizedTaskAllocator(bus, matrix=matrix)
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 1.0)
        result = allocator.allocate("t", "style_review")
        assert result.assigned_agent == "planner"

    def test_custom_threshold_tightens_escalation(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus, threshold=0.99)
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 0.9)
        # With threshold 0.99 and best bid 0.9, escalate.
        result = allocator.allocate("t", "plan")
        assert result.escalated is True
        assert result.threshold == pytest.approx(0.99)


# ===========================================================================
# Thread safety smoke test
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_publish_and_allocate(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)

        stop_event = threading.Event()
        errors: list[BaseException] = []

        def publisher() -> None:
            try:
                i = 0
                while not stop_event.is_set():
                    _publish_dict(bus, "planner", 0.5 + (i % 2) * 0.4)
                    _publish_dict(bus, "security_reviewer", 0.6)
                    _publish_dict(bus, "style_reviewer", 0.6)
                    _publish_dict(bus, "summarizer", 0.6)
                    i += 1
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        def caller() -> None:
            try:
                for _ in range(500):
                    result = allocator.allocate("t", "plan")
                    # Under any valid interleaving, planner wins or the
                    # round is escalated; the pure negotiation function
                    # never raises for valid inputs.
                    assert result.escalated or result.assigned_agent in set(AGENT_IDS)
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        pubs = [threading.Thread(target=publisher) for _ in range(2)]
        calls = [threading.Thread(target=caller) for _ in range(2)]
        for t in pubs + calls:
            t.start()
        for t in calls:
            t.join()
        stop_event.set()
        for t in pubs:
            t.join()

        assert not errors


# ===========================================================================
# Integration with the health scorer publish shape
# ===========================================================================


class TestHealthScorerIntegration:
    def test_accepts_scorer_vars_payload_shape(self) -> None:
        """The scorer publishes ``vars(HealthUpdate)``; we must accept it."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        update = _full_health_update("summarizer", 0.42)
        bus.publish(HEALTH_CHANNEL, vars(update))
        cached = allocator.get_snapshot()
        assert cached["summarizer"] == update

    def test_handler_survives_bad_payload_mixed_with_good(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        with caplog.at_level(logging.WARNING, logger="chronoagent.allocator.task_allocator"):
            bus.publish(HEALTH_CHANNEL, None)  # bad
            _publish_dict(bus, "planner", 0.75)  # good
            bus.publish(HEALTH_CHANNEL, "nope")  # bad
        snapshot = allocator.get_health_snapshot()
        assert snapshot == {"planner": 0.75}


# ===========================================================================
# Exports
# ===========================================================================


def test_decentralized_task_allocator_reexported() -> None:
    from chronoagent import allocator as pkg

    assert pkg.DecentralizedTaskAllocator is DecentralizedTaskAllocator


def test_base_handler_never_raises_for_arbitrary_payloads() -> None:
    """The base class's bus handler must swallow parse errors rather than
    propagate them to the publisher.  LocalBus does not trap handler
    exceptions, so raising here would break any other subscriber on the
    same channel."""
    bus = LocalBus()
    DecentralizedTaskAllocator(bus)  # subscribes as a side effect
    payloads: list[Any] = [
        None,
        42,
        3.14,
        "string",
        ("tuple",),
        ["list"],
        {"agent_id": "planner"},  # missing required fields
        {"agent_id": "planner", "health": 0.5, "bogus": True},  # extra field
    ]
    for payload in payloads:
        bus.publish(HEALTH_CHANNEL, payload)  # must not raise
