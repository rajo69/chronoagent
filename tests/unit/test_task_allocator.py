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
- Round-robin fallback (task 5.6): negotiation timeout/error -> the
  allocator logs a warning and synthesizes a NegotiationResult whose
  ``assigned_agent`` cycles through ``AGENT_IDS`` deterministically.
- Phase 5.7 integrated sweep: Hypothesis property tests at the
  bus + cache + negotiation layer (parity with the pure negotiation
  function on random valid snapshots; exactly-one-outcome invariant;
  last-write-wins cache invariant under random publish streams) and a
  deterministic mock-health "degradation walk" exercising the
  full-health -> peer -> escalation transitions.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chronoagent.allocator import (
    AGENT_IDS,
    DEFAULT_BID_THRESHOLD,
    DEFAULT_CAPABILITY_MATRIX,
    DEFAULT_MISSING_HEALTH,
    TASK_TYPES,
    CapabilityMatrix,
    DecentralizedTaskAllocator,
    NegotiationResult,
    run_contract_net,
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


# ===========================================================================
# Round-robin fallback (Phase 5 task 5.6)
# ===========================================================================


class TestRoundRobinFallback:
    """The allocator's graceful-degradation path.

    ``allocate`` wraps ``run_contract_net`` in a try/except.  When the
    negotiation function raises (timeout, corrupt snapshot, programming
    bug, etc.) the allocator must:

    * log a WARNING that names the failure,
    * synthesize a :class:`NegotiationResult` whose ``assigned_agent``
      is picked from :data:`AGENT_IDS` in round-robin order,
    * advance the cursor so subsequent failures cycle through agents,
    * and never propagate the underlying exception.

    Tests use ``monkeypatch`` to swap ``run_contract_net`` for a stub
    that raises the desired exception type.  Patching the symbol on
    the ``task_allocator`` module is the right scope: ``allocate``
    looks it up by name there, so the patch only affects code under
    test.
    """

    @staticmethod
    def _patch_negotiation_to_raise(
        monkeypatch: pytest.MonkeyPatch,
        exc: BaseException,
    ) -> list[tuple[Any, ...]]:
        """Replace ``run_contract_net`` with a stub that raises *exc*.

        Returns the call-log list so tests can assert how often the
        allocator invoked the (failing) negotiation function.
        """
        from chronoagent.allocator import task_allocator as ta_mod

        calls: list[tuple[Any, ...]] = []

        def _raising(*args: Any, **kwargs: Any) -> NegotiationResult:
            calls.append((args, kwargs))
            raise exc

        monkeypatch.setattr(ta_mod, "run_contract_net", _raising)
        return calls

    def test_generic_exception_triggers_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._patch_negotiation_to_raise(monkeypatch, RuntimeError("boom"))
        allocator = DecentralizedTaskAllocator(LocalBus())
        with caplog.at_level(logging.WARNING, logger="chronoagent.allocator.task_allocator"):
            result = allocator.allocate("t1", "plan")
        # Synthesized result, never raised.
        assert isinstance(result, NegotiationResult)
        assert result.assigned_agent == AGENT_IDS[0]  # first round-robin pick
        assert result.escalated is False
        assert result.winning_bid is None
        assert result.all_bids == ()
        assert result.task_id == "t1"
        assert result.task_type == "plan"
        assert result.threshold == DEFAULT_BID_THRESHOLD
        assert "round-robin fallback" in result.rationale
        assert "RuntimeError" in result.rationale
        assert "boom" in result.rationale
        # Warning log captured.
        assert any(
            "falling back to round-robin" in rec.message and rec.levelname == "WARNING"
            for rec in caplog.records
        )

    def test_timeout_error_triggers_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The plan calls out "negotiation timeout" explicitly; even though
        # the current run_contract_net is synchronous, the allocator must
        # treat TimeoutError exactly the same as any other failure.
        self._patch_negotiation_to_raise(monkeypatch, TimeoutError("negotiation deadline exceeded"))
        allocator = DecentralizedTaskAllocator(LocalBus())
        with caplog.at_level(logging.WARNING, logger="chronoagent.allocator.task_allocator"):
            result = allocator.allocate("t-timeout", "security_review")
        assert result.assigned_agent == AGENT_IDS[0]
        assert result.escalated is False
        assert "TimeoutError" in result.rationale
        assert "negotiation deadline exceeded" in result.rationale
        assert any("falling back to round-robin" in rec.message for rec in caplog.records)

    def test_round_robin_cursor_advances_across_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_negotiation_to_raise(monkeypatch, RuntimeError("nope"))
        allocator = DecentralizedTaskAllocator(LocalBus())
        picks = [
            allocator.allocate(f"t{i}", "plan").assigned_agent for i in range(len(AGENT_IDS) * 2)
        ]
        # Two full passes through AGENT_IDS in canonical order.
        assert picks == list(AGENT_IDS) * 2

    def test_fallback_records_task_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_negotiation_to_raise(monkeypatch, RuntimeError("x"))
        allocator = DecentralizedTaskAllocator(LocalBus())
        # Cycle the cursor a couple of times to verify task_id/task_type
        # are echoed back faithfully on every fallback.
        first = allocator.allocate("pr-1::style", "style_review")
        second = allocator.allocate("pr-2::summarize", "summarize")
        assert (first.task_id, first.task_type) == ("pr-1::style", "style_review")
        assert (second.task_id, second.task_type) == ("pr-2::summarize", "summarize")
        assert first.assigned_agent == AGENT_IDS[0]
        assert second.assigned_agent == AGENT_IDS[1]

    def test_happy_path_unaffected_by_fallback_machinery(self) -> None:
        # No monkeypatch: real negotiation runs.  The cursor must NOT
        # advance on a successful allocation.
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 1.0)
        result = allocator.allocate("t-happy", "plan")
        assert result.assigned_agent == "planner"
        assert result.winning_bid is not None
        # Internal cursor untouched.
        assert allocator._round_robin_cursor == 0

    def test_concurrent_fallback_cursor_is_thread_safe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Multiple threads each forcing a fallback; every AGENT_IDS slot
        # should be hit exactly N times after N*len(AGENT_IDS) calls,
        # which proves the cursor advances atomically.
        self._patch_negotiation_to_raise(monkeypatch, RuntimeError("race"))
        allocator = DecentralizedTaskAllocator(LocalBus())

        rounds_per_thread = 25
        n_threads = 4
        results: list[str] = []
        results_lock = threading.Lock()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                local: list[str] = []
                for i in range(rounds_per_thread):
                    r = allocator.allocate(f"t-{i}", "plan")
                    assert r.assigned_agent is not None
                    local.append(r.assigned_agent)
                with results_lock:
                    results.extend(local)
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        total = rounds_per_thread * n_threads
        assert len(results) == total
        # Even distribution across all 4 agents (cursor advanced atomically).
        per_agent = total // len(AGENT_IDS)
        for agent_id in AGENT_IDS:
            assert results.count(agent_id) == per_agent

    def test_fallback_does_not_propagate_exception_subclasses(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Negotiation may raise its own validation errors.  Verify that
        # InvalidHealthError (a ValueError subclass) is also caught.
        from chronoagent.allocator.negotiation import InvalidHealthError

        self._patch_negotiation_to_raise(monkeypatch, InvalidHealthError("planner", float("nan")))
        allocator = DecentralizedTaskAllocator(LocalBus())
        result = allocator.allocate("t-bad-health", "plan")
        assert result.assigned_agent == AGENT_IDS[0]
        assert "InvalidHealthError" in result.rationale


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


# ===========================================================================
# Phase 5.7: integrated allocator sweep
# ===========================================================================
#
# The Phase 5.7 deliverable is a *test sweep* rather than new product code.
# What was already covered before this task:
#
# * ``tests/unit/test_negotiation.py`` exercises the pure
#   :func:`run_contract_net` function with Hypothesis (highest-bid wins,
#   exactly-one-outcome, deterministic tie-break, all-low-health
#   escalation, snapshot edge cases, threshold validation).
# * ``TestAllocate`` above covers the happy path, the degraded-specialist
#   redistribution, and the all-low-health escalation at the allocator
#   layer using deterministic snapshots published through the bus.
# * ``TestRoundRobinFallback`` above covers the "timeout handling"
#   item from the 5.7 spec via monkeypatched exceptions (RuntimeError,
#   TimeoutError, InvalidHealthError).
#
# What 5.7 adds here is the missing slice: Hypothesis property tests at
# the **integrated** layer (bus -> cache -> negotiation -> result).  The
# pure-function properties already verify negotiation; these properties
# verify that the bus + cache projection feeds it faithfully and that
# allocator outputs match an independent recomputation against the
# expected cache state.  Plus a deterministic "degradation walk" that
# steps a single task type through the three regimes (full health ->
# specialist; degraded specialist -> peer; everyone degraded -> escalate)
# to give the spec's narrative a single readable test.

# Strategy mirrors the one in test_negotiation.py: every canonical
# agent_id maps to a finite float in [0, 1].  No NaN / out-of-range
# values - those have their own targeted tests in test_negotiation.py.
_healthy_float = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_full_snapshot_strategy = st.fixed_dictionaries(
    {agent_id: _healthy_float for agent_id in AGENT_IDS}
)
_threshold_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_task_type_strategy = st.sampled_from(TASK_TYPES)


def _publish_full_snapshot(bus: LocalBus, snapshot: dict[str, float]) -> None:
    """Publish a full ``{agent_id: health}`` snapshot via the bus.

    Uses the same dict shape the real ``TemporalHealthScorer`` publishes
    (``vars(HealthUpdate)``) so the integrated tests cover the realistic
    payload, not just the dataclass-instance shortcut.
    """
    for agent_id, health in snapshot.items():
        _publish_dict(bus, agent_id, health)


class TestPhase57IntegratedSweep:
    """Hypothesis property tests at the bus + cache + negotiation layer.

    The settings disable Hypothesis's ``function_scoped_fixture`` health
    check because each example creates its own ``LocalBus`` and
    ``DecentralizedTaskAllocator`` inside the test body, not via a
    pytest fixture, so the warning would not apply.  ``deadline=None``
    matches the convention used in test_negotiation.py.
    """

    @given(
        snapshot=_full_snapshot_strategy,
        task_type=_task_type_strategy,
        threshold=_threshold_strategy,
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_exactly_one_outcome_via_bus(
        self,
        snapshot: dict[str, float],
        task_type: str,
        threshold: float,
    ) -> None:
        """For any valid snapshot published over the bus, the allocator
        produces exactly one of (assigned, escalated)."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus, threshold=threshold)
        _publish_full_snapshot(bus, snapshot)
        result = allocator.allocate("t", task_type)
        if result.escalated:
            assert result.assigned_agent is None
            assert result.winning_bid is None
        else:
            assert result.assigned_agent is not None
            assert result.winning_bid is not None
            assert result.winning_bid.agent_id == result.assigned_agent
            # Highest-bid invariant: no other bid in the ledger beats
            # the winner's score (the implementation enforces strict
            # >, so equal scores resolve to the AGENT_IDS-first agent).
            for bid in result.all_bids:
                assert bid.score <= result.winning_bid.score

    @given(
        snapshot=_full_snapshot_strategy,
        task_type=_task_type_strategy,
        threshold=_threshold_strategy,
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_allocator_matches_pure_negotiation(
        self,
        snapshot: dict[str, float],
        task_type: str,
        threshold: float,
    ) -> None:
        """The integrated allocator's result is byte-equivalent (modulo
        the task_id field) to a direct ``run_contract_net`` call against
        the same snapshot.  This pins down the bus + cache + projection
        layer: any drift between published-and-cached health and the
        snapshot consumed by negotiation would surface here."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus, threshold=threshold)
        _publish_full_snapshot(bus, snapshot)
        actual = allocator.allocate("t", task_type)
        expected = run_contract_net(
            task_id="t",
            task_type=task_type,
            health_snapshot=snapshot,
            threshold=threshold,
        )
        assert actual.assigned_agent == expected.assigned_agent
        assert actual.escalated == expected.escalated
        assert actual.task_type == expected.task_type
        assert actual.threshold == expected.threshold
        assert len(actual.all_bids) == len(expected.all_bids)
        for actual_bid, expected_bid in zip(actual.all_bids, expected.all_bids, strict=True):
            assert actual_bid.agent_id == expected_bid.agent_id
            assert actual_bid.capability == pytest.approx(expected_bid.capability)
            assert actual_bid.health == pytest.approx(expected_bid.health)
            assert actual_bid.score == pytest.approx(expected_bid.score)
        if expected.winning_bid is None:
            assert actual.winning_bid is None
        else:
            assert actual.winning_bid is not None
            assert actual.winning_bid.agent_id == expected.winning_bid.agent_id
            assert actual.winning_bid.score == pytest.approx(expected.winning_bid.score)

    @given(
        # A non-empty sequence of (agent_id, health) publish events; the
        # cache must reflect the *last* publish for each agent.
        events=st.lists(
            st.tuples(st.sampled_from(AGENT_IDS), _healthy_float),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(
        max_examples=150,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_last_write_wins_under_random_publish_stream(
        self,
        events: list[tuple[str, float]],
    ) -> None:
        """The cache projection equals the agent-wise last write of the
        publish stream.  Guards the bus handler against ordering bugs or
        accidental dict mutation."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id, health in events:
            _publish_dict(bus, agent_id, health)
        expected: dict[str, float] = {}
        for agent_id, health in events:
            expected[agent_id] = health
        assert allocator.get_health_snapshot() == expected


class TestPhase57DegradationWalk:
    """Deterministic three-step walk through the three allocation regimes
    for one task type.

    Documents the Phase 5.7 acceptance narrative as a single readable
    test: a healthy specialist gets its own task; if the specialist
    degrades the contract-net protocol redistributes to the highest-
    capability peer; if every agent's health falls below the
    capability-weighted threshold the round escalates.

    Uses ``security_review`` because the default capability matrix has
    a clear second-best (``style_reviewer`` at 0.55) for that task,
    making the peer-redistribution step deterministic.
    """

    def test_full_health_to_peer_to_escalation(self) -> None:
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)

        # Step 1: full health -> specialist wins.
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 1.0)
        step1 = allocator.allocate("walk-1", "security_review")
        assert step1.escalated is False
        assert step1.assigned_agent == "security_reviewer"
        assert step1.winning_bid is not None
        assert step1.winning_bid.score == pytest.approx(1.0)

        # Step 2: degrade only the specialist -> highest-capability
        # peer (style_reviewer @ 0.55) wins because its bid 1.0 * 0.55
        # beats security_reviewer's 1.0 * 0.05.
        _publish_dict(bus, "security_reviewer", 0.05)
        step2 = allocator.allocate("walk-2", "security_review")
        assert step2.escalated is False
        assert step2.assigned_agent == "style_reviewer"
        assert step2.winning_bid is not None
        # style_reviewer's capability on security_review is 0.55 in the
        # default matrix; with health 1.0 the bid is exactly 0.55.
        assert step2.winning_bid.score == pytest.approx(0.55)

        # Step 3: degrade everyone below the default 0.25 threshold
        # divided by every off-diagonal weight; 0.05 across the board
        # sends every bid below the threshold and the round escalates.
        for agent_id in AGENT_IDS:
            _publish_dict(bus, agent_id, 0.05)
        step3 = allocator.allocate("walk-3", "security_review")
        assert step3.escalated is True
        assert step3.assigned_agent is None
        assert step3.winning_bid is None
        assert "below threshold" in step3.rationale
        # The full bid ledger is still recorded so the audit layer can
        # explain *why* every agent fell short.
        assert len(step3.all_bids) == len(AGENT_IDS)
