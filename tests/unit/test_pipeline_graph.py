"""Tests for pipeline/graph.py — ReviewPipeline LangGraph StateGraph."""

from __future__ import annotations

import chromadb
import pytest

from chronoagent.agents.planner import PlannerAgent
from chronoagent.agents.security_reviewer import SecurityReviewerAgent, SyntheticPR
from chronoagent.agents.style_reviewer import StyleReviewerAgent
from chronoagent.agents.summarizer import ReviewReport, SummarizerAgent
from chronoagent.allocator.task_allocator import DecentralizedTaskAllocator
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.pipeline.graph import PipelineState, ReviewPipeline
from chronoagent.scorer.health_scorer import HEALTH_CHANNEL, HealthUpdate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pr() -> SyntheticPR:
    """A synthetic PR fixture reused across all test cases."""
    return SyntheticPR(
        pr_id="pr_graph_test",
        title="Add file upload endpoint with path construction",
        description="Allows users to upload profile documents.",
        diff="+def upload(file):\n+    path = base_dir + file.filename\n+    save(path, file)",
        files_changed=["api/upload.py"],
    )


@pytest.fixture
def chroma_client() -> chromadb.ClientAPI:
    """Ephemeral ChromaDB client shared within a single test."""
    return chromadb.EphemeralClient()


@pytest.fixture
def pipeline(chroma_client: chromadb.ClientAPI) -> ReviewPipeline:
    """ReviewPipeline built with MockBackend agents."""
    return ReviewPipeline.create(seed=42, chroma_client=chroma_client)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestReviewPipelineConstruction:
    def test_create_returns_pipeline(self) -> None:
        """.create() factory returns a ReviewPipeline instance."""
        p = ReviewPipeline.create()
        assert isinstance(p, ReviewPipeline)

    def test_create_with_shared_chroma_client(self, chroma_client: chromadb.ClientAPI) -> None:
        """Providing an explicit chroma client does not raise."""
        p = ReviewPipeline.create(chroma_client=chroma_client)
        assert isinstance(p, ReviewPipeline)

    def test_graph_is_compiled(self, pipeline: ReviewPipeline) -> None:
        """The internal graph attribute is not None after construction."""
        assert pipeline._graph is not None

    def test_manual_construction(self, chroma_client: chromadb.ClientAPI) -> None:
        """ReviewPipeline can be built from manually constructed agents."""
        planner = PlannerAgent.create(chroma_client=chroma_client)
        sec = SecurityReviewerAgent.create(chroma_client=chroma_client)
        sty = StyleReviewerAgent.create(chroma_client=chroma_client)
        summ = SummarizerAgent.create(chroma_client=chroma_client)
        p = ReviewPipeline(
            planner=planner,
            security_reviewer=sec,
            style_reviewer=sty,
            summarizer=summ,
        )
        assert isinstance(p, ReviewPipeline)


# ---------------------------------------------------------------------------
# run() output structure tests
# ---------------------------------------------------------------------------


class TestReviewPipelineRun:
    def test_run_returns_review_report(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """run() returns a ReviewReport instance."""
        report = pipeline.run(sample_pr)
        assert isinstance(report, ReviewReport)

    def test_report_pr_id_matches(self, pipeline: ReviewPipeline, sample_pr: SyntheticPR) -> None:
        """The returned report's pr_id matches the input PR."""
        report = pipeline.run(sample_pr)
        assert report.pr_id == sample_pr.pr_id

    def test_report_title_matches(self, pipeline: ReviewPipeline, sample_pr: SyntheticPR) -> None:
        """The report title matches the input PR title."""
        report = pipeline.run(sample_pr)
        assert report.title == sample_pr.title

    def test_report_has_security_findings_list(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """security_findings is a list (may be empty)."""
        report = pipeline.run(sample_pr)
        assert isinstance(report.security_findings, list)

    def test_report_has_style_findings_list(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """style_findings is a list (may be empty)."""
        report = pipeline.run(sample_pr)
        assert isinstance(report.style_findings, list)

    def test_report_overall_risk_valid(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """overall_risk is one of the known severity levels."""
        valid = {"none", "low", "medium", "high", "critical"}
        report = pipeline.run(sample_pr)
        assert report.overall_risk in valid

    def test_report_markdown_is_string(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """The markdown field is a non-empty string."""
        report = pipeline.run(sample_pr)
        assert isinstance(report.markdown, str)
        assert len(report.markdown) > 0

    def test_report_markdown_contains_pr_title(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """The markdown body includes the PR title."""
        report = pipeline.run(sample_pr)
        assert sample_pr.title in report.markdown

    def test_report_latencies_non_negative(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """llm_latency_ms and retrieval_latency_ms are >= 0."""
        report = pipeline.run(sample_pr)
        assert report.llm_latency_ms >= 0.0
        assert report.retrieval_latency_ms >= 0.0

    def test_run_is_deterministic(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        """Two pipelines with the same seed produce identical overall_risk."""
        p1 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        p2 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        r1 = p1.run(sample_pr)
        r2 = p2.run(sample_pr)
        assert r1.overall_risk == r2.overall_risk
        assert r1.security_findings == r2.security_findings
        assert r1.style_findings == r2.style_findings

    def test_run_different_prs_may_differ(self, pipeline: ReviewPipeline) -> None:
        """Running on two distinct PRs does not raise and returns two reports."""
        pr1 = SyntheticPR(
            pr_id="pr_a",
            title="Add SQL query builder",
            description="Raw SQL string concatenation.",
            diff="+query = 'SELECT * FROM users WHERE id = ' + user_id",
            files_changed=["db/queries.py"],
        )
        pr2 = SyntheticPR(
            pr_id="pr_b",
            title="Fix typo in README",
            description="Minor documentation fix.",
            diff="-Helo world\n+Hello world",
            files_changed=["README.md"],
        )
        r1 = pipeline.run(pr1)
        r2 = pipeline.run(pr2)
        assert r1.pr_id == "pr_a"
        assert r2.pr_id == "pr_b"


# ---------------------------------------------------------------------------
# PipelineState TypedDict
# ---------------------------------------------------------------------------


class TestPipelineState:
    def test_state_is_typed_dict(self) -> None:
        """PipelineState can be instantiated as a plain dict."""
        pr = SyntheticPR(
            pr_id="s1",
            title="T",
            description="D",
            diff="",
            files_changed=[],
        )
        state: PipelineState = {"pr": pr}  # type: ignore[typeddict-item]
        assert state["pr"] is pr

    def test_state_keys(self) -> None:
        """PipelineState defines the expected keys."""
        annotations = PipelineState.__annotations__
        assert "pr" in annotations
        assert "decomposition" in annotations
        assert "security_allocation" in annotations
        assert "style_allocation" in annotations
        assert "security_review" in annotations
        assert "style_review" in annotations
        assert "report" in annotations


# ---------------------------------------------------------------------------
# Allocator-gated routing (Phase 5.4)
# ---------------------------------------------------------------------------


def _publish_health(bus: LocalBus, agent_id: str, health: float) -> None:
    bus.publish(
        HEALTH_CHANNEL,
        HealthUpdate(
            agent_id=agent_id,
            health=health,
            bocpd_score=0.0,
            chronos_score=0.0,
        ),
    )


class TestAllocatorGating:
    """Verify the allocate_{security,style} nodes and conditional routing."""

    def test_default_allocator_routes_to_specialists(
        self, pipeline: ReviewPipeline, sample_pr: SyntheticPR
    ) -> None:
        """With the built-in default allocator (empty health cache,
        missing_health_default=1.0) the specialist branch always wins
        and the produced report must contain real findings from the
        specialists (not the escalation placeholder)."""
        state = pipeline._graph.invoke({"pr": sample_pr})
        # Both allocations were recorded on the state.
        assert "security_allocation" in state
        assert "style_allocation" in state
        assert state["security_allocation"].assigned_agent == "security_reviewer"
        assert state["style_allocation"].assigned_agent == "style_reviewer"
        assert state["security_allocation"].escalated is False
        assert state["style_allocation"].escalated is False
        # Winning scores are 1.0 (capability * health = 1.0 * 1.0).
        assert state["security_allocation"].winning_bid is not None
        assert state["security_allocation"].winning_bid.score == pytest.approx(1.0)
        # Reviews are the real outputs, not placeholders — their
        # raw_response should not be the negotiation rationale.
        assert state["security_review"].raw_response != state["security_allocation"].rationale
        assert state["style_review"].raw_response != state["style_allocation"].rationale

    def test_healthy_snapshot_routes_to_specialists(
        self, sample_pr: SyntheticPR, chroma_client: chromadb.ClientAPI
    ) -> None:
        """An explicit allocator with full-health updates routes the
        same way as the default allocator."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id in ("planner", "security_reviewer", "style_reviewer", "summarizer"):
            _publish_health(bus, agent_id, 1.0)
        pipeline = ReviewPipeline.create(chroma_client=chroma_client, allocator=allocator)
        state = pipeline._graph.invoke({"pr": sample_pr})
        assert state["security_allocation"].assigned_agent == "security_reviewer"
        assert state["style_allocation"].assigned_agent == "style_reviewer"
        assert isinstance(state["report"], ReviewReport)

    def test_non_specialist_winner_is_escalated(
        self, sample_pr: SyntheticPR, chroma_client: chromadb.ClientAPI
    ) -> None:
        """When the allocator picks a non-specialist, _route_{security,
        style} drops the task into the escalation branch.  With the
        default matrix, health={planner: 0.1, security_reviewer: 0.0,
        style_reviewer: 0.1, summarizer: 1.0} gives these bids:

        security_review:
            planner     0.30 * 0.1 = 0.03
            security    1.00 * 0.0 = 0.00
            style       0.55 * 0.1 = 0.055
            summarizer  0.35 * 1.0 = 0.35   <- wins, non-specialist
        style_review:
            planner     0.30 * 0.1 = 0.03
            security    0.55 * 0.0 = 0.00
            style       1.00 * 0.1 = 0.10
            summarizer  0.35 * 1.0 = 0.35   <- wins, non-specialist

        Both branches produce a non-specialist winner above the default
        0.25 threshold, so neither is strictly escalated but both are
        routed to the escalation nodes because the router only treats
        the matching specialist as "runnable"."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        _publish_health(bus, "planner", 0.1)
        _publish_health(bus, "security_reviewer", 0.0)
        _publish_health(bus, "style_reviewer", 0.1)
        _publish_health(bus, "summarizer", 1.0)
        pipeline = ReviewPipeline.create(chroma_client=chroma_client, allocator=allocator)
        state = pipeline._graph.invoke({"pr": sample_pr})
        # Allocator winners are non-specialists and not strictly escalated.
        assert state["security_allocation"].assigned_agent == "summarizer"
        assert state["security_allocation"].escalated is False
        assert state["style_allocation"].assigned_agent == "summarizer"
        assert state["style_allocation"].escalated is False
        # But the graph routed both to the escalation nodes.
        assert state["security_review"].findings == []
        assert state["security_review"].recommendation.startswith("escalated:")
        assert state["style_review"].findings == []
        assert state["style_review"].recommendation.startswith("escalated:")
        # Summarizer still produced a report.
        assert isinstance(state["report"], ReviewReport)
        assert state["report"].pr_id == sample_pr.pr_id

    def test_all_low_health_both_branches_escalate(
        self, sample_pr: SyntheticPR, chroma_client: chromadb.ClientAPI
    ) -> None:
        """Every agent crushed to 0.01 -> both branches escalate
        (strict escalation, assigned_agent=None)."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id in ("planner", "security_reviewer", "style_reviewer", "summarizer"):
            _publish_health(bus, agent_id, 0.01)
        pipeline = ReviewPipeline.create(chroma_client=chroma_client, allocator=allocator)
        state = pipeline._graph.invoke({"pr": sample_pr})
        assert state["security_allocation"].escalated is True
        assert state["security_allocation"].assigned_agent is None
        assert state["style_allocation"].escalated is True
        assert state["style_allocation"].assigned_agent is None
        # Placeholders emitted.
        assert state["security_review"].findings == []
        assert state["style_review"].findings == []
        # raw_response carries the negotiation rationale verbatim.
        assert state["security_review"].raw_response == state["security_allocation"].rationale
        assert state["style_review"].raw_response == state["style_allocation"].rationale
        # The report is still produced via the summarizer.
        assert isinstance(state["report"], ReviewReport)

    def test_run_public_api_survives_full_escalation(
        self, sample_pr: SyntheticPR, chroma_client: chromadb.ClientAPI
    ) -> None:
        """`run()` continues to return a ReviewReport even when every
        task is escalated — the escalation stays inside the graph."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        for agent_id in ("planner", "security_reviewer", "style_reviewer", "summarizer"):
            _publish_health(bus, agent_id, 0.0)
        pipeline = ReviewPipeline.create(chroma_client=chroma_client, allocator=allocator)
        report = pipeline.run(sample_pr)
        assert isinstance(report, ReviewReport)
        assert report.pr_id == sample_pr.pr_id
        assert sample_pr.title in report.markdown

    def test_allocations_task_ids_include_pr_id_and_task_type(
        self, sample_pr: SyntheticPR, chroma_client: chromadb.ClientAPI
    ) -> None:
        """Task IDs carry PR + task type so the audit layer (5.5) can
        correlate allocations back to the PR."""
        pipeline = ReviewPipeline.create(chroma_client=chroma_client)
        state = pipeline._graph.invoke({"pr": sample_pr})
        assert state["security_allocation"].task_id == f"{sample_pr.pr_id}::security_review"
        assert state["security_allocation"].task_type == "security_review"
        assert state["style_allocation"].task_id == f"{sample_pr.pr_id}::style_review"
        assert state["style_allocation"].task_type == "style_review"

    def test_explicit_allocator_injected_into_pipeline_instance(
        self, chroma_client: chromadb.ClientAPI
    ) -> None:
        """Constructor honours a supplied allocator (not a default)."""
        bus = LocalBus()
        allocator = DecentralizedTaskAllocator(bus)
        pipeline = ReviewPipeline.create(chroma_client=chroma_client, allocator=allocator)
        assert pipeline._allocator is allocator

    def test_default_allocator_constructed_when_none_supplied(
        self, chroma_client: chromadb.ClientAPI
    ) -> None:
        """Omitting the allocator builds a default one internally."""
        pipeline = ReviewPipeline.create(chroma_client=chroma_client)
        assert isinstance(pipeline._allocator, DecentralizedTaskAllocator)

    def test_graph_contains_allocator_and_escalation_nodes(self, pipeline: ReviewPipeline) -> None:
        """The compiled graph exposes the new nodes by name."""
        node_names = set(pipeline._graph.get_graph().nodes.keys())
        for expected in (
            "plan",
            "allocate_security",
            "allocate_style",
            "security_review",
            "escalate_security",
            "style_review",
            "escalate_style",
            "summarize",
        ):
            assert expected in node_names, f"missing node: {expected}"
