"""Tests for pipeline/graph.py — ReviewPipeline LangGraph StateGraph."""

from __future__ import annotations

import chromadb
import pytest

from chronoagent.agents.planner import PlannerAgent
from chronoagent.agents.security_reviewer import SecurityReviewerAgent, SyntheticPR
from chronoagent.agents.style_reviewer import StyleReviewerAgent
from chronoagent.agents.summarizer import ReviewReport, SummarizerAgent
from chronoagent.pipeline.graph import PipelineState, ReviewPipeline

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
        assert "security_review" in annotations
        assert "style_review" in annotations
        assert "report" in annotations
