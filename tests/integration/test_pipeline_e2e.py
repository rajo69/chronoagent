"""End-to-end integration tests for the full review pipeline.

Uses MockBackend exclusively — zero API calls, zero cost, fully deterministic.

Test groups
-----------
* ``TestE2EReviewReport``     — ReviewReport structure and finding content
* ``TestE2EAgentOrder``       — plan → (security_review ∥ style_review) → summarize
* ``TestE2EMultiplePRs``      — independent reports for distinct PR ids
* ``TestE2EDeterminism``      — same seed → identical reports across pipeline instances
"""

from __future__ import annotations

from typing import Any

import chromadb
import pytest

from chronoagent.agents.planner import DecompositionResult
from chronoagent.agents.security_reviewer import SecurityReview, SyntheticPR
from chronoagent.agents.style_reviewer import StyleReview
from chronoagent.agents.summarizer import ReviewReport
from chronoagent.pipeline.graph import ReviewPipeline

# ---------------------------------------------------------------------------
# Shared PR fixtures
# ---------------------------------------------------------------------------

_SQL_INJECTION_PR = SyntheticPR(
    pr_id="e2e_sql_001",
    title="Add dynamic query builder",
    description="Builds SQL queries by concatenating user-supplied parameters.",
    diff=(
        "+def get_user(user_id: str) -> dict:\n"
        "+    query = 'SELECT * FROM users WHERE id = ' + user_id\n"
        "+    return db.execute(query).fetchone()\n"
    ),
    files_changed=["db/queries.py"],
)

_PATH_TRAVERSAL_PR = SyntheticPR(
    pr_id="e2e_path_001",
    title="Add file upload endpoint with path construction",
    description="Allows users to upload profile documents using a user-supplied filename.",
    diff=(
        "+def upload(file: UploadFile) -> None:\n"
        "+    path = base_dir + file.filename\n"
        "+    with open(path, 'wb') as f:\n"
        "+        f.write(file.read())\n"
    ),
    files_changed=["api/upload.py"],
)

_JWT_AUTH_PR = SyntheticPR(
    pr_id="e2e_jwt_001",
    title="Implement JWT authentication middleware",
    description="Adds JWT-based auth to all protected routes; validates algorithm field.",
    diff=(
        "+import jwt\n"
        "+def verify_token(token: str) -> dict:\n"
        "+    return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n"
    ),
    files_changed=["auth/middleware.py"],
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> ReviewPipeline:
    """Shared pipeline using MockBackend + ephemeral ChromaDB (seed=42)."""
    return ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())


# ---------------------------------------------------------------------------
# Helper: execution-order tracker
# ---------------------------------------------------------------------------


def _make_tracked_pipeline(seed: int = 42) -> tuple[ReviewPipeline, list[str]]:
    """Return a pipeline whose agent methods are wrapped to record call order.

    Each of the four main agent methods is replaced with a thin wrapper that
    appends a label to *call_log* before delegating to the original.

    Args:
        seed: Forwarded to :meth:`ReviewPipeline.create`.

    Returns:
        ``(pipeline, call_log)`` — the pipeline is ready to run; *call_log* will
        be populated in chronological order during :meth:`ReviewPipeline.run`.
    """
    p = ReviewPipeline.create(seed=seed, chroma_client=chromadb.EphemeralClient())
    call_log: list[str] = []

    # Wrap PlannerAgent.decompose
    _orig_decompose = p._planner.decompose

    def _tracked_decompose(pr: SyntheticPR) -> DecompositionResult:
        call_log.append("plan")
        return _orig_decompose(pr)

    p._planner.decompose = _tracked_decompose  # type: ignore[method-assign]

    # Wrap SecurityReviewerAgent.review
    _orig_sec = p._security_reviewer.review

    def _tracked_sec_review(pr: SyntheticPR) -> SecurityReview:
        call_log.append("security_review")
        return _orig_sec(pr)

    p._security_reviewer.review = _tracked_sec_review  # type: ignore[method-assign]

    # Wrap StyleReviewerAgent.review
    _orig_style = p._style_reviewer.review

    def _tracked_style_review(pr: SyntheticPR) -> StyleReview:
        call_log.append("style_review")
        return _orig_style(pr)

    p._style_reviewer.review = _tracked_style_review  # type: ignore[method-assign]

    # Wrap SummarizerAgent.synthesize
    _orig_synth = p._summarizer.synthesize

    def _tracked_synthesize(
        pr: SyntheticPR, sec: SecurityReview, sty: StyleReview
    ) -> ReviewReport:
        call_log.append("summarize")
        return _orig_synth(pr, sec, sty)

    p._summarizer.synthesize = _tracked_synthesize  # type: ignore[method-assign]

    # LangGraph compiles node callables at construction time; rebuild so the
    # wrapped methods are captured by the new node closures.
    p._graph = p._build_graph()

    return p, call_log


# ---------------------------------------------------------------------------
# TestE2EReviewReport — report structure and finding content
# ---------------------------------------------------------------------------


class TestE2EReviewReport:
    """Full-pipeline output is a well-formed ReviewReport with real findings."""

    def test_returns_review_report_type(self, pipeline: ReviewPipeline) -> None:
        """pipeline.run() returns a ReviewReport instance."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert isinstance(report, ReviewReport)

    def test_pr_id_propagates(self, pipeline: ReviewPipeline) -> None:
        """report.pr_id matches the submitted PR's id."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert report.pr_id == _SQL_INJECTION_PR.pr_id

    def test_title_propagates(self, pipeline: ReviewPipeline) -> None:
        """report.title matches the submitted PR's title."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert report.title == _SQL_INJECTION_PR.title

    def test_security_findings_non_empty(self, pipeline: ReviewPipeline) -> None:
        """A diff with SQL injection risk produces at least one security finding."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert len(report.security_findings) > 0, (
            "Expected at least one SecurityFinding for a SQL-injection diff"
        )

    def test_style_findings_non_empty(self, pipeline: ReviewPipeline) -> None:
        """The style reviewer produces at least one style finding."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert len(report.style_findings) > 0, (
            "Expected at least one StyleFinding from the MockBackend response library"
        )

    def test_security_finding_fields(self, pipeline: ReviewPipeline) -> None:
        """Every SecurityFinding has the required fields with valid values."""
        valid_severities = {"none", "low", "medium", "high", "critical"}
        report = pipeline.run(_SQL_INJECTION_PR)
        for f in report.security_findings:
            assert isinstance(f.severity, str) and f.severity in valid_severities
            assert isinstance(f.description, str) and len(f.description) > 0
            assert isinstance(f.line_ref, str)
            assert isinstance(f.cwe_id, str)

    def test_style_finding_fields(self, pipeline: ReviewPipeline) -> None:
        """Every StyleFinding has category, description, and line_ref fields."""
        report = pipeline.run(_SQL_INJECTION_PR)
        for f in report.style_findings:
            assert isinstance(f.category, str) and len(f.category) > 0
            assert isinstance(f.description, str) and len(f.description) > 0
            assert isinstance(f.line_ref, str)

    def test_overall_risk_is_valid(self, pipeline: ReviewPipeline) -> None:
        """overall_risk is one of the expected severity levels."""
        valid = {"none", "low", "medium", "high", "critical"}
        report = pipeline.run(_SQL_INJECTION_PR)
        assert report.overall_risk in valid

    def test_markdown_is_nonempty_string(self, pipeline: ReviewPipeline) -> None:
        """markdown field is a non-empty string."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert isinstance(report.markdown, str) and len(report.markdown) > 0

    def test_markdown_contains_pr_title(self, pipeline: ReviewPipeline) -> None:
        """Markdown report body includes the PR title."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert _SQL_INJECTION_PR.title in report.markdown

    def test_markdown_contains_overall_risk(self, pipeline: ReviewPipeline) -> None:
        """Markdown report body includes the overall risk level."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert report.overall_risk.upper() in report.markdown.upper()

    def test_path_traversal_risk_is_high_or_critical(
        self, pipeline: ReviewPipeline
    ) -> None:
        """A path-traversal diff produces high or critical overall_risk."""
        report = pipeline.run(_PATH_TRAVERSAL_PR)
        assert report.overall_risk in {"high", "critical"}, (
            f"Expected high/critical for path traversal diff; got '{report.overall_risk}'"
        )

    def test_latencies_non_negative(self, pipeline: ReviewPipeline) -> None:
        """llm_latency_ms and retrieval_latency_ms in the report are >= 0."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert report.llm_latency_ms >= 0.0
        assert report.retrieval_latency_ms >= 0.0

    def test_retrieved_docs_non_negative(self, pipeline: ReviewPipeline) -> None:
        """retrieved_docs is a non-negative integer."""
        report = pipeline.run(_SQL_INJECTION_PR)
        assert isinstance(report.retrieved_docs, int)
        assert report.retrieved_docs >= 0


# ---------------------------------------------------------------------------
# TestE2EAgentOrder — plan → (security_review ∥ style_review) → summarize
# ---------------------------------------------------------------------------


class TestE2EAgentOrder:
    """The pipeline executes agents in the correct topological order."""

    def _run_and_get_log(self, pr: SyntheticPR) -> list[str]:
        pipeline, call_log = _make_tracked_pipeline(seed=42)
        pipeline.run(pr)
        return call_log

    def test_all_four_nodes_execute(self) -> None:
        """All four agent methods are called exactly once per run."""
        log = self._run_and_get_log(_SQL_INJECTION_PR)
        assert log.count("plan") == 1
        assert log.count("security_review") == 1
        assert log.count("style_review") == 1
        assert log.count("summarize") == 1

    def test_plan_executes_before_security_review(self) -> None:
        """plan runs before security_review."""
        log = self._run_and_get_log(_SQL_INJECTION_PR)
        assert log.index("plan") < log.index("security_review")

    def test_plan_executes_before_style_review(self) -> None:
        """plan runs before style_review."""
        log = self._run_and_get_log(_SQL_INJECTION_PR)
        assert log.index("plan") < log.index("style_review")

    def test_summarize_executes_after_security_review(self) -> None:
        """summarize runs after security_review."""
        log = self._run_and_get_log(_SQL_INJECTION_PR)
        assert log.index("summarize") > log.index("security_review")

    def test_summarize_executes_after_style_review(self) -> None:
        """summarize runs after style_review."""
        log = self._run_and_get_log(_SQL_INJECTION_PR)
        assert log.index("summarize") > log.index("style_review")

    def test_plan_executes_before_summarize(self) -> None:
        """plan runs before summarize (transitivity check)."""
        log = self._run_and_get_log(_SQL_INJECTION_PR)
        assert log.index("plan") < log.index("summarize")

    def test_order_consistent_across_pr_types(self) -> None:
        """Agent order holds for a different PR (path traversal)."""
        log = self._run_and_get_log(_PATH_TRAVERSAL_PR)
        assert log.index("plan") < log.index("security_review")
        assert log.index("plan") < log.index("style_review")
        assert log.index("summarize") > log.index("security_review")
        assert log.index("summarize") > log.index("style_review")

    def test_order_consistent_for_jwt_pr(self) -> None:
        """Agent order holds for a JWT authentication PR."""
        log = self._run_and_get_log(_JWT_AUTH_PR)
        assert log.index("plan") < log.index("security_review")
        assert log.index("plan") < log.index("style_review")
        assert log.index("summarize") > log.index("security_review")
        assert log.index("summarize") > log.index("style_review")


# ---------------------------------------------------------------------------
# TestE2EMultiplePRs — independent reports for distinct PR ids
# ---------------------------------------------------------------------------


class TestE2EMultiplePRs:
    """Each PR produces an independent, correctly-labelled report."""

    def test_three_prs_return_three_independent_reports(
        self, pipeline: ReviewPipeline
    ) -> None:
        """Running three PRs produces three reports with matching pr_ids."""
        prs = [_SQL_INJECTION_PR, _PATH_TRAVERSAL_PR, _JWT_AUTH_PR]
        reports = [pipeline.run(pr) for pr in prs]
        for pr, report in zip(prs, reports):
            assert report.pr_id == pr.pr_id

    def test_reports_are_distinct_objects(self, pipeline: ReviewPipeline) -> None:
        """Each call to run() returns a new ReviewReport object."""
        r1 = pipeline.run(_SQL_INJECTION_PR)
        r2 = pipeline.run(_PATH_TRAVERSAL_PR)
        assert r1 is not r2

    def test_pr_ids_do_not_cross_contaminate(self, pipeline: ReviewPipeline) -> None:
        """The pr_id in each report matches its own submission, not a neighbour's."""
        r_sql = pipeline.run(_SQL_INJECTION_PR)
        r_path = pipeline.run(_PATH_TRAVERSAL_PR)
        assert r_sql.pr_id != r_path.pr_id
        assert r_sql.pr_id == _SQL_INJECTION_PR.pr_id
        assert r_path.pr_id == _PATH_TRAVERSAL_PR.pr_id

    def test_titles_do_not_cross_contaminate(self, pipeline: ReviewPipeline) -> None:
        """Each report carries its own PR's title."""
        r_sql = pipeline.run(_SQL_INJECTION_PR)
        r_jwt = pipeline.run(_JWT_AUTH_PR)
        assert r_sql.title == _SQL_INJECTION_PR.title
        assert r_jwt.title == _JWT_AUTH_PR.title


# ---------------------------------------------------------------------------
# TestE2EDeterminism — same seed → identical reports across instances
# ---------------------------------------------------------------------------


class TestE2EDeterminism:
    """MockBackend determinism: same seed → same findings on fresh pipeline instances."""

    def test_same_seed_produces_identical_overall_risk(self) -> None:
        """Two fresh pipelines with seed=42 produce the same overall_risk."""
        p1 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        p2 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        r1 = p1.run(_SQL_INJECTION_PR)
        r2 = p2.run(_SQL_INJECTION_PR)
        assert r1.overall_risk == r2.overall_risk

    def test_same_seed_produces_identical_security_findings(self) -> None:
        """Same seed → identical security finding list (count + severities)."""
        p1 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        p2 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        r1 = p1.run(_SQL_INJECTION_PR)
        r2 = p2.run(_SQL_INJECTION_PR)
        assert len(r1.security_findings) == len(r2.security_findings)
        for f1, f2 in zip(r1.security_findings, r2.security_findings):
            assert f1.severity == f2.severity
            assert f1.cwe_id == f2.cwe_id

    def test_same_seed_produces_identical_style_findings(self) -> None:
        """Same seed → identical style finding list (count + categories)."""
        p1 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        p2 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        r1 = p1.run(_SQL_INJECTION_PR)
        r2 = p2.run(_SQL_INJECTION_PR)
        assert len(r1.style_findings) == len(r2.style_findings)
        for f1, f2 in zip(r1.style_findings, r2.style_findings):
            assert f1.category == f2.category

    def test_same_seed_produces_identical_markdown(self) -> None:
        """Same seed → identical markdown body."""
        p1 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        p2 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        r1 = p1.run(_SQL_INJECTION_PR)
        r2 = p2.run(_SQL_INJECTION_PR)
        assert r1.markdown == r2.markdown

    def test_different_seeds_may_differ(self) -> None:
        """Pipelines with different seeds do not always produce the same report."""
        p_42 = ReviewPipeline.create(seed=42, chroma_client=chromadb.EphemeralClient())
        p_99 = ReviewPipeline.create(seed=99, chroma_client=chromadb.EphemeralClient())
        r_42 = p_42.run(_SQL_INJECTION_PR)
        r_99 = p_99.run(_SQL_INJECTION_PR)
        # At least the pr_id must be the same; report content may differ.
        assert r_42.pr_id == r_99.pr_id
        # (We do not assert they differ — determinism of seeds is tested above.)
