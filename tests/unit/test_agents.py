"""Tests for BaseAgent ABC, SecurityReviewerAgent, SummarizerAgent, StyleReviewerAgent, and AgentRegistry."""

from __future__ import annotations

import chromadb
import pytest

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.agents.base import BaseAgent, Task, TaskResult
from chronoagent.agents.security_reviewer import (
    SecurityFinding,
    SecurityReview,
    SecurityReviewerAgent,
    SyntheticPR,
    _parse_finding,
)
from chronoagent.agents.planner import DecompositionResult, PlannerAgent, ReviewSubtask
from chronoagent.agents.style_reviewer import (
    StyleFinding,
    StyleReview,
    StyleReviewerAgent,
    _parse_finding as _parse_style_finding,
)
from chronoagent.agents.summarizer import ReviewReport, SummarizerAgent, Summary
from chronoagent.agents.registry import AgentRegistry, UnknownTaskTypeError


@pytest.fixture
def chroma_client() -> chromadb.ClientAPI:
    """Shared in-memory ChromaDB client per test."""
    return chromadb.EphemeralClient()


@pytest.fixture
def sample_pr() -> SyntheticPR:
    return SyntheticPR(
        pr_id="pr_test",
        title="Add user file upload endpoint",
        description="Allows users to upload documents to their profile.",
        diff="+def upload(file):\n+    path = base_dir + file.filename\n+    save(path, file)",
        files_changed=["api/upload.py"],
    )


class TestBaseAgentABC:
    def test_cannot_instantiate_base_agent(self) -> None:
        """BaseAgent is abstract — direct instantiation must raise TypeError."""
        with pytest.raises(TypeError):
            BaseAgent(  # type: ignore[abstract]
                agent_id="x",
                backend=None,  # type: ignore[arg-type]
                collection=None,  # type: ignore[arg-type]
            )

    def test_task_defaults(self) -> None:
        task = Task(task_id="t1", task_type="security_review")
        assert task.payload == {}

    def test_task_result_fields(self) -> None:
        result = TaskResult(
            task_id="t1",
            agent_id="agent",
            status="success",
            output={"key": "value"},
            llm_latency_ms=10.0,
            retrieval_latency_ms=5.0,
            timestamp=0.0,
        )
        assert result.status == "success"
        assert result.output["key"] == "value"


class TestSyntheticPR:
    def test_default_files_changed(self) -> None:
        pr = SyntheticPR(pr_id="x", title="t", description="d", diff="")
        assert pr.files_changed == []


class TestSecurityFinding:
    def test_dataclass_fields(self) -> None:
        f = SecurityFinding(severity="high", description="SQL injection", line_ref="line 42", cwe_id="CWE-89")
        assert f.severity == "high"
        assert f.description == "SQL injection"
        assert f.line_ref == "line 42"
        assert f.cwe_id == "CWE-89"

    def test_defaults(self) -> None:
        f = SecurityFinding(severity="low", description="minor issue")
        assert f.line_ref == ""
        assert f.cwe_id == ""

    def test_parse_finding_full_format(self) -> None:
        line = "1. [CWE-89] SQL injection in query builder (line 42) - SEVERITY: HIGH"
        f = _parse_finding(line)
        assert f.cwe_id == "CWE-89"
        assert f.severity == "high"
        assert "line 42" in f.line_ref
        assert "SQL injection" in f.description

    def test_parse_finding_no_cwe(self) -> None:
        line = "2. Missing rate limiting on auth route - SEVERITY: MEDIUM"
        f = _parse_finding(line)
        assert f.cwe_id == ""
        assert f.severity == "medium"
        assert "Missing rate limiting" in f.description

    def test_parse_finding_no_line_ref(self) -> None:
        line = "3. [CWE-79] XSS via template variable - SEVERITY: HIGH"
        f = _parse_finding(line)
        assert f.line_ref == ""
        assert f.cwe_id == "CWE-79"
        assert f.severity == "high"

    def test_parse_finding_severity_none(self) -> None:
        line = "1. No significant security issues found - SEVERITY: NONE"
        f = _parse_finding(line)
        assert f.severity == "none"

    def test_parse_finding_severity_critical(self) -> None:
        line = "1. [CWE-22] Path traversal in upload handler (line 67) - SEVERITY: CRITICAL"
        f = _parse_finding(line)
        assert f.severity == "critical"

    def test_parse_finding_info_maps_to_low(self) -> None:
        line = "2. [CWE-532] Minor unused import (line 3) - SEVERITY: INFO"
        f = _parse_finding(line)
        assert f.severity == "low"


class TestSecurityReviewerAgent:
    def test_create_returns_agent(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        assert agent.agent_id == "security_reviewer"

    def test_review_returns_security_review(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert isinstance(review, SecurityReview)
        assert review.pr_id == "pr_test"

    def test_review_retrieves_docs(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(top_k=3, chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.retrieved_docs == 3
        assert len(review.retrieval_distances) == 3

    def test_review_records_latencies(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.retrieval_latency_ms >= 0.0
        assert review.llm_latency_ms >= 0.0

    def test_review_has_severity(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.severity in ("none", "low", "medium", "high", "critical")

    def test_review_deterministic_same_seed(
        self, sample_pr: SyntheticPR
    ) -> None:
        c1 = chromadb.EphemeralClient()
        c2 = chromadb.EphemeralClient()
        a1 = SecurityReviewerAgent.create(seed=42, chroma_client=c1)
        a2 = SecurityReviewerAgent.create(seed=42, chroma_client=c2)
        r1 = a1.review(sample_pr)
        r2 = a2.review(sample_pr)
        assert r1.raw_response == r2.raw_response

    def test_review_raw_response_non_empty(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.raw_response.strip() != ""

    def test_call_llm_returns_text_and_latency(
        self, chroma_client: chromadb.ClientAPI
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        text, latency_ms = agent._call_llm("test prompt")
        assert isinstance(text, str)
        assert len(text) > 0
        assert latency_ms >= 0.0

    def test_retrieve_memory_returns_result(
        self, chroma_client: chromadb.ClientAPI
    ) -> None:
        agent = SecurityReviewerAgent.create(top_k=2, chroma_client=chroma_client)
        result = agent._retrieve_memory("SQL injection")
        assert len(result.documents) == 2
        assert result.latency_ms >= 0.0

    def test_execute_returns_task_result(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        task = Task(
            task_id="t1",
            task_type="security_review",
            payload={"pr": sample_pr},
        )
        result = agent.execute(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "t1"
        assert result.agent_id == "security_reviewer"
        assert result.status == "success"
        assert isinstance(result.output["review"], SecurityReview)
        assert result.llm_latency_ms >= 0.0
        assert result.retrieval_latency_ms >= 0.0

    def test_review_findings_are_security_findings(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert isinstance(review.findings, list)
        assert all(isinstance(f, SecurityFinding) for f in review.findings)

    def test_review_finding_fields_populated(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        for finding in review.findings:
            assert finding.severity in ("none", "low", "medium", "high", "critical")
            assert isinstance(finding.description, str)
            assert isinstance(finding.line_ref, str)
            assert isinstance(finding.cwe_id, str)

    def test_review_findings_have_cwe_ids(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        # At least one finding should carry a CWE ID (mock responses include them)
        cwe_findings = [f for f in review.findings if f.cwe_id]
        assert len(cwe_findings) >= 1

    def test_multiple_prs_processed(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        prs = [
            SyntheticPR(pr_id=f"pr_{i}", title=f"PR {i}", description="desc", diff="")
            for i in range(5)
        ]
        reviews = [agent.review(pr) for pr in prs]
        assert len(reviews) == 5
        assert all(isinstance(r, SecurityReview) for r in reviews)


class TestSummarizerAgent:
    def test_create_returns_agent(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = SummarizerAgent.create(chroma_client=chroma_client)
        assert agent.agent_id == "summarizer"

    def test_summarize_returns_summary(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        reviewer = SecurityReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        review = reviewer.review(sample_pr)
        summary = summarizer.summarize(sample_pr, review)
        assert isinstance(summary, Summary)
        assert summary.pr_id == "pr_test"

    def test_summarize_retrieves_docs(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        reviewer = SecurityReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(top_k=3, chroma_client=chroma_client)
        review = reviewer.review(sample_pr)
        summary = summarizer.summarize(sample_pr, review)
        assert summary.retrieved_docs == 3

    def test_summarize_records_latencies(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        reviewer = SecurityReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        review = reviewer.review(sample_pr)
        summary = summarizer.summarize(sample_pr, review)
        assert summary.retrieval_latency_ms >= 0.0
        assert summary.llm_latency_ms >= 0.0

    def test_summarize_risk_level_valid(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        reviewer = SecurityReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        review = reviewer.review(sample_pr)
        summary = summarizer.summarize(sample_pr, review)
        assert summary.risk_level in ("none", "low", "medium", "high", "critical")

    def test_execute_returns_task_result(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        sec_agent = SecurityReviewerAgent.create(chroma_client=chroma_client)
        style_agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        security_review = sec_agent.review(sample_pr)
        style_review = style_agent.review(sample_pr)
        task = Task(
            task_id="t2",
            task_type="synthesize",
            payload={
                "pr": sample_pr,
                "security_review": security_review,
                "style_review": style_review,
            },
        )
        result = summarizer.execute(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "t2"
        assert result.agent_id == "summarizer"
        assert result.status == "success"
        assert isinstance(result.output["report"], ReviewReport)

    def test_end_to_end_pipeline(self, chroma_client: chromadb.ClientAPI) -> None:
        reviewer = SecurityReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        prs = [
            SyntheticPR(
                pr_id=f"pr_{i:03d}",
                title=f"PR title {i}",
                description=f"Description for PR {i}",
                diff=f"+line {i}",
            )
            for i in range(10)
        ]
        for pr in prs:
            review = reviewer.review(pr)
            summary = summarizer.summarize(pr, review)
            assert summary.pr_id == pr.pr_id


class TestPlannerAgent:
    def test_create_returns_agent(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        assert agent.agent_id == "planner"

    def test_decompose_returns_decomposition_result(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        result = agent.decompose(sample_pr)
        assert isinstance(result, DecompositionResult)
        assert result.pr_id == "pr_test"

    def test_decompose_returns_subtasks(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        result = agent.decompose(sample_pr)
        assert len(result.subtasks) > 0
        assert all(isinstance(s, ReviewSubtask) for s in result.subtasks)

    def test_subtask_fields(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        result = agent.decompose(sample_pr)
        for subtask in result.subtasks:
            assert subtask.subtask_id.startswith("s")
            assert subtask.task_type in ("security_review", "style_review")
            assert subtask.code_segment != ""

    def test_decompose_retrieves_docs(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(top_k=3, chroma_client=chroma_client)
        result = agent.decompose(sample_pr)
        assert result.retrieved_docs == 3
        assert len(result.retrieval_distances) == 3

    def test_decompose_records_latencies(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        result = agent.decompose(sample_pr)
        assert result.retrieval_latency_ms >= 0.0
        assert result.llm_latency_ms >= 0.0

    def test_decompose_raw_response_non_empty(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        result = agent.decompose(sample_pr)
        assert result.raw_response.strip() != ""

    def test_decompose_deterministic_same_seed(
        self, sample_pr: SyntheticPR
    ) -> None:
        c1 = chromadb.EphemeralClient()
        c2 = chromadb.EphemeralClient()
        a1 = PlannerAgent.create(seed=42, chroma_client=c1)
        a2 = PlannerAgent.create(seed=42, chroma_client=c2)
        r1 = a1.decompose(sample_pr)
        r2 = a2.decompose(sample_pr)
        assert r1.raw_response == r2.raw_response

    def test_execute_returns_task_result(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        task = Task(
            task_id="t3",
            task_type="plan",
            payload={"pr": sample_pr},
        )
        result = agent.execute(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "t3"
        assert result.agent_id == "planner"
        assert result.status == "success"
        assert isinstance(result.output["decomposition"], DecompositionResult)
        assert result.llm_latency_ms >= 0.0
        assert result.retrieval_latency_ms >= 0.0

    def test_multiple_prs_decomposed(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = PlannerAgent.create(chroma_client=chroma_client)
        prs = [
            SyntheticPR(pr_id=f"pr_{i}", title=f"PR {i}", description="desc", diff="")
            for i in range(5)
        ]
        results = [agent.decompose(pr) for pr in prs]
        assert len(results) == 5
        assert all(isinstance(r, DecompositionResult) for r in results)


class TestStyleFinding:
    def test_dataclass_fields(self) -> None:
        f = StyleFinding(category="complexity", description="Function too long", line_ref="line 55")
        assert f.category == "complexity"
        assert f.description == "Function too long"
        assert f.line_ref == "line 55"

    def test_defaults(self) -> None:
        f = StyleFinding(category="naming", description="Non-descriptive variable name")
        assert f.line_ref == ""

    def test_parse_finding_complexity(self) -> None:
        line = "1. Function `process_data` exceeds 50 lines — refactor into smaller units"
        f = _parse_style_finding(line)
        assert f.category == "complexity"
        assert "process_data" in f.description

    def test_parse_finding_naming(self) -> None:
        line = "2. Variable names `x`, `y`, `tmp` are non-descriptive"
        f = _parse_style_finding(line)
        assert f.category == "naming"

    def test_parse_finding_documentation(self) -> None:
        line = "3. Missing docstrings on 3 public methods"
        f = _parse_style_finding(line)
        assert f.category == "documentation"

    def test_parse_finding_formatting(self) -> None:
        line = "2. Minor: trailing whitespace on line 47"
        f = _parse_style_finding(line)
        assert f.category == "formatting"
        assert "line 47" in f.line_ref

    def test_parse_finding_readability(self) -> None:
        line = "3. Duplicate code block detected — extract into shared helper"
        f = _parse_style_finding(line)
        assert f.category == "readability"

    def test_parse_finding_other(self) -> None:
        line = "1. No significant style issues found"
        f = _parse_style_finding(line)
        assert f.category == "other"

    def test_parse_finding_strips_leading_number(self) -> None:
        line = "5. Magic numbers used without named constants"
        f = _parse_style_finding(line)
        assert not f.description.startswith("5.")


class TestStyleReviewerAgent:
    def test_create_returns_agent(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        assert agent.agent_id == "style_reviewer"

    def test_review_returns_style_review(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert isinstance(review, StyleReview)
        assert review.pr_id == "pr_test"

    def test_review_retrieves_docs(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(top_k=3, chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.retrieved_docs == 3
        assert len(review.retrieval_distances) == 3

    def test_review_records_latencies(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.retrieval_latency_ms >= 0.0
        assert review.llm_latency_ms >= 0.0

    def test_review_has_recommendation(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert isinstance(review.recommendation, str)

    def test_review_findings_are_style_findings(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert isinstance(review.findings, list)
        assert all(isinstance(f, StyleFinding) for f in review.findings)

    def test_review_finding_fields_populated(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        valid_categories = {"complexity", "naming", "documentation", "formatting", "readability", "other"}
        for finding in review.findings:
            assert finding.category in valid_categories
            assert isinstance(finding.description, str)
            assert isinstance(finding.line_ref, str)

    def test_review_raw_response_non_empty(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        review = agent.review(sample_pr)
        assert review.raw_response.strip() != ""

    def test_review_deterministic_same_seed(self, sample_pr: SyntheticPR) -> None:
        c1 = chromadb.EphemeralClient()
        c2 = chromadb.EphemeralClient()
        a1 = StyleReviewerAgent.create(seed=42, chroma_client=c1)
        a2 = StyleReviewerAgent.create(seed=42, chroma_client=c2)
        r1 = a1.review(sample_pr)
        r2 = a2.review(sample_pr)
        assert r1.raw_response == r2.raw_response

    def test_execute_returns_task_result(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        task = Task(
            task_id="t4",
            task_type="style_review",
            payload={"pr": sample_pr},
        )
        result = agent.execute(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "t4"
        assert result.agent_id == "style_reviewer"
        assert result.status == "success"
        assert isinstance(result.output["review"], StyleReview)
        assert result.llm_latency_ms >= 0.0
        assert result.retrieval_latency_ms >= 0.0

    def test_multiple_prs_reviewed(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = StyleReviewerAgent.create(chroma_client=chroma_client)
        prs = [
            SyntheticPR(pr_id=f"pr_{i}", title=f"PR {i}", description="desc", diff="")
            for i in range(5)
        ]
        reviews = [agent.review(pr) for pr in prs]
        assert len(reviews) == 5
        assert all(isinstance(r, StyleReview) for r in reviews)

    def test_custom_agent_id(self, chroma_client: chromadb.ClientAPI) -> None:
        agent = StyleReviewerAgent.create(agent_id="my_style_agent", chroma_client=chroma_client)
        assert agent.agent_id == "my_style_agent"


# ---------------------------------------------------------------------------
# ReviewReport dataclass
# ---------------------------------------------------------------------------


class TestReviewReport:
    def test_dataclass_fields(self) -> None:
        report = ReviewReport(
            pr_id="pr_1",
            title="Add upload endpoint",
            overall_risk="high",
            security_findings=[],
            style_findings=[],
            markdown="# Report",
            retrieved_docs=3,
            retrieval_distances=[0.1, 0.2, 0.3],
            retrieval_latency_ms=5.0,
            llm_latency_ms=12.0,
            raw_response="raw",
        )
        assert report.pr_id == "pr_1"
        assert report.title == "Add upload endpoint"
        assert report.overall_risk == "high"
        assert report.markdown == "# Report"
        assert report.retrieved_docs == 3

    def test_overall_risk_valid_values(self) -> None:
        valid = {"none", "low", "medium", "high", "critical"}
        for risk in valid:
            r = ReviewReport(
                pr_id="x", title="t", overall_risk=risk,
                security_findings=[], style_findings=[], markdown="",
                retrieved_docs=0, retrieval_distances=[], retrieval_latency_ms=0.0,
                llm_latency_ms=0.0, raw_response="",
            )
            assert r.overall_risk == risk


# ---------------------------------------------------------------------------
# SummarizerAgent — synthesize() (Phase 2 interface)
# ---------------------------------------------------------------------------


class TestSummarizerAgentSynthesize:
    @pytest.fixture
    def agents(
        self, chroma_client: chromadb.ClientAPI
    ) -> tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent]:
        return (
            SecurityReviewerAgent.create(chroma_client=chroma_client),
            StyleReviewerAgent.create(chroma_client=chroma_client),
            SummarizerAgent.create(chroma_client=chroma_client),
        )

    def test_synthesize_returns_review_report(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert isinstance(report, ReviewReport)
        assert report.pr_id == "pr_test"
        assert report.title == sample_pr.title

    def test_synthesize_overall_risk_valid(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert report.overall_risk in ("none", "low", "medium", "high", "critical")

    def test_synthesize_markdown_contains_pr_title(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert sample_pr.title in report.markdown

    def test_synthesize_markdown_has_sections(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert "## Security Findings" in report.markdown
        assert "## Style Findings" in report.markdown
        assert "## Executive Summary" in report.markdown

    def test_synthesize_retrieves_templates(
        self,
        chroma_client: chromadb.ClientAPI,
        sample_pr: SyntheticPR,
    ) -> None:
        sec = SecurityReviewerAgent.create(chroma_client=chroma_client)
        style = StyleReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(top_k=3, chroma_client=chroma_client)
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert report.retrieved_docs == 3
        assert len(report.retrieval_distances) == 3

    def test_synthesize_records_latencies(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert report.retrieval_latency_ms >= 0.0
        assert report.llm_latency_ms >= 0.0

    def test_synthesize_propagates_security_findings(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        sec_review = sec.review(sample_pr)
        report = summarizer.synthesize(sample_pr, sec_review, style.review(sample_pr))
        assert report.security_findings == sec_review.findings

    def test_synthesize_propagates_style_findings(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        style_review = style.review(sample_pr)
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style_review)
        assert report.style_findings == style_review.findings

    def test_synthesize_raw_response_non_empty(
        self,
        agents: tuple[SecurityReviewerAgent, StyleReviewerAgent, SummarizerAgent],
        sample_pr: SyntheticPR,
    ) -> None:
        sec, style, summarizer = agents
        report = summarizer.synthesize(sample_pr, sec.review(sample_pr), style.review(sample_pr))
        assert report.raw_response.strip() != ""

    def test_synthesize_deterministic_same_seed(self, sample_pr: SyntheticPR) -> None:
        c1, c2 = chromadb.EphemeralClient(), chromadb.EphemeralClient()
        s1 = SummarizerAgent.create(seed=42, chroma_client=c1)
        s2 = SummarizerAgent.create(seed=42, chroma_client=c2)
        sec1 = SecurityReviewerAgent.create(seed=42, chroma_client=c1)
        sec2 = SecurityReviewerAgent.create(seed=42, chroma_client=c2)
        style1 = StyleReviewerAgent.create(seed=42, chroma_client=c1)
        style2 = StyleReviewerAgent.create(seed=42, chroma_client=c2)
        r1 = s1.synthesize(sample_pr, sec1.review(sample_pr), style1.review(sample_pr))
        r2 = s2.synthesize(sample_pr, sec2.review(sample_pr), style2.review(sample_pr))
        assert r1.raw_response == r2.raw_response

    def test_synthesize_empty_findings(
        self, chroma_client: chromadb.ClientAPI, sample_pr: SyntheticPR
    ) -> None:
        from chronoagent.agents.security_reviewer import SecurityFinding, SecurityReview
        from chronoagent.agents.style_reviewer import StyleReview

        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        empty_sec = SecurityReview(
            pr_id=sample_pr.pr_id,
            findings=[],
            severity="none",
            recommendation="Approve.",
            retrieved_docs=0,
            retrieval_distances=[],
            retrieval_latency_ms=0.0,
            llm_latency_ms=0.0,
            raw_response="",
        )
        empty_style = StyleReview(
            pr_id=sample_pr.pr_id,
            findings=[],
            recommendation="Approve.",
            retrieved_docs=0,
            retrieval_distances=[],
            retrieval_latency_ms=0.0,
            llm_latency_ms=0.0,
            raw_response="",
        )
        report = summarizer.synthesize(sample_pr, empty_sec, empty_style)
        assert report.overall_risk == "none"
        assert "No security findings." in report.markdown
        assert "No style findings." in report.markdown

    def test_multiple_prs_synthesized(self, chroma_client: chromadb.ClientAPI) -> None:
        sec = SecurityReviewerAgent.create(chroma_client=chroma_client)
        style = StyleReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        prs = [
            SyntheticPR(pr_id=f"pr_{i}", title=f"PR {i}", description="desc", diff="")
            for i in range(5)
        ]
        reports = [
            summarizer.synthesize(pr, sec.review(pr), style.review(pr)) for pr in prs
        ]
        assert len(reports) == 5
        assert all(isinstance(r, ReviewReport) for r in reports)


# ---------------------------------------------------------------------------
# AgentRegistry (task 2.7)
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    """Tests for AgentRegistry capability map and lookup."""

    def test_get_class_plan(self) -> None:
        registry = AgentRegistry()
        assert registry.get_class("plan") is PlannerAgent

    def test_get_class_security_review(self) -> None:
        registry = AgentRegistry()
        assert registry.get_class("security_review") is SecurityReviewerAgent

    def test_get_class_style_review(self) -> None:
        registry = AgentRegistry()
        assert registry.get_class("style_review") is StyleReviewerAgent

    def test_get_class_summarize(self) -> None:
        registry = AgentRegistry()
        assert registry.get_class("summarize") is SummarizerAgent

    def test_unknown_task_type_raises(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(UnknownTaskTypeError) as exc_info:
            registry.get_class("nonexistent_type")
        assert exc_info.value.task_type == "nonexistent_type"

    def test_unknown_task_type_message_contains_known_types(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(UnknownTaskTypeError, match="plan"):
            registry.get_class("bad_type")

    def test_supported_task_types_sorted(self) -> None:
        registry = AgentRegistry()
        types = registry.supported_task_types()
        assert types == sorted(types)
        assert set(types) == {"plan", "security_review", "style_review", "summarize"}

    def test_has_returns_true_for_known(self) -> None:
        registry = AgentRegistry()
        for t in ("plan", "security_review", "style_review", "summarize"):
            assert registry.has(t) is True

    def test_has_returns_false_for_unknown(self) -> None:
        assert AgentRegistry().has("unknown") is False

    def test_capabilities_returns_all_four(self) -> None:
        registry = AgentRegistry()
        caps = registry.capabilities()
        assert set(caps.keys()) == {"plan", "security_review", "style_review", "summarize"}
        assert caps["plan"] is PlannerAgent
        assert caps["security_review"] is SecurityReviewerAgent
        assert caps["style_review"] is StyleReviewerAgent
        assert caps["summarize"] is SummarizerAgent

    def test_capabilities_snapshot_is_copy(self) -> None:
        """Mutating the returned dict must not affect the registry."""
        registry = AgentRegistry()
        caps = registry.capabilities()
        caps["plan"] = StyleReviewerAgent  # type: ignore[assignment]
        assert registry.get_class("plan") is PlannerAgent

    def test_get_class_returns_baseagent_subclass(self) -> None:
        registry = AgentRegistry()
        for task_type in registry.supported_task_types():
            cls = registry.get_class(task_type)
            assert issubclass(cls, BaseAgent), f"{cls} is not a BaseAgent subclass"

    def test_multiple_registry_instances_share_same_map(self) -> None:
        """Two independent instances see the same capability map."""
        r1 = AgentRegistry()
        r2 = AgentRegistry()
        assert r1.capabilities() == r2.capabilities()
