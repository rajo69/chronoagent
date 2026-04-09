"""Tests for BaseAgent ABC, SecurityReviewerAgent, and SummarizerAgent."""

from __future__ import annotations

import chromadb
import pytest

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.agents.base import BaseAgent, Task, TaskResult
from chronoagent.agents.security_reviewer import (
    SecurityReview,
    SecurityReviewerAgent,
    SyntheticPR,
)
from chronoagent.agents.summarizer import SummarizerAgent, Summary


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
        reviewer = SecurityReviewerAgent.create(chroma_client=chroma_client)
        summarizer = SummarizerAgent.create(chroma_client=chroma_client)
        review = reviewer.review(sample_pr)
        task = Task(
            task_id="t2",
            task_type="summarize",
            payload={"pr": sample_pr, "review": review},
        )
        result = summarizer.execute(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "t2"
        assert result.agent_id == "summarizer"
        assert result.status == "success"
        assert isinstance(result.output["summary"], Summary)

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
