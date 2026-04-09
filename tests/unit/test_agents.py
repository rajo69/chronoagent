"""Tests for SecurityReviewerAgent and SummarizerAgent."""

from __future__ import annotations

import chromadb
import pytest

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
