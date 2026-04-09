"""SummarizerAgent: summarizes security reviews for downstream consumers.

Takes the output of :class:`SecurityReviewerAgent` and a PR, retrieves relevant
historical context from ChromaDB, and produces a concise structured summary.
Designed to run with MockBackend for deterministic, cost-free experiments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI

from chronoagent.agents.backends.mock import MockBackend, MockBackendVariant
from chronoagent.agents.base import BaseAgent, RetrievalResult, Task, TaskResult
from chronoagent.agents.security_reviewer import SecurityReview, SyntheticPR

# ---------------------------------------------------------------------------
# Historical summary corpus pre-populated into ChromaDB
# ---------------------------------------------------------------------------

_SUMMARY_CONTEXT_DOCS: list[str] = [
    "Historical pattern: PRs introducing new database query methods have a 40% "
    "rate of SQL injection findings when input validation is absent.",
    "Historical pattern: Authentication refactors have a 35% rate of session "
    "management issues, particularly on logout and token refresh paths.",
    "Historical pattern: Frontend template changes carry a 30% XSS risk when "
    "the templating engine's auto-escape is disabled or bypassed.",
    "Historical pattern: Dependency updates frequently introduce transitive CVEs; "
    "always run a dependency audit before merging.",
    "Historical pattern: File upload handlers are a frequent source of path "
    "traversal and unrestricted file type vulnerabilities.",
    "Historical pattern: Serialization changes using pickle or Java serialization "
    "have caused multiple critical deserialization vulnerabilities.",
    "Historical pattern: PRs with only refactoring changes have an 8% finding "
    "rate — mostly informational items like unused imports.",
    "Historical pattern: SSO / JWT integrations are high-risk; algorithm "
    "confusion and key confusion attacks are common implementation mistakes.",
    "Remediation SLA: CRITICAL findings require patch within 24 hours. "
    "HIGH findings require patch within 7 days. MEDIUM within 30 days.",
    "Review process: All CRITICAL and HIGH findings require a follow-up review "
    "after the fix before the PR can be merged.",
    "Escalation: CRITICAL findings in authentication or data access paths must "
    "be escalated to the security team lead and logged in the incident tracker.",
    "Context: The team has seen a 20% increase in dependency-related findings "
    "over the past quarter; stricter version pinning has been recommended.",
]


@dataclass
class Summary:
    """Result of the SummarizerAgent processing a PR review.

    Attributes:
        pr_id: Identifier of the summarized PR.
        summary: One-paragraph plain-language summary.
        key_points: Bullet-point key takeaways.
        risk_level: Derived risk level ('low'/'medium'/'high'/'critical').
        retrieved_docs: Number of ChromaDB documents retrieved.
        retrieval_distances: Cosine distances of retrieved docs.
        retrieval_latency_ms: Time taken for ChromaDB query.
        llm_latency_ms: Time taken for LLM call.
        raw_response: Raw LLM output.
    """

    pr_id: str
    summary: str
    key_points: list[str]
    risk_level: str
    retrieved_docs: int
    retrieval_distances: list[float]
    retrieval_latency_ms: float
    llm_latency_ms: float
    raw_response: str


def _parse_summary(
    raw: str, pr_id: str, retrieval: RetrievalResult, review_severity: str
) -> Summary:
    """Parse raw LLM output into a :class:`Summary`.

    Args:
        raw: Raw LLM response text.
        pr_id: PR identifier.
        retrieval: ChromaDB retrieval result for this step.
        review_severity: Severity from the upstream :class:`SecurityReview`.

    Returns:
        Parsed :class:`Summary`.
    """
    summary_text = ""
    key_points: list[str] = []
    risk_level = review_severity  # default to upstream severity

    lines = raw.splitlines()
    in_key_points = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SUMMARY:"):
            summary_text = stripped.removeprefix("SUMMARY:").strip()
        elif stripped.startswith("KEY POINTS:"):
            in_key_points = True
        elif in_key_points and stripped.startswith("-"):
            key_points.append(stripped.lstrip("- ").strip())
        elif stripped.startswith("NEXT STEPS:"):
            in_key_points = False
        # Extract risk level from response if present
        upper = stripped.upper()
        if "OVERALL RISK LEVEL: CRITICAL" in upper:
            risk_level = "critical"
        elif "OVERALL RISK LEVEL: HIGH" in upper and risk_level not in ("critical",):
            risk_level = "high"
        elif "OVERALL RISK LEVEL: MEDIUM" in upper and risk_level not in ("critical", "high"):
            risk_level = "medium"
        elif "OVERALL RISK LEVEL: LOW" in upper and risk_level not in (
            "critical", "high", "medium"
        ):
            risk_level = "low"

    if not summary_text:
        summary_text = raw.split("\n")[0]

    return Summary(
        pr_id=pr_id,
        summary=summary_text,
        key_points=key_points,
        risk_level=risk_level,
        retrieved_docs=len(retrieval.documents),
        retrieval_distances=retrieval.distances,
        retrieval_latency_ms=retrieval.latency_ms,
        llm_latency_ms=0.0,  # filled in by caller
        raw_response=raw,
    )


class SummarizerAgent(BaseAgent):
    """Agent that summarizes security reviews into concise reports.

    Retrieves historical context from a ChromaDB knowledge base, augments
    the prompt with the upstream security review, and calls the LLM backend
    to produce a structured summary.

    Args:
        agent_id: Unique identifier for this agent instance.
        backend: :class:`~chronoagent.agents.backends.base.LLMBackend` for
            generation and embeddings.
        collection: ChromaDB collection containing historical context.
        top_k: Number of historical patterns to retrieve per step.
    """

    SYSTEM_PROMPT = (
        "You are a security team lead producing concise summaries of PR security reviews "
        "for engineering managers and developers. Use the historical context to calibrate "
        "risk levels. Structure your response with SUMMARY: followed by a paragraph, "
        "KEY POINTS: followed by bullet points, and NEXT STEPS: at the end."
    )

    def summarize(self, pr: SyntheticPR, review: SecurityReview) -> Summary:
        """Summarize a security review into a concise report.

        Args:
            pr: The original synthetic pull request.
            review: The upstream security review to summarize.

        Returns:
            :class:`Summary` with key points and retrieval metadata.
        """
        findings_text = (
            "\n".join(
                f"[{f.cwe_id}] {f.description} ({f.line_ref}) - {f.severity.upper()}"
                if f.cwe_id
                else f"{f.description} - {f.severity.upper()}"
                for f in review.findings
            )
            or "No findings."
        )
        query = f"{pr.title} {review.severity} {findings_text[:200]}"
        retrieval = self._retrieve_memory(query)

        context_block = "\n".join(
            f"- {doc}" for doc in retrieval.documents
        ) or "No historical context available."

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"HISTORICAL CONTEXT:\n{context_block}\n\n"
            f"PR TITLE: {pr.title}\n"
            f"SECURITY REVIEW FINDINGS:\n{findings_text}\n"
            f"SEVERITY: {review.severity.upper()}\n"
            f"RECOMMENDATION: {review.recommendation}\n\n"
            "Provide your summary:"
        )

        raw_response, llm_latency_ms = self._call_llm(prompt)

        summary = _parse_summary(raw_response, pr.pr_id, retrieval, review.severity)
        summary.llm_latency_ms = llm_latency_ms
        return summary

    def execute(self, task: Task) -> TaskResult:
        """Execute a summarization task.

        Args:
            task: Task with ``payload["pr"]`` set to a :class:`SyntheticPR`
                and ``payload["review"]`` set to a :class:`SecurityReview`.

        Returns:
            :class:`TaskResult` with ``output["summary"]`` containing the
            :class:`Summary`.
        """
        pr: SyntheticPR = task.payload["pr"]
        review: SecurityReview = task.payload["review"]
        summary = self.summarize(pr, review)
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            output={"summary": summary},
            llm_latency_ms=summary.llm_latency_ms,
            retrieval_latency_ms=summary.retrieval_latency_ms,
            timestamp=time.time(),
        )

    @classmethod
    def create(
        cls,
        agent_id: str = "summarizer",
        seed: int = 42,
        top_k: int = 3,
        chroma_client: ClientAPI | None = None,
    ) -> SummarizerAgent:
        """Factory method creating a ready-to-use agent with MockBackend.

        Args:
            agent_id: Unique identifier for this agent instance.
            seed: Seed for the :class:`MockBackend` response selection.
            top_k: Number of historical patterns to retrieve per step.
            chroma_client: ChromaDB client; ephemeral in-memory client used if None.

        Returns:
            Configured :class:`SummarizerAgent`.
        """
        client = chroma_client or chromadb.EphemeralClient()
        backend = MockBackend(seed=seed, variant=MockBackendVariant.SUMMARY)
        collection = cls.build_collection(
            client,
            name=f"{agent_id}_history_kb",
            documents=_SUMMARY_CONTEXT_DOCS,
            backend=backend,
        )
        return cls(agent_id=agent_id, backend=backend, collection=collection, top_k=top_k)
