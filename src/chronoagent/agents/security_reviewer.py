"""SecurityReviewerAgent: reviews synthetic PRs for security issues.

This agent retrieves relevant security patterns from ChromaDB, constructs a
prompt, and returns a structured security review.  It is designed to run
against MockBackend for deterministic, cost-free experiments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chromadb
from chromadb.api import ClientAPI
from langchain_core.language_models.llms import LLM

from chronoagent.agents.base import BaseAgent, RetrievalResult
from chronoagent.llm.mock_backend import MockBackend

# ---------------------------------------------------------------------------
# Synthetic PR corpus used to pre-populate the security knowledge base
# ---------------------------------------------------------------------------

_SECURITY_KNOWLEDGE_BASE: list[str] = [
    "SQL Injection: Always use parameterized queries or ORM abstractions. "
    "Never interpolate user input directly into SQL strings.",
    "XSS Prevention: Escape all user-supplied data before rendering in HTML. "
    "Use Content-Security-Policy headers to restrict script execution.",
    "Authentication: Use strong password hashing (bcrypt, argon2). "
    "Enforce MFA on privileged accounts. Invalidate sessions on logout.",
    "Secrets Management: Never hardcode API keys, passwords, or tokens. "
    "Use environment variables or a secrets manager (Vault, AWS Secrets Manager).",
    "Input Validation: Validate all external inputs at the application boundary. "
    "Use allowlists, not denylists, for permitted values.",
    "CSRF Protection: Require CSRF tokens on all state-modifying requests. "
    "Validate the Origin and Referer headers as a secondary check.",
    "Path Traversal: Normalize and validate all file paths. "
    "Restrict uploads to a designated safe directory; reject any '../' sequences.",
    "Dependency Security: Scan dependencies for known CVEs on every build. "
    "Pin transitive dependencies; review changelogs before upgrading.",
    "JWT Security: Always validate the algorithm field; reject 'none'. "
    "Use RS256 or ES256 for asymmetric signing. Rotate keys on a schedule.",
    "Rate Limiting: Apply rate limits to authentication endpoints to prevent "
    "credential stuffing. Use token-bucket or sliding-window algorithms.",
    "Insecure Deserialization: Never deserialize untrusted data with pickle. "
    "Prefer JSON with a strict schema or protobuf.",
    "IDOR Prevention: Always verify the requesting user owns or has access to "
    "the requested resource. Do not rely solely on object IDs in URLs.",
    "Security Headers: Set Strict-Transport-Security, X-Frame-Options, "
    "X-Content-Type-Options, and Referrer-Policy on all responses.",
    "Logging: Never log sensitive data (passwords, tokens, PII). "
    "Use structured logging; set appropriate retention and access controls.",
    "Race Conditions: Protect shared state with locks or atomic operations. "
    "Use database transactions for multi-step read-modify-write patterns.",
]


@dataclass
class SyntheticPR:
    """Represents a synthetic pull request used in experiments.

    Attributes:
        pr_id: Unique identifier (e.g. 'pr_001').
        title: Short PR title.
        description: Longer PR description.
        diff: Code diff string (simplified for simulation).
        files_changed: List of file paths modified.
    """

    pr_id: str
    title: str
    description: str
    diff: str
    files_changed: list[str] = field(default_factory=list)


@dataclass
class SecurityReview:
    """Result of the SecurityReviewerAgent processing a PR.

    Attributes:
        pr_id: Identifier of the reviewed PR.
        findings: List of individual security findings.
        severity: Highest severity level found ('none'/'low'/'medium'/'high'/'critical').
        recommendation: Final recommendation string.
        retrieved_docs: Number of ChromaDB documents retrieved.
        retrieval_distances: Cosine distances of retrieved docs.
        retrieval_latency_ms: Time taken for ChromaDB query.
        llm_latency_ms: Time taken for LLM call.
        raw_response: Raw LLM output.
    """

    pr_id: str
    findings: list[str]
    severity: str
    recommendation: str
    retrieved_docs: int
    retrieval_distances: list[float]
    retrieval_latency_ms: float
    llm_latency_ms: float
    raw_response: str


def _parse_review(raw: str, pr_id: str, retrieval: RetrievalResult) -> SecurityReview:
    """Parse raw LLM output into a :class:`SecurityReview`.

    Args:
        raw: Raw LLM response text.
        pr_id: PR identifier.
        retrieval: ChromaDB retrieval result for this step.

    Returns:
        Parsed :class:`SecurityReview`.
    """
    findings: list[str] = []
    severity = "none"
    recommendation = ""

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("1.", "2.", "3.", "4.", "5.")):
            findings.append(stripped)
            # Infer highest severity from finding text
            upper = stripped.upper()
            if "CRITICAL" in upper and severity not in ("critical",):
                severity = "critical"
            elif "HIGH" in upper and severity not in ("critical", "high"):
                severity = "high"
            elif "MEDIUM" in upper and severity not in ("critical", "high", "medium"):
                severity = "medium"
            elif "LOW" in upper and severity not in ("critical", "high", "medium", "low"):
                severity = "low"
        elif stripped.startswith("RECOMMENDATION:"):
            recommendation = stripped.removeprefix("RECOMMENDATION:").strip()

    return SecurityReview(
        pr_id=pr_id,
        findings=findings,
        severity=severity,
        recommendation=recommendation,
        retrieved_docs=len(retrieval.documents),
        retrieval_distances=retrieval.distances,
        retrieval_latency_ms=retrieval.latency_ms,
        llm_latency_ms=0.0,  # filled in by the caller
        raw_response=raw,
    )


class SecurityReviewerAgent(BaseAgent):
    """Agent that reviews synthetic PRs for security vulnerabilities.

    Retrieves relevant security patterns from a ChromaDB knowledge base,
    augments the prompt with retrieved context, and calls the LLM backend
    to produce a structured security review.

    Args:
        agent_id: Unique identifier for this agent instance.
        llm: LangChain-compatible language model backend.
        collection: ChromaDB collection containing security patterns.
        top_k: Number of security patterns to retrieve per PR.
    """

    SYSTEM_PROMPT = (
        "You are a senior security engineer reviewing pull requests for vulnerabilities. "
        "Use the provided security patterns to guide your analysis. "
        "Structure your response with SECURITY REVIEW FINDINGS: followed by numbered items, "
        "and end with RECOMMENDATION:."
    )

    def review(self, pr: SyntheticPR) -> SecurityReview:
        """Review a synthetic PR and return a structured security review.

        Args:
            pr: The synthetic pull request to review.

        Returns:
            :class:`SecurityReview` with findings, severity, and retrieval metadata.
        """
        query = f"{pr.title} {pr.description} {pr.diff[:200]}"
        retrieval = self.retrieve(query)

        context_block = "\n".join(
            f"- {doc}" for doc in retrieval.documents
        ) or "No relevant patterns retrieved."

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"SECURITY PATTERNS:\n{context_block}\n\n"
            f"PR TITLE: {pr.title}\n"
            f"PR DESCRIPTION: {pr.description}\n"
            f"FILES CHANGED: {', '.join(pr.files_changed) or 'unknown'}\n"
            f"DIFF EXCERPT:\n{pr.diff[:500]}\n\n"
            "Provide your security review:"
        )

        t0 = time.perf_counter()
        raw_response = self.llm.invoke(prompt)
        llm_latency_ms = (time.perf_counter() - t0) * 1_000

        review = _parse_review(str(raw_response), pr.pr_id, retrieval)
        review.llm_latency_ms = llm_latency_ms
        return review

    @classmethod
    def create(
        cls,
        agent_id: str = "security_reviewer",
        seed: int = 42,
        top_k: int = 3,
        chroma_client: ClientAPI | None = None,
    ) -> SecurityReviewerAgent:
        """Factory method creating a ready-to-use agent with MockBackend.

        Args:
            agent_id: Unique identifier for this agent instance.
            seed: Seed for the :class:`MockBackend` response selection.
            top_k: Number of security patterns to retrieve per PR.
            chroma_client: ChromaDB client; ephemeral in-memory client used if None.

        Returns:
            Configured :class:`SecurityReviewerAgent`.
        """
        client = chroma_client or chromadb.EphemeralClient()
        collection = cls.build_collection(
            client,
            name=f"{agent_id}_security_kb",
            documents=_SECURITY_KNOWLEDGE_BASE,
        )
        llm: LLM = MockBackend(seed=seed)
        return cls(agent_id=agent_id, llm=llm, collection=collection, top_k=top_k)
