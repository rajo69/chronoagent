"""SecurityReviewerAgent: reviews synthetic PRs for security issues.

This agent retrieves relevant CWE-mapped security patterns from ChromaDB,
constructs a prompt, and returns a structured security review with per-finding
severity, description, line reference, and CWE identifier.  It is designed to
run against MockBackend for deterministic, cost-free experiments.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import chromadb
from chromadb.api import ClientAPI

from chronoagent.agents.backends.mock import MockBackend, MockBackendVariant
from chronoagent.agents.base import BaseAgent, RetrievalResult, Task, TaskResult

# ---------------------------------------------------------------------------
# CWE top-25 knowledge base pre-populated into the security ChromaDB collection.
# Each entry maps a CWE identifier to a concise detection + mitigation note.
# ---------------------------------------------------------------------------

_SECURITY_KNOWLEDGE_BASE: list[str] = [
    "CWE-89 SQL Injection: Always use parameterized queries or ORM abstractions. "
    "Never interpolate user input directly into SQL strings. "
    "Detection: look for string concatenation in DB query construction.",
    "CWE-79 Cross-site Scripting (XSS): Escape all user-supplied data before rendering "
    "in HTML. Use Content-Security-Policy headers to restrict script execution. "
    "Detection: unsanitized template variables, missing output encoding.",
    "CWE-287 Improper Authentication: Use strong password hashing (bcrypt, argon2). "
    "Enforce MFA on privileged accounts. Invalidate sessions on logout. "
    "Detection: plain-text password storage, missing session invalidation.",
    "CWE-798 Hardcoded Credentials: Never hardcode API keys, passwords, or tokens. "
    "Use environment variables or a secrets manager (Vault, AWS Secrets Manager). "
    "Detection: literals matching /password|api_key|secret/ in source code.",
    "CWE-20 Improper Input Validation: Validate all external inputs at the application "
    "boundary. Use allowlists, not denylists, for permitted values. "
    "Detection: missing schema validation on request bodies or query parameters.",
    "CWE-352 Cross-Site Request Forgery (CSRF): Require CSRF tokens on all "
    "state-modifying requests. Validate the Origin and Referer headers. "
    "Detection: POST/PUT/DELETE handlers missing token verification.",
    "CWE-22 Path Traversal: Normalize and validate all file paths. Restrict uploads "
    "to a designated safe directory; reject any '../' sequences. "
    "Detection: user-controlled path components without normalization.",
    "CWE-937 Using Components with Known Vulnerabilities: Scan dependencies for known "
    "CVEs on every build. Pin transitive dependencies; review changelogs before upgrading. "
    "Detection: outdated package versions in lock files.",
    "CWE-347 Improper Verification of Cryptographic Signature (JWT): Always validate "
    "the algorithm field; reject 'none'. Use RS256 or ES256 for asymmetric signing. "
    "Detection: algorithm field not explicitly allowlisted in JWT validation.",
    "CWE-307 Improper Restriction of Excessive Authentication Attempts: Apply rate "
    "limits to authentication endpoints to prevent credential stuffing. "
    "Detection: login/auth endpoints missing throttling middleware.",
    "CWE-502 Deserialization of Untrusted Data: Never deserialize untrusted data with "
    "pickle. Prefer JSON with a strict schema or protobuf. "
    "Detection: pickle.loads / yaml.load called on external data.",
    "CWE-639 Insecure Direct Object Reference (IDOR): Always verify the requesting "
    "user owns or has access to the requested resource. "
    "Detection: resource IDs from URL/body used without ownership check.",
    "CWE-16 Security Misconfiguration (Headers): Set Strict-Transport-Security, "
    "X-Frame-Options, X-Content-Type-Options, and Referrer-Policy on all responses. "
    "Detection: missing security-header middleware in response pipeline.",
    "CWE-532 Insertion of Sensitive Information into Log File: Never log passwords, "
    "tokens, or PII. Use structured logging with field-level redaction. "
    "Detection: log statements containing auth headers, passwords, or user data.",
    "CWE-362 Race Condition: Protect shared state with locks or atomic operations. "
    "Use database transactions for multi-step read-modify-write patterns. "
    "Detection: counter increments or token refresh logic without locking.",
    "CWE-434 Unrestricted Upload of File with Dangerous Type: Validate uploaded file "
    "type and extension. Reject executable types; store outside web root. "
    "Detection: file upload handlers with no MIME or extension validation.",
    "CWE-611 XML External Entity (XXE): Disable external entity processing in XML "
    "parsers. Use a hardened parser configuration. "
    "Detection: lxml / ElementTree with external entity resolution enabled.",
    "CWE-918 Server-Side Request Forgery (SSRF): Validate and allowlist outbound URLs. "
    "Do not forward user-supplied URLs to internal services. "
    "Detection: HTTP client calls using user-controlled URL parameters.",
    "CWE-284 Improper Access Control: Enforce role-based or attribute-based access "
    "control on every protected resource. "
    "Detection: admin-only endpoints lacking authorization middleware.",
    "CWE-476 NULL Pointer Dereference: Validate return values before dereferencing. "
    "Use Optional types and null checks where APIs may return None. "
    "Detection: chained attribute access on values from external or optional sources.",
    "CWE-190 Integer Overflow: Use safe arithmetic or checked-integer libraries for "
    "untrusted numeric inputs. "
    "Detection: arithmetic on user-supplied integers without bounds checks.",
    "CWE-416 Use After Free: Avoid manual memory management; prefer garbage-collected "
    "or ownership-typed languages. In C/C++, audit all free() call sites. "
    "Detection: pointer reuse after deallocation in low-level code.",
    "CWE-306 Missing Authentication for Critical Function: Require authentication on "
    "all state-modifying endpoints. Unauthenticated access must be an explicit design "
    "decision. Detection: routes without auth middleware or session check.",
    "CWE-119 Buffer Overflow: Validate all buffer lengths before copy operations. "
    "Use safe string functions (strlcpy, snprintf). "
    "Detection: memcpy / strcpy calls with user-controlled length arguments.",
    "CWE-601 Open Redirect: Validate redirect targets against an allowlist of trusted "
    "domains. Reject or encode user-supplied redirect URLs. "
    "Detection: redirect() calls using unvalidated user input as the destination.",
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
class SecurityFinding:
    """A single structured security finding produced by the reviewer.

    Attributes:
        severity: Finding severity: 'none', 'low', 'medium', 'high', or 'critical'.
        description: Human-readable description of the vulnerability.
        line_ref: Source-level reference (e.g. ``"line 42"`` or ``""`` if unknown).
        cwe_id: CWE identifier (e.g. ``"CWE-89"``), or ``""`` if not mapped.
    """

    severity: str
    description: str
    line_ref: str = ""
    cwe_id: str = ""


# Pre-compiled patterns for _parse_finding.
_CWE_RE = re.compile(r"\[(CWE-\d+)\]")
_LINE_RE = re.compile(r"\(lines?\s+[\d,\s–-]+\)", re.IGNORECASE)
_SEVERITY_RE = re.compile(r"-\s*SEVERITY:\s*(\w+)", re.IGNORECASE)
_LEADING_NUM_RE = re.compile(r"^\d+\.\s*")
_SEVERITY_MAP = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "info": "low",
    "none": "none",
}


def _parse_finding(line: str) -> SecurityFinding:
    """Extract a :class:`SecurityFinding` from a single numbered finding line.

    Expected format (produced by :data:`SecurityReviewerAgent.SYSTEM_PROMPT`)::

        1. [CWE-89] SQL injection risk in user input handling (line 42) - SEVERITY: HIGH

    All fields degrade gracefully if absent.

    Args:
        line: A single numbered finding line from the LLM response.

    Returns:
        Parsed :class:`SecurityFinding`.
    """
    text = _LEADING_NUM_RE.sub("", line).strip()

    # Extract CWE identifier
    cwe_match = _CWE_RE.search(text)
    cwe_id = cwe_match.group(1) if cwe_match else ""
    if cwe_match:
        text = text[: cwe_match.start()] + text[cwe_match.end() :]

    # Extract severity
    sev_match = _SEVERITY_RE.search(text)
    raw_sev = sev_match.group(1).lower() if sev_match else "none"
    severity = _SEVERITY_MAP.get(raw_sev, "none")
    if sev_match:
        text = text[: sev_match.start()] + text[sev_match.end() :]

    # Extract line reference
    line_match = _LINE_RE.search(text)
    line_ref = line_match.group(0).strip("()") if line_match else ""
    if line_match:
        text = text[: line_match.start()] + text[line_match.end() :]

    description = text.strip(" .-")
    return SecurityFinding(
        severity=severity,
        description=description,
        line_ref=line_ref,
        cwe_id=cwe_id,
    )


_SEVERITY_ORDER = ("none", "low", "medium", "high", "critical")


def _highest_severity(findings: list[SecurityFinding]) -> str:
    """Return the highest severity across all findings.

    Args:
        findings: List of :class:`SecurityFinding` objects.

    Returns:
        The highest severity string, or ``"none"`` if *findings* is empty.
    """
    best = "none"
    for f in findings:
        if _SEVERITY_ORDER.index(f.severity) > _SEVERITY_ORDER.index(best):
            best = f.severity
    return best


@dataclass
class SecurityReview:
    """Result of the SecurityReviewerAgent processing a PR.

    Attributes:
        pr_id: Identifier of the reviewed PR.
        findings: List of structured :class:`SecurityFinding` objects.
        severity: Highest severity level found ('none'/'low'/'medium'/'high'/'critical').
        recommendation: Final recommendation string.
        retrieved_docs: Number of ChromaDB documents retrieved.
        retrieval_distances: Cosine distances of retrieved docs.
        retrieval_latency_ms: Time taken for ChromaDB query.
        llm_latency_ms: Time taken for LLM call.
        raw_response: Raw LLM output.
    """

    pr_id: str
    findings: list[SecurityFinding]
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
    findings: list[SecurityFinding] = []
    recommendation = ""

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and ". " in stripped:
            findings.append(_parse_finding(stripped))
        elif stripped.startswith("RECOMMENDATION:"):
            recommendation = stripped.removeprefix("RECOMMENDATION:").strip()

    severity = _highest_severity(findings)

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
        backend: :class:`~chronoagent.agents.backends.base.LLMBackend` for
            generation and embeddings.
        collection: ChromaDB collection containing security patterns.
        top_k: Number of security patterns to retrieve per PR.
    """

    SYSTEM_PROMPT = (
        "You are a senior security engineer reviewing pull requests for vulnerabilities. "
        "Use the provided CWE-mapped security patterns to guide your analysis. "
        "Structure your response with SECURITY REVIEW FINDINGS: followed by numbered items "
        "in this exact format:\n"
        "  N. [CWE-NNN] <description> (<line ref>) - SEVERITY: <LEVEL>\n"
        "where LEVEL is one of: CRITICAL, HIGH, MEDIUM, LOW, NONE. "
        "Include a CWE identifier whenever applicable. Include a line reference "
        "(e.g. 'line 42') when the diff pinpoints the vulnerable line. "
        "End with RECOMMENDATION: followed by a single-line action item."
    )

    def review(self, pr: SyntheticPR) -> SecurityReview:
        """Review a synthetic PR and return a structured security review.

        Args:
            pr: The synthetic pull request to review.

        Returns:
            :class:`SecurityReview` with findings, severity, and retrieval metadata.
        """
        query = f"{pr.title} {pr.description} {pr.diff[:200]}"
        retrieval = self._retrieve_memory(query)

        context_block = (
            "\n".join(f"- {doc}" for doc in retrieval.documents)
            or "No relevant patterns retrieved."
        )

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"SECURITY PATTERNS:\n{context_block}\n\n"
            f"PR TITLE: {pr.title}\n"
            f"PR DESCRIPTION: {pr.description}\n"
            f"FILES CHANGED: {', '.join(pr.files_changed) or 'unknown'}\n"
            f"DIFF EXCERPT:\n{pr.diff[:500]}\n\n"
            "Provide your security review:"
        )

        raw_response, llm_latency_ms = self._call_llm(prompt)

        review = _parse_review(raw_response, pr.pr_id, retrieval)
        review.llm_latency_ms = llm_latency_ms
        return review

    def execute(self, task: Task) -> TaskResult:
        """Execute a security review task.

        Args:
            task: Task with ``payload["pr"]`` set to a :class:`SyntheticPR`.

        Returns:
            :class:`TaskResult` with ``output["review"]`` containing the
            :class:`SecurityReview`.
        """
        pr: SyntheticPR = task.payload["pr"]
        review = self.review(pr)
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            output={"review": review},
            llm_latency_ms=review.llm_latency_ms,
            retrieval_latency_ms=review.retrieval_latency_ms,
            timestamp=time.time(),
        )

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
        backend = MockBackend(seed=seed, variant=MockBackendVariant.SECURITY)
        collection = cls.build_collection(
            client,
            name=f"{agent_id}_security_kb",
            documents=_SECURITY_KNOWLEDGE_BASE,
            backend=backend,
        )
        return cls(agent_id=agent_id, backend=backend, collection=collection, top_k=top_k)
