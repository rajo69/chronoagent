"""SummarizerAgent: synthesizes all review findings into a ReviewReport.

Combines :class:`~chronoagent.agents.security_reviewer.SecurityReview` and
:class:`~chronoagent.agents.style_reviewer.StyleReview` into a single structured
:class:`ReviewReport` with a markdown body.  Retrieves report templates from
ChromaDB to guide formatting and risk calibration.

The Phase 1 :class:`Summary` dataclass and :meth:`SummarizerAgent.summarize`
method are preserved for the signal-validation experiment pipeline in
:mod:`chronoagent.experiments.runner`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI

from chronoagent.agents.backends.mock import MockBackend, MockBackendVariant
from chronoagent.agents.base import BaseAgent, RetrievalResult, Task, TaskResult
from chronoagent.agents.security_reviewer import SecurityFinding, SecurityReview, SyntheticPR
from chronoagent.agents.style_reviewer import StyleFinding, StyleReview

# ---------------------------------------------------------------------------
# Report template knowledge base — 10 entries guiding report structure and
# risk calibration for different finding profiles.
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE_DOCS: list[str] = [
    "Template CRITICAL_REPORT: Open with 'MERGE BLOCKED — CRITICAL FINDINGS DETECTED'. "
    "List CRITICAL findings first, then HIGH. State that all CRITICAL issues require "
    "same-day remediation and must be escalated to the security team lead.",
    "Template HIGH_RISK_REPORT: Open with 'MERGE BLOCKED — HIGH SEVERITY FINDINGS'. "
    "Enumerate each HIGH finding with its CWE ID and line reference. "
    "State 7-day SLA for remediation before re-review.",
    "Template MEDIUM_RISK_REPORT: Open with 'CHANGES REQUESTED'. "
    "List MEDIUM findings with suggested fixes. "
    "Approve conditionally upon resolution within 30 days.",
    "Template LOW_RISK_REPORT: Open with 'APPROVED WITH NITS'. "
    "List LOW and informational findings as optional cleanup items. No merge block.",
    "Template CLEAN_REPORT: Open with 'APPROVED'. "
    "State no significant security or style issues found. "
    "List any trivial nits inline; encourage follow-up on deferred items.",
    "Template MIXED_SECURITY_STYLE: Lead with security section (risk-ordered findings), "
    "then style section (category-grouped findings). Provide separate recommendations "
    "for security and style tracks.",
    "Template DEPENDENCY_REPORT: List each CVE with its NVD score and affected version range. "
    "Recommend pinning to a patched version. Cross-reference SBOM and dependency audit logs.",
    "Template AUTH_CHANGE_REPORT: Apply elevated scrutiny to JWT, session, and credential "
    "findings first regardless of severity. Include references to OWASP ASVS Level 2.",
    "Template REFACTOR_REPORT: Security risk is typically low for refactoring PRs. "
    "Focus on complexity metrics, naming quality, and documentation completeness.",
    "Template STYLE_HEAVY_REPORT: No security findings. Detail each style category: "
    "complexity, naming, documentation, formatting, readability. "
    "Provide a code quality score summary.",
]

# ---------------------------------------------------------------------------
# Phase 1 summary context (kept for backward compat with experiments/runner.py)
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

# ---------------------------------------------------------------------------
# Risk ordering shared by Phase 1 and Phase 2 helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ("none", "low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# Phase 1: Summary dataclass + parser
# ---------------------------------------------------------------------------


@dataclass
class Summary:
    """Result of the SummarizerAgent processing a PR review (Phase 1).

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
    risk_level = review_severity

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
        llm_latency_ms=0.0,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Phase 2: ReviewReport dataclass + helpers
# ---------------------------------------------------------------------------


@dataclass
class ReviewReport:
    """Full synthesized review report combining security and style findings.

    Attributes:
        pr_id: Identifier of the reviewed PR.
        title: PR title.
        overall_risk: Highest risk level across all security findings
            (``'none'``, ``'low'``, ``'medium'``, ``'high'``, or ``'critical'``).
        security_findings: List of :class:`~chronoagent.agents.security_reviewer.SecurityFinding`
            from :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent`.
        style_findings: List of :class:`~chronoagent.agents.style_reviewer.StyleFinding`
            from :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent`.
        markdown: Complete markdown-formatted review report body.
        retrieved_docs: Number of ChromaDB template documents retrieved.
        retrieval_distances: Cosine distances of retrieved templates.
        retrieval_latency_ms: Time taken for ChromaDB query.
        llm_latency_ms: Time taken for LLM call.
        raw_response: Raw LLM output (executive summary + next steps).
    """

    pr_id: str
    title: str
    overall_risk: str
    security_findings: list[SecurityFinding]
    style_findings: list[StyleFinding]
    markdown: str
    retrieved_docs: int
    retrieval_distances: list[float]
    retrieval_latency_ms: float
    llm_latency_ms: float
    raw_response: str


def _compute_overall_risk(security_review: SecurityReview) -> str:
    """Derive overall risk from the security review severity.

    Args:
        security_review: The upstream :class:`SecurityReview`.

    Returns:
        Risk level string — one of ``'none'``, ``'low'``, ``'medium'``,
        ``'high'``, or ``'critical'``.
    """
    return security_review.severity


def _parse_synthesis(raw: str) -> tuple[str, list[str], str]:
    """Extract executive summary, next steps, and overall risk from LLM output.

    Expected format (produced by :data:`SummarizerAgent.SYSTEM_PROMPT`)::

        EXECUTIVE SUMMARY: ...
        NEXT STEPS:
        - action item 1
        - action item 2
        OVERALL RISK: HIGH

    Args:
        raw: Raw LLM response text.

    Returns:
        Tuple of ``(exec_summary, next_steps, overall_risk)``.
    """
    exec_summary = ""
    next_steps: list[str] = []
    overall_risk = "none"
    in_next_steps = False

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("EXECUTIVE SUMMARY:"):
            exec_summary = stripped.removeprefix("EXECUTIVE SUMMARY:").strip()
            in_next_steps = False
        elif stripped.startswith("NEXT STEPS:"):
            in_next_steps = True
        elif in_next_steps and stripped.startswith("-"):
            next_steps.append(stripped.lstrip("- ").strip())
        elif stripped.startswith("OVERALL RISK:"):
            in_next_steps = False
            risk_text = stripped.removeprefix("OVERALL RISK:").strip().lower()
            if risk_text in _SEVERITY_ORDER:
                overall_risk = risk_text

    if not exec_summary:
        exec_summary = raw.split("\n")[0]

    return exec_summary, next_steps, overall_risk


def _build_markdown(
    pr: SyntheticPR,
    security_review: SecurityReview,
    style_review: StyleReview,
    exec_summary: str,
    next_steps: list[str],
    overall_risk: str,
) -> str:
    """Build a complete markdown review report from all components.

    Args:
        pr: The pull request being reviewed.
        security_review: Security review result.
        style_review: Style review result.
        exec_summary: LLM-generated executive summary paragraph.
        next_steps: List of action items from LLM synthesis.
        overall_risk: Computed overall risk level string.

    Returns:
        Formatted markdown string.
    """
    lines: list[str] = [
        f"# PR Review Report: {pr.title}",
        "",
        f"**PR ID:** {pr.pr_id}  ",
        f"**Overall Risk:** {overall_risk.upper()}  ",
        f"**Files Changed:** {', '.join(pr.files_changed) or 'unknown'}",
        "",
        "---",
        "",
        "## Security Findings",
        "",
    ]

    if security_review.findings:
        for i, f in enumerate(security_review.findings, 1):
            cwe = f" [{f.cwe_id}]" if f.cwe_id else ""
            ref = f" ({f.line_ref})" if f.line_ref else ""
            lines.append(f"{i}.{cwe} **{f.severity.upper()}**{ref} — {f.description}")
    else:
        lines.append("No security findings.")

    lines += [
        "",
        f"**Security Recommendation:** {security_review.recommendation}",
        "",
        "---",
        "",
        "## Style Findings",
        "",
    ]

    if style_review.findings:
        for i, f in enumerate(style_review.findings, 1):
            ref = f" ({f.line_ref})" if f.line_ref else ""
            lines.append(f"{i}. **[{f.category}]**{ref} — {f.description}")
    else:
        lines.append("No style findings.")

    lines += [
        "",
        f"**Style Recommendation:** {style_review.recommendation}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        exec_summary,
    ]

    if next_steps:
        lines += [
            "",
            "---",
            "",
            "## Next Steps",
            "",
        ]
        for step in next_steps:
            lines.append(f"- {step}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SummarizerAgent
# ---------------------------------------------------------------------------


class SummarizerAgent(BaseAgent):
    """Agent that synthesizes all review findings into a :class:`ReviewReport`.

    Retrieves report templates from a ChromaDB knowledge base, builds a prompt
    from the security and style review outputs, and calls the LLM backend to
    produce an executive summary.  The final :class:`ReviewReport` includes a
    fully rendered markdown body.

    Also exposes :meth:`summarize` (Phase 1 interface) for use in the
    signal-validation experiment pipeline.

    Args:
        agent_id: Unique identifier for this agent instance.
        backend: :class:`~chronoagent.agents.backends.base.LLMBackend` for
            generation and embeddings.
        collection: ChromaDB collection containing report templates.
        top_k: Number of templates to retrieve per synthesis.
    """

    SYSTEM_PROMPT = (
        "You are a tech lead producing a final review report that synthesizes security and "
        "style findings for engineering managers. Use the retrieved report templates to guide "
        "structure and risk calibration. "
        "Structure your response with EXECUTIVE SUMMARY: followed by a paragraph, "
        "NEXT STEPS: followed by bullet points (one per line, prefixed with '-'), "
        "and OVERALL RISK: followed by one word (none/low/medium/high/critical)."
    )

    def synthesize(
        self,
        pr: SyntheticPR,
        security_review: SecurityReview,
        style_review: StyleReview,
    ) -> ReviewReport:
        """Synthesize security and style reviews into a full :class:`ReviewReport`.

        Args:
            pr: The pull request being reviewed.
            security_review: Output of :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent`.
            style_review: Output of :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent`.

        Returns:
            :class:`ReviewReport` with markdown body, overall risk, and retrieval metadata.
        """
        query = (
            f"{pr.title} {security_review.severity} "
            f"{len(security_review.findings)} security findings "
            f"{len(style_review.findings)} style findings"
        )
        retrieval = self._retrieve_memory(query)

        context_block = (
            "\n".join(f"- {doc}" for doc in retrieval.documents)
            or "No templates retrieved."
        )

        sec_text = (
            "\n".join(
                f"[{f.cwe_id}] {f.description} - {f.severity.upper()}"
                if f.cwe_id
                else f"{f.description} - {f.severity.upper()}"
                for f in security_review.findings
            )
            or "No security findings."
        )
        style_text = (
            "\n".join(
                f"[{f.category}] {f.description}" for f in style_review.findings
            )
            or "No style findings."
        )

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"REPORT TEMPLATES:\n{context_block}\n\n"
            f"PR TITLE: {pr.title}\n"
            f"SECURITY FINDINGS:\n{sec_text}\n"
            f"SECURITY RECOMMENDATION: {security_review.recommendation}\n"
            f"STYLE FINDINGS:\n{style_text}\n"
            f"STYLE RECOMMENDATION: {style_review.recommendation}\n\n"
            "Provide your executive summary, next steps, and overall risk:"
        )

        raw_response, llm_latency_ms = self._call_llm(prompt)

        exec_summary, next_steps, _ = _parse_synthesis(raw_response)
        overall_risk = _compute_overall_risk(security_review)
        markdown = _build_markdown(
            pr, security_review, style_review, exec_summary, next_steps, overall_risk
        )

        return ReviewReport(
            pr_id=pr.pr_id,
            title=pr.title,
            overall_risk=overall_risk,
            security_findings=security_review.findings,
            style_findings=style_review.findings,
            markdown=markdown,
            retrieved_docs=len(retrieval.documents),
            retrieval_distances=retrieval.distances,
            retrieval_latency_ms=retrieval.latency_ms,
            llm_latency_ms=llm_latency_ms,
            raw_response=raw_response,
        )

    def summarize(self, pr: SyntheticPR, review: SecurityReview) -> Summary:
        """Summarize a security review into a concise report (Phase 1 interface).

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

        context_block = (
            "\n".join(f"- {doc}" for doc in retrieval.documents)
            or "No historical context available."
        )

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
        """Execute a report synthesis task.

        Args:
            task: Task with ``payload["pr"]`` set to a :class:`SyntheticPR`,
                ``payload["security_review"]`` set to a :class:`SecurityReview`,
                and ``payload["style_review"]`` set to a :class:`StyleReview`.

        Returns:
            :class:`TaskResult` with ``output["report"]`` containing the
            :class:`ReviewReport`.
        """
        pr: SyntheticPR = task.payload["pr"]
        security_review: SecurityReview = task.payload["security_review"]
        style_review: StyleReview = task.payload["style_review"]
        report = self.synthesize(pr, security_review, style_review)
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            output={"report": report},
            llm_latency_ms=report.llm_latency_ms,
            retrieval_latency_ms=report.retrieval_latency_ms,
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
            top_k: Number of report templates to retrieve per synthesis.
            chroma_client: ChromaDB client; ephemeral in-memory client used if None.

        Returns:
            Configured :class:`SummarizerAgent`.
        """
        client = chroma_client or chromadb.EphemeralClient()
        backend = MockBackend(seed=seed, variant=MockBackendVariant.REPORT)
        collection = cls.build_collection(
            client,
            name=f"{agent_id}_report_templates_kb",
            documents=_REPORT_TEMPLATE_DOCS,
            backend=backend,
        )
        return cls(agent_id=agent_id, backend=backend, collection=collection, top_k=top_k)
