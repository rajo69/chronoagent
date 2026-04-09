"""Review router — POST /api/v1/review and GET /api/v1/review/{review_id}.

``POST /api/v1/review``
    Accepts a PR payload, runs it through :class:`~chronoagent.pipeline.graph.ReviewPipeline`,
    persists the :class:`~chronoagent.agents.summarizer.ReviewReport` in the in-memory store
    keyed by *pr_id*, and returns the serialised report.

``GET /api/v1/review/{review_id}``
    Returns the previously computed report for *review_id* (= pr_id), or HTTP 404 if
    no review with that id has been submitted.

The pipeline instance and the review store are stored on ``app.state`` and injected
via FastAPI dependency functions.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from chronoagent.agents.security_reviewer import SyntheticPR
from chronoagent.agents.summarizer import ReviewReport
from chronoagent.pipeline.graph import ReviewPipeline

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["review"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class PRSubmitRequest(BaseModel):
    """Request body for submitting a PR for review.

    Attributes:
        pr_id: Unique pull-request identifier (e.g. ``'pr_001'``).
        title: Short PR title.
        description: Longer PR description / body.
        diff: Raw code diff text.
        files_changed: Optional list of file paths modified by the PR.
    """

    pr_id: str
    title: str
    description: str
    diff: str
    files_changed: list[str] = []


class SecurityFindingOut(BaseModel):
    """Serialised security finding.

    Attributes:
        severity: ``'none'``, ``'low'``, ``'medium'``, ``'high'``, or ``'critical'``.
        description: Human-readable description of the vulnerability.
        line_ref: Source-level reference, or empty string if unknown.
        cwe_id: CWE identifier (e.g. ``'CWE-89'``), or empty string if not mapped.
    """

    severity: str
    description: str
    line_ref: str
    cwe_id: str


class StyleFindingOut(BaseModel):
    """Serialised style finding.

    Attributes:
        category: ``'complexity'``, ``'naming'``, ``'documentation'``,
            ``'formatting'``, ``'readability'``, or ``'other'``.
        description: Human-readable description of the style issue.
        line_ref: Source-level reference, or empty string if unknown.
    """

    category: str
    description: str
    line_ref: str


class ReviewResponse(BaseModel):
    """Serialised review report returned by both endpoints.

    Attributes:
        pr_id: Identifier of the reviewed PR.
        title: PR title.
        overall_risk: Highest risk across all security findings.
        security_findings: Structured security findings list.
        style_findings: Structured style findings list.
        markdown: Full markdown-formatted review body.
    """

    pr_id: str
    title: str
    overall_risk: str
    security_findings: list[SecurityFindingOut]
    style_findings: list[StyleFindingOut]
    markdown: str


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_pipeline(request: Request) -> ReviewPipeline:
    """Inject the shared :class:`~chronoagent.pipeline.graph.ReviewPipeline`.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The :class:`ReviewPipeline` stored on ``app.state.pipeline``.
    """
    return request.app.state.pipeline  # type: ignore[no-any-return]


def _get_review_store(request: Request) -> dict[str, ReviewReport]:
    """Inject the in-memory review store.

    Args:
        request: Incoming FastAPI request (injected automatically).

    Returns:
        The ``dict[str, ReviewReport]`` stored on ``app.state.review_store``.
    """
    return request.app.state.review_store  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Helper: dataclass → Pydantic response model
# ---------------------------------------------------------------------------


def _to_response(report: ReviewReport) -> ReviewResponse:
    """Convert a :class:`~chronoagent.agents.summarizer.ReviewReport` to a JSON-safe response.

    Args:
        report: The dataclass produced by the pipeline.

    Returns:
        :class:`ReviewResponse` Pydantic model ready for serialisation.
    """
    return ReviewResponse(
        pr_id=report.pr_id,
        title=report.title,
        overall_risk=report.overall_risk,
        security_findings=[
            SecurityFindingOut(
                severity=f.severity,
                description=f.description,
                line_ref=f.line_ref,
                cwe_id=f.cwe_id,
            )
            for f in report.security_findings
        ],
        style_findings=[
            StyleFindingOut(
                category=f.category,
                description=f.description,
                line_ref=f.line_ref,
            )
            for f in report.style_findings
        ],
        markdown=report.markdown,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/review", response_model=ReviewResponse, status_code=201)
def submit_review(
    body: PRSubmitRequest,
    pipeline: Annotated[ReviewPipeline, Depends(_get_pipeline)],
    store: Annotated[dict[str, ReviewReport], Depends(_get_review_store)],
) -> ReviewResponse:
    """Submit a PR for review and return the generated report.

    Runs the full LangGraph pipeline (plan → security ∥ style → summarize) and
    persists the result so it can be retrieved via ``GET /api/v1/review/{review_id}``.

    Args:
        body: PR payload containing the diff and metadata.
        pipeline: Shared :class:`~chronoagent.pipeline.graph.ReviewPipeline` (injected).
        store: In-memory review store keyed by *pr_id* (injected).

    Returns:
        :class:`ReviewResponse` with findings and the markdown report.
    """
    log = logger.bind(pr_id=body.pr_id)
    log.info("review.submit")

    pr = SyntheticPR(
        pr_id=body.pr_id,
        title=body.title,
        description=body.description,
        diff=body.diff,
        files_changed=body.files_changed,
    )

    report = pipeline.run(pr)
    store[body.pr_id] = report

    log.info("review.complete", overall_risk=report.overall_risk)
    return _to_response(report)


@router.get("/review/{review_id}", response_model=ReviewResponse)
def get_review(
    review_id: str,
    store: Annotated[dict[str, ReviewReport], Depends(_get_review_store)],
) -> ReviewResponse:
    """Retrieve a previously computed review report by PR id.

    Args:
        review_id: The ``pr_id`` used when submitting the review.
        store: In-memory review store (injected).

    Returns:
        :class:`ReviewResponse` for the given *review_id*.

    Raises:
        :class:`~fastapi.HTTPException` 404 if no review exists for *review_id*.
    """
    report = store.get(review_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found")
    return _to_response(report)
