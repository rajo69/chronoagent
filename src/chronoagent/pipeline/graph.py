"""LangGraph StateGraph wiring the 4-agent code-review pipeline.

Topology
--------
The graph implements a fan-out / fan-in pattern::

    START
      |
    plan
     / \\
    /   \\
  sec   style   (parallel)
    \\   /
     \\ /
   summarize
      |
     END

``plan`` decomposes the incoming :class:`~chronoagent.agents.security_reviewer.SyntheticPR`
into :class:`~chronoagent.agents.planner.ReviewSubtask` objects.  ``security_review`` and
``style_review`` run **concurrently** on the same PR.  ``summarize`` waits for both and
synthesises the final :class:`~chronoagent.agents.summarizer.ReviewReport`.

Usage
-----
::

    pipeline = ReviewPipeline.create()
    report = pipeline.run(pr)
"""

from __future__ import annotations

import structlog
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from chronoagent.agents.planner import DecompositionResult, PlannerAgent
from chronoagent.agents.security_reviewer import (
    SecurityReview,
    SecurityReviewerAgent,
    SyntheticPR,
)
from chronoagent.agents.style_reviewer import StyleReview, StyleReviewerAgent
from chronoagent.agents.summarizer import ReviewReport, SummarizerAgent

logger: structlog.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """Mutable state carried through every node in the review pipeline.

    All fields except *pr* are populated progressively as nodes execute.

    Attributes:
        pr: The pull request under review (required at graph entry).
        decomposition: Output of the :class:`~chronoagent.agents.planner.PlannerAgent`.
        security_review: Output of the
            :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent`.
        style_review: Output of the
            :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent`.
        report: Final synthesised :class:`~chronoagent.agents.summarizer.ReviewReport`.
    """

    pr: SyntheticPR
    decomposition: DecompositionResult
    security_review: SecurityReview
    style_review: StyleReview
    report: ReviewReport


# ---------------------------------------------------------------------------
# Node factory helpers
# ---------------------------------------------------------------------------


def _make_plan_node(
    agent: PlannerAgent,
) -> Any:
    """Return a LangGraph node function that runs the planner.

    Args:
        agent: A fully constructed :class:`~chronoagent.agents.planner.PlannerAgent`.

    Returns:
        Callable ``(state) -> partial PipelineState`` for use as a graph node.
    """

    def plan_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        log = logger.bind(node="plan", pr_id=pr.pr_id)
        log.info("plan_node.start")
        decomposition = agent.decompose(pr)
        log.info(
            "plan_node.done",
            subtasks=len(decomposition.subtasks),
            llm_latency_ms=round(decomposition.llm_latency_ms, 1),
        )
        return {"decomposition": decomposition}

    return plan_node


def _make_security_review_node(
    agent: SecurityReviewerAgent,
) -> Any:
    """Return a LangGraph node function that runs the security reviewer.

    Args:
        agent: A fully constructed :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent`.

    Returns:
        Callable ``(state) -> partial PipelineState``.
    """

    def security_review_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        log = logger.bind(node="security_review", pr_id=pr.pr_id)
        log.info("security_review_node.start")
        review = agent.review(pr)
        log.info(
            "security_review_node.done",
            findings=len(review.findings),
            severity=review.severity,
            llm_latency_ms=round(review.llm_latency_ms, 1),
        )
        return {"security_review": review}

    return security_review_node


def _make_style_review_node(
    agent: StyleReviewerAgent,
) -> Any:
    """Return a LangGraph node function that runs the style reviewer.

    Args:
        agent: A fully constructed :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent`.

    Returns:
        Callable ``(state) -> partial PipelineState``.
    """

    def style_review_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        log = logger.bind(node="style_review", pr_id=pr.pr_id)
        log.info("style_review_node.start")
        review = agent.review(pr)
        log.info(
            "style_review_node.done",
            findings=len(review.findings),
            recommendation=review.recommendation,
            llm_latency_ms=round(review.llm_latency_ms, 1),
        )
        return {"style_review": review}

    return style_review_node


def _make_summarize_node(
    agent: SummarizerAgent,
) -> Any:
    """Return a LangGraph node function that runs the summarizer.

    Args:
        agent: A fully constructed :class:`~chronoagent.agents.summarizer.SummarizerAgent`.

    Returns:
        Callable ``(state) -> partial PipelineState``.
    """

    def summarize_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        security_review: SecurityReview = state["security_review"]
        style_review: StyleReview = state["style_review"]
        log = logger.bind(node="summarize", pr_id=pr.pr_id)
        log.info("summarize_node.start")
        report = agent.synthesize(pr, security_review, style_review)
        log.info(
            "summarize_node.done",
            overall_risk=report.overall_risk,
            llm_latency_ms=round(report.llm_latency_ms, 1),
        )
        return {"report": report}

    return summarize_node


# ---------------------------------------------------------------------------
# ReviewPipeline
# ---------------------------------------------------------------------------


class ReviewPipeline:
    """Compiled LangGraph pipeline: plan → (security_review ∥ style_review) → summarize.

    Args:
        planner: :class:`~chronoagent.agents.planner.PlannerAgent` instance.
        security_reviewer: :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent` instance.
        style_reviewer: :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent` instance.
        summarizer: :class:`~chronoagent.agents.summarizer.SummarizerAgent` instance.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        security_reviewer: SecurityReviewerAgent,
        style_reviewer: StyleReviewerAgent,
        summarizer: SummarizerAgent,
    ) -> None:
        self._planner = planner
        self._security_reviewer = security_reviewer
        self._style_reviewer = style_reviewer
        self._summarizer = summarizer
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph ``StateGraph``.

        Returns:
            Compiled LangGraph runnable.
        """
        builder: StateGraph = StateGraph(PipelineState)

        # Register nodes
        builder.add_node("plan", _make_plan_node(self._planner))
        builder.add_node("security_review", _make_security_review_node(self._security_reviewer))
        builder.add_node("style_review", _make_style_review_node(self._style_reviewer))
        builder.add_node("summarize", _make_summarize_node(self._summarizer))

        # Edges — fan-out from plan, fan-in to summarize
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "security_review")
        builder.add_edge("plan", "style_review")
        builder.add_edge("security_review", "summarize")
        builder.add_edge("style_review", "summarize")
        builder.add_edge("summarize", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, pr: SyntheticPR) -> ReviewReport:
        """Execute the full pipeline for a single pull request.

        Args:
            pr: The :class:`~chronoagent.agents.security_reviewer.SyntheticPR` to review.

        Returns:
            :class:`~chronoagent.agents.summarizer.ReviewReport` with security findings,
            style findings, overall risk, and a markdown report body.
        """
        initial_state: PipelineState = {"pr": pr}  # type: ignore[typeddict-item]
        final_state: PipelineState = self._graph.invoke(initial_state)
        return final_state["report"]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        seed: int = 42,
        chroma_client: ClientAPI | None = None,
    ) -> ReviewPipeline:
        """Factory method that builds all 4 agents with :class:`~chronoagent.agents.backends.mock.MockBackend`.

        All agents share the same ephemeral ChromaDB client so their collections
        are isolated by name but backed by the same in-memory store.

        Args:
            seed: Seed forwarded to every :class:`~chronoagent.agents.backends.mock.MockBackend`
                instance for deterministic responses.
            chroma_client: Optional ChromaDB client; an ephemeral in-memory client is
                created if ``None``.

        Returns:
            Fully constructed :class:`ReviewPipeline` ready to call :meth:`run`.
        """
        client: ClientAPI = chroma_client or chromadb.EphemeralClient()
        planner = PlannerAgent.create(seed=seed, chroma_client=client)
        security_reviewer = SecurityReviewerAgent.create(seed=seed, chroma_client=client)
        style_reviewer = StyleReviewerAgent.create(seed=seed, chroma_client=client)
        summarizer = SummarizerAgent.create(seed=seed, chroma_client=client)
        return cls(
            planner=planner,
            security_reviewer=security_reviewer,
            style_reviewer=style_reviewer,
            summarizer=summarizer,
        )
