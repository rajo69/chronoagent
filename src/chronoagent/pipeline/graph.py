"""LangGraph StateGraph wiring the 4-agent code-review pipeline.

Topology
--------
The graph implements a fan-out / allocator-gated / fan-in pattern::

    START
      |
    plan
     / \\
    /   \\
 allocate_security     allocate_style       (parallel)
    |  \\                 |  \\
    |   \\                |   \\
  sec   escalate_sec   style  escalate_sty   (conditional)
    \\   /               \\   /
     \\ /                 \\ /
          \\      |      /
           \\     |     /
              summarize
                  |
                 END

``plan`` decomposes the incoming
:class:`~chronoagent.agents.security_reviewer.SyntheticPR` into
:class:`~chronoagent.agents.planner.ReviewSubtask` objects.  Each
review dimension then passes through an *allocation node* that asks
:class:`~chronoagent.allocator.task_allocator.DecentralizedTaskAllocator`
to run a contract-net round for the ``(pr, task_type)`` pair.  A
conditional edge inspects the :class:`NegotiationResult`:

* if the winning agent is the expected specialist, the task is routed
  to the specialist node (``security_review`` / ``style_review``);
* otherwise (non-specialist winner or escalation), the task is routed
  to the matching ``escalate_*`` node, which emits a placeholder
  review so the summarizer can still produce a report.

When :class:`ReviewPipeline` is constructed without an explicit
allocator, a default one is built on a fresh in-process
:class:`~chronoagent.messaging.local_bus.LocalBus`.  With no health
updates in the cache, ``missing_health_default=1.0`` makes every bid
1.0, so specialists always win and the pipeline behaves identically to
the pre-5.4 hard-wired topology.  Supplying an allocator whose cache
already has low health scores activates the health gate.

Usage
-----
::

    pipeline = ReviewPipeline.create()
    report = pipeline.run(pr)
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api import ClientAPI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from chronoagent.agents.planner import DecompositionResult, PlannerAgent
from chronoagent.agents.security_reviewer import (
    SecurityReview,
    SecurityReviewerAgent,
    SyntheticPR,
)
from chronoagent.agents.style_reviewer import StyleReview, StyleReviewerAgent
from chronoagent.agents.summarizer import ReviewReport, SummarizerAgent
from chronoagent.allocator.negotiation import NegotiationResult
from chronoagent.allocator.task_allocator import DecentralizedTaskAllocator
from chronoagent.messaging.local_bus import LocalBus
from chronoagent.observability.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """Mutable state carried through every node in the review pipeline.

    All fields except *pr* are populated progressively as nodes execute.

    Attributes:
        pr: The pull request under review (required at graph entry).
        decomposition: Output of the :class:`~chronoagent.agents.planner.PlannerAgent`.
        security_allocation: :class:`NegotiationResult` from the allocator's
            contract-net round for the security_review task.  Written by
            the ``allocate_security`` node and consumed by the conditional
            edge that routes to either ``security_review`` or
            ``escalate_security``.  Also consumed by the Phase 5.5 audit
            layer.
        style_allocation: :class:`NegotiationResult` for the style_review
            task, analogous to *security_allocation*.
        security_review: Output of the
            :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent`
            (or a placeholder emitted by ``escalate_security``).
        style_review: Output of the
            :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent`
            (or a placeholder emitted by ``escalate_style``).
        report: Final synthesised :class:`~chronoagent.agents.summarizer.ReviewReport`.
    """

    pr: SyntheticPR
    decomposition: DecompositionResult
    security_allocation: NegotiationResult
    style_allocation: NegotiationResult
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
        agent: A fully constructed
            :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent`.

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


def _make_allocate_node(
    allocator: DecentralizedTaskAllocator,
    task_type: str,
    state_key: str,
) -> Any:
    """Return a LangGraph node that runs one contract-net round.

    The node reads ``state["pr"]``, builds the task id
    ``f"{pr_id}::{task_type}"``, calls
    :meth:`DecentralizedTaskAllocator.allocate`, and writes the
    :class:`NegotiationResult` into *state_key*.  No other state is
    touched, so the two allocation branches (security + style) can run
    in parallel without merge conflicts on a shared key.

    Args:
        allocator: Shared allocator instance owned by the pipeline.
        task_type: One of
            :data:`chronoagent.allocator.capability_weights.TASK_TYPES`.
            In practice this is ``"security_review"`` or
            ``"style_review"`` when called from :class:`ReviewPipeline`.
        state_key: Name of the :class:`PipelineState` field to write.

    Returns:
        Callable ``(state) -> {state_key: NegotiationResult}``.
    """

    def allocate_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        task_id = f"{pr.pr_id}::{task_type}"
        log = logger.bind(node=f"allocate_{task_type}", pr_id=pr.pr_id, task_id=task_id)
        log.info("allocate_node.start")
        result = allocator.allocate(task_id=task_id, task_type=task_type)
        log.info(
            "allocate_node.done",
            assigned_agent=result.assigned_agent,
            escalated=result.escalated,
            winning_score=(result.winning_bid.score if result.winning_bid is not None else None),
            threshold=result.threshold,
        )
        return {state_key: result}

    return allocate_node


def _placeholder_security_review(pr: SyntheticPR, rationale: str) -> SecurityReview:
    """Build a zero-finding :class:`SecurityReview` for the escalation path.

    The Phase 5 4-agent pipeline has specialized, non-swappable reviewer
    interfaces, so a non-specialist winner cannot actually run the task.
    When the allocator declines to assign the specialist, we emit a
    placeholder review so the summarizer can still produce a report.
    The ``raw_response`` carries the negotiation rationale so the audit
    layer (Phase 5.5) can trace *why* the task was escalated.

    Args:
        pr: The PR under review (only ``pr_id`` is used).
        rationale: Human-readable explanation copied from the
            :class:`NegotiationResult`.

    Returns:
        A :class:`SecurityReview` with empty findings, severity
        ``"none"``, a recommendation that explains the escalation, and
        zero latencies.
    """
    return SecurityReview(
        pr_id=pr.pr_id,
        findings=[],
        severity="none",
        recommendation=f"escalated: {rationale}",
        retrieved_docs=0,
        retrieval_distances=[],
        retrieval_latency_ms=0.0,
        llm_latency_ms=0.0,
        raw_response=rationale,
    )


def _placeholder_style_review(pr: SyntheticPR, rationale: str) -> StyleReview:
    """Build a zero-finding :class:`StyleReview` for the escalation path.

    Companion to :func:`_placeholder_security_review`; see its docstring
    for the motivation and field semantics.
    """
    return StyleReview(
        pr_id=pr.pr_id,
        findings=[],
        recommendation=f"escalated: {rationale}",
        retrieved_docs=0,
        retrieval_distances=[],
        retrieval_latency_ms=0.0,
        llm_latency_ms=0.0,
        raw_response=rationale,
    )


def _make_escalate_security_node() -> Any:
    """Return a node that emits a placeholder security review.

    Triggered when the allocator escalates the ``security_review`` task
    or assigns it to a non-specialist.  The node is a pure function of
    ``state["pr"]`` and ``state["security_allocation"]``, so it has no
    LLM or ChromaDB side effects and will never raise in normal
    operation.
    """

    def escalate_security_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        allocation = state["security_allocation"]
        log = logger.bind(
            node="escalate_security",
            pr_id=pr.pr_id,
            assigned_agent=allocation.assigned_agent,
            escalated=allocation.escalated,
        )
        log.warning("escalate_security_node.emit_placeholder", rationale=allocation.rationale)
        return {"security_review": _placeholder_security_review(pr, allocation.rationale)}

    return escalate_security_node


def _make_escalate_style_node() -> Any:
    """Return a node that emits a placeholder style review.

    Triggered when the allocator escalates the ``style_review`` task or
    assigns it to a non-specialist.  See
    :func:`_make_escalate_security_node` for the rationale.
    """

    def escalate_style_node(state: PipelineState) -> dict[str, Any]:
        pr = state["pr"]
        allocation = state["style_allocation"]
        log = logger.bind(
            node="escalate_style",
            pr_id=pr.pr_id,
            assigned_agent=allocation.assigned_agent,
            escalated=allocation.escalated,
        )
        log.warning("escalate_style_node.emit_placeholder", rationale=allocation.rationale)
        return {"style_review": _placeholder_style_review(pr, allocation.rationale)}

    return escalate_style_node


def _route_security(state: PipelineState) -> str:
    """Conditional-edge function for the security allocation branch.

    Returns the name of the next LangGraph node based on the allocator's
    decision.  The specialist branch fires only when the winning bid
    belongs to ``"security_reviewer"``; any other outcome (escalation,
    or a non-specialist winner whose interface cannot run a security
    review) drops into the escalation branch.
    """
    allocation = state["security_allocation"]
    if not allocation.escalated and allocation.assigned_agent == "security_reviewer":
        return "security_review"
    return "escalate_security"


def _route_style(state: PipelineState) -> str:
    """Conditional-edge function for the style allocation branch.

    Symmetrical to :func:`_route_security`; the specialist branch fires
    only when the winning bid belongs to ``"style_reviewer"``.
    """
    allocation = state["style_allocation"]
    if not allocation.escalated and allocation.assigned_agent == "style_reviewer":
        return "style_review"
    return "escalate_style"


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
    """Compiled LangGraph pipeline with an allocator-gated review stage.

    Topology: ``plan → (allocate_security ∥ allocate_style) → {specialist |
    escalate} → summarize``.  See the module docstring for the full
    diagram.

    Args:
        planner: :class:`~chronoagent.agents.planner.PlannerAgent` instance.
        security_reviewer:
            :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent` instance.
        style_reviewer: :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent` instance.
        summarizer: :class:`~chronoagent.agents.summarizer.SummarizerAgent` instance.
        allocator: Optional
            :class:`~chronoagent.allocator.task_allocator.DecentralizedTaskAllocator`
            used to gate review routing.  When ``None``, a default
            allocator is constructed on a fresh in-process
            :class:`~chronoagent.messaging.local_bus.LocalBus`; with an
            empty health cache and ``missing_health_default=1.0`` every
            bid is 1.0 and specialists always win, so the behaviour is
            byte-for-byte identical to the pre-5.4 hard-wired topology.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        security_reviewer: SecurityReviewerAgent,
        style_reviewer: StyleReviewerAgent,
        summarizer: SummarizerAgent,
        allocator: DecentralizedTaskAllocator | None = None,
    ) -> None:
        self._planner = planner
        self._security_reviewer = security_reviewer
        self._style_reviewer = style_reviewer
        self._summarizer = summarizer
        self._allocator = (
            allocator if allocator is not None else DecentralizedTaskAllocator(LocalBus())
        )
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph ``StateGraph``.

        Returns:
            Compiled LangGraph runnable.
        """
        builder: StateGraph[PipelineState] = StateGraph(PipelineState)

        # Register nodes
        builder.add_node("plan", _make_plan_node(self._planner))
        builder.add_node(
            "allocate_security",
            _make_allocate_node(
                self._allocator,
                task_type="security_review",
                state_key="security_allocation",
            ),
        )
        builder.add_node(
            "allocate_style",
            _make_allocate_node(
                self._allocator,
                task_type="style_review",
                state_key="style_allocation",
            ),
        )
        builder.add_node("security_review", _make_security_review_node(self._security_reviewer))
        builder.add_node("style_review", _make_style_review_node(self._style_reviewer))
        builder.add_node("escalate_security", _make_escalate_security_node())
        builder.add_node("escalate_style", _make_escalate_style_node())
        builder.add_node("summarize", _make_summarize_node(self._summarizer))

        # Edges — fan-out from plan, conditional gate, fan-in to summarize
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "allocate_security")
        builder.add_edge("plan", "allocate_style")
        builder.add_conditional_edges(
            "allocate_security",
            _route_security,
            {
                "security_review": "security_review",
                "escalate_security": "escalate_security",
            },
        )
        builder.add_conditional_edges(
            "allocate_style",
            _route_style,
            {
                "style_review": "style_review",
                "escalate_style": "escalate_style",
            },
        )
        builder.add_edge("security_review", "summarize")
        builder.add_edge("escalate_security", "summarize")
        builder.add_edge("style_review", "summarize")
        builder.add_edge("escalate_style", "summarize")
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
        initial_state: PipelineState = {"pr": pr}
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
        allocator: DecentralizedTaskAllocator | None = None,
    ) -> ReviewPipeline:
        """Factory method that builds all 4 agents with
        :class:`~chronoagent.agents.backends.mock.MockBackend`.

        All agents share the same ephemeral ChromaDB client so their collections
        are isolated by name but backed by the same in-memory store.

        Args:
            seed: Seed forwarded to every :class:`~chronoagent.agents.backends.mock.MockBackend`
                instance for deterministic responses.
            chroma_client: Optional ChromaDB client; an ephemeral in-memory client is
                created if ``None``.
            allocator: Optional
                :class:`~chronoagent.allocator.task_allocator.DecentralizedTaskAllocator`
                forwarded to :class:`ReviewPipeline`; see the class docstring
                for the default-allocator behaviour.

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
            allocator=allocator,
        )
