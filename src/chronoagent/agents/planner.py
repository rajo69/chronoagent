"""PlannerAgent: decomposes a PR diff into a list of ReviewSubtask objects.

The planner retrieves similar past decompositions from ChromaDB, augments the
prompt with that context, and asks the LLM to produce a structured task plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chromadb
from chromadb.api import ClientAPI

from chronoagent.agents.backends.mock import MockBackend, MockBackendVariant
from chronoagent.agents.base import BaseAgent, RetrievalResult, Task, TaskResult
from chronoagent.agents.security_reviewer import SyntheticPR

# ---------------------------------------------------------------------------
# Past-decomposition knowledge base used to seed the planner's ChromaDB
# collection.  Each entry is a narrative description of a prior planning
# decision, giving the LLM useful precedents.
# ---------------------------------------------------------------------------

_PLANNER_KNOWLEDGE_BASE: list[str] = [
    "Authentication PRs should always include both a security_review subtask for "
    "the login/token logic and a style_review subtask for the surrounding helpers.",
    "File-upload endpoints are high-risk: create a security_review subtask for the "
    "upload handler and a separate security_review subtask for path-handling code.",
    "Database migration PRs require a security_review subtask for any raw-SQL "
    "additions and a style_review subtask for the migration script itself.",
    "Dependency-bump PRs need a security_review subtask scanning for new CVEs and a "
    "style_review subtask checking that pinned versions follow project conventions.",
    "JWT or session-management changes warrant two security_review subtasks: one for "
    "token generation and one for token validation.",
    "API-endpoint additions should produce a security_review subtask for input "
    "validation and a style_review subtask for naming and HTTP-verb conventions.",
    "Configuration-file changes (env vars, secrets) need a security_review subtask "
    "to check for hardcoded secrets.",
    "Middleware additions require a security_review subtask for request interception "
    "logic and a style_review subtask for error-handling patterns.",
    "Refactoring PRs with no new logic still need a style_review subtask to verify "
    "that the restructured code meets project conventions.",
    "PRs touching logging or observability code need a security_review subtask to "
    "ensure no PII or secrets leak into log output.",
]


@dataclass
class ReviewSubtask:
    """A single unit of review work produced by the planner.

    Attributes:
        subtask_id: Unique identifier for this subtask (e.g. ``"s1"``).
        task_type: Review category: ``"security_review"`` or ``"style_review"``.
        code_segment: Human-readable description of the code area to review.
    """

    subtask_id: str
    task_type: str
    code_segment: str


@dataclass
class DecompositionResult:
    """Output of :meth:`PlannerAgent.decompose`.

    Attributes:
        pr_id: Identifier of the PR that was decomposed.
        subtasks: Ordered list of :class:`ReviewSubtask` objects.
        rationale: Planner's brief rationale for the chosen decomposition.
        retrieved_docs: Number of ChromaDB documents retrieved.
        retrieval_distances: Cosine distances of retrieved docs (lower = closer).
        retrieval_latency_ms: Time taken for the ChromaDB query.
        llm_latency_ms: Time taken for the LLM call.
        raw_response: Raw LLM output for debugging.
    """

    pr_id: str
    subtasks: list[ReviewSubtask] = field(default_factory=list)
    rationale: str = ""
    retrieved_docs: int = 0
    retrieval_distances: list[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    raw_response: str = ""


def _parse_decomposition(
    raw: str,
    pr_id: str,
    retrieval: RetrievalResult,
) -> DecompositionResult:
    """Parse raw LLM output into a :class:`DecompositionResult`.

    Expected format produced by the MockBackend (and requested from real LLMs)::

        PLAN:
        1. subtask_type=security_review code_segment=<description>
        2. subtask_type=style_review code_segment=<description>
        RATIONALE: <single-line rationale>

    Lines that do not match either pattern are silently skipped so the parser
    degrades gracefully on partial LLM output.

    Args:
        raw: Raw LLM response text.
        pr_id: PR identifier.
        retrieval: ChromaDB retrieval result for this step.

    Returns:
        Parsed :class:`DecompositionResult`.
    """
    subtasks: list[ReviewSubtask] = []
    rationale = ""
    counter = 0

    for line in raw.splitlines():
        stripped = line.strip()

        # Numbered subtask lines: "1. subtask_type=X code_segment=Y"
        if stripped and stripped[0].isdigit() and "subtask_type=" in stripped:
            counter += 1
            task_type = ""
            code_segment = ""

            # Parse key=value pairs that may appear anywhere in the line.
            for token in stripped.split():
                if token.startswith("subtask_type="):
                    task_type = token.removeprefix("subtask_type=")
                elif token.startswith("code_segment="):
                    # code_segment value may be multi-word — collect the rest.
                    idx = stripped.index("code_segment=")
                    code_segment = stripped[idx:].removeprefix("code_segment=").strip()
                    break

            subtasks.append(
                ReviewSubtask(
                    subtask_id=f"s{counter}",
                    task_type=task_type,
                    code_segment=code_segment,
                )
            )

        elif stripped.startswith("RATIONALE:"):
            rationale = stripped.removeprefix("RATIONALE:").strip()

    return DecompositionResult(
        pr_id=pr_id,
        subtasks=subtasks,
        rationale=rationale,
        retrieved_docs=len(retrieval.documents),
        retrieval_distances=retrieval.distances,
        retrieval_latency_ms=retrieval.latency_ms,
        llm_latency_ms=0.0,  # filled in by the caller
        raw_response=raw,
    )


class PlannerAgent(BaseAgent):
    """Agent that decomposes a PR diff into an ordered list of review subtasks.

    Retrieves similar past decompositions from a ChromaDB knowledge base,
    augments the prompt with that context, and calls the LLM to produce a
    structured :class:`DecompositionResult`.

    Args:
        agent_id: Unique identifier for this agent instance.
        backend: :class:`~chronoagent.agents.backends.base.LLMBackend` for
            generation and embeddings.
        collection: ChromaDB collection containing past decomposition examples.
        top_k: Number of past decompositions to retrieve per PR.
    """

    SYSTEM_PROMPT = (
        "You are a senior code-review planner. "
        "Given a pull request diff, decompose it into an ordered list of review subtasks. "
        "Each subtask must specify a task_type (security_review or style_review) and a "
        "code_segment (short description of the code area). "
        "Use the provided past decomposition examples to guide your decisions. "
        "Format your response as:\n"
        "PLAN:\n"
        "1. subtask_type=<type> code_segment=<description>\n"
        "2. subtask_type=<type> code_segment=<description>\n"
        "...\n"
        "RATIONALE: <one sentence explaining your choices>"
    )

    def decompose(self, pr: SyntheticPR) -> DecompositionResult:
        """Decompose a PR into an ordered list of :class:`ReviewSubtask` objects.

        Args:
            pr: The synthetic pull request to plan.

        Returns:
            :class:`DecompositionResult` with subtasks, rationale, and
            retrieval + LLM latency metadata.
        """
        query = f"{pr.title} {pr.description} {pr.diff[:200]}"
        retrieval = self._retrieve_memory(query)

        context_block = "\n".join(
            f"- {doc}" for doc in retrieval.documents
        ) or "No relevant past decompositions retrieved."

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"PAST DECOMPOSITION EXAMPLES:\n{context_block}\n\n"
            f"PR TITLE: {pr.title}\n"
            f"PR DESCRIPTION: {pr.description}\n"
            f"FILES CHANGED: {', '.join(pr.files_changed) or 'unknown'}\n"
            f"DIFF EXCERPT:\n{pr.diff[:500]}\n\n"
            "Provide your decomposition plan:"
        )

        raw_response, llm_latency_ms = self._call_llm(prompt)

        result = _parse_decomposition(raw_response, pr.pr_id, retrieval)
        result.llm_latency_ms = llm_latency_ms
        return result

    def execute(self, task: Task) -> TaskResult:
        """Execute a planning task.

        Args:
            task: Task with ``payload["pr"]`` set to a :class:`SyntheticPR`.

        Returns:
            :class:`TaskResult` with ``output["decomposition"]`` containing the
            :class:`DecompositionResult`.
        """
        pr: SyntheticPR = task.payload["pr"]
        decomposition = self.decompose(pr)
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            output={"decomposition": decomposition},
            llm_latency_ms=decomposition.llm_latency_ms,
            retrieval_latency_ms=decomposition.retrieval_latency_ms,
            timestamp=time.time(),
        )

    @classmethod
    def create(
        cls,
        agent_id: str = "planner",
        seed: int = 42,
        top_k: int = 3,
        chroma_client: ClientAPI | None = None,
    ) -> PlannerAgent:
        """Factory method creating a ready-to-use agent with MockBackend.

        Args:
            agent_id: Unique identifier for this agent instance.
            seed: Seed for the :class:`MockBackend` response selection.
            top_k: Number of past decompositions to retrieve per PR.
            chroma_client: ChromaDB client; ephemeral in-memory client used if None.

        Returns:
            Configured :class:`PlannerAgent`.
        """
        client = chroma_client or chromadb.EphemeralClient()
        backend = MockBackend(seed=seed, variant=MockBackendVariant.PLANNER)
        collection = cls.build_collection(
            client,
            name=f"{agent_id}_planner_kb",
            documents=_PLANNER_KNOWLEDGE_BASE,
            backend=backend,
        )
        return cls(agent_id=agent_id, backend=backend, collection=collection, top_k=top_k)
