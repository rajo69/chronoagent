"""StyleReviewerAgent: reviews PRs for code quality and style issues.

Retrieves relevant style conventions from a ChromaDB knowledge base,
constructs a prompt, and returns a structured style review with per-finding
category, description, and optional line reference.  Designed to run against
MockBackend for deterministic, cost-free experiments.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI

from chronoagent.agents.backends.mock import MockBackend, MockBackendVariant
from chronoagent.agents.base import BaseAgent, RetrievalResult, Task, TaskResult
from chronoagent.agents.security_reviewer import SyntheticPR

# ---------------------------------------------------------------------------
# Style conventions knowledge base pre-populated into the style ChromaDB
# collection.  Each entry documents a convention with rationale and detection
# heuristics so the LLM can reference them when reviewing diffs.
# ---------------------------------------------------------------------------

_STYLE_KNOWLEDGE_BASE: list[str] = [
    "Function length: Keep functions under 50 lines. Longer functions obscure intent "
    "and are hard to test in isolation. "
    "Detection: count non-blank, non-comment lines per function; flag >50.",
    "Naming — variables: Use descriptive snake_case names. Single-letter names (`x`, `tmp`) "
    "are only acceptable as loop indices or mathematical variables. "
    "Detection: variable names shorter than 3 characters in non-loop context.",
    "Naming — functions: Verb-noun naming for functions (`calculate_total`, `validate_user`). "
    "Avoid vague names like `do_stuff`, `process`, `handle`. "
    "Detection: function names without a verb prefix.",
    "Naming — consistency: Use one convention per module. Do not mix camelCase and snake_case. "
    "Classes use PascalCase; everything else uses snake_case in Python. "
    "Detection: camelCase identifiers outside class definitions in Python files.",
    "Cyclomatic complexity: Keep cyclomatic complexity below 10 per function. "
    "High complexity correlates with defect density and reduces testability. "
    "Detection: count decision points (if/elif/for/while/except/and/or per function) + 1.",
    "Nesting depth: Limit nesting to 3 levels. Deep nesting signals missing abstractions. "
    "Use early-return (guard clauses) and helper functions to flatten control flow. "
    "Detection: count indent levels; flag blocks nested >3 deep.",
    "Magic numbers: Replace numeric literals with named constants or enums. "
    "Magic numbers make code opaque and fragile under change. "
    "Detection: numeric literals other than 0, 1, -1 outside test files.",
    "Docstrings: Every public function, class, and module must have a docstring. "
    "One-line summary + Args + Returns for functions with parameters. "
    "Detection: `def` or `class` statement not followed by a string literal.",
    "TODO comments: Do not merge code with TODO/FIXME/HACK comments unless tracked in an issue. "
    "Unresolved TODOs indicate incomplete work. "
    "Detection: grep for TODO, FIXME, HACK, XXX, NOCOMMIT in changed lines.",
    "Dead code: Remove commented-out code blocks before merging. "
    "Use version control history instead of comment tombstones. "
    "Detection: blocks of consecutive commented lines (>3) that look like code.",
    "Import order: Follow isort conventions — stdlib, third-party, local, each group separated "
    "by a blank line. Use `from __future__ import annotations` at the top of every file. "
    "Detection: import groups not separated by blank lines; local before stdlib.",
    "Line length: Lines must not exceed 100 characters. Long lines reduce readability "
    "in side-by-side diff views and on narrow terminals. "
    "Detection: any line >100 characters in non-auto-generated files.",
    "Trailing whitespace: Remove trailing spaces and tabs. They pollute diffs and confuse "
    "some editors. Detection: lines ending with `\\s+$`.",
    "Type annotations: All public functions must have full type annotations on parameters "
    "and return types. Use `Optional[X]` or `X | None` for nullable values. "
    "Detection: function signatures missing annotations; missing return type.",
    "Exception handling: Avoid bare `except:` clauses. Catch specific exceptions and "
    "log or re-raise. Swallowing exceptions silently hides bugs. "
    "Detection: `except:` or `except Exception:` without a logged message or re-raise.",
    "Test coverage: Every new function should have a corresponding unit test. "
    "Added logic paths without tests are a coverage regression. "
    "Detection: new `def` statements in non-test files without matching test file changes.",
    "Constants: Module-level constants should be ALL_CAPS. "
    "Lowercase module-level names that are never reassigned should be constants. "
    "Detection: module-level assignments to lowercase names that are never mutated.",
    "String formatting: Use f-strings (Python ≥3.6) rather than `%` or `.format()`. "
    "F-strings are more readable and slightly faster. "
    "Detection: `%s` formatting or `.format(` calls in new code.",
    "Return types: Functions must not have multiple implicit `None` returns mixed with "
    "value returns. Use explicit `return None` or restructure with a single exit point. "
    "Detection: function body with both `return <value>` and bare `return` or fall-through.",
    "Dataclass vs dict: Use dataclasses or Pydantic models instead of plain dicts for "
    "structured data passed between functions. Typed structures are self-documenting. "
    "Detection: functions that return `dict[str, Any]` where a named type would fit.",
    "Global state: Avoid mutable module-level globals. Use dependency injection or "
    "configuration objects instead. "
    "Detection: module-level `list`, `dict`, or custom objects mutated in function bodies.",
    "Logging: Use structured logging (`structlog` or `logging`) instead of `print`. "
    "Print statements do not respect log levels or output routing. "
    "Detection: bare `print(` calls in non-script, non-test code.",
    "Assertions: Do not use `assert` for runtime validation in production code. "
    "Assertions are removed with `-O` and should be reserved for invariants in tests. "
    "Detection: `assert` statements outside test files.",
    "Comprehension clarity: List/dict/set comprehensions should fit on one line (≤80 chars). "
    "Nested comprehensions are almost always clearer as explicit loops. "
    "Detection: comprehensions with more than one `for` clause.",
    "Context managers: Use `with` statements for all resource acquisition (files, locks, "
    "DB connections). Manual `try/finally` for cleanup is error-prone. "
    "Detection: explicit `f.close()` or `lock.release()` outside a `finally` block.",
    "DRY principle: Avoid duplicated code blocks (>5 lines appearing more than once). "
    "Extract into a shared helper or utility function. "
    "Detection: identical or near-identical code segments repeated in the same diff.",
    "Class cohesion: Classes should have a single responsibility. God classes with >10 "
    "public methods often signal the need for decomposition. "
    "Detection: class with more than 10 public methods (not prefixed with `_`).",
    "Mutable default arguments: Never use mutable objects (list, dict, set) as default "
    "argument values. They are shared across all calls and cause subtle bugs. "
    "Detection: `def f(x=[])` or `def f(x={})` patterns.",
    "Shadowing builtins: Do not use builtin names (`list`, `dict`, `id`, `type`, `input`) "
    "as variable names. Shadowing them makes the builtins inaccessible in that scope. "
    "Detection: assignment to a name that matches a Python builtin.",
    "Unused imports: Remove all unused imports. They add noise and slow import time. "
    "Detection: imported names never referenced in the module body.",
]


@dataclass
class StyleFinding:
    """A single structured style finding produced by the reviewer.

    Attributes:
        category: Finding category: 'complexity', 'naming', 'documentation',
            'formatting', 'readability', or 'other'.
        description: Human-readable description of the style issue.
        line_ref: Source-level reference (e.g. ``"line 47"``), or ``""`` if unknown.
    """

    category: str
    description: str
    line_ref: str = ""


# Pre-compiled patterns for _parse_finding and _categorise.
_LEADING_NUM_RE = re.compile(r"^\d+\.\s*")
_LINE_RE = re.compile(r"\blines?\s+\d+\b", re.IGNORECASE)

_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"complex|cyclomat|nest|depth|too long|exceed", re.IGNORECASE), "complexity"),
    (
        re.compile(r"name|naming|camel|snake_case|variable|identifier|descriptive", re.IGNORECASE),
        "naming",
    ),
    (
        re.compile(r"docstring|comment|documentation|missing doc|annotati", re.IGNORECASE),
        "documentation",
    ),
    (
        re.compile(
            r"whitespace|trailing|indent|line length|import|format|blank line", re.IGNORECASE
        ),
        "formatting",
    ),
    (
        re.compile(r"readab|magic number|TODO|FIXME|dead code|duplicate|repeat", re.IGNORECASE),
        "readability",
    ),
]


def _categorise(description: str) -> str:
    """Infer a category from the *description* text.

    Args:
        description: Finding description string.

    Returns:
        Category string — one of ``'complexity'``, ``'naming'``,
        ``'documentation'``, ``'formatting'``, ``'readability'``,
        or ``'other'``.
    """
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(description):
            return category
    return "other"


def _parse_finding(line: str) -> StyleFinding:
    """Extract a :class:`StyleFinding` from a single numbered finding line.

    Expected format (produced by :data:`StyleReviewerAgent.SYSTEM_PROMPT`)::

        1. Function `process_data` exceeds 50 lines — refactor into smaller units

    Args:
        line: A single numbered finding line from the LLM response.

    Returns:
        Parsed :class:`StyleFinding`.
    """
    text = _LEADING_NUM_RE.sub("", line).strip()

    line_match = _LINE_RE.search(text)
    line_ref = line_match.group(0) if line_match else ""

    description = text.strip()
    category = _categorise(description)

    return StyleFinding(category=category, description=description, line_ref=line_ref)


@dataclass
class StyleReview:
    """Result of the StyleReviewerAgent processing a PR.

    Attributes:
        pr_id: Identifier of the reviewed PR.
        findings: List of structured :class:`StyleFinding` objects.
        recommendation: Final recommendation string.
        retrieved_docs: Number of ChromaDB documents retrieved.
        retrieval_distances: Cosine distances of retrieved docs.
        retrieval_latency_ms: Time taken for ChromaDB query.
        llm_latency_ms: Time taken for LLM call.
        raw_response: Raw LLM output.
    """

    pr_id: str
    findings: list[StyleFinding]
    recommendation: str
    retrieved_docs: int
    retrieval_distances: list[float]
    retrieval_latency_ms: float
    llm_latency_ms: float
    raw_response: str


def _parse_review(raw: str, pr_id: str, retrieval: RetrievalResult) -> StyleReview:
    """Parse raw LLM output into a :class:`StyleReview`.

    Args:
        raw: Raw LLM response text.
        pr_id: PR identifier.
        retrieval: ChromaDB retrieval result for this step.

    Returns:
        Parsed :class:`StyleReview`.
    """
    findings: list[StyleFinding] = []
    recommendation = ""

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and ". " in stripped:
            findings.append(_parse_finding(stripped))
        elif stripped.startswith("RECOMMENDATION:"):
            recommendation = stripped.removeprefix("RECOMMENDATION:").strip()

    return StyleReview(
        pr_id=pr_id,
        findings=findings,
        recommendation=recommendation,
        retrieved_docs=len(retrieval.documents),
        retrieval_distances=retrieval.distances,
        retrieval_latency_ms=retrieval.latency_ms,
        llm_latency_ms=0.0,  # filled in by the caller
        raw_response=raw,
    )


class StyleReviewerAgent(BaseAgent):
    """Agent that reviews PRs for code quality and style issues.

    Retrieves relevant style conventions from a ChromaDB knowledge base,
    augments the prompt with retrieved context, and calls the LLM backend
    to produce a structured style review.

    Args:
        agent_id: Unique identifier for this agent instance.
        backend: :class:`~chronoagent.agents.backends.base.LLMBackend` for
            generation and embeddings.
        collection: ChromaDB collection containing style conventions.
        top_k: Number of style conventions to retrieve per PR.
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineer reviewing pull requests for code quality and style. "
        "Use the provided style conventions to guide your analysis. "
        "Structure your response with STYLE REVIEW FINDINGS: followed by numbered items, "
        "one finding per line.  Each item should clearly describe the issue. "
        "When the diff pinpoints a specific line, include a reference such as 'line 47'. "
        "End with RECOMMENDATION: followed by a single-line action item "
        "(e.g. 'Approve.', 'Request changes.', 'Approve with minor nits.')."
    )

    def review(self, pr: SyntheticPR) -> StyleReview:
        """Review a synthetic PR and return a structured style review.

        Args:
            pr: The synthetic pull request to review.

        Returns:
            :class:`StyleReview` with findings, recommendation, and retrieval
            metadata.
        """
        query = f"{pr.title} {pr.description} {pr.diff[:200]}"
        retrieval = self._retrieve_memory(query)

        context_block = "\n".join(
            f"- {doc}" for doc in retrieval.documents
        ) or "No relevant conventions retrieved."

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"STYLE CONVENTIONS:\n{context_block}\n\n"
            f"PR TITLE: {pr.title}\n"
            f"PR DESCRIPTION: {pr.description}\n"
            f"FILES CHANGED: {', '.join(pr.files_changed) or 'unknown'}\n"
            f"DIFF EXCERPT:\n{pr.diff[:500]}\n\n"
            "Provide your style review:"
        )

        raw_response, llm_latency_ms = self._call_llm(prompt)

        review = _parse_review(raw_response, pr.pr_id, retrieval)
        review.llm_latency_ms = llm_latency_ms
        return review

    def execute(self, task: Task) -> TaskResult:
        """Execute a style review task.

        Args:
            task: Task with ``payload["pr"]`` set to a :class:`SyntheticPR`.

        Returns:
            :class:`TaskResult` with ``output["review"]`` containing the
            :class:`StyleReview`.
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
        agent_id: str = "style_reviewer",
        seed: int = 42,
        top_k: int = 3,
        chroma_client: ClientAPI | None = None,
    ) -> StyleReviewerAgent:
        """Factory method creating a ready-to-use agent with MockBackend.

        Args:
            agent_id: Unique identifier for this agent instance.
            seed: Seed for the :class:`MockBackend` response selection.
            top_k: Number of style conventions to retrieve per PR.
            chroma_client: ChromaDB client; ephemeral in-memory client used if None.

        Returns:
            Configured :class:`StyleReviewerAgent`.
        """
        client = chroma_client or chromadb.EphemeralClient()
        backend = MockBackend(seed=seed, variant=MockBackendVariant.STYLE)
        collection = cls.build_collection(
            client,
            name=f"{agent_id}_style_kb",
            documents=_STYLE_KNOWLEDGE_BASE,
            backend=backend,
        )
        return cls(agent_id=agent_id, backend=backend, collection=collection, top_k=top_k)
