"""Base agent with ChromaDB retrieval.

All ChronoAgent agents inherit from :class:`BaseAgent` which wires together
an LLM backend and a ChromaDB collection for retrieval-augmented generation.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from chromadb import Collection
from chromadb.api import ClientAPI
from langchain_core.language_models.llms import LLM

from chronoagent.llm.mock_backend import MockEmbeddingFunction


@dataclass
class Task:
    """Unit of work submitted to an agent.

    Attributes:
        task_id: Unique identifier for this task.
        task_type: Agent-specific type tag (e.g. ``"security_review"``, ``"summarize"``).
        payload: Agent-specific input data.
    """

    task_id: str
    task_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Outcome produced by an agent executing a :class:`Task`.

    Attributes:
        task_id: Matches :attr:`Task.task_id`.
        agent_id: ID of the agent that produced this result.
        status: ``"success"`` or ``"error"``.
        output: Agent-specific output data.
        llm_latency_ms: Wall-clock time of the LLM call in milliseconds.
        retrieval_latency_ms: Wall-clock time of the memory query in milliseconds.
        timestamp: Unix epoch (seconds) when the result was produced.
    """

    task_id: str
    agent_id: str
    status: Literal["success", "error"]
    output: dict[str, Any]
    llm_latency_ms: float
    retrieval_latency_ms: float
    timestamp: float


@dataclass
class RetrievalResult:
    """Result of a ChromaDB similarity search.

    Attributes:
        documents: Retrieved document strings.
        distances: Cosine distances to query (lower = more similar).
        ids: ChromaDB document IDs.
        latency_ms: Time taken for the retrieval query in milliseconds.
    """

    documents: list[str]
    distances: list[float]
    ids: list[str]
    latency_ms: float


class BaseAgent(abc.ABC):
    """Foundation class for all ChronoAgent agents.

    Provides instrumented LLM calls, ChromaDB memory retrieval, and an
    abstract :meth:`execute` interface that every concrete agent must implement.

    Args:
        agent_id: Unique identifier for this agent instance.
        llm: LangChain-compatible language model backend.
        collection: ChromaDB collection used for retrieval.
        top_k: Number of documents to retrieve per query.
    """

    def __init__(
        self,
        agent_id: str,
        llm: LLM,
        collection: Collection,
        top_k: int = 3,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm
        self.collection = collection
        self.top_k = top_k
        self._embed = MockEmbeddingFunction()

    @abc.abstractmethod
    def execute(self, task: Task) -> TaskResult:
        """Execute a task and return a structured result.

        Args:
            task: The :class:`Task` to process.

        Returns:
            :class:`TaskResult` with output, latencies, and status.
        """

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        """Call the LLM backend and measure latency.

        Args:
            prompt: Input prompt string.

        Returns:
            Tuple of ``(response_text, latency_ms)``.
        """
        t0 = time.perf_counter()
        response = self.llm.invoke(prompt)
        latency_ms = (time.perf_counter() - t0) * 1_000
        return str(response), latency_ms

    def _retrieve_memory(self, query: str) -> RetrievalResult:
        """Query the ChromaDB collection for context relevant to *query*.

        Args:
            query: Natural language query string.

        Returns:
            :class:`RetrievalResult` containing retrieved documents,
            distances, IDs, and retrieval latency.
        """
        t0 = time.perf_counter()
        results = self.collection.query(
            query_texts=[query],
            n_results=min(self.top_k, self.collection.count()),
            include=["documents", "distances"],
        )
        latency_ms = (time.perf_counter() - t0) * 1_000

        raw_docs: list[str] = (results.get("documents") or [[]])[0]
        raw_dists: list[float] = (results.get("distances") or [[]])[0]
        raw_ids: list[str] = (results.get("ids") or [[]])[0]

        return RetrievalResult(
            documents=raw_docs,
            distances=raw_dists,
            ids=raw_ids,
            latency_ms=latency_ms,
        )

    @staticmethod
    def build_collection(
        client: ClientAPI,
        name: str,
        documents: list[str],
        ids: list[str] | None = None,
    ) -> Collection:
        """Create (or get) a ChromaDB collection and populate it with documents.

        Args:
            client: ChromaDB client instance.
            name: Collection name.
            documents: Documents to add.
            ids: Optional explicit IDs; auto-generated if omitted.

        Returns:
            Populated :class:`Collection`.
        """
        embed_fn = MockEmbeddingFunction()
        collection: Collection = client.get_or_create_collection(
            name=name,
            embedding_function=embed_fn,  # type: ignore[arg-type]
        )
        if collection.count() == 0:
            doc_ids = ids if ids else [f"doc_{i}" for i in range(len(documents))]
            collection.add(documents=documents, ids=doc_ids)
        return collection
