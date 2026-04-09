"""Base agent with ChromaDB retrieval.

All ChronoAgent agents inherit from :class:`BaseAgent` which wires together
an LLM backend and a ChromaDB collection for retrieval-augmented generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from chromadb import Collection
from chromadb.api import ClientAPI
from langchain_core.language_models.llms import LLM

from chronoagent.llm.mock_backend import MockEmbeddingFunction


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


class BaseAgent:
    """Foundation class for all ChronoAgent agents.

    Provides ChromaDB retrieval utilities and a common interface for
    subclasses to implement task-specific processing.

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

    def retrieve(self, query: str) -> RetrievalResult:
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
