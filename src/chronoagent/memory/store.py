"""ChromaDB-backed memory store for ChronoAgent.

:class:`MemoryStore` is a thin, typed wrapper around a ChromaDB
:class:`~chromadb.Collection` that provides the four operations used
throughout the agent pipeline:

* :meth:`add` — upsert documents (with optional pre-computed embeddings).
* :meth:`query` — nearest-neighbour lookup by natural-language query.
* :meth:`delete` — remove documents by ID.
* :meth:`get_all_embeddings` — return every stored embedding vector.

The store owns its embedding logic via the injected
:class:`~chronoagent.agents.backends.base.LLMBackend`; callers never need to
manage embedding dimensions or ChromaDB internals directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from chromadb import Collection

from chronoagent.agents.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Result of a :meth:`MemoryStore.query` call.

    Attributes:
        documents: Retrieved document strings in ranked order.
        distances: Cosine distances to the query vector (lower = more similar).
        ids: ChromaDB document IDs, aligned with *documents* and *distances*.
        metadatas: Per-document metadata dicts (empty dict if none stored).
    """

    documents: list[str]
    distances: list[float]
    ids: list[str]
    metadatas: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """ChromaDB collection wrapper with typed add / query / delete / embed API.

    Args:
        collection: The ChromaDB :class:`~chromadb.Collection` to wrap.
        backend: :class:`~chronoagent.agents.backends.base.LLMBackend` used to
            embed query strings and documents that arrive without pre-computed
            embeddings.
    """

    def __init__(self, collection: Collection, backend: LLMBackend) -> None:
        self._collection = collection
        self._backend = backend

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of documents currently stored in the collection.

        Returns:
            Document count as reported by ChromaDB.
        """
        return int(self._collection.count())

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add documents to the memory store.

        If *embeddings* are not provided they are computed via the injected
        backend.  Documents are upserted — adding an ID that already exists
        overwrites the previous record.

        Args:
            documents: Document strings to store.
            ids: Unique ChromaDB IDs, one per document.
            metadatas: Optional metadata dicts, one per document.
            embeddings: Optional pre-computed float vectors.  Must have the
                same length as *documents* if supplied.

        Raises:
            ValueError: If ``len(documents) != len(ids)``, or if *embeddings*
                is provided but ``len(embeddings) != len(documents)``.
        """
        if len(documents) != len(ids):
            raise ValueError(
                f"documents and ids must have equal length (got {len(documents)} vs {len(ids)})"
            )
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(
                f"embeddings length must match documents length "
                f"(got {len(embeddings)} vs {len(documents)})"
            )

        resolved_embeddings: list[list[float]] = (
            embeddings if embeddings is not None else self._backend.embed(documents)
        )

        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "embeddings": resolved_embeddings,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas

        self._collection.upsert(**kwargs)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        include: list[str] | None = None,
    ) -> QueryResult:
        """Retrieve the *n_results* most relevant documents for *query_text*.

        The query string is embedded via the backend and used for
        approximate nearest-neighbour search in the ChromaDB collection.

        Args:
            query_text: Natural-language query string.
            n_results: Maximum number of documents to return.  Clamped to
                :attr:`count` so the call never errors on small collections.
            include: ChromaDB ``include`` list controlling what fields are
                returned.  Defaults to ``["documents", "distances", "metadatas"]``.

        Returns:
            :class:`QueryResult` with ranked documents, distances, IDs, and
            metadata.
        """
        if include is None:
            include = ["documents", "distances", "metadatas"]

        actual_n = min(n_results, self.count)
        if actual_n == 0:
            return QueryResult(documents=[], distances=[], ids=[], metadatas=[])

        query_vec: list[float] = self._backend.embed([query_text])[0]
        query_arr = np.array([query_vec], dtype=np.float32)

        raw = self._collection.query(
            query_embeddings=query_arr,
            n_results=actual_n,
            include=include,  # type: ignore[arg-type]
        )

        docs: list[str] = (raw.get("documents") or [[]])[0]
        dists: list[float] = (raw.get("distances") or [[]])[0]
        ids: list[str] = (raw.get("ids") or [[]])[0]
        raw_meta = (raw.get("metadatas") or [[]])[0]
        metas: list[dict[str, Any]] = [dict(m) if m is not None else {} for m in raw_meta]

        return QueryResult(
            documents=docs,
            distances=dists,
            ids=ids,
            metadatas=metas,
        )

    def get_all_embeddings(self) -> list[list[float]]:
        """Return every embedding vector stored in the collection.

        Useful for drift detection, anomaly scoring, and memory-integrity
        analysis (e.g. comparing centroid shift before and after a poisoning
        attack).

        Returns:
            List of float vectors in insertion order.  Returns an empty list
            if the collection is empty.
        """
        if self.count == 0:
            return []

        raw = self._collection.get(include=["embeddings"])
        stored = raw.get("embeddings")
        if stored is None or len(stored) == 0:
            return []
        # ChromaDB may return numpy arrays — normalise to plain Python lists.
        return [list(map(float, vec)) for vec in stored]

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete(self, ids: list[str]) -> int:
        """Remove documents by ID.

        Silently ignores IDs that do not exist in the collection.

        Args:
            ids: ChromaDB document IDs to remove.

        Returns:
            Number of IDs submitted for deletion (not the number actually
            found — ChromaDB does not return a matched count).
        """
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)
