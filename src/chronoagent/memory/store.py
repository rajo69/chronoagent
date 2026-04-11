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
from chronoagent.retry import chroma_retry

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


@dataclass
class StoredDoc:
    """A single document fetched by ID from a :class:`MemoryStore`.

    Mirrors the columns ChromaDB stores per record so callers can move a
    document between collections (the quarantine round-trip in
    :class:`~chronoagent.memory.quarantine.QuarantineStore`) without losing
    its embedding or metadata.

    Attributes:
        doc_id: ChromaDB document identifier.
        text: Document content as stored.
        embedding: Stored embedding vector.  Empty list if the collection
            holds no embedding for the record (should not occur for stores
            populated through :meth:`MemoryStore.add`).
        metadata: Per-document metadata dict; empty dict if none was stored.
    """

    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


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
        return self._count_raw()

    @chroma_retry
    def _count_raw(self) -> int:
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

        self._upsert_raw(kwargs)

    @chroma_retry
    def _upsert_raw(self, kwargs: dict[str, Any]) -> None:
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

        raw = self._query_raw(query_arr, actual_n, include)

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

    @chroma_retry
    def _query_raw(
        self,
        query_arr: np.ndarray[Any, Any],
        actual_n: int,
        include: list[str],
    ) -> dict[str, Any]:
        raw: dict[str, Any] = self._collection.query(  # type: ignore[assignment]
            query_embeddings=query_arr,
            n_results=actual_n,
            include=include,  # type: ignore[arg-type]
        )
        return raw

    def get_by_ids(self, ids: list[str]) -> list[StoredDoc]:
        """Fetch full records (text + embedding + metadata) for *ids*.

        IDs that do not exist in the collection are silently dropped, so the
        returned list may be shorter than the input list.  Order follows
        whatever ChromaDB reports, which is generally insertion order rather
        than the input order; callers that need a specific ordering should
        re-key by :attr:`StoredDoc.doc_id`.

        Args:
            ids: ChromaDB document IDs to look up.

        Returns:
            List of :class:`StoredDoc` records, one per ID actually found in
            the collection.  Empty list if *ids* is empty.
        """
        if not ids:
            return []

        raw = self._get_by_ids_raw(ids)
        out_ids: list[str] = list(raw.get("ids") or [])
        if not out_ids:
            return []

        # ChromaDB always returns the requested ``include`` fields when at
        # least one ID matches, so we can rely on documents/embeddings being
        # populated.  Per-row metadata may still be ``None`` (when the doc
        # was inserted without a metadata dict), so that one stays guarded.
        documents: list[str] = list(raw.get("documents") or [])
        raw_embeddings = raw.get("embeddings")
        raw_metadatas = raw.get("metadatas")
        assert raw_embeddings is not None

        results: list[StoredDoc] = []
        for i, doc_id in enumerate(out_ids):
            metadata: dict[str, Any]
            if raw_metadatas is not None and raw_metadatas[i]:
                metadata = dict(raw_metadatas[i])
            else:
                metadata = {}
            results.append(
                StoredDoc(
                    doc_id=doc_id,
                    text=documents[i],
                    # ChromaDB may return numpy arrays; coerce to plain floats.
                    embedding=[float(v) for v in raw_embeddings[i]],
                    metadata=metadata,
                )
            )
        return results

    @chroma_retry
    def _get_by_ids_raw(self, ids: list[str]) -> dict[str, Any]:
        raw: dict[str, Any] = self._collection.get(  # type: ignore[assignment]
            ids=ids,
            include=["documents", "embeddings", "metadatas"],
        )
        return raw

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

        raw = self._get_all_raw()
        stored = raw.get("embeddings")
        if stored is None or len(stored) == 0:
            return []
        # ChromaDB may return numpy arrays — normalise to plain Python lists.
        return [list(map(float, vec)) for vec in stored]

    @chroma_retry
    def _get_all_raw(self) -> dict[str, Any]:
        raw: dict[str, Any] = self._collection.get(include=["embeddings"])  # type: ignore[assignment]
        return raw

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
        self._delete_raw(ids)
        return len(ids)

    @chroma_retry
    def _delete_raw(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
