"""Quarantine collection for flagged memory documents (Phase 6 task 6.3).

:class:`QuarantineStore` is a thin wrapper around a *separate* ChromaDB
:class:`~chromadb.Collection` that holds documents flagged by
:class:`~chronoagent.memory.integrity.MemoryIntegrityModule`.  The active
:class:`~chronoagent.memory.store.MemoryStore` and the quarantine store live
in two distinct collections; retrieval only ever queries the active store, so
flagged documents are excluded from agent context until a human approves them.

The store exposes two transactional moves:

* :meth:`QuarantineStore.quarantine` -- pulls full records (text + embedding
  + metadata) from the active store via
  :meth:`~chronoagent.memory.store.MemoryStore.get_by_ids`, copies them into
  the quarantine collection (annotating each with ``quarantined_at`` and an
  optional ``quarantine_reason``), and deletes them from the active store.
* :meth:`QuarantineStore.approve` -- the inverse: pulls records from the
  quarantine collection, strips the quarantine metadata, restores them to the
  active store, and deletes them from the quarantine collection.

Both methods are idempotent at the ID level: quarantining a document that is
already in the quarantine collection (or that does not exist in the active
store) is a no-op for that ID, and approving an ID that is not currently
quarantined is likewise a no-op.  This keeps the API safe to call from a
retry loop or from an HTTP handler that may receive duplicate requests.

The store does not own a backend or an integrity module; it is intentionally
a passive bookkeeping layer that the API router (task 6.4) drives from the
output of :class:`~chronoagent.memory.integrity.IntegrityResult`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from chromadb import Collection

from chronoagent.memory.store import MemoryStore, StoredDoc

# ---------------------------------------------------------------------------
# Metadata keys
# ---------------------------------------------------------------------------

#: Metadata key holding the Unix epoch seconds at which a document entered
#: quarantine.  Stripped on :meth:`QuarantineStore.approve` so the restored
#: record looks identical to its pre-quarantine state.
QUARANTINED_AT_KEY = "quarantined_at"

#: Optional metadata key holding a free-form reason string passed to
#: :meth:`QuarantineStore.quarantine`.  Stripped on approve.
QUARANTINE_REASON_KEY = "quarantine_reason"


# ---------------------------------------------------------------------------
# QuarantineStore
# ---------------------------------------------------------------------------


class QuarantineStore:
    """ChromaDB-backed holding pen for flagged documents.

    Args:
        collection: A ChromaDB :class:`~chromadb.Collection` dedicated to
            quarantined records.  Must be a *different* collection from the
            active :class:`MemoryStore`; the caller is responsible for
            providing the right one (typically via
            ``client.get_or_create_collection("memory_quarantine")``).
        now_fn: Injectable wall-clock source used for the ``quarantined_at``
            metadata stamp.  Defaults to :func:`time.time`.
    """

    def __init__(
        self,
        collection: Collection,
        *,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._collection = collection
        self._now_fn = now_fn

    # ------------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of documents currently held in quarantine."""
        return int(self._collection.count())

    def list_ids(self) -> list[str]:
        """Return every quarantined document ID.

        Returns:
            List of ChromaDB IDs in whatever order ChromaDB reports them.
            Empty list if the collection is empty.
        """
        if self.count == 0:
            return []
        raw = self._collection.get(include=[])
        return list(raw.get("ids") or [])

    def get_doc(self, doc_id: str) -> StoredDoc | None:
        """Fetch a single quarantined document by ID for human review.

        Args:
            doc_id: ChromaDB ID of the quarantined document.

        Returns:
            :class:`StoredDoc` if the document is currently quarantined,
            ``None`` otherwise.  The returned record still carries the
            ``quarantined_at`` (and optional ``quarantine_reason``) metadata
            keys, which an approval flow can surface to the reviewer.
        """
        raw = self._collection.get(
            ids=[doc_id],
            include=["documents", "embeddings", "metadatas"],
        )
        out_ids: list[str] = list(raw.get("ids") or [])
        if not out_ids:
            return None

        # Records in this collection are always upserted via :meth:`quarantine`,
        # which writes a document, an embedding, and a non-empty metadata dict
        # (it always stamps ``quarantined_at``).  Trust those invariants here.
        documents = raw.get("documents") or []
        embeddings_field = raw.get("embeddings")
        metadatas_field = raw.get("metadatas")
        assert embeddings_field is not None and len(embeddings_field) > 0
        assert metadatas_field is not None and len(metadatas_field) > 0
        return StoredDoc(
            doc_id=out_ids[0],
            text=documents[0],
            embedding=[float(v) for v in embeddings_field[0]],
            metadata=dict(metadatas_field[0]),
        )

    # ------------------------------------------------------------------
    # State-changing moves
    # ------------------------------------------------------------------

    def quarantine(
        self,
        active_store: MemoryStore,
        ids: list[str],
        *,
        reason: str | None = None,
    ) -> list[str]:
        """Move documents from *active_store* into the quarantine collection.

        For each ID actually present in *active_store* and not already
        quarantined the method copies the full record (text, embedding,
        metadata) into the quarantine collection, stamps it with
        ``quarantined_at`` (and ``quarantine_reason`` if provided), and
        removes it from the active store.  IDs that do not exist in the
        active store, that are already quarantined, or that appear more than
        once in *ids* are silently deduplicated and skipped.

        Args:
            active_store: The :class:`MemoryStore` documents are being
                pulled out of.
            ids: ChromaDB IDs to quarantine.  Empty list is a no-op.
            reason: Optional human-readable reason recorded as metadata on
                each quarantined document.

        Returns:
            The IDs that were actually moved into quarantine, in the same
            order they were processed.  Empty list if nothing moved.
        """
        if not ids:
            return []

        # Deduplicate while preserving order so the caller can replay the
        # list verbatim from a flagged_ids array without surprises.
        seen: set[str] = set()
        unique_ids: list[str] = []
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique_ids.append(doc_id)

        already_quarantined = set(self._existing_ids(unique_ids))
        candidate_ids = [d for d in unique_ids if d not in already_quarantined]
        if not candidate_ids:
            return []

        records = active_store.get_by_ids(candidate_ids)
        if not records:
            return []

        # Order records to match candidate_ids so the returned list reflects
        # the caller's intent rather than ChromaDB's internal ordering.
        record_by_id = {r.doc_id: r for r in records}
        ordered: list[StoredDoc] = [record_by_id[d] for d in candidate_ids if d in record_by_id]

        ts = float(self._now_fn())
        moved_ids: list[str] = []
        moved_documents: list[str] = []
        moved_embeddings: list[list[float]] = []
        moved_metadatas: list[dict[str, Any]] = []
        for rec in ordered:
            meta = dict(rec.metadata)
            meta[QUARANTINED_AT_KEY] = ts
            if reason is not None:
                meta[QUARANTINE_REASON_KEY] = reason
            moved_ids.append(rec.doc_id)
            moved_documents.append(rec.text)
            moved_embeddings.append(list(rec.embedding))
            moved_metadatas.append(meta)

        # Route through an ``Any``-typed kwargs dict to bypass ChromaDB's
        # narrow ndarray-flavoured type stubs (the runtime accepts plain
        # ``list[list[float]]`` and ``list[dict[str, Any]]`` just fine).
        upsert_kwargs: dict[str, Any] = {
            "ids": moved_ids,
            "documents": moved_documents,
            "embeddings": moved_embeddings,
            "metadatas": moved_metadatas,
        }
        self._collection.upsert(**upsert_kwargs)
        active_store.delete(moved_ids)
        return moved_ids

    def approve(
        self,
        active_store: MemoryStore,
        ids: list[str],
    ) -> list[str]:
        """Restore previously quarantined documents to *active_store*.

        For each ID actually present in the quarantine collection the
        method strips the quarantine bookkeeping metadata
        (:data:`QUARANTINED_AT_KEY`, :data:`QUARANTINE_REASON_KEY`),
        re-inserts the record into *active_store* using the original
        embedding (so the round-trip does not perturb the vector space),
        and deletes it from the quarantine collection.  IDs that are not
        currently quarantined are silently skipped.

        Args:
            active_store: The :class:`MemoryStore` to restore documents to.
            ids: ChromaDB IDs to approve.  Empty list is a no-op.

        Returns:
            The IDs that were actually restored, in the same order they
            were processed.  Empty list if nothing was restored.
        """
        if not ids:
            return []

        seen: set[str] = set()
        unique_ids: list[str] = []
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique_ids.append(doc_id)

        raw = self._collection.get(
            ids=unique_ids,
            include=["documents", "embeddings", "metadatas"],
        )
        found_ids: list[str] = list(raw.get("ids") or [])
        if not found_ids:
            return []

        documents = raw.get("documents") or []
        embeddings_field = raw.get("embeddings")
        metadatas_field = raw.get("metadatas")
        # Records here are always written by :meth:`quarantine`, which
        # always upserts a document, an embedding, and a non-empty metadata
        # dict.  Trust the structure on read.
        assert embeddings_field is not None
        assert metadatas_field is not None

        records_by_id: dict[str, StoredDoc] = {}
        for i, doc_id in enumerate(found_ids):
            metadata = dict(metadatas_field[i])
            metadata.pop(QUARANTINED_AT_KEY, None)
            metadata.pop(QUARANTINE_REASON_KEY, None)
            records_by_id[doc_id] = StoredDoc(
                doc_id=doc_id,
                text=documents[i],
                embedding=[float(v) for v in embeddings_field[i]],
                metadata=metadata,
            )

        ordered = [records_by_id[d] for d in unique_ids if d in records_by_id]

        # ChromaDB rejects empty metadata dicts on upsert, so split the
        # restore into a "has surviving metadata" batch and a "metadata is
        # empty after stripping quarantine keys" batch.  The latter is added
        # with ``metadatas=None`` so MemoryStore.add omits the field entirely.
        with_meta: list[StoredDoc] = [r for r in ordered if r.metadata]
        without_meta: list[StoredDoc] = [r for r in ordered if not r.metadata]

        if with_meta:
            active_store.add(
                documents=[r.text for r in with_meta],
                ids=[r.doc_id for r in with_meta],
                metadatas=[r.metadata for r in with_meta],
                embeddings=[list(r.embedding) for r in with_meta],
            )
        if without_meta:
            active_store.add(
                documents=[r.text for r in without_meta],
                ids=[r.doc_id for r in without_meta],
                embeddings=[list(r.embedding) for r in without_meta],
            )

        restored_ids: list[str] = [r.doc_id for r in ordered]
        self._collection.delete(ids=restored_ids)
        return restored_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _existing_ids(self, ids: list[str]) -> list[str]:
        """Return the subset of *ids* already present in the quarantine collection.

        Used to make :meth:`quarantine` idempotent without raising on
        already-quarantined IDs.  Caller is responsible for passing a
        non-empty list (the only call site is :meth:`quarantine`, which
        gates on emptiness up front).
        """
        raw = self._collection.get(ids=ids, include=[])
        return list(raw.get("ids") or [])
