"""Tests for Phase 6 task 6.3: QuarantineStore.

Coverage:
- ``MemoryStore.get_by_ids`` round-trips text + embedding + metadata, drops
  missing IDs, and is empty-safe.
- :meth:`QuarantineStore.quarantine` moves matching IDs out of the active
  store, stamps ``quarantined_at`` (and an optional reason) onto the
  quarantined record, leaves the active retrieval path blind to flagged
  docs, and is idempotent / dedup-safe / no-op on empty + missing inputs.
- :meth:`QuarantineStore.approve` restores text + embedding + original
  metadata (with quarantine bookkeeping stripped), removes the record from
  the quarantine collection, and is idempotent on unknown IDs.
- ``count``, ``list_ids``, and ``get_doc`` reflect the live state of the
  quarantine collection.
- End-to-end: a poisoned document flagged by
  :class:`MemoryIntegrityModule.check_retrieval` can be moved through the
  quarantine round-trip from its ``flagged_ids`` output without losing the
  vector that originally triggered the flag.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.memory.integrity import MemoryIntegrityModule, RetrievedDoc
from chronoagent.memory.quarantine import (
    QUARANTINE_REASON_KEY,
    QUARANTINED_AT_KEY,
    QuarantineStore,
)
from chronoagent.memory.store import MemoryStore, StoredDoc

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def client() -> chromadb.api.ClientAPI:
    return chromadb.EphemeralClient()


@pytest.fixture
def active_store(client: chromadb.api.ClientAPI, backend: MockBackend) -> MemoryStore:
    collection = client.get_or_create_collection(f"active_{uuid.uuid4().hex[:8]}")
    return MemoryStore(collection=collection, backend=backend)


@pytest.fixture
def quarantine_store(client: chromadb.api.ClientAPI) -> QuarantineStore:
    collection = client.get_or_create_collection(f"quarantine_{uuid.uuid4().hex[:8]}")
    return QuarantineStore(collection=collection, now_fn=lambda: 1_700_000_000.0)


def _seed_active(store: MemoryStore, n: int = 4) -> list[str]:
    docs = [f"clean document {i}" for i in range(n)]
    ids = [f"doc_{i}" for i in range(n)]
    metas = [{"source": "seed", "idx": i} for i in range(n)]
    store.add(documents=docs, ids=ids, metadatas=metas)
    return ids


# ===========================================================================
# MemoryStore.get_by_ids
# ===========================================================================


class TestGetByIds:
    def test_empty_input_returns_empty(self, active_store: MemoryStore) -> None:
        _seed_active(active_store, 3)
        assert active_store.get_by_ids([]) == []

    def test_returns_stored_doc_per_existing_id(self, active_store: MemoryStore) -> None:
        ids = _seed_active(active_store, 3)
        records = active_store.get_by_ids(ids)
        assert len(records) == 3
        for r in records:
            assert isinstance(r, StoredDoc)
            assert r.doc_id in ids
            assert r.text.startswith("clean document")
            assert len(r.embedding) > 0
            assert r.metadata.get("source") == "seed"

    def test_missing_ids_silently_dropped(self, active_store: MemoryStore) -> None:
        _seed_active(active_store, 2)
        records = active_store.get_by_ids(["doc_0", "does-not-exist"])
        ids_returned = [r.doc_id for r in records]
        assert "doc_0" in ids_returned
        assert "does-not-exist" not in ids_returned

    def test_unknown_ids_only_returns_empty(self, active_store: MemoryStore) -> None:
        _seed_active(active_store, 2)
        assert active_store.get_by_ids(["nope_a", "nope_b"]) == []

    def test_embedding_round_trips_within_tolerance(
        self, active_store: MemoryStore, backend: MockBackend
    ) -> None:
        original = backend.embed(["round trip"])
        active_store.add(
            documents=["round trip"],
            ids=["rt"],
            embeddings=original,
            metadatas=[{"k": "v"}],
        )
        records = active_store.get_by_ids(["rt"])
        assert len(records) == 1
        for a, b in zip(records[0].embedding, original[0], strict=False):
            assert abs(a - b) < 1e-5

    def test_no_metadata_yields_empty_dict(self, active_store: MemoryStore) -> None:
        active_store.add(documents=["no meta"], ids=["nm"])
        records = active_store.get_by_ids(["nm"])
        assert records[0].metadata == {}


# ===========================================================================
# QuarantineStore introspection
# ===========================================================================


class TestQuarantineIntrospection:
    def test_count_starts_zero(self, quarantine_store: QuarantineStore) -> None:
        assert quarantine_store.count == 0

    def test_list_ids_empty(self, quarantine_store: QuarantineStore) -> None:
        assert quarantine_store.list_ids() == []

    def test_get_doc_missing_returns_none(self, quarantine_store: QuarantineStore) -> None:
        assert quarantine_store.get_doc("nope") is None

    def test_count_and_list_after_quarantine(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 3)
        quarantine_store.quarantine(active_store, [ids[0], ids[2]])
        assert quarantine_store.count == 2
        assert set(quarantine_store.list_ids()) == {ids[0], ids[2]}


# ===========================================================================
# QuarantineStore.quarantine
# ===========================================================================


class TestQuarantine:
    def test_empty_ids_noop(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        _seed_active(active_store, 2)
        moved = quarantine_store.quarantine(active_store, [])
        assert moved == []
        assert quarantine_store.count == 0
        assert active_store.count == 2

    def test_moves_records_out_of_active(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 4)
        moved = quarantine_store.quarantine(active_store, [ids[1]])
        assert moved == [ids[1]]
        assert active_store.count == 3
        assert quarantine_store.count == 1
        assert active_store.get_by_ids([ids[1]]) == []

    def test_unknown_ids_silently_skipped(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 2)
        moved = quarantine_store.quarantine(active_store, [ids[0], "ghost-id-xyz"])
        assert moved == [ids[0]]
        assert active_store.count == 1
        assert quarantine_store.count == 1

    def test_only_unknown_ids_noop(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        _seed_active(active_store, 2)
        moved = quarantine_store.quarantine(active_store, ["a", "b"])
        assert moved == []
        assert active_store.count == 2
        assert quarantine_store.count == 0

    def test_idempotent_on_already_quarantined(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 2)
        first = quarantine_store.quarantine(active_store, [ids[0]])
        second = quarantine_store.quarantine(active_store, [ids[0]])
        assert first == [ids[0]]
        assert second == []
        assert quarantine_store.count == 1

    def test_dedupes_repeated_ids(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 2)
        moved = quarantine_store.quarantine(active_store, [ids[0], ids[0], ids[0]])
        assert moved == [ids[0]]
        assert quarantine_store.count == 1

    def test_quarantined_at_metadata_stamped(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 1)
        quarantine_store.quarantine(active_store, ids)
        doc = quarantine_store.get_doc(ids[0])
        assert doc is not None
        assert doc.metadata.get(QUARANTINED_AT_KEY) == 1_700_000_000.0
        # Original metadata is preserved alongside the quarantine stamp.
        assert doc.metadata.get("source") == "seed"

    def test_reason_recorded_when_provided(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 1)
        quarantine_store.quarantine(active_store, ids, reason="content_mismatch")
        doc = quarantine_store.get_doc(ids[0])
        assert doc is not None
        assert doc.metadata.get(QUARANTINE_REASON_KEY) == "content_mismatch"

    def test_reason_omitted_when_none(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 1)
        quarantine_store.quarantine(active_store, ids)
        doc = quarantine_store.get_doc(ids[0])
        assert doc is not None
        assert QUARANTINE_REASON_KEY not in doc.metadata

    def test_active_query_excludes_quarantined(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 4)
        # Quarantine doc_2 then verify it cannot resurface from the active store.
        quarantine_store.quarantine(active_store, [ids[2]])
        result = active_store.query("clean document 2", n_results=10)
        assert ids[2] not in result.ids

    def test_embedding_preserved_round_trip_into_quarantine(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
        backend: MockBackend,
    ) -> None:
        # Use a hand-crafted embedding so we can verify byte-level preservation.
        custom_emb = backend.embed(["payload"])
        active_store.add(
            documents=["payload"],
            ids=["payload-1"],
            embeddings=custom_emb,
            metadatas=[{"tag": "before"}],
        )
        quarantine_store.quarantine(active_store, ["payload-1"])
        doc = quarantine_store.get_doc("payload-1")
        assert doc is not None
        assert doc.text == "payload"
        for a, b in zip(doc.embedding, custom_emb[0], strict=False):
            assert abs(a - b) < 1e-5


# ===========================================================================
# QuarantineStore.approve
# ===========================================================================


class TestApprove:
    def test_empty_ids_noop(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 2)
        quarantine_store.quarantine(active_store, [ids[0]])
        restored = quarantine_store.approve(active_store, [])
        assert restored == []
        assert quarantine_store.count == 1
        assert active_store.count == 1

    def test_unknown_ids_noop(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        _seed_active(active_store, 1)
        restored = quarantine_store.approve(active_store, ["never-quarantined"])
        assert restored == []

    def test_round_trip_restores_to_active(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 3)
        quarantine_store.quarantine(active_store, [ids[1]], reason="test")
        assert active_store.count == 2

        restored = quarantine_store.approve(active_store, [ids[1]])
        assert restored == [ids[1]]
        assert quarantine_store.count == 0
        assert active_store.count == 3
        # Document is queryable from the active store again.
        result = active_store.query("clean document 1", n_results=5)
        assert ids[1] in result.ids

    def test_quarantine_metadata_stripped_on_approve(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 1)
        quarantine_store.quarantine(active_store, ids, reason="suspect")
        quarantine_store.approve(active_store, ids)

        records = active_store.get_by_ids(ids)
        assert len(records) == 1
        meta = records[0].metadata
        assert QUARANTINED_AT_KEY not in meta
        assert QUARANTINE_REASON_KEY not in meta
        # Original metadata survives the round trip.
        assert meta.get("source") == "seed"

    def test_embedding_preserved_round_trip(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
        backend: MockBackend,
    ) -> None:
        original = backend.embed(["payload"])
        active_store.add(
            documents=["payload"],
            ids=["payload-rt"],
            embeddings=original,
            metadatas=[{"tag": "x"}],
        )
        quarantine_store.quarantine(active_store, ["payload-rt"])
        quarantine_store.approve(active_store, ["payload-rt"])

        restored = active_store.get_by_ids(["payload-rt"])
        assert len(restored) == 1
        for a, b in zip(restored[0].embedding, original[0], strict=False):
            assert abs(a - b) < 1e-5

    def test_dedupes_repeated_ids(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
    ) -> None:
        ids = _seed_active(active_store, 1)
        quarantine_store.quarantine(active_store, ids)
        restored = quarantine_store.approve(active_store, [ids[0], ids[0]])
        assert restored == [ids[0]]


# ===========================================================================
# End-to-end: integrity flag -> quarantine -> approve
# ===========================================================================


class TestIntegrityFlowEndToEnd:
    def test_flagged_id_can_be_quarantined_and_restored(
        self,
        active_store: MemoryStore,
        quarantine_store: QuarantineStore,
        backend: MockBackend,
    ) -> None:
        # Seed a clean corpus and one mismatched poison doc.
        clean_texts = [f"the quick brown fox jumped over log {i}" for i in range(6)]
        clean_ids = [f"clean_{i}" for i in range(6)]
        active_store.add(documents=clean_texts, ids=clean_ids)

        # Poison: stored embedding is from text A, but the visible text is text B.
        poison_text = "completely unrelated topic about astrophysics"
        decoy_emb = backend.embed(["the quick brown fox jumped over log 0"])
        active_store.add(
            documents=[poison_text],
            ids=["poison_1"],
            embeddings=decoy_emb,
        )

        # Run integrity check on the poison doc -- content mismatch should fire.
        # Weight everything onto the mismatch signal so the test does not depend
        # on the (noisy) embedding-outlier or retrieval-frequency baselines.
        module = MemoryIntegrityModule(
            backend=backend,
            flag_threshold=0.3,
            weights={
                "embedding_outlier": 0.0,
                "freshness_anomaly": 0.0,
                "retrieval_frequency": 0.0,
                "content_embedding_mismatch": 1.0,
            },
        )
        retrieved = [
            RetrievedDoc(
                doc_id="poison_1",
                text=poison_text,
                embedding=decoy_emb[0],
            )
        ]
        result = module.check_retrieval("anything", retrieved)
        assert "poison_1" in result.flagged_ids

        # Hand the flagged_ids straight to the quarantine store.
        moved = quarantine_store.quarantine(
            active_store, result.flagged_ids, reason="integrity_module"
        )
        assert moved == ["poison_1"]
        assert active_store.count == 6
        assert "poison_1" not in active_store.query("astrophysics", n_results=10).ids

        # Reviewer approves -- doc lands back in active with original metadata clean.
        restored = quarantine_store.approve(active_store, ["poison_1"])
        assert restored == ["poison_1"]
        records = active_store.get_by_ids(["poison_1"])
        assert len(records) == 1
        assert QUARANTINED_AT_KEY not in records[0].metadata
        assert QUARANTINE_REASON_KEY not in records[0].metadata
        # Embedding survived the active -> quarantine -> active round trip.
        for a, b in zip(records[0].embedding, decoy_emb[0], strict=False):
            assert abs(a - b) < 1e-5
