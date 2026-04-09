"""Unit tests for MemoryStore (memory/store.py)."""

from __future__ import annotations

import uuid

import chromadb
import pytest

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.memory.store import MemoryStore, QueryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def store(backend: MockBackend) -> MemoryStore:
    client = chromadb.EphemeralClient()
    # Use a unique name per test to avoid clashes with the shared EphemeralClient.
    collection = client.get_or_create_collection(f"test_store_{uuid.uuid4().hex[:8]}")
    return MemoryStore(collection=collection, backend=backend)


def _seed(store: MemoryStore, n: int = 5) -> list[str]:
    """Add *n* simple documents and return their IDs."""
    docs = [f"document about topic {i}" for i in range(n)]
    ids = [f"doc_{i}" for i in range(n)]
    store.add(documents=docs, ids=ids)
    return ids


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_empty_store_count_zero(self, store: MemoryStore) -> None:
        assert store.count == 0

    def test_count_reflects_adds(self, store: MemoryStore) -> None:
        _seed(store, 3)
        assert store.count == 3


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_documents_increments_count(self, store: MemoryStore) -> None:
        store.add(documents=["hello world"], ids=["id1"])
        assert store.count == 1

    def test_add_multiple(self, store: MemoryStore) -> None:
        _seed(store, 5)
        assert store.count == 5

    def test_add_with_metadatas(self, store: MemoryStore) -> None:
        store.add(
            documents=["doc with meta"],
            ids=["m1"],
            metadatas=[{"source": "unit-test"}],
        )
        assert store.count == 1

    def test_add_with_precomputed_embeddings(
        self, store: MemoryStore, backend: MockBackend
    ) -> None:
        emb = backend.embed(["custom doc"])
        store.add(documents=["custom doc"], ids=["e1"], embeddings=emb)
        assert store.count == 1

    def test_upsert_overwrites_existing_id(self, store: MemoryStore) -> None:
        store.add(documents=["original"], ids=["dup"])
        store.add(documents=["updated"], ids=["dup"])
        # Count stays at 1 after upsert.
        assert store.count == 1

    def test_mismatched_docs_ids_raises(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError, match="equal length"):
            store.add(documents=["a", "b"], ids=["only_one"])

    def test_mismatched_embeddings_raises(
        self, store: MemoryStore, backend: MockBackend
    ) -> None:
        emb = backend.embed(["vec"])
        with pytest.raises(ValueError, match="embeddings length"):
            store.add(documents=["a", "b"], ids=["x", "y"], embeddings=emb)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_query_result(self, store: MemoryStore) -> None:
        _seed(store)
        result = store.query("topic 2", n_results=3)
        assert isinstance(result, QueryResult)

    def test_query_respects_n_results(self, store: MemoryStore) -> None:
        _seed(store, 10)
        result = store.query("topic", n_results=4)
        assert len(result.documents) == 4

    def test_query_clamps_to_count(self, store: MemoryStore) -> None:
        _seed(store, 2)
        result = store.query("topic", n_results=100)
        assert len(result.documents) == 2

    def test_query_empty_store_returns_empty(self, store: MemoryStore) -> None:
        result = store.query("anything")
        assert result.documents == []
        assert result.distances == []
        assert result.ids == []
        assert result.metadatas == []

    def test_query_ids_aligned_with_documents(self, store: MemoryStore) -> None:
        _seed(store, 5)
        result = store.query("topic 3", n_results=3)
        assert len(result.ids) == len(result.documents)

    def test_query_distances_are_floats(self, store: MemoryStore) -> None:
        _seed(store, 3)
        result = store.query("topic 0", n_results=2)
        assert all(isinstance(d, float) for d in result.distances)

    def test_query_metadatas_returned(self, store: MemoryStore) -> None:
        store.add(
            documents=["meta doc"],
            ids=["md1"],
            metadatas=[{"tag": "test"}],
        )
        result = store.query("meta", n_results=1)
        assert result.metadatas[0].get("tag") == "test"

    def test_query_custom_include(self, store: MemoryStore) -> None:
        _seed(store, 3)
        result = store.query("topic", n_results=2, include=["documents", "distances"])
        assert len(result.documents) == 2


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_documents(self, store: MemoryStore) -> None:
        ids = _seed(store, 5)
        removed = store.delete([ids[0], ids[1]])
        assert removed == 2
        assert store.count == 3

    def test_delete_empty_list_returns_zero(self, store: MemoryStore) -> None:
        _seed(store, 3)
        assert store.delete([]) == 0

    def test_delete_all(self, store: MemoryStore) -> None:
        ids = _seed(store, 4)
        store.delete(ids)
        assert store.count == 0

    def test_delete_returns_submitted_count(self, store: MemoryStore) -> None:
        _seed(store, 3)
        # Attempt to delete a non-existent ID alongside a real one.
        n = store.delete(["doc_0", "nonexistent_xyz"])
        assert n == 2  # returns len(ids), not matched count

    def test_deleted_docs_not_retrieved(self, store: MemoryStore) -> None:
        store.add(documents=["target doc"], ids=["target"])
        store.add(documents=["other doc"], ids=["other"])
        store.delete(["target"])
        result = store.query("target", n_results=5)
        assert "target" not in result.ids


# ---------------------------------------------------------------------------
# get_all_embeddings
# ---------------------------------------------------------------------------


class TestGetAllEmbeddings:
    def test_empty_returns_empty_list(self, store: MemoryStore) -> None:
        assert store.get_all_embeddings() == []

    def test_returns_one_vector_per_document(self, store: MemoryStore) -> None:
        _seed(store, 4)
        vecs = store.get_all_embeddings()
        assert len(vecs) == 4

    def test_each_vector_is_list_of_floats(self, store: MemoryStore) -> None:
        _seed(store, 2)
        vecs = store.get_all_embeddings()
        for vec in vecs:
            assert isinstance(vec, list)
            assert all(isinstance(v, float) for v in vec)

    def test_embedding_dimension_consistent(
        self, store: MemoryStore, backend: MockBackend
    ) -> None:
        _seed(store, 3)
        vecs = store.get_all_embeddings()
        expected_dim = len(backend.embed(["any"])[0])
        assert all(len(v) == expected_dim for v in vecs)

    def test_precomputed_embeddings_round_trip(
        self, store: MemoryStore, backend: MockBackend
    ) -> None:
        original = backend.embed(["round trip doc"])
        store.add(documents=["round trip doc"], ids=["rt1"], embeddings=original)
        vecs = store.get_all_embeddings()
        assert len(vecs) == 1
        # Values should be numerically close.
        for a, b in zip(vecs[0], original[0]):
            assert abs(a - b) < 1e-5
