"""Tests for MINJAStyleAttack and AGENTPOISONStyleAttack (task 1.5)."""

from __future__ import annotations

import chromadb
import numpy as np
import pytest

from chronoagent.agents.base import BaseAgent
from chronoagent.memory.poisoning import AGENTPOISONStyleAttack, MINJAStyleAttack

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def collection() -> chromadb.Collection:
    """Ephemeral in-memory ChromaDB collection with 10 seed documents."""
    client = chromadb.EphemeralClient()
    seed_docs = [
        "Check for SQL injection vulnerabilities in user-facing endpoints.",
        "Validate CSRF token presence on all state-modifying requests.",
        "Ensure secrets are not committed to the repository.",
        "Audit dependency versions against known CVE databases.",
        "Review authentication and session management logic.",
        "Confirm rate limiting is in place on auth endpoints.",
        "Verify file upload handlers restrict paths and types.",
        "Check for XSS via unsanitized template variables.",
        "Review JWT algorithm enforcement to prevent confusion attacks.",
        "Ensure verbose error messages do not leak stack traces.",
    ]
    col = BaseAgent.build_collection(
        client=client,
        name="test_collection",
        documents=seed_docs,
    )
    return col  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MINJAStyleAttack
# ---------------------------------------------------------------------------


class TestMINJAStyleAttack:
    def test_inject_returns_expected_count(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack(seed=0)
        ids = attack.inject(
            collection,
            target_queries=["security review findings"],
            n_poison=3,
        )
        assert len(ids) == 3

    def test_ids_added_to_collection(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack(seed=1)
        initial_count = collection.count()
        ids = attack.inject(
            collection,
            target_queries=["vulnerability scan"],
            n_poison=4,
        )
        assert collection.count() == initial_count + 4
        assert attack.injected_ids == ids

    def test_unique_ids_per_inject_call(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack(seed=2)
        ids_a = attack.inject(collection, target_queries=["q1"], n_poison=3)
        ids_b = attack.inject(collection, target_queries=["q2"], n_poison=3)
        assert len(set(ids_a) & set(ids_b)) == 0, "IDs must be globally unique"

    def test_rollback_removes_injected_docs(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack(seed=3)
        initial_count = collection.count()
        attack.inject(collection, target_queries=["code review"], n_poison=5)
        removed = attack.rollback(collection)
        assert removed == 5
        assert collection.count() == initial_count
        assert attack.injected_ids == []

    def test_rollback_idempotent_when_empty(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack()
        assert attack.rollback(collection) == 0

    def test_empty_target_queries_raises(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack()
        with pytest.raises(ValueError, match="target_queries"):
            attack.inject(collection, target_queries=[], n_poison=1)

    def test_custom_malicious_content(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack(seed=4)
        content = ["EVIL CONTENT A", "EVIL CONTENT B"]
        ids = attack.inject(
            collection,
            target_queries=["auth review"],
            malicious_content=content,
            n_poison=4,
        )
        # Verify injected docs are retrievable and contain the malicious content.
        result = collection.get(ids=ids, include=["documents"])
        docs: list[str] = (result.get("documents") or [])
        assert all(d in content for d in docs)

    def test_poison_embedding_is_close_to_query(self, collection: chromadb.Collection) -> None:
        """Injected embedding should be closer to the target query than a random doc."""
        from chronoagent.llm.mock_backend import MockEmbeddingFunction

        embed_fn = MockEmbeddingFunction()
        query = "detect SQL injection in user input"
        query_emb = np.array(embed_fn([query])[0], dtype=np.float64)

        attack = MINJAStyleAttack(seed=7, noise_scale=0.01)
        ids = attack.inject(collection, target_queries=[query], n_poison=1)

        # Retrieve injected doc's embedding from ChromaDB.
        result = collection.get(ids=ids, include=["embeddings"])
        embeddings = result.get("embeddings")
        assert embeddings is not None and len(embeddings) > 0
        poison_emb = np.array(embeddings[0], dtype=np.float64)

        # Cosine similarity: unit vectors, so dot product == cosine sim.
        sim = float(np.dot(query_emb, poison_emb))
        # With noise_scale=0.01, similarity should be very high (> 0.90).
        assert sim > 0.90, f"Expected cosine sim > 0.90, got {sim:.3f}"

    def test_repr_shows_state(self, collection: chromadb.Collection) -> None:
        attack = MINJAStyleAttack(seed=5, noise_scale=0.1)
        attack.inject(collection, target_queries=["q"], n_poison=2)
        r = repr(attack)
        assert "MINJAStyleAttack" in r
        assert "n_injected=2" in r


# ---------------------------------------------------------------------------
# AGENTPOISONStyleAttack
# ---------------------------------------------------------------------------


class TestAGENTPOISONStyleAttack:
    def test_inject_returns_expected_count(self, collection: chromadb.Collection) -> None:
        attack = AGENTPOISONStyleAttack(trigger_phrase="TRG_001", seed=10)
        ids = attack.inject(collection, n_poison=3)
        assert len(ids) == 3

    def test_ids_added_to_collection(self, collection: chromadb.Collection) -> None:
        attack = AGENTPOISONStyleAttack(trigger_phrase="TRG_002", seed=11)
        initial_count = collection.count()
        ids = attack.inject(collection, n_poison=4)
        assert collection.count() == initial_count + 4
        assert attack.injected_ids == ids

    def test_rollback_removes_injected_docs(self, collection: chromadb.Collection) -> None:
        attack = AGENTPOISONStyleAttack(trigger_phrase="TRG_003", seed=12)
        initial_count = collection.count()
        attack.inject(collection, n_poison=3)
        removed = attack.rollback(collection)
        assert removed == 3
        assert collection.count() == initial_count
        assert attack.injected_ids == []

    def test_rollback_idempotent_when_empty(self, collection: chromadb.Collection) -> None:
        attack = AGENTPOISONStyleAttack()
        assert attack.rollback(collection) == 0

    def test_metadata_stored(self, collection: chromadb.Collection) -> None:
        trigger = "TRIG_META"
        attack = AGENTPOISONStyleAttack(trigger_phrase=trigger, seed=13)
        ids = attack.inject(collection, n_poison=2, metadata={"phase": "1"})
        result = collection.get(ids=ids, include=["metadatas"])
        metas: list[dict[str, object]] = result.get("metadatas") or []
        for m in metas:
            assert m.get("attack") == "agentpoison"
            assert m.get("trigger") == trigger
            assert m.get("phase") == "1"

    def test_custom_malicious_content(self, collection: chromadb.Collection) -> None:
        attack = AGENTPOISONStyleAttack(trigger_phrase="TRG_CONTENT", seed=14)
        content = ["BACKDOOR DOC"]
        ids = attack.inject(collection, malicious_content=content, n_poison=3)
        result = collection.get(ids=ids, include=["documents"])
        docs: list[str] = result.get("documents") or []
        assert all(d == "BACKDOOR DOC" for d in docs)

    def test_triggered_query_prepends_trigger(self) -> None:
        attack = AGENTPOISONStyleAttack(trigger_phrase="MAGIC_TRIGGER")
        q = attack.triggered_query("review this PR")
        assert q == "MAGIC_TRIGGER review this PR"

    def test_poison_embedding_is_close_to_trigger(self, collection: chromadb.Collection) -> None:
        """Injected embedding should be near the trigger phrase embedding."""
        from chronoagent.llm.mock_backend import MockEmbeddingFunction

        embed_fn = MockEmbeddingFunction()
        trigger = "UNIQUE_TRIGGER_XYZ"
        trigger_emb = np.array(embed_fn([trigger])[0], dtype=np.float64)

        attack = AGENTPOISONStyleAttack(trigger_phrase=trigger, seed=20, noise_scale=0.01)
        ids = attack.inject(collection, n_poison=1)

        result = collection.get(ids=ids, include=["embeddings"])
        embeddings = result.get("embeddings")
        assert embeddings is not None and len(embeddings) > 0
        poison_emb = np.array(embeddings[0], dtype=np.float64)

        sim = float(np.dot(trigger_emb, poison_emb))
        assert sim > 0.90, f"Expected cosine sim > 0.90, got {sim:.3f}"

    def test_unique_ids_across_attacks(self, collection: chromadb.Collection) -> None:
        a1 = AGENTPOISONStyleAttack(trigger_phrase="T1", seed=30)
        a2 = AGENTPOISONStyleAttack(trigger_phrase="T2", seed=31)
        ids1 = a1.inject(collection, n_poison=3)
        ids2 = a2.inject(collection, n_poison=3)
        assert len(set(ids1) & set(ids2)) == 0

    def test_repr_shows_state(self, collection: chromadb.Collection) -> None:
        attack = AGENTPOISONStyleAttack(trigger_phrase="TRG_REPR", seed=40)
        attack.inject(collection, n_poison=2)
        r = repr(attack)
        assert "AGENTPOISONStyleAttack" in r
        assert "TRG_REPR" in r
        assert "n_injected=2" in r


# ---------------------------------------------------------------------------
# Combined / interaction tests
# ---------------------------------------------------------------------------


class TestCombinedAttacks:
    def test_both_attacks_can_coexist(self, collection: chromadb.Collection) -> None:
        minja = MINJAStyleAttack(seed=50)
        agentpoison = AGENTPOISONStyleAttack(trigger_phrase="TRG_COMBO", seed=51)
        initial = collection.count()

        minja.inject(collection, target_queries=["security"], n_poison=3)
        agentpoison.inject(collection, n_poison=3)
        assert collection.count() == initial + 6

        minja.rollback(collection)
        agentpoison.rollback(collection)
        assert collection.count() == initial

    def test_rollback_is_scoped_per_attack_instance(
        self, collection: chromadb.Collection
    ) -> None:
        a1 = MINJAStyleAttack(seed=60)
        a2 = MINJAStyleAttack(seed=61)
        initial = collection.count()

        a1.inject(collection, target_queries=["q"], n_poison=2)
        ids2 = a2.inject(collection, target_queries=["q"], n_poison=3)

        a1.rollback(collection)
        # ids1 gone, ids2 still there
        assert collection.count() == initial + 3
        result = collection.get(ids=ids2)
        assert len(result.get("ids") or []) == 3
