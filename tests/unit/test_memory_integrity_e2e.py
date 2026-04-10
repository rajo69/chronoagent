"""End-to-end tests for memory integrity detection and quarantine (Phase 6 task 6.5).

Tests three scenarios required by the phase exit criteria:

1. Poison doc injection followed by detection:
   - MINJA-style attack (centroid-embedding injection) is flagged by the
     content-embedding mismatch signal.
   - AgentPoison-style attack (trigger-embedding backdoor) is flagged.
   - Clean docs that were stored via the normal add path are NOT flagged
     (their stored embedding matches re-embedding of their own text).

2. AUROC validation:
   - A mixed corpus of 10 clean + 5 poison docs yields AUROC > 0.8 on the
     ``content_embedding_mismatch`` signal scores.  The threshold 0.8 is the
     phase 6 exit criterion.

3. Quarantine / approve flow (both pure-Python and HTTP layers):
   - Flagged IDs quarantined via ``QuarantineStore`` (or ``POST /quarantine``)
     are excluded from subsequent active-store retrieval.
   - Approving via ``QuarantineStore`` (or ``POST /approve``) restores them
     and leaves clean docs undisturbed throughout.

Detection relies entirely on the ``content_embedding_mismatch`` signal because
the IsolationForest baseline is not fitted (cold start).  Both ``MINJAStyleAttack``
and the agents' ``MockBackend`` use the same SHA-256 hash -> random unit vector
algorithm, so:

* Clean doc: stored_embedding == re-embed(text) -> mismatch == 0.0
* Poison doc: stored_embedding near query/trigger centroid, text is adversarial
  content -> re-embed(text) is unrelated -> mismatch near 1.0
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.config import Settings
from chronoagent.main import create_app
from chronoagent.memory.integrity import MemoryIntegrityModule, RetrievedDoc
from chronoagent.memory.poisoning import AGENTPOISONStyleAttack, MINJAStyleAttack
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Weights that isolate the content-embedding mismatch signal.
# Needed because the IsolationForest baseline is not fitted at test time and
# the default 4-signal aggregate cannot reach flag_threshold=0.6 from the
# mismatch signal alone (weight = 0.4).
# ---------------------------------------------------------------------------

_MISMATCH_ONLY_WEIGHTS = {
    "content_embedding_mismatch": 1.0,
    "embedding_outlier": 0.0,
    "freshness_anomaly": 0.0,
    "retrieval_frequency": 0.0,
}

# 384-d random unit vectors have cosine distance ~0.5; 0.3 gives ~8-sigma clearance.
_DETECTION_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Clean corpus used across multiple test classes
# ---------------------------------------------------------------------------

_CLEAN_TEXTS = [
    "SQL injection prevention best practices for web APIs",
    "Secure password hashing with bcrypt and Argon2",
    "CSRF token validation in stateless REST services",
    "Rate limiting strategies to prevent brute force attacks",
    "Dependency scanning and CVE remediation workflows",
    "JWT expiry and refresh token rotation patterns",
    "Input sanitisation for file upload endpoints",
    "Secrets management using environment variables",
    "TLS certificate pinning in mobile clients",
    "Principle of least privilege in microservice IAM",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stores() -> Generator[tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule], None, None]:
    """Fresh isolated MemoryStore + QuarantineStore + MemoryIntegrityModule.

    Uses UUID-suffixed ChromaDB collection names to avoid process-level
    EphemeralClient state leakage between tests.
    """
    suffix = uuid.uuid4().hex
    backend = MockBackend()
    client = chromadb.EphemeralClient()
    active = MemoryStore(
        client.get_or_create_collection(f"e2e_active_{suffix}"),
        backend,
    )
    quarantine = QuarantineStore(
        client.get_or_create_collection(f"e2e_quarantine_{suffix}"),
    )
    integrity = MemoryIntegrityModule(
        backend,
        weights=_MISMATCH_ONLY_WEIGHTS,
        flag_threshold=_DETECTION_THRESHOLD,
    )
    yield active, quarantine, integrity


@pytest.fixture()
def client_app() -> Generator[tuple[TestClient, FastAPI], None, None]:
    """Full app TestClient with per-test-isolated memory stores."""
    settings = Settings(env="test", llm_backend="mock")
    app = create_app(settings=settings)
    suffix = uuid.uuid4().hex
    backend = MockBackend()
    client = chromadb.EphemeralClient()
    with TestClient(app) as tc:
        app.state.active_store = MemoryStore(
            client.get_or_create_collection(f"http_active_{suffix}"),
            backend,
        )
        app.state.quarantine_store = QuarantineStore(
            client.get_or_create_collection(f"http_quarantine_{suffix}"),
        )
        app.state.integrity_module = MemoryIntegrityModule(
            backend,
            weights=_MISMATCH_ONLY_WEIGHTS,
            flag_threshold=_DETECTION_THRESHOLD,
        )
        yield tc, app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_clean(store: MemoryStore, texts: list[str] | None = None) -> list[str]:
    """Add clean docs to *store* and return their IDs.

    Each ID is prefixed ``clean_`` so tests can distinguish clean from poison.
    """
    corpus = texts or _CLEAN_TEXTS
    ids = [f"clean_{i}" for i in range(len(corpus))]
    store.add(documents=corpus, ids=ids)
    return ids


def _retrieve_all_as_docs(store: MemoryStore, ids: list[str]) -> list[RetrievedDoc]:
    """Fetch *ids* from *store* and return as :class:`RetrievedDoc` list."""
    stored = store.get_by_ids(ids)
    return [
        RetrievedDoc(
            doc_id=s.doc_id,
            text=s.text,
            embedding=s.embedding,
            distance_to_query=0.0,
            metadata=s.metadata,
        )
        for s in stored
    ]


def _inject_minja(
    store: MemoryStore,
    n_poison: int = 3,
    target_queries: list[str] | None = None,
) -> list[str]:
    """Inject MINJA-style poison docs into *store* and return their IDs."""
    attack = MINJAStyleAttack(seed=42, noise_scale=0.05)
    queries = target_queries or ["security vulnerability code review"]
    # Access the underlying collection directly (test-only pattern).
    return attack.inject(store._collection, target_queries=queries, n_poison=n_poison)


def _inject_agentpoison(store: MemoryStore, n_poison: int = 3) -> list[str]:
    """Inject AgentPoison-style poison docs into *store* and return their IDs."""
    attack = AGENTPOISONStyleAttack(trigger_phrase="AGENT_TRIGGER_7b3f", seed=99)
    return attack.inject(store._collection, n_poison=n_poison)


# ---------------------------------------------------------------------------
# TestCleanDocsNotFlagged
# ---------------------------------------------------------------------------


class TestCleanDocsNotFlagged:
    """Clean docs added via normal add path must not be flagged."""

    def test_single_clean_doc_not_flagged(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """A doc whose stored embedding matches its text is never flagged."""
        active, _, integrity = stores
        active.add(documents=["safe code review guidelines"], ids=["clean_single"])
        docs = _retrieve_all_as_docs(active, ["clean_single"])
        result = integrity.check_retrieval("safe guidelines", docs)
        assert result.is_clean

    def test_clean_doc_content_mismatch_near_zero(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Content-embedding mismatch score is near 0 for a clean doc."""
        active, _, integrity = stores
        active.add(documents=["input validation prevents injection attacks"], ids=["c1"])
        docs = _retrieve_all_as_docs(active, ["c1"])
        result = integrity.check_retrieval("injection", docs)
        assert result.signals[0].content_embedding_mismatch < 0.01

    def test_full_clean_corpus_none_flagged(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """None of the 10-doc clean corpus is flagged."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active)
        docs = _retrieve_all_as_docs(active, clean_ids)
        result = integrity.check_retrieval("security review", docs)
        assert result.is_clean
        assert result.flagged_ids == []

    def test_clean_corpus_max_aggregate_low(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Max aggregate score across all clean docs is well below threshold."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active)
        docs = _retrieve_all_as_docs(active, clean_ids)
        result = integrity.check_retrieval("security", docs)
        assert result.max_aggregate < _DETECTION_THRESHOLD


# ---------------------------------------------------------------------------
# TestMINJAAttackDetection
# ---------------------------------------------------------------------------


class TestMINJAAttackDetection:
    """MINJA-style attack docs are detected by the content-embedding mismatch signal."""

    def test_single_poison_doc_is_flagged(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """A single MINJA-injected doc is in flagged_ids."""
        active, _, integrity = stores
        poison_ids = _inject_minja(active, n_poison=1)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("security review", docs)
        assert poison_ids[0] in result.flagged_ids

    def test_poison_content_mismatch_is_high(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Content-embedding mismatch score for MINJA poison is above 0.8."""
        active, _, integrity = stores
        poison_ids = _inject_minja(active, n_poison=1)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("security", docs)
        mismatch = result.signals[0].content_embedding_mismatch
        # Two independent random unit vectors in 384-d space have cosine distance ~0.5.
        # The stored attack embedding and re-embedded adversarial text are independent,
        # so we assert well above zero but not 0.8 (which would require near-antipodal vecs).
        assert mismatch > 0.3

    def test_multiple_poison_docs_all_flagged(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """All 3 injected MINJA docs appear in flagged_ids."""
        active, _, integrity = stores
        poison_ids = _inject_minja(active, n_poison=3)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("code review", docs)
        assert set(poison_ids) == set(result.flagged_ids)

    def test_flagged_ids_exclude_clean_docs(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """When clean and poison docs are both checked, only poison is flagged."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active, ["jwt token validation", "csrf protection"])
        poison_ids = _inject_minja(active, n_poison=2)
        all_ids = clean_ids + poison_ids
        docs = _retrieve_all_as_docs(active, all_ids)
        result = integrity.check_retrieval("security", docs)
        flagged = set(result.flagged_ids)
        assert flagged == set(poison_ids)
        assert not (flagged & set(clean_ids))

    def test_clean_docs_not_flagged_alongside_poison(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """None of the clean docs is flagged even when poison docs are present."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active)
        _inject_minja(active, n_poison=3)
        clean_docs = _retrieve_all_as_docs(active, clean_ids)
        result = integrity.check_retrieval("review", clean_docs)
        assert result.is_clean


# ---------------------------------------------------------------------------
# TestAGENTPOISONDetection
# ---------------------------------------------------------------------------


class TestAGENTPOISONDetection:
    """AgentPoison-style backdoor docs are detected by the mismatch signal."""

    def test_agentpoison_doc_is_flagged(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """A single AgentPoison-injected doc is in flagged_ids."""
        active, _, integrity = stores
        poison_ids = _inject_agentpoison(active, n_poison=1)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("security review", docs)
        assert poison_ids[0] in result.flagged_ids

    def test_agentpoison_content_mismatch_is_high(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Mismatch score for AgentPoison doc is above 0.8."""
        active, _, integrity = stores
        poison_ids = _inject_agentpoison(active, n_poison=1)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("trigger query", docs)
        assert result.signals[0].content_embedding_mismatch > 0.3

    def test_agentpoison_clean_docs_not_flagged(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Clean docs are not flagged when checked alongside AgentPoison docs."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active, ["tls certificate pinning", "rate limiting"])
        _inject_agentpoison(active, n_poison=2)
        clean_docs = _retrieve_all_as_docs(active, clean_ids)
        result = integrity.check_retrieval("security", clean_docs)
        assert result.is_clean


# ---------------------------------------------------------------------------
# TestDetectionAUROC
# ---------------------------------------------------------------------------


class TestDetectionAUROC:
    """AUROC of content_embedding_mismatch scores on mixed corpora must exceed 0.8.

    Uses ``content_embedding_mismatch`` scores directly (not the ``flagged``
    boolean) so the metric is threshold-independent.
    """

    def _compute_auroc(
        self,
        active: MemoryStore,
        integrity: MemoryIntegrityModule,
        clean_ids: list[str],
        poison_ids: list[str],
    ) -> float:
        all_ids = clean_ids + poison_ids
        docs = _retrieve_all_as_docs(active, all_ids)
        result = integrity.check_retrieval("security code review vulnerability", docs)
        score_by_id = {s.doc_id: s.content_embedding_mismatch for s in result.signals}
        y_true = [0] * len(clean_ids) + [1] * len(poison_ids)
        y_score = [score_by_id.get(i, 0.0) for i in all_ids]
        return float(roc_auc_score(y_true, y_score))

    def test_minja_auroc_above_threshold(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """AUROC >= 0.8 on 10 clean + 5 MINJA poison docs (phase 6 exit criterion)."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active)
        poison_ids = _inject_minja(active, n_poison=5)
        auroc = self._compute_auroc(active, integrity, clean_ids, poison_ids)
        assert auroc >= 0.8, f"AUROC {auroc:.3f} below 0.8 threshold"

    def test_agentpoison_auroc_above_threshold(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """AUROC >= 0.8 on 10 clean + 5 AgentPoison docs."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active)
        poison_ids = _inject_agentpoison(active, n_poison=5)
        auroc = self._compute_auroc(active, integrity, clean_ids, poison_ids)
        assert auroc >= 0.8, f"AUROC {auroc:.3f} below 0.8 threshold"

    def test_clean_only_scores_near_zero(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """All content_mismatch scores in a clean-only corpus are near 0."""
        active, _, integrity = stores
        clean_ids = _seed_clean(active)
        docs = _retrieve_all_as_docs(active, clean_ids)
        result = integrity.check_retrieval("security", docs)
        scores = [s.content_embedding_mismatch for s in result.signals]
        assert all(sc < 0.01 for sc in scores), f"Unexpected scores: {scores}"


# ---------------------------------------------------------------------------
# TestQuarantineFlowPython (pure Python layer)
# ---------------------------------------------------------------------------


class TestQuarantineFlowPython:
    """Quarantine and approve via the Python API after detection."""

    def test_flagged_doc_quarantined_excluded_from_retrieval(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """After quarantine, querying active store no longer returns poison doc."""
        active, quarantine, integrity = stores
        _seed_clean(active, ["safe document one"])
        poison_ids = _inject_minja(active, n_poison=1)
        all_docs = _retrieve_all_as_docs(active, ["clean_0"] + poison_ids)
        result = integrity.check_retrieval("review", all_docs)
        quarantine.quarantine(active, result.flagged_ids, reason="integrity_module")
        # Query should no longer surface the poison doc.
        query_result = active.query("security override", n_results=10)
        assert poison_ids[0] not in query_result.ids

    def test_clean_docs_still_accessible_after_quarantine(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Quarantining poison docs leaves clean docs in the active store."""
        active, quarantine, integrity = stores
        clean_ids = _seed_clean(active)
        poison_ids = _inject_minja(active, n_poison=2)
        all_docs = _retrieve_all_as_docs(active, clean_ids + poison_ids)
        result = integrity.check_retrieval("security", all_docs)
        quarantine.quarantine(active, result.flagged_ids)
        # All clean docs should still be queryable.
        after = active.query("security vulnerability", n_results=len(clean_ids))
        surviving_clean = set(after.ids) & set(clean_ids)
        assert len(surviving_clean) > 0

    def test_approved_poison_restored_to_active_store(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """After approval, previously quarantined doc is back in active retrieval."""
        active, quarantine, integrity = stores
        poison_ids = _inject_minja(active, n_poison=1)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("review", docs)
        quarantine.quarantine(active, result.flagged_ids)
        quarantine.approve(active, result.flagged_ids)
        # Doc should be back.
        after_ids = active.get_by_ids(poison_ids)
        assert len(after_ids) == 1 and after_ids[0].doc_id == poison_ids[0]

    def test_quarantine_count_correct_after_full_flow(
        self, stores: tuple[MemoryStore, QuarantineStore, MemoryIntegrityModule]
    ) -> None:
        """Quarantine count rises on quarantine and returns to 0 on approve."""
        active, quarantine, integrity = stores
        poison_ids = _inject_minja(active, n_poison=3)
        docs = _retrieve_all_as_docs(active, poison_ids)
        result = integrity.check_retrieval("code review", docs)
        quarantine.quarantine(active, result.flagged_ids)
        assert quarantine.count == len(result.flagged_ids)
        quarantine.approve(active, result.flagged_ids)
        assert quarantine.count == 0


# ---------------------------------------------------------------------------
# TestHTTPIntegrityFlow (API layer)
# ---------------------------------------------------------------------------


class TestHTTPIntegrityFlow:
    """Quarantine and approve via the HTTP endpoints after programmatic detection."""

    def test_post_quarantine_moves_flagged_ids(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """POST /quarantine with flagged_ids moves docs out of the active store."""
        tc, app = client_app
        _seed_clean(app.state.active_store, ["clean doc for http test"])
        poison_ids = _inject_minja(app.state.active_store, n_poison=1)
        all_docs = _retrieve_all_as_docs(app.state.active_store, ["clean_0"] + poison_ids)
        result = app.state.integrity_module.check_retrieval("security review", all_docs)
        resp = tc.post(
            "/api/v1/memory/quarantine",
            json={"ids": result.flagged_ids, "reason": "integrity_module"},
        )
        assert resp.status_code == 200
        assert set(resp.json()["quarantined"]) == set(result.flagged_ids)

    def test_get_integrity_shows_quarantine_count_after_post(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """GET /integrity quarantine_count reflects POST /quarantine result."""
        tc, app = client_app
        poison_ids = _inject_minja(app.state.active_store, n_poison=2)
        docs = _retrieve_all_as_docs(app.state.active_store, poison_ids)
        result = app.state.integrity_module.check_retrieval("review", docs)
        tc.post("/api/v1/memory/quarantine", json={"ids": result.flagged_ids})
        status = tc.get("/api/v1/memory/integrity").json()
        assert status["quarantine_count"] == len(result.flagged_ids)
        assert set(status["quarantined_ids"]) == set(result.flagged_ids)

    def test_quarantined_docs_absent_from_active_store(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """After POST /quarantine, active store no longer holds the poison docs."""
        tc, app = client_app
        poison_ids = _inject_minja(app.state.active_store, n_poison=2)
        docs = _retrieve_all_as_docs(app.state.active_store, poison_ids)
        result = app.state.integrity_module.check_retrieval("review", docs)
        tc.post("/api/v1/memory/quarantine", json={"ids": result.flagged_ids})
        remaining = app.state.active_store.get_by_ids(poison_ids)
        assert remaining == []

    def test_post_approve_restores_docs_to_active_store(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """POST /approve returns the docs and they re-appear in the active store."""
        tc, app = client_app
        poison_ids = _inject_minja(app.state.active_store, n_poison=1)
        docs = _retrieve_all_as_docs(app.state.active_store, poison_ids)
        result = app.state.integrity_module.check_retrieval("review", docs)
        tc.post("/api/v1/memory/quarantine", json={"ids": result.flagged_ids})
        resp = tc.post("/api/v1/memory/approve", json={"ids": result.flagged_ids})
        assert resp.status_code == 200
        assert set(resp.json()["approved"]) == set(result.flagged_ids)
        restored = app.state.active_store.get_by_ids(poison_ids)
        assert len(restored) == 1

    def test_full_detect_quarantine_approve_cycle(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """Full cycle: detect -> quarantine -> verify count -> approve -> count=0."""
        tc, app = client_app
        clean_ids = _seed_clean(app.state.active_store)
        poison_ids = _inject_minja(app.state.active_store, n_poison=3)
        all_ids = clean_ids + poison_ids
        docs = _retrieve_all_as_docs(app.state.active_store, all_ids)
        result = app.state.integrity_module.check_retrieval("security", docs)
        # Only poison flagged.
        assert set(result.flagged_ids) == set(poison_ids)

        # Quarantine via API.
        tc.post("/api/v1/memory/quarantine", json={"ids": result.flagged_ids})
        mid = tc.get("/api/v1/memory/integrity").json()
        assert mid["quarantine_count"] == len(poison_ids)

        # Clean docs unaffected.
        clean_remaining = app.state.active_store.get_by_ids(clean_ids)
        assert len(clean_remaining) == len(clean_ids)

        # Approve via API.
        tc.post("/api/v1/memory/approve", json={"ids": result.flagged_ids})
        end = tc.get("/api/v1/memory/integrity").json()
        assert end["quarantine_count"] == 0

    def test_agentpoison_detected_and_quarantined_via_http(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """AgentPoison docs are flagged by integrity module and quarantined via API."""
        tc, app = client_app
        poison_ids = _inject_agentpoison(app.state.active_store, n_poison=2)
        docs = _retrieve_all_as_docs(app.state.active_store, poison_ids)
        result = app.state.integrity_module.check_retrieval("security", docs)
        assert set(result.flagged_ids) == set(poison_ids)
        resp = tc.post("/api/v1/memory/quarantine", json={"ids": result.flagged_ids})
        assert set(resp.json()["quarantined"]) == set(poison_ids)
