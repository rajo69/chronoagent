"""Unit tests for the memory integrity and quarantine router (Phase 6 task 6.4).

Covers:
    GET  /api/v1/memory/integrity
    POST /api/v1/memory/quarantine
    POST /api/v1/memory/approve
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chronoagent.agents.backends.mock import MockBackend
from chronoagent.config import Settings
from chronoagent.main import create_app
from chronoagent.memory.integrity import MemoryIntegrityModule
from chronoagent.memory.quarantine import QuarantineStore
from chronoagent.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_app() -> Generator[tuple[TestClient, FastAPI], None, None]:
    """TestClient backed by a fresh app with MockBackend and isolated stores.

    ChromaDB's EphemeralClient uses a process-level shared segment manager, so
    collections named ``"memory_active"`` / ``"memory_quarantine"`` accumulate
    state across test instances within the same pytest session.  To guarantee
    full isolation, the fixture replaces the lifespan-created stores with fresh
    ones backed by UUID-suffixed collection names that no prior test has touched.

    Yields both the TestClient and the FastAPI app so tests can seed
    ``app.state.active_store`` directly without going through HTTP.
    """
    settings = Settings(env="test", llm_backend="mock")
    app = create_app(settings=settings)
    suffix = uuid.uuid4().hex
    memory_backend = MockBackend()
    chroma_client = chromadb.EphemeralClient()
    with TestClient(app) as tc:
        # Replace lifespan stores with per-test-isolated instances.
        app.state.active_store = MemoryStore(
            chroma_client.get_or_create_collection(f"test_active_{suffix}"),
            memory_backend,
        )
        app.state.quarantine_store = QuarantineStore(
            chroma_client.get_or_create_collection(f"test_quarantine_{suffix}"),
        )
        app.state.integrity_module = MemoryIntegrityModule(memory_backend)
        yield tc, app


def _seed(app: FastAPI, doc_id: str, text: str = "hello world") -> None:
    """Add a single document to the active store for test setup."""
    app.state.active_store.add(documents=[text], ids=[doc_id])


# ---------------------------------------------------------------------------
# GET /api/v1/memory/integrity
# ---------------------------------------------------------------------------


class TestGetIntegrityStatus:
    """GET /api/v1/memory/integrity -- read-only status endpoint."""

    def test_returns_200(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Status endpoint returns HTTP 200."""
        tc, _ = client_app
        assert tc.get("/api/v1/memory/integrity").status_code == 200

    def test_response_shape(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Response contains all expected top-level keys."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        expected_keys = {
            "baseline_fitted",
            "baseline_size",
            "pending_refit_count",
            "total_retrievals",
            "flag_threshold",
            "weights",
            "quarantine_count",
            "quarantined_ids",
        }
        assert expected_keys <= body.keys()

    def test_initial_baseline_not_fitted(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Freshly created module has no fitted baseline."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["baseline_fitted"] is False

    def test_initial_baseline_size_zero(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Baseline size is 0 before fitting."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["baseline_size"] == 0

    def test_initial_quarantine_count_zero(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """No documents are quarantined initially."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["quarantine_count"] == 0

    def test_initial_quarantined_ids_empty(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """quarantined_ids is an empty list initially."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["quarantined_ids"] == []

    def test_initial_total_retrievals_zero(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """total_retrievals is 0 before any check_retrieval calls."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["total_retrievals"] == 0

    def test_weights_sum_to_one(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Signal weights are normalised and sum to 1.0."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        weights: dict[str, float] = body["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_contain_all_signals(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """weights dict contains all four signal keys."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        expected = {
            "embedding_outlier",
            "freshness_anomaly",
            "retrieval_frequency",
            "content_embedding_mismatch",
        }
        assert expected == set(body["weights"].keys())

    def test_flag_threshold_in_range(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """flag_threshold is in [0, 1]."""
        tc, _ = client_app
        body = tc.get("/api/v1/memory/integrity").json()
        assert 0.0 <= body["flag_threshold"] <= 1.0

    def test_quarantine_count_reflects_quarantined_doc(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """quarantine_count increments after a document is quarantined."""
        tc, app = client_app
        _seed(app, "q_status_1", "poison doc")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["q_status_1"]})
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["quarantine_count"] == 1

    def test_quarantined_ids_reflects_quarantined_doc(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """quarantined_ids contains the ID of a quarantined document."""
        tc, app = client_app
        _seed(app, "q_status_2", "suspicious doc")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["q_status_2"]})
        body = tc.get("/api/v1/memory/integrity").json()
        assert "q_status_2" in body["quarantined_ids"]


# ---------------------------------------------------------------------------
# POST /api/v1/memory/quarantine
# ---------------------------------------------------------------------------


class TestQuarantineDocs:
    """POST /api/v1/memory/quarantine -- move docs from active store."""

    def test_returns_200(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Quarantine endpoint returns HTTP 200."""
        tc, _ = client_app
        assert tc.post("/api/v1/memory/quarantine", json={"ids": []}).status_code == 200

    def test_empty_ids_returns_empty_list(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Empty ids list produces an empty quarantined list."""
        tc, _ = client_app
        body = tc.post("/api/v1/memory/quarantine", json={"ids": []}).json()
        assert body["quarantined"] == []

    def test_nonexistent_id_returns_empty_list(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """ID not in active store is silently skipped."""
        tc, _ = client_app
        body = tc.post("/api/v1/memory/quarantine", json={"ids": ["ghost_doc_999"]}).json()
        assert body["quarantined"] == []

    def test_quarantines_real_doc(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """A document present in the active store is moved to quarantine."""
        tc, app = client_app
        _seed(app, "qtest_1")
        body = tc.post("/api/v1/memory/quarantine", json={"ids": ["qtest_1"]}).json()
        assert body["quarantined"] == ["qtest_1"]

    def test_quarantined_doc_removed_from_active_store(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """After quarantine the document is absent from the active store."""
        tc, app = client_app
        _seed(app, "qtest_2")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["qtest_2"]})
        # Active store should no longer contain the document.
        results = app.state.active_store.query("hello world", n_results=10)
        assert "qtest_2" not in results.ids

    def test_idempotent_second_quarantine(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Quarantining the same ID twice is idempotent: second call returns []."""
        tc, app = client_app
        _seed(app, "qtest_3")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["qtest_3"]})
        body = tc.post("/api/v1/memory/quarantine", json={"ids": ["qtest_3"]}).json()
        assert body["quarantined"] == []

    def test_with_reason(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Reason is accepted and the doc lands in quarantine."""
        tc, app = client_app
        _seed(app, "qtest_4")
        body = tc.post(
            "/api/v1/memory/quarantine",
            json={"ids": ["qtest_4"], "reason": "integrity_module"},
        ).json()
        assert body["quarantined"] == ["qtest_4"]
        # Reason should be stored as quarantine metadata.
        doc = app.state.quarantine_store.get_doc("qtest_4")
        assert doc is not None
        assert doc.metadata.get("quarantine_reason") == "integrity_module"

    def test_multiple_ids(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Multiple IDs are all quarantined in a single call."""
        tc, app = client_app
        _seed(app, "qtest_5a")
        _seed(app, "qtest_5b")
        body = tc.post("/api/v1/memory/quarantine", json={"ids": ["qtest_5a", "qtest_5b"]}).json()
        assert set(body["quarantined"]) == {"qtest_5a", "qtest_5b"}

    def test_mixed_existing_and_missing_ids(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Only existing IDs are quarantined; missing IDs are silently skipped."""
        tc, app = client_app
        _seed(app, "qtest_6_real")
        body = tc.post(
            "/api/v1/memory/quarantine",
            json={"ids": ["qtest_6_real", "qtest_6_ghost"]},
        ).json()
        assert body["quarantined"] == ["qtest_6_real"]

    def test_quarantine_without_reason(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Omitting reason is valid; quarantined_at is still stamped on the doc."""
        tc, app = client_app
        _seed(app, "qtest_7")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["qtest_7"]})
        doc = app.state.quarantine_store.get_doc("qtest_7")
        assert doc is not None
        assert "quarantined_at" in doc.metadata


# ---------------------------------------------------------------------------
# POST /api/v1/memory/approve
# ---------------------------------------------------------------------------


class TestApproveDocs:
    """POST /api/v1/memory/approve -- restore quarantined docs to active store."""

    def test_returns_200(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Approve endpoint returns HTTP 200."""
        tc, _ = client_app
        assert tc.post("/api/v1/memory/approve", json={"ids": []}).status_code == 200

    def test_empty_ids_returns_empty_list(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Empty ids list produces an empty approved list."""
        tc, _ = client_app
        body = tc.post("/api/v1/memory/approve", json={"ids": []}).json()
        assert body["approved"] == []

    def test_nonexistent_id_returns_empty_list(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """ID not in quarantine is silently skipped."""
        tc, _ = client_app
        body = tc.post("/api/v1/memory/approve", json={"ids": ["ghost_999"]}).json()
        assert body["approved"] == []

    def test_approves_quarantined_doc(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """A quarantined document is restored to the active store."""
        tc, app = client_app
        _seed(app, "atest_1")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["atest_1"]})
        body = tc.post("/api/v1/memory/approve", json={"ids": ["atest_1"]}).json()
        assert body["approved"] == ["atest_1"]

    def test_approved_doc_removed_from_quarantine(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """After approval the document is no longer in quarantine."""
        tc, app = client_app
        _seed(app, "atest_2")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["atest_2"]})
        tc.post("/api/v1/memory/approve", json={"ids": ["atest_2"]})
        assert app.state.quarantine_store.get_doc("atest_2") is None

    def test_approved_doc_back_in_active_store(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """Approved document is queryable from the active store again."""
        tc, app = client_app
        _seed(app, "atest_3", text="restore me")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["atest_3"]})
        tc.post("/api/v1/memory/approve", json={"ids": ["atest_3"]})
        results = app.state.active_store.query("restore me", n_results=5)
        assert "atest_3" in results.ids

    def test_idempotent_second_approve(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Approving the same ID twice is idempotent: second call returns []."""
        tc, app = client_app
        _seed(app, "atest_4")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["atest_4"]})
        tc.post("/api/v1/memory/approve", json={"ids": ["atest_4"]})
        body = tc.post("/api/v1/memory/approve", json={"ids": ["atest_4"]}).json()
        assert body["approved"] == []

    def test_approve_multiple_ids(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """Multiple quarantined IDs are all approved in a single call."""
        tc, app = client_app
        _seed(app, "atest_5a")
        _seed(app, "atest_5b")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["atest_5a", "atest_5b"]})
        body = tc.post("/api/v1/memory/approve", json={"ids": ["atest_5a", "atest_5b"]}).json()
        assert set(body["approved"]) == {"atest_5a", "atest_5b"}

    def test_approve_strips_quarantine_metadata(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """After approval the restored doc carries no quarantine metadata."""
        tc, app = client_app
        _seed(app, "atest_6")
        tc.post(
            "/api/v1/memory/quarantine",
            json={"ids": ["atest_6"], "reason": "test_reason"},
        )
        tc.post("/api/v1/memory/approve", json={"ids": ["atest_6"]})
        # Fetch from active store by querying; presence in ids is enough to confirm.
        docs = app.state.active_store.get_by_ids(["atest_6"])
        assert len(docs) == 1
        meta = docs[0].metadata
        assert "quarantined_at" not in meta
        assert "quarantine_reason" not in meta


# ---------------------------------------------------------------------------
# Full round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Quarantine then approve a document and verify end state."""

    def test_quarantine_approve_round_trip(self, client_app: tuple[TestClient, FastAPI]) -> None:
        """GET integrity count returns to 0 after quarantine + approve."""
        tc, app = client_app
        _seed(app, "rt_1", "some content")

        tc.post("/api/v1/memory/quarantine", json={"ids": ["rt_1"]})
        mid_status = tc.get("/api/v1/memory/integrity").json()
        assert mid_status["quarantine_count"] == 1

        tc.post("/api/v1/memory/approve", json={"ids": ["rt_1"]})
        end_status = tc.get("/api/v1/memory/integrity").json()
        assert end_status["quarantine_count"] == 0
        assert end_status["quarantined_ids"] == []

    def test_integrity_status_reflects_multiple_quarantined_docs(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """GET integrity lists all IDs currently in quarantine."""
        tc, app = client_app
        _seed(app, "rt_2a")
        _seed(app, "rt_2b")
        tc.post("/api/v1/memory/quarantine", json={"ids": ["rt_2a", "rt_2b"]})
        body = tc.get("/api/v1/memory/integrity").json()
        assert body["quarantine_count"] == 2
        assert set(body["quarantined_ids"]) == {"rt_2a", "rt_2b"}

    def test_approve_not_in_quarantine_leaves_active_store_unchanged(
        self, client_app: tuple[TestClient, FastAPI]
    ) -> None:
        """Approving a doc that was never quarantined does nothing to active store."""
        tc, app = client_app
        _seed(app, "rt_3", "control doc")
        before_count = app.state.active_store.count
        body = tc.post("/api/v1/memory/approve", json={"ids": ["rt_3"]}).json()
        assert body["approved"] == []
        # Document is still in active store; count unchanged.
        assert app.state.active_store.count == before_count
