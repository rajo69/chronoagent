"""Unit tests for scripts/seed_memory.py (task 2.9)."""

from __future__ import annotations

import chromadb
import pytest
from scripts.seed_memory import (
    COLLECTION_REVIEWS,
    COLLECTION_SECURITY,
    COLLECTION_STYLE,
    COLLECTION_TEMPLATES,
    REPORT_TEMPLATES,
    SAMPLE_REVIEWS,
    SECURITY_PATTERNS,
    STYLE_CONVENTIONS,
    _upsert_collection,
    seed,
)

from chronoagent.agents.backends.mock import MockBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ephemeral_client() -> chromadb.ClientAPI:
    """Return a fresh in-memory ChromaDB client."""
    return chromadb.EphemeralClient()


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=0)


# ---------------------------------------------------------------------------
# Data integrity tests (no ChromaDB needed)
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_security_patterns_count(self) -> None:
        assert len(SECURITY_PATTERNS) >= 50

    def test_style_conventions_count(self) -> None:
        assert len(STYLE_CONVENTIONS) >= 30

    def test_report_templates_count(self) -> None:
        assert len(REPORT_TEMPLATES) >= 10

    def test_sample_reviews_count(self) -> None:
        assert len(SAMPLE_REVIEWS) >= 20

    def test_security_patterns_no_duplicates(self) -> None:
        assert len(SECURITY_PATTERNS) == len(set(SECURITY_PATTERNS))

    def test_style_conventions_no_duplicates(self) -> None:
        assert len(STYLE_CONVENTIONS) == len(set(STYLE_CONVENTIONS))

    def test_report_templates_no_duplicates(self) -> None:
        assert len(REPORT_TEMPLATES) == len(set(REPORT_TEMPLATES))

    def test_sample_reviews_no_duplicates(self) -> None:
        assert len(SAMPLE_REVIEWS) == len(set(SAMPLE_REVIEWS))

    def test_security_patterns_all_non_empty(self) -> None:
        assert all(doc.strip() for doc in SECURITY_PATTERNS)

    def test_style_conventions_all_non_empty(self) -> None:
        assert all(doc.strip() for doc in STYLE_CONVENTIONS)

    def test_report_templates_all_non_empty(self) -> None:
        assert all(doc.strip() for doc in REPORT_TEMPLATES)

    def test_sample_reviews_all_non_empty(self) -> None:
        assert all(doc.strip() for doc in SAMPLE_REVIEWS)

    def test_security_patterns_have_cwe_ids(self) -> None:
        """Every security pattern should reference at least one CWE identifier."""
        for doc in SECURITY_PATTERNS:
            assert "CWE-" in doc, f"Missing CWE ID in: {doc[:60]}"

    def test_sample_reviews_have_pr_ids(self) -> None:
        """Every sample review should reference a PR identifier."""
        for doc in SAMPLE_REVIEWS:
            assert "PR-" in doc, f"Missing PR ID in: {doc[:60]}"


# ---------------------------------------------------------------------------
# _upsert_collection tests
# ---------------------------------------------------------------------------


class TestUpsertCollection:
    def test_creates_collection_and_returns_count(
        self,
        ephemeral_client: chromadb.ClientAPI,
        mock_backend: MockBackend,
    ) -> None:
        docs = ["doc alpha", "doc beta", "doc gamma"]
        count = _upsert_collection(ephemeral_client, "test_col", docs, mock_backend)
        assert count == 3

    def test_upsert_is_idempotent(
        self,
        ephemeral_client: chromadb.ClientAPI,
        mock_backend: MockBackend,
    ) -> None:
        docs = ["doc a", "doc b"]
        _upsert_collection(ephemeral_client, "idempotent_col", docs, mock_backend)
        count = _upsert_collection(ephemeral_client, "idempotent_col", docs, mock_backend)
        assert count == 2  # upsert; no duplicates

    def test_reset_wipes_and_reseeds(
        self,
        ephemeral_client: chromadb.ClientAPI,
        mock_backend: MockBackend,
    ) -> None:
        docs = ["doc one", "doc two", "doc three"]
        _upsert_collection(ephemeral_client, "reset_col", docs, mock_backend)
        # Reset with fewer documents
        count = _upsert_collection(
            ephemeral_client, "reset_col", docs[:1], mock_backend, reset=True
        )
        assert count == 1

    def test_collection_name_used(
        self,
        ephemeral_client: chromadb.ClientAPI,
        mock_backend: MockBackend,
    ) -> None:
        _upsert_collection(ephemeral_client, "named_col", ["x"], mock_backend)
        names = [c.name for c in ephemeral_client.list_collections()]
        assert "named_col" in names


# ---------------------------------------------------------------------------
# seed() integration tests (uses tmp_path for persistence)
# ---------------------------------------------------------------------------


class TestSeed:
    def test_seed_returns_four_collections(self, tmp_path: pytest.TempPathFactory) -> None:
        counts = seed(chroma_dir=tmp_path)
        assert set(counts.keys()) == {
            COLLECTION_SECURITY,
            COLLECTION_STYLE,
            COLLECTION_TEMPLATES,
            COLLECTION_REVIEWS,
        }

    def test_seed_correct_counts(self, tmp_path: pytest.TempPathFactory) -> None:
        counts = seed(chroma_dir=tmp_path)
        assert counts[COLLECTION_SECURITY] == len(SECURITY_PATTERNS)
        assert counts[COLLECTION_STYLE] == len(STYLE_CONVENTIONS)
        assert counts[COLLECTION_TEMPLATES] == len(REPORT_TEMPLATES)
        assert counts[COLLECTION_REVIEWS] == len(SAMPLE_REVIEWS)

    def test_seed_idempotent(self, tmp_path: pytest.TempPathFactory) -> None:
        counts_first = seed(chroma_dir=tmp_path)
        counts_second = seed(chroma_dir=tmp_path)
        assert counts_first == counts_second

    def test_seed_reset_repopulates(self, tmp_path: pytest.TempPathFactory) -> None:
        seed(chroma_dir=tmp_path)
        counts = seed(chroma_dir=tmp_path, reset=True)
        assert counts[COLLECTION_SECURITY] == len(SECURITY_PATTERNS)
        assert counts[COLLECTION_REVIEWS] == len(SAMPLE_REVIEWS)

    def test_seed_creates_chroma_dir(self, tmp_path: pytest.TempPathFactory) -> None:
        target = tmp_path / "nested" / "chroma"
        seed(chroma_dir=target)
        assert target.exists()

    def test_seed_collections_queryable(self, tmp_path: pytest.TempPathFactory) -> None:
        """Documents seeded should be retrievable by chromadb query."""
        seed(chroma_dir=tmp_path)
        client = chromadb.PersistentClient(path=str(tmp_path))
        col = client.get_collection(COLLECTION_SECURITY)
        result = col.get(limit=1)
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# CLI (_parse_args) tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self) -> None:
        from scripts.seed_memory import _parse_args

        args = _parse_args([])
        assert args.chroma_dir == "./chroma_data"
        assert args.reset is False

    def test_custom_dir(self) -> None:
        from scripts.seed_memory import _parse_args

        args = _parse_args(["--chroma-dir", "/tmp/mydb"])
        assert args.chroma_dir == "/tmp/mydb"

    def test_reset_flag(self) -> None:
        from scripts.seed_memory import _parse_args

        args = _parse_args(["--reset"])
        assert args.reset is True
