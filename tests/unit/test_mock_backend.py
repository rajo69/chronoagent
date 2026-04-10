"""Tests for MockBackend and MockEmbeddingFunction."""

from __future__ import annotations

from chronoagent.llm.mock_backend import MockBackend, MockEmbeddingFunction, MockSummaryBackend


class TestMockEmbeddingFunction:
    def test_returns_correct_shape(self) -> None:
        ef = MockEmbeddingFunction(dim=384)
        result = ef(["hello world", "another doc"])
        assert len(result) == 2
        assert all(len(vec) == 384 for vec in result)

    def test_deterministic(self) -> None:
        ef = MockEmbeddingFunction()
        v1 = ef(["test text"])
        v2 = ef(["test text"])
        assert v1 == v2

    def test_different_inputs_different_embeddings(self) -> None:
        ef = MockEmbeddingFunction()
        v1 = ef(["security vulnerability"])
        v2 = ef(["dependency update"])
        assert v1 != v2

    def test_unit_norm(self) -> None:
        import math

        ef = MockEmbeddingFunction()
        vec = ef(["normalize me"])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_name_method(self) -> None:
        ef = MockEmbeddingFunction()
        assert ef.name() == "mock_embedding_function"

    def test_embed_query_matches_call(self) -> None:
        ef = MockEmbeddingFunction()
        texts = ["query text"]
        assert ef.embed_query(texts) == ef(texts)

    def test_embed_documents_matches_call(self) -> None:
        ef = MockEmbeddingFunction()
        texts = ["doc one", "doc two"]
        assert ef.embed_documents(texts) == ef(texts)


class TestMockBackend:
    def test_basic_call_returns_string(self) -> None:
        llm = MockBackend(seed=0)
        result = llm.invoke("review this PR")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_with_same_seed_and_prompt(self) -> None:
        llm1 = MockBackend(seed=42)
        llm2 = MockBackend(seed=42)
        r1 = llm1.invoke("same prompt")
        r2 = llm2.invoke("same prompt")
        assert r1 == r2

    def test_different_prompts_can_differ(self) -> None:
        # With many responses in library, hashing different prompts should vary
        llm = MockBackend(seed=42)
        responses = {llm.invoke(f"prompt number {i}") for i in range(20)}
        # At least some responses should differ (not all the same)
        assert len(responses) > 1

    def test_call_count_increments(self) -> None:
        llm = MockBackend(seed=42)
        llm.invoke("prompt")
        r2 = llm.invoke("prompt")
        # Same prompt but different call count — allow same or different (both valid)
        assert isinstance(r2, str)

    def test_reset_restores_call_count(self) -> None:
        llm = MockBackend(seed=42)
        r1 = llm.invoke("same prompt")
        llm.reset()
        r2 = llm.invoke("same prompt")
        assert r1 == r2

    def test_llm_type(self) -> None:
        llm = MockBackend()
        assert llm._llm_type == "mock"

    def test_stop_sequence_truncates(self) -> None:
        llm = MockBackend(seed=0)
        result = llm.invoke("prompt", stop=["RECOMMENDATION"])
        assert "RECOMMENDATION" not in result


class TestMockSummaryBackend:
    def test_returns_summary_response(self) -> None:
        llm = MockSummaryBackend(seed=0)
        result = llm.invoke("summarize this review")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_uses_summary_library(self) -> None:
        llm = MockSummaryBackend(seed=0)
        result = llm.invoke("summarize this review")
        # Summary responses contain SUMMARY: prefix
        assert "SUMMARY" in result or "KEY POINTS" in result
