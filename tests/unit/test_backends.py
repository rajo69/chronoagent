"""Tests for agents/backends: LLMBackend ABC, MockBackend, and backend adapters."""

from __future__ import annotations

import pytest

from chronoagent.agents.backends.base import LLMBackend
from chronoagent.agents.backends.mock import (
    MockBackend,
    MockBackendVariant,
    _mock_embed,
)


class TestLLMBackendABC:
    def test_cannot_instantiate_abstract(self) -> None:
        """LLMBackend is abstract — direct instantiation must raise TypeError."""
        with pytest.raises(TypeError):
            LLMBackend()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_generate_and_embed(self) -> None:
        """A subclass missing generate or embed must raise TypeError on init."""

        class IncompleteBackend(LLMBackend):  # type: ignore[abstract]
            def generate(self, prompt: str) -> str:
                return ""

        with pytest.raises(TypeError):
            IncompleteBackend()  # type: ignore[abstract]


class TestMockEmbedHelper:
    def test_embed_returns_list_of_vectors(self) -> None:
        vecs = _mock_embed(["hello", "world"])
        assert len(vecs) == 2
        assert all(isinstance(v, list) for v in vecs)

    def test_embed_vectors_have_correct_dim(self) -> None:
        vecs = _mock_embed(["test"], dim=128)
        assert len(vecs[0]) == 128

    def test_embed_same_text_same_vector(self) -> None:
        v1 = _mock_embed(["consistent text"])[0]
        v2 = _mock_embed(["consistent text"])[0]
        assert v1 == v2

    def test_embed_different_texts_different_vectors(self) -> None:
        v1 = _mock_embed(["alpha"])[0]
        v2 = _mock_embed(["beta"])[0]
        assert v1 != v2

    def test_embed_unit_norm(self) -> None:
        import math

        vec = _mock_embed(["norm check"])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5


class TestMockBackend:
    def test_generate_returns_string(self) -> None:
        b = MockBackend()
        result = b.generate("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_deterministic_same_seed(self) -> None:
        b1 = MockBackend(seed=42)
        b2 = MockBackend(seed=42)
        assert b1.generate("same prompt") == b2.generate("same prompt")

    def test_generate_different_seeds_may_differ(self) -> None:
        responses = {MockBackend(seed=s).generate("prompt") for s in range(20)}
        # At least two distinct responses across different seeds
        assert len(responses) > 1

    def test_call_count_advances(self) -> None:
        b = MockBackend(seed=0)
        b.generate("p")
        b.generate("p")
        # Same prompt, different call count — may or may not differ but count changed
        assert b._call_count == 2

    def test_reset_call_count(self) -> None:
        b = MockBackend(seed=42)
        b.generate("x")
        b.generate("x")
        b.reset()
        assert b._call_count == 0

    def test_reset_produces_same_response(self) -> None:
        b = MockBackend(seed=42)
        r1 = b.generate("same prompt")
        b.reset()
        r2 = b.generate("same prompt")
        assert r1 == r2

    def test_security_variant_contains_findings(self) -> None:
        b = MockBackend(variant=MockBackendVariant.SECURITY)
        response = b.generate("any prompt")
        assert "SECURITY REVIEW FINDINGS" in response

    def test_summary_variant_contains_summary(self) -> None:
        b = MockBackend(variant=MockBackendVariant.SUMMARY)
        response = b.generate("any prompt")
        assert "SUMMARY:" in response

    def test_planner_variant_contains_plan(self) -> None:
        b = MockBackend(variant=MockBackendVariant.PLANNER)
        response = b.generate("any prompt")
        assert "PLAN:" in response

    def test_style_variant_contains_findings(self) -> None:
        b = MockBackend(variant=MockBackendVariant.STYLE)
        response = b.generate("any prompt")
        assert "STYLE REVIEW FINDINGS" in response

    def test_string_variant_accepted(self) -> None:
        b = MockBackend(variant="summary")
        response = b.generate("x")
        assert "SUMMARY:" in response

    def test_custom_response_library(self) -> None:
        b = MockBackend(response_library=["response_a", "response_b"])
        responses = {b.generate(f"prompt {i}") for i in range(10)}
        assert responses <= {"response_a", "response_b"}

    def test_embed_returns_correct_shape(self) -> None:
        b = MockBackend()
        vecs = b.embed(["text one", "text two"])
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_embed_deterministic(self) -> None:
        b = MockBackend()
        v1 = b.embed(["stable"])[0]
        v2 = b.embed(["stable"])[0]
        assert v1 == v2

    def test_embed_dim_property(self) -> None:
        b = MockBackend()
        assert b.embed_dim == 384

    def test_implements_llm_backend(self) -> None:
        b = MockBackend()
        assert isinstance(b, LLMBackend)
