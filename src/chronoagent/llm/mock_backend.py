"""MockBackend: deterministic LLM for experiments and tests.

Uses a seeded response library so experiments are reproducible without any
API keys or GPU.  Response selection is based on a hash of the prompt combined
with a call counter, giving controlled variability across steps.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import numpy as np
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

# ---------------------------------------------------------------------------
# Realistic response libraries
# ---------------------------------------------------------------------------

_SECURITY_REVIEW_RESPONSES: list[str] = [
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. SQL injection risk in user input handling (line 42) - SEVERITY: HIGH\n"
        "2. Missing input validation on API endpoint parameters - SEVERITY: MEDIUM\n"
        "3. Hardcoded credentials detected in config module - SEVERITY: CRITICAL\n"
        "RECOMMENDATION: Block merge. Address critical and high findings before re-review."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. Insecure direct object reference in resource endpoint - SEVERITY: HIGH\n"
        "2. Missing rate limiting on authentication route - SEVERITY: MEDIUM\n"
        "3. Debug logging exposes sensitive user data - SEVERITY: MEDIUM\n"
        "RECOMMENDATION: Request changes. Fix high severity items and re-submit."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. Cross-site scripting (XSS) via unsanitized template variable - SEVERITY: HIGH\n"
        "2. CSRF protection bypassed on state-modifying endpoint - SEVERITY: HIGH\n"
        "RECOMMENDATION: Block merge. Both findings require immediate remediation."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. Dependency with known CVE-2024-1234 (severity 8.1) - SEVERITY: HIGH\n"
        "2. Overly broad file permissions set in deployment script - SEVERITY: LOW\n"
        "3. Missing security headers (HSTS, CSP) in HTTP response - SEVERITY: MEDIUM\n"
        "RECOMMENDATION: Update vulnerable dependency before merge. Other items in follow-up."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. Path traversal vulnerability in file upload handler - SEVERITY: CRITICAL\n"
        "2. Weak password hashing algorithm (MD5) used for token generation - SEVERITY: HIGH\n"
        "RECOMMENDATION: Block merge. Path traversal is exploitable immediately."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. Insecure deserialization of untrusted pickle data - SEVERITY: CRITICAL\n"
        "2. Race condition in token refresh logic allows session fixation - SEVERITY: HIGH\n"
        "3. Verbose error messages leak stack traces to client - SEVERITY: LOW\n"
        "RECOMMENDATION: Block merge. Deserialization vulnerability is remotely exploitable."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. No significant security issues found in diff scope - SEVERITY: NONE\n"
        "2. Minor: unused import of deprecated crypto module - SEVERITY: INFO\n"
        "RECOMMENDATION: Approve with minor nit. Clean up unused import."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. Authentication bypass via JWT algorithm confusion - SEVERITY: CRITICAL\n"
        "2. Missing authorization check on admin-only resource - SEVERITY: HIGH\n"
        "3. Session tokens not invalidated on logout - SEVERITY: MEDIUM\n"
        "RECOMMENDATION: Block merge. JWT vulnerability allows full authentication bypass."
    ),
]

_SUMMARY_RESPONSES: list[str] = [
    (
        "SUMMARY: This PR introduces a new data processing pipeline.\n"
        "KEY POINTS:\n"
        "- Security review flagged critical hardcoded credentials requiring immediate fix\n"
        "- SQL injection risk in user input path must be addressed\n"
        "- Overall risk level: HIGH — merge blocked pending remediation\n"
        "NEXT STEPS: Developer must rotate credentials and parameterize queries."
    ),
    (
        "SUMMARY: Authentication refactor across the user service.\n"
        "KEY POINTS:\n"
        "- IDOR vulnerability identified in resource endpoint\n"
        "- Rate limiting gap on auth route is an availability risk\n"
        "- Overall risk level: MEDIUM — changes requested\n"
        "NEXT STEPS: Add object-level authorization and throttle auth endpoints."
    ),
    (
        "SUMMARY: Frontend template engine upgrade.\n"
        "KEY POINTS:\n"
        "- XSS vulnerability via unsanitized template variable is exploitable\n"
        "- CSRF protection gap compounds frontend risk\n"
        "- Overall risk level: HIGH — merge blocked\n"
        "NEXT STEPS: Sanitize all template outputs; verify CSRF token enforcement."
    ),
    (
        "SUMMARY: Dependency update and deployment script changes.\n"
        "KEY POINTS:\n"
        "- Known CVE in updated dependency must be patched to a safe version\n"
        "- File permission issue in deployment script is low priority\n"
        "- Overall risk level: MEDIUM — conditional approval\n"
        "NEXT STEPS: Pin dependency to patched version, then merge."
    ),
    (
        "SUMMARY: File upload feature for document management.\n"
        "KEY POINTS:\n"
        "- Path traversal in upload handler is critically exploitable\n"
        "- Weak token hashing undermines session security\n"
        "- Overall risk level: CRITICAL — merge blocked\n"
        "NEXT STEPS: Restrict upload paths to safe directory; replace MD5 with BLAKE2."
    ),
    (
        "SUMMARY: Serialization layer refactor for caching.\n"
        "KEY POINTS:\n"
        "- Pickle deserialization of external data is a remote code execution risk\n"
        "- Race condition in token refresh needs mutex or atomic operation\n"
        "- Overall risk level: CRITICAL — merge blocked\n"
        "NEXT STEPS: Replace pickle with JSON schema; add lock to refresh logic."
    ),
    (
        "SUMMARY: Routine code cleanup and minor refactoring.\n"
        "KEY POINTS:\n"
        "- No significant security issues in changed scope\n"
        "- Cosmetic: unused deprecated import can be removed\n"
        "- Overall risk level: LOW — approved\n"
        "NEXT STEPS: Remove deprecated import as optional follow-up."
    ),
    (
        "SUMMARY: JWT-based SSO integration.\n"
        "KEY POINTS:\n"
        "- Algorithm confusion attack allows complete authentication bypass\n"
        "- Admin endpoint missing authorization enforcement\n"
        "- Overall risk level: CRITICAL — merge blocked\n"
        "NEXT STEPS: Enforce algorithm whitelist in JWT validation; add authz guard."
    ),
]


class MockEmbeddingFunction:
    """Deterministic embedding function for ChromaDB — no model downloads.

    Produces 384-dimensional unit vectors derived from the SHA-256 hash of each
    input string.  Same text always yields the same embedding; different texts
    yield different embeddings in a pseudo-random but reproducible manner.

    Args:
        dim: Embedding dimensionality (default 384, matching all-MiniLM-L6-v2).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def name(self) -> str:
        """Return the embedding function name required by ChromaDB.

        Returns:
            Identifier string for this embedding function.
        """
        return "mock_embedding_function"

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a list of documents (alias for ``__call__``).

        Args:
            input: Document strings to embed.

        Returns:
            List of unit-norm float vectors.
        """
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a list of query strings (alias for ``__call__``).

        Args:
            input: Query strings to embed.

        Returns:
            List of unit-norm float vectors.
        """
        return self(input)

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a list of strings.

        Args:
            input: Texts to embed.

        Returns:
            List of unit-norm float vectors of length ``self.dim``.
        """
        result: list[list[float]] = []
        for text in input:
            digest = hashlib.sha256(text.encode()).hexdigest()
            seed = int(digest[:16], 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec: np.ndarray = rng.standard_normal(self.dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            vec = vec / norm
            result.append(vec.tolist())
        return result


class MockBackend(LLM):
    """Deterministic LLM backend for experiments and offline testing.

    Selects responses from a fixed library using a deterministic function of
    the prompt hash and an internal call counter, producing realistic but
    reproducible outputs without any API calls.

    Args:
        seed: Master seed controlling response selection order.
        response_library: Pool of response strings.  If not provided, defaults
            to :data:`_SECURITY_REVIEW_RESPONSES`.
    """

    seed: int = 42
    response_library: list[str] = []
    _call_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        """Populate default response library after model init."""
        if not self.response_library:
            object.__setattr__(self, "response_library", list(_SECURITY_REVIEW_RESPONSES))

    @property
    def _llm_type(self) -> str:
        """LangChain LLM type identifier."""
        return "mock"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a deterministic response for *prompt*.

        Args:
            prompt: Input prompt string.
            stop: Unused (MockBackend does not support stop sequences).
            run_manager: LangChain callback manager (unused).
            **kwargs: Absorbed for interface compatibility.

        Returns:
            A response string chosen from :attr:`response_library`.
        """
        prompt_hash = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        idx = (prompt_hash + self._call_count + self.seed) % len(self.response_library)
        object.__setattr__(self, "_call_count", self._call_count + 1)
        response = self.response_library[idx]
        # Honour stop sequences if provided
        if stop:
            for token in stop:
                cut = response.find(token)
                if cut != -1:
                    response = response[:cut]
        return response

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the mock response word by word.

        Args:
            prompt: Input prompt string.
            stop: Optional stop sequences.
            run_manager: LangChain callback manager.
            **kwargs: Absorbed for interface compatibility.

        Yields:
            :class:`GenerationChunk` for each whitespace-separated token.
        """
        full = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        for word in full.split(" "):
            chunk = GenerationChunk(text=word + " ")
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)
            yield chunk

    def reset(self) -> None:
        """Reset the call counter to its initial state."""
        object.__setattr__(self, "_call_count", 0)


class MockSummaryBackend(MockBackend):
    """MockBackend pre-loaded with summary responses.

    Args:
        seed: Master seed controlling response selection order.
    """

    def model_post_init(self, __context: Any) -> None:
        """Populate summary response library after model init."""
        object.__setattr__(self, "response_library", list(_SUMMARY_RESPONSES))
