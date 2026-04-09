"""Deterministic MockBackend — zero cost, no API keys, reproducible.

Uses the same seeded hash-based response selection as the Phase 1
:mod:`chronoagent.llm.mock_backend` but exposed through the clean
:class:`~chronoagent.agents.backends.base.LLMBackend` interface.  This is
the primary backend for all unit tests and experiment pipelines.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

import numpy as np

from chronoagent.agents.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Response libraries (same as in llm/mock_backend.py — single source of truth
# for fixture text lives here; the legacy module re-exports them)
# ---------------------------------------------------------------------------

_SECURITY_REVIEW_RESPONSES: list[str] = [
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-89] SQL injection risk in user input handling (line 42) - SEVERITY: HIGH\n"
        "2. [CWE-20] Missing input validation on API endpoint parameters (line 17) - SEVERITY: MEDIUM\n"
        "3. [CWE-798] Hardcoded credentials detected in config module (line 8) - SEVERITY: CRITICAL\n"
        "RECOMMENDATION: Block merge. Address critical and high findings before re-review."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-639] Insecure direct object reference in resource endpoint (line 55) - SEVERITY: HIGH\n"
        "2. [CWE-307] Missing rate limiting on authentication route - SEVERITY: MEDIUM\n"
        "3. [CWE-532] Debug logging exposes sensitive user data (line 31) - SEVERITY: MEDIUM\n"
        "RECOMMENDATION: Request changes. Fix high severity items and re-submit."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-79] Cross-site scripting (XSS) via unsanitized template variable (line 23) - SEVERITY: HIGH\n"
        "2. [CWE-352] CSRF protection bypassed on state-modifying endpoint - SEVERITY: HIGH\n"
        "RECOMMENDATION: Block merge. Both findings require immediate remediation."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-937] Dependency with known CVE-2024-1234 in requirements.txt - SEVERITY: HIGH\n"
        "2. [CWE-284] Overly broad file permissions set in deployment script (line 12) - SEVERITY: LOW\n"
        "3. [CWE-16] Missing security headers (HSTS, CSP) in HTTP response middleware - SEVERITY: MEDIUM\n"
        "RECOMMENDATION: Update vulnerable dependency before merge. Other items in follow-up."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-22] Path traversal vulnerability in file upload handler (line 67) - SEVERITY: CRITICAL\n"
        "2. [CWE-287] Weak password hashing algorithm (MD5) used for token generation (line 14) - SEVERITY: HIGH\n"
        "RECOMMENDATION: Block merge. Path traversal is exploitable immediately."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-502] Insecure deserialization of untrusted pickle data (line 88) - SEVERITY: CRITICAL\n"
        "2. [CWE-362] Race condition in token refresh logic allows session fixation (line 103) - SEVERITY: HIGH\n"
        "3. [CWE-532] Verbose error messages leak stack traces to client (line 76) - SEVERITY: LOW\n"
        "RECOMMENDATION: Block merge. Deserialization vulnerability is remotely exploitable."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. No significant security issues found in diff scope - SEVERITY: NONE\n"
        "2. [CWE-532] Minor: unused import of deprecated crypto module (line 3) - SEVERITY: LOW\n"
        "RECOMMENDATION: Approve with minor nit. Clean up unused import."
    ),
    (
        "SECURITY REVIEW FINDINGS:\n"
        "1. [CWE-347] Authentication bypass via JWT algorithm confusion (line 29) - SEVERITY: CRITICAL\n"
        "2. [CWE-284] Missing authorization check on admin-only resource (line 51) - SEVERITY: HIGH\n"
        "3. [CWE-287] Session tokens not invalidated on logout (line 38) - SEVERITY: MEDIUM\n"
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

_PLANNER_RESPONSES: list[str] = [
    (
        "PLAN:\n"
        "1. subtask_type=security_review code_segment=authentication module\n"
        "2. subtask_type=style_review code_segment=authentication module\n"
        "3. subtask_type=security_review code_segment=database queries\n"
        "4. subtask_type=style_review code_segment=database queries\n"
        "RATIONALE: Authentication and database access are high-risk areas."
    ),
    (
        "PLAN:\n"
        "1. subtask_type=security_review code_segment=file upload handler\n"
        "2. subtask_type=style_review code_segment=file upload handler\n"
        "3. subtask_type=security_review code_segment=API endpoint validation\n"
        "RATIONALE: File uploads are high-risk; API validation is mandatory."
    ),
    (
        "PLAN:\n"
        "1. subtask_type=security_review code_segment=dependency changes\n"
        "2. subtask_type=style_review code_segment=configuration files\n"
        "RATIONALE: Dependency updates require CVE scanning; config files need review."
    ),
    (
        "PLAN:\n"
        "1. subtask_type=security_review code_segment=JWT handling\n"
        "2. subtask_type=security_review code_segment=session management\n"
        "3. subtask_type=style_review code_segment=middleware layer\n"
        "RATIONALE: JWT and session code are critical security surfaces."
    ),
]

_STYLE_REVIEW_RESPONSES: list[str] = [
    (
        "STYLE REVIEW FINDINGS:\n"
        "1. Function `process_data` exceeds 50 lines — refactor into smaller units\n"
        "2. Variable names `x`, `y`, `tmp` are non-descriptive\n"
        "3. Missing docstrings on 3 public methods\n"
        "RECOMMENDATION: Request changes. Improve readability before merge."
    ),
    (
        "STYLE REVIEW FINDINGS:\n"
        "1. Cyclomatic complexity of `validate_user` is 12 — exceeds threshold of 10\n"
        "2. Inconsistent naming: mix of camelCase and snake_case in same module\n"
        "RECOMMENDATION: Request changes. Reduce complexity and standardize naming."
    ),
    (
        "STYLE REVIEW FINDINGS:\n"
        "1. No significant style issues found\n"
        "2. Minor: trailing whitespace on line 47\n"
        "RECOMMENDATION: Approve. Fix trailing whitespace as optional cleanup."
    ),
    (
        "STYLE REVIEW FINDINGS:\n"
        "1. Deeply nested conditionals (depth 5) reduce readability\n"
        "2. Magic numbers used without named constants\n"
        "3. TODO comment left in production code\n"
        "RECOMMENDATION: Request changes. Extract constants and flatten nesting."
    ),
]


class MockBackendVariant(StrEnum):
    """Response library variant for :class:`MockBackend`.

    Attributes:
        SECURITY: Security review responses.
        SUMMARY: Summary responses.
        PLANNER: Planner decomposition responses.
        STYLE: Style review responses.
    """

    SECURITY = "security"
    SUMMARY = "summary"
    PLANNER = "planner"
    STYLE = "style"


_VARIANT_LIBRARIES: dict[MockBackendVariant, list[str]] = {
    MockBackendVariant.SECURITY: _SECURITY_REVIEW_RESPONSES,
    MockBackendVariant.SUMMARY: _SUMMARY_RESPONSES,
    MockBackendVariant.PLANNER: _PLANNER_RESPONSES,
    MockBackendVariant.STYLE: _STYLE_REVIEW_RESPONSES,
}

_EMBED_DIM = 384


def _mock_embed(texts: list[str], dim: int = _EMBED_DIM) -> list[list[float]]:
    """Produce deterministic unit-norm embeddings from text hashes.

    Args:
        texts: Input strings to embed.
        dim: Embedding dimensionality.

    Returns:
        List of unit-norm float vectors of length *dim*.
    """
    result: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).hexdigest()
        seed = int(digest[:16], 16) % (2**31)
        rng = np.random.default_rng(seed)
        vec: np.ndarray = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        result.append((vec / norm).tolist())
    return result


class MockBackend(LLMBackend):
    """Deterministic LLM backend for tests and experiments.

    Selects responses from a fixed response library using a deterministic
    function of the prompt hash and a call counter.  Produces 384-dim unit
    embeddings via SHA-256 hashing — no model downloads required.

    Args:
        seed: Master seed controlling response selection.
        variant: Which response library to use (default ``SECURITY``).
        response_library: Override the full response pool; takes precedence
            over *variant* when provided.
    """

    def __init__(
        self,
        seed: int = 42,
        variant: MockBackendVariant | Literal["security", "summary", "planner", "style"] = (
            MockBackendVariant.SECURITY
        ),
        response_library: list[str] | None = None,
    ) -> None:
        self.seed = seed
        if response_library is not None:
            self._library = list(response_library)
        else:
            v = MockBackendVariant(variant) if isinstance(variant, str) else variant
            self._library = list(_VARIANT_LIBRARIES[v])
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        """Return a deterministic response for *prompt*.

        Args:
            prompt: Input prompt string.

        Returns:
            A response string from the configured response library.
        """
        prompt_hash = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        idx = (prompt_hash + self._call_count + self.seed) % len(self._library)
        self._call_count += 1
        return self._library[idx]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Produce deterministic unit-norm embeddings for *texts*.

        Args:
            texts: Input strings to embed.

        Returns:
            List of 384-dimensional unit-norm float vectors.
        """
        return _mock_embed(texts)

    def reset(self) -> None:
        """Reset the call counter to its initial state."""
        self._call_count = 0

    @property
    def embed_dim(self) -> int:
        """Embedding dimensionality produced by :meth:`embed`.

        Returns:
            Integer dimensionality (always 384 for :class:`MockBackend`).
        """
        return _EMBED_DIM
