"""Abstract base class for LLM backends.

Every agent backend must implement :meth:`generate` for text generation and
:meth:`embed` for producing vector embeddings.  Both operations are
synchronous; async wrappers can be layered on top if needed.
"""

from __future__ import annotations

import abc


class LLMBackend(abc.ABC):
    """Abstract interface for language model backends.

    Concrete implementations provide text generation and embedding
    capabilities.  The interface intentionally stays narrow — agents need
    only these two operations.
    """

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response for *prompt*.

        Args:
            prompt: The full input prompt string.

        Returns:
            The generated text response.
        """

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Produce vector embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.  All vectors share
            the same dimensionality.
        """
