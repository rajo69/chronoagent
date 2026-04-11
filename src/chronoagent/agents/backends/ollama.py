"""OllamaBackend — optional local LLM backend via Ollama.

Calls a locally running Ollama server.  This backend is **optional** and
requires a GPU-capable machine with Ollama installed and a model pulled.
The default generation model is ``phi3:mini``; embeddings use ``nomic-embed-text``.

Skip this backend entirely in CPU-only environments — use :class:`MockBackend`
for tests and :class:`TogetherAIBackend` for production instead.
"""

from __future__ import annotations

import httpx

from chronoagent.agents.backends.base import LLMBackend
from chronoagent.retry import llm_retry

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_GEN_MODEL = "phi3:mini"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_REQUEST_TIMEOUT = 120.0


class OllamaBackend(LLMBackend):
    """LLM backend that calls a locally running Ollama server.

    Both generation and embedding requests use Ollama's REST API.
    The server must be running and the requested models must be pulled
    (``ollama pull <model>``) before use.

    Args:
        base_url: Ollama server base URL.
        gen_model: Model name used for generation.
        embed_model: Model name used for embeddings.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        gen_model: str = _DEFAULT_GEN_MODEL,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.gen_model = gen_model
        self.embed_model = embed_model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    @llm_retry
    def generate(self, prompt: str) -> str:
        """Generate a response using Ollama's ``/api/generate`` endpoint.

        Args:
            prompt: The full prompt string.

        Returns:
            The generated text from Ollama.

        Raises:
            httpx.HTTPStatusError: If Ollama returns a non-2xx status.
            httpx.ConnectError: If the Ollama server is not reachable.
        """
        response = self._client.post(
            "/api/generate",
            json={"model": self.gen_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return str(response.json()["response"])

    @llm_retry
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Produce embeddings via Ollama's ``/api/embed`` endpoint.

        Args:
            texts: Texts to embed (batched in a single request).

        Returns:
            List of float vectors, one per input text.

        Raises:
            httpx.HTTPStatusError: If Ollama returns a non-2xx status.
            httpx.ConnectError: If the Ollama server is not reachable.
        """
        response = self._client.post(
            "/api/embed",
            json={"model": self.embed_model, "input": texts},
        )
        response.raise_for_status()
        return [list(vec) for vec in response.json()["embeddings"]]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> OllamaBackend:
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the HTTP client on context manager exit."""
        self.close()
