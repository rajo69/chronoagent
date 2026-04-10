"""TogetherAIBackend — production LLM backend using the Together.ai REST API.

Requires a valid ``CHRONO_TOGETHER_API_KEY`` environment variable (or explicit
*api_key* parameter).  Both generation and embedding are routed through the
Together.ai v1 API using :mod:`httpx` (synchronous).

Embedding model: ``togethercomputer/m2-bert-80M-8k-retrieval`` (768 dims).
Generation model: configurable, default ``mistralai/Mixtral-8x7B-Instruct-v0.1``.
"""

from __future__ import annotations

import os

import httpx

from chronoagent.agents.backends.base import LLMBackend

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
_DEFAULT_GEN_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
_DEFAULT_EMBED_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"
_DEFAULT_MAX_TOKENS = 512
_REQUEST_TIMEOUT = 60.0


class TogetherAIBackend(LLMBackend):
    """LLM backend backed by the Together.ai API.

    Sends synchronous HTTP requests to ``https://api.together.xyz/v1``.
    Use this backend in production or for experiments with a real LLM.

    Args:
        api_key: Together.ai API key.  Falls back to the
            ``CHRONO_TOGETHER_API_KEY`` environment variable when omitted.
        gen_model: Model used for text generation.
        embed_model: Model used for embeddings.
        max_tokens: Maximum tokens in the generated response.
        timeout: HTTP request timeout in seconds.

    Raises:
        ValueError: If *api_key* is not provided and the environment variable
            is not set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        gen_model: str = _DEFAULT_GEN_MODEL,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.getenv("CHRONO_TOGETHER_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "TogetherAIBackend requires an API key. "
                "Set CHRONO_TOGETHER_API_KEY or pass api_key=."
            )
        self._api_key = resolved_key
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=_TOGETHER_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def generate(self, prompt: str) -> str:
        """Generate a response from the Together.ai chat completions endpoint.

        Args:
            prompt: The full prompt string (sent as a user message).

        Returns:
            The generated text content from the first choice.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self.gen_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Produce embeddings via the Together.ai embeddings endpoint.

        Args:
            texts: Texts to embed (batched in a single API call).

        Returns:
            List of float vectors ordered to match *texts*.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        response = self._client.post(
            "/embeddings",
            json={"model": self.embed_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        ordered: list[tuple[int, list[float]]] = [
            (item["index"], item["embedding"]) for item in data["data"]
        ]
        ordered.sort(key=lambda t: t[0])
        return [vec for _, vec in ordered]

    def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        self._client.close()

    def __enter__(self) -> TogetherAIBackend:
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the HTTP client on context manager exit."""
        self.close()
