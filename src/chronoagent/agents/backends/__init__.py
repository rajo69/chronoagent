"""LLM backend implementations for ChronoAgent.

All agents use a :class:`LLMBackend` to call the language model and produce
embeddings.  The concrete backends are:

- :class:`MockBackend` — deterministic fixtures; used in all tests and experiments.
- :class:`TogetherAIBackend` — default production backend (Together.ai REST API).
- :class:`OllamaBackend` — optional local backend; requires Ollama running with a GPU.
"""

from chronoagent.agents.backends.base import LLMBackend
from chronoagent.agents.backends.mock import MockBackend, MockBackendVariant
from chronoagent.agents.backends.ollama import OllamaBackend
from chronoagent.agents.backends.together import TogetherAIBackend

__all__ = [
    "LLMBackend",
    "MockBackend",
    "MockBackendVariant",
    "OllamaBackend",
    "TogetherAIBackend",
]
