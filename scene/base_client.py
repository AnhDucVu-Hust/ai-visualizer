"""
Abstract base class that every LLM client must implement.

Adding a new provider:
  1. Subclass BaseLLMClient.
  2. Implement chat() and close().
  3. Register it in scene/__init__.py and the generate_scenes.py factory.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class LLMError(Exception):
    """Unified error raised by any LLM client."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"LLM error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class BaseLLMClient(abc.ABC):
    """
    Minimal interface shared by all LLM clients.

    The contract is intentionally narrow: accept an OpenAI-style messages
    list, return the assistant's text content.  Clients map this to whatever
    wire format their provider requires.
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. 'openrouter' or 'gemini'."""

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat request and return the assistant's text content.

        Parameters
        ----------
        messages:
            OpenAI-style list:
            ``[{"role": "system"|"user"|"assistant", "content": "..."}]``
        model:
            Provider-specific model identifier.
        temperature:
            Sampling temperature (0 = deterministic).
        max_tokens:
            Maximum tokens to generate.
        response_format:
            Optional hint; clients that support JSON mode use
            ``{"type": "json_object"}``.
        """

    def close(self) -> None:
        """Release underlying HTTP connections.  Override if needed."""

    def __enter__(self) -> "BaseLLMClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r}>"
