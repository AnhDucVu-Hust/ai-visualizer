"""
Google Gemini LLM client — uses the Gemini REST API directly via httpx.
No google-generativeai SDK required.

Free-tier limits (as of 2026):
  gemini-2.0-flash        15 RPM  1 500 RPD  1 M TPM
  gemini-2.0-flash-lite   30 RPM  1 500 RPD  1 M TPM
  gemini-1.5-flash        15 RPM  1 500 RPD  1 M TPM
  gemini-1.5-flash-8b     15 RPM  1 500 RPD  1 M TPM

Reference: https://ai.google.dev/api/generate-content
Get a free API key: https://aistudio.google.com/app/apikey

Multiple API keys / key rotation
---------------------------------
Pass a list (or comma-separated string) of API keys to use round-robin
rotation whenever any key hits the 429 quota/rate-limit.  All keys are
tried before a ``LLMError(429, …)`` is raised.

  client = GeminiClient(api_key=["KEY_A", "KEY_B", "KEY_C"])

  # or comma-separated string / env var
  # GEMINI_API_KEY=KEY_A,KEY_B,KEY_C
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv

from .base_client import BaseLLMClient, LLMError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_TIMEOUT = 60.0

GeminiError = LLMError


class GeminiClient(BaseLLMClient):
    """
    Synchronous Google Gemini chat client with automatic API-key rotation.

    Accepts the same OpenAI-style ``messages`` list as ``OpenRouterClient``
    and converts it to Gemini's ``contents`` + ``system_instruction`` format
    internally, so it is a drop-in replacement in ``PromptGenerator``.

    Parameters
    ----------
    api_key:
        One or more Gemini API keys.  Accepts:

        * A single key string — ``"AIza..."``
        * A comma-separated string — ``"AIza...,AIzb...,AIzc..."``
        * A list of key strings — ``["AIza...", "AIzb..."]``

        Falls back to the ``GEMINI_API_KEY`` environment variable (also
        supports a comma-separated list of keys there).
    timeout:
        Per-request timeout in seconds.
    quota_retry_wait:
        Seconds to wait when *all* keys have hit 429 before trying the
        whole rotation again.  Defaults to 60s (enough for per-minute
        rate limits to reset).
    max_quota_retries:
        Maximum number of wait-and-retry cycles after all keys are
        exhausted.  Set to 0 to disable waiting entirely, or to a large
        value for near-unlimited retries.  Defaults to 3
        (so up to 3 × ~60s ≈ 3 min total wait) — this prevents the
        pipeline from hanging indefinitely when a *daily* quota (RPD)
        is the one being hit (those only reset at midnight PT).
    """

    def __init__(
        self,
        api_key: Optional[Union[str, List[str]]] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        quota_retry_wait: float = 60.0,
        max_quota_retries: int = 3,
    ) -> None:
        raw = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not raw:
            raise ValueError(
                "No Gemini API key provided. "
                "Pass api_key= or set GEMINI_API_KEY in .env"
            )

        if isinstance(raw, str):
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        else:
            keys = [k.strip() for k in raw if k.strip()]

        if not keys:
            raise ValueError(
                "No valid Gemini API key found after parsing. "
                "Check your api_key value or GEMINI_API_KEY env var."
            )

        self._api_keys: List[str] = keys
        self._key_idx: int = 0
        self._quota_retry_wait: float = quota_retry_wait
        self._max_quota_retries: int = max_quota_retries
        self._http = httpx.Client(
            base_url=_BASE_URL,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Key rotation helpers
    # ------------------------------------------------------------------

    @property
    def _current_key(self) -> str:
        return self._api_keys[self._key_idx]

    def _rotate_key(self) -> None:
        """Advance to the next API key (wraps around)."""
        self._key_idx = (self._key_idx + 1) % len(self._api_keys)

    # ------------------------------------------------------------------
    # BaseLLMClient
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "gemini"

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a request to Gemini and return the model's text response.

        System messages are lifted out of ``messages`` and placed in
        Gemini's ``system_instruction`` field automatically.
        ``response_format={"type": "json_object"}`` maps to
        ``responseMimeType: "application/json"``.
        """
        contents = self._convert_messages(messages)

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if response_format and response_format.get("type") == "json_object":
            if self._supports_json_mode(model):
                generation_config["responseMimeType"] = "application/json"
            else:
                logger.debug(
                    "Model %r does not support JSON mode — skipping responseMimeType.",
                    model,
                )

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        endpoint = f"/models/{model}:generateContent"
        logger.debug("Gemini → model=%r  turns=%d", model, len(contents))

        last_error: Optional[LLMError] = None
        total_cycles = self._max_quota_retries + 1

        for cycle in range(total_cycles):
            for _ in range(len(self._api_keys)):
                key = self._current_key
                response = self._http.post(
                    endpoint,
                    json=payload,
                    params={"key": key},
                )

                if response.status_code == 429:
                    try:
                        detail = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        detail = response.text
                    key_num = self._key_idx + 1
                    logger.warning(
                        "Gemini key #%d hit quota/rate-limit (429): %s — rotating to next key.",
                        key_num,
                        detail,
                    )
                    last_error = LLMError(429, detail)
                    self._rotate_key()
                    continue

                if response.status_code != 200:
                    try:
                        detail = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        detail = response.text
                    raise LLMError(response.status_code, detail)

                body = response.json()
                content = self._extract_text(body)
                logger.debug(
                    "Gemini ← key=#%d  tokens_used=%s",
                    self._key_idx + 1,
                    body.get("usageMetadata", {}).get("totalTokenCount", "?"),
                )
                return content

            # All keys are exhausted this cycle — wait and try again if allowed.
            if cycle < total_cycles - 1:
                logger.warning(
                    "All %d Gemini key(s) hit rate-limit (cycle %d/%d). "
                    "Sleeping %.1fs before retrying from the first key …",
                    len(self._api_keys),
                    cycle + 1,
                    total_cycles,
                    self._quota_retry_wait,
                )
                time.sleep(self._quota_retry_wait)
                self._key_idx = 0

        raise LLMError(
            429,
            f"All {len(self._api_keys)} Gemini API key(s) still rate-limited after "
            f"{total_cycles} cycle(s) with {self._quota_retry_wait:.0f}s waits. "
            f"Last error: {last_error}",
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _supports_json_mode(model: str) -> bool:
        """
        Return False for model families known to reject responseMimeType.
        Gemma models and a few others only support plain text generation.
        """
        _NO_JSON_MODE_PREFIXES = ("gemma-", "lyria-", "nano-banana")
        return not any(model.lower().startswith(p) for p in _NO_JSON_MODE_PREFIXES)

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Gemini's ``contents`` format.

        System messages are prepended to the first user turn instead of using
        ``system_instruction``, which is not supported by Gemma models and some
        other providers accessed via the Gemini API.
        """
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")

            if role == "system":
                system_parts.append(text)
                continue

            # Gemini uses "model" instead of "assistant".
            gemini_role = "model" if role == "assistant" else "user"

            # Prepend collected system text into the very first user turn.
            if system_parts and gemini_role == "user" and not any(
                c["role"] == "user" for c in contents
            ):
                text = "\n\n".join(system_parts) + "\n\n" + text
                system_parts = []

            # Merge consecutive same-role turns (Gemini requires alternation).
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].append({"text": text})
            else:
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        return contents

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> str:
        """Pull the generated text from a Gemini response body."""
        try:
            candidates = body.get("candidates", [])
            if not candidates:
                raise LLMError(200, f"Gemini returned no candidates: {body}")

            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason not in ("STOP", "MAX_TOKENS", ""):
                raise LLMError(
                    200,
                    f"Gemini stopped with reason={finish_reason!r}: {body}",
                )

            parts = candidates[0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(200, f"Unexpected Gemini response shape: {exc}") from exc
