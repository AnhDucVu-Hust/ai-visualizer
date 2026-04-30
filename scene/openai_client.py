"""
Generic OpenAI-compatible chat-completions client.

Works with any provider that speaks the OpenAI ``/chat/completions`` protocol
— OpenAI, OpenRouter, Groq, Together, DeepSeek, Mistral, Fireworks, …
Just point ``base_url`` at their endpoint and pass the matching API key(s).

Common base URLs:
    OpenAI      https://api.openai.com/v1
    OpenRouter  https://openrouter.ai/api/v1
    Groq        https://api.groq.com/openai/v1
    Together    https://api.together.xyz/v1
    DeepSeek    https://api.deepseek.com/v1
    Mistral     https://api.mistral.ai/v1

Multiple API keys / key rotation
---------------------------------
Pass a list (or comma-separated string) of API keys and the client will
round-robin rotate whenever any key hits 429 quota/rate-limit.  All keys
are tried before an ``LLMError(429, …)`` is raised.

    client = OpenAICompatibleClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=["gsk_A...", "gsk_B...", "gsk_C..."],
    )
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

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT = 60.0
_MAX_ERROR_BODY_CHARS = 4000

OpenAIError = LLMError


class OpenAICompatibleClient(BaseLLMClient):
    """
    Synchronous OpenAI-compatible chat-completions client with automatic
    API-key rotation.

    Parameters
    ----------
    api_key:
        One or more API keys.  Accepts:

        * A single key string — ``"sk-..."``
        * A comma-separated string — ``"sk-a,sk-b,sk-c"``
        * A list of key strings — ``["sk-a", "sk-b"]``

        Falls back to the ``env_var`` environment variable (defaults to
        ``OPENAI_API_KEY``), which may itself be a comma-separated list.
    base_url:
        Fully-qualified base URL of the OpenAI-compatible endpoint, e.g.
        ``"https://api.groq.com/openai/v1"``.  Defaults to OpenRouter.
    env_var:
        Name of the environment variable to use when ``api_key`` is not
        passed.  Defaults to ``"OPENAI_API_KEY"``.
    extra_headers:
        Optional dictionary of additional headers to send with every
        request (e.g. ``HTTP-Referer`` / ``X-Title`` for OpenRouter
        free-tier ranking).
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
        pipeline from hanging indefinitely when a *daily* quota is the
        one being hit (since those only reset at midnight PT).
    """

    def __init__(
        self,
        api_key: Optional[Union[str, List[str]]] = None,
        base_url: str = _DEFAULT_BASE_URL,
        env_var: str = "OPENAI_API_KEY",
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        quota_retry_wait: float = 60.0,
        max_quota_retries: int = 3,
    ) -> None:
        raw = api_key or os.environ.get(env_var, "")
        if not raw:
            raise ValueError(
                f"No API key provided. Pass api_key= or set {env_var} in .env"
            )

        if isinstance(raw, str):
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        else:
            keys = [k.strip() for k in raw if k.strip()]

        if not keys:
            raise ValueError(
                f"No valid API key found after parsing. "
                f"Check your api_key value or {env_var} env var."
            )

        self._api_keys: List[str] = keys
        self._key_idx: int = 0
        self._base_url: str = base_url.rstrip("/")
        self._quota_retry_wait: float = quota_retry_wait
        self._max_quota_retries: int = max_quota_retries

        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        self._http = httpx.Client(
            base_url=self._base_url,
            headers=headers,
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
        return "openai_compatible"

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        logger.debug(
            "OpenAI-compat → base=%s  model=%r  messages=%d",
            self._base_url,
            model,
            len(messages),
        )

        last_error: Optional[LLMError] = None
        total_cycles = self._max_quota_retries + 1

        for cycle in range(total_cycles):
            saw_quota_error = False
            for _ in range(len(self._api_keys)):
                key = self._current_key
                try:
                    response = self._http.post(
                        "/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {key}"},
                    )
                except httpx.HTTPError as exc:
                    key_num = self._key_idx + 1
                    logger.error(
                        "OpenAI-compat request transport error on key #%d: %s — rotating to next key.",
                        key_num,
                        exc,
                    )
                    last_error = LLMError(0, f"Transport error: {exc}")
                    self._rotate_key()
                    continue

                if response.status_code == 429:
                    try:
                        detail = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        detail = response.text
                    key_num = self._key_idx + 1
                    logger.warning(
                        "API key #%d hit quota/rate-limit (429): %s — rotating to next key.",
                        key_num,
                        detail,
                    )
                    saw_quota_error = True
                    last_error = LLMError(429, detail)
                    self._rotate_key()
                    continue

                if response.status_code != 200:
                    try:
                        body = response.json()
                        detail = body.get("error", {}).get("message", response.text)
                    except Exception:
                        body = None
                        detail = response.text
                    request_id = (
                        response.headers.get("x-request-id")
                        or response.headers.get("x-openrouter-request-id")
                    )
                    raw_body = response.text or ""
                    if len(raw_body) > _MAX_ERROR_BODY_CHARS:
                        raw_body = raw_body[:_MAX_ERROR_BODY_CHARS] + "...(truncated)"
                    logger.error(
                        "OpenAI-compat request failed: status=%s base=%s model=%r key_index=%d request_id=%s detail=%s raw_body=%s parsed_body=%s",
                        response.status_code,
                        self._base_url,
                        model,
                        self._key_idx + 1,
                        request_id or "-",
                        detail,
                        raw_body,
                        body,
                    )
                    last_error = LLMError(response.status_code, detail)
                    self._rotate_key()
                    continue

                body = response.json()
                if "choices" not in body or not body["choices"]:
                    msg = body.get("error", {}).get("message", str(body))
                    key_num = self._key_idx + 1
                    logger.error(
                        "OpenAI-compat malformed response on key #%d: missing choices: %s — rotating to next key.",
                        key_num,
                        msg,
                    )
                    last_error = LLMError(200, f"Response missing 'choices': {msg}")
                    self._rotate_key()
                    continue

                content: str = body["choices"][0]["message"]["content"]
                logger.debug(
                    "OpenAI-compat ← key=#%d  tokens_used=%s",
                    self._key_idx + 1,
                    body.get("usage", {}).get("total_tokens", "?"),
                )
                return content

            # All keys are exhausted this cycle.
            # Retry-with-wait only for quota/rate-limit cases.
            if saw_quota_error and cycle < total_cycles - 1:
                logger.warning(
                    "All %d API key(s) hit rate-limit (cycle %d/%d). "
                    "Sleeping %.1fs before retrying from the first key …",
                    len(self._api_keys),
                    cycle + 1,
                    total_cycles,
                    self._quota_retry_wait,
                )
                time.sleep(self._quota_retry_wait)
                self._key_idx = 0
                continue

            if last_error is not None:
                raise last_error
            raise LLMError(500, "All API keys failed without a detailed error.")

        raise LLMError(
            429,
            f"All {len(self._api_keys)} API key(s) still rate-limited after "
            f"{total_cycles} cycle(s) with {self._quota_retry_wait:.0f}s waits. "
            f"Last error: {last_error}",
        )

    def close(self) -> None:
        self._http.close()


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

class OpenRouterClient(OpenAICompatibleClient):
    """
    Backward-compatible wrapper that defaults to OpenRouter's base URL and
    attaches the OpenRouter-specific ranking headers.
    """

    def __init__(
        self,
        api_key: Optional[Union[str, List[str]]] = None,
        site_url: str = "https://github.com/AI-Visualizer",
        site_name: str = "AI Visualizer",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            env_var="OPENROUTER_API_KEY",
            extra_headers={"HTTP-Referer": site_url, "X-Title": site_name},
            timeout=timeout,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"


OpenRouterError = LLMError
