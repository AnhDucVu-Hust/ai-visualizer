"""
CharacterExtractor: analyses the full transcript with an LLM and returns
a CharacterRoster of named characters with visual descriptions.

The roster is produced once per audio file and then passed to every
scene-level prompt so the image generator always draws consistent characters.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .base_client import BaseLLMClient, LLMError
from .models import Character, CharacterRoster, SceneConfig
from stt.models import TranscriptionResult

logger = logging.getLogger(__name__)

# Max characters of transcript text sent to the LLM.
# Enough for ~30–60 min of speech; keeps tokens reasonable.
_MAX_TRANSCRIPT_CHARS = 12_000

_SYSTEM_PROMPT = """\
You are a story analyst specializing in visual character design.

Read the following transcript and identify every character or recurring group \
who appears or participates in the story.

Rules:
- All descriptions MUST be in English.
- If a character has a proper name, set "name" to that name and leave "role" null.
- If a character has no name but has a clear role or group label \
  (e.g. "office workers", "the interviewer", "the boss"), \
  set "role" to that label and leave "name" null.
- MUST READ THE PARAGRAPH CAREFULLY FOR RECOGNIZING THE CHARACTER'S GENDER
- In case there's character's name, the role should be what the character is referred to as in the transcript.
- For off-screen narrators who are never visually depicted, set \
  "role": "narrator" and "description": "off-screen voice, no visual appearance".
- For each entry give a detailed VISUAL description suitable for image generation in English: \
  approximate age, body type, hairstyle/hair length, skin tone, clothing/uniform, \
  accessories, facial features, and overall vibe.
- If the transcript does not clearly specify some visual details, infer them \
  plausibly from context and role. Favor attractive, eye-catching, cinematic \
  character design (e.g. graceful posture, well-styled hair, confident presence), \
  while remaining believable for the story.
- Do not contradict explicit transcript facts. Inferred details must stay consistent \
  with the character's role, setting, and tone.

Output ONLY a JSON object in exactly this format (no markdown, no extra keys):
{
  "characters": [
    {"name": "<proper name or null>", "role": "<role/group label or null>", "description": "<visual description>"},
    ...
  ]
}
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _build_user_message(transcription: TranscriptionResult, language: str) -> str:
    full_text = " ".join(s.text.strip() for s in transcription.segments)
    if len(full_text) > _MAX_TRANSCRIPT_CHARS:
        full_text = full_text[:_MAX_TRANSCRIPT_CHARS] + "\n[...transcript truncated...]"

    return (
        f"Language: {language}\n"
        f"Audio duration: {transcription.duration:.1f}s\n\n"
        f"TRANSCRIPT:\n{full_text}"
    )


def _extract_json(raw: Optional[str]) -> str:
    if not raw:
        return ""
    fence = _JSON_FENCE_RE.search(raw)
    if fence:
        return fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _parse_roster(raw: Optional[str]) -> CharacterRoster:
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Character roster parse failed (%s); returning empty roster.", exc)
        return CharacterRoster()

    characters = []
    for item in data.get("characters", []):
        name = (item.get("name") or "").strip() or None
        role = (item.get("role") or "").strip() or None
        description = (item.get("description") or "").strip()

        if not description:
            continue
        if not name and not role:
            continue

        characters.append(Character(name=name, role=role, description=description))

    return CharacterRoster(characters=characters)


class CharacterExtractor:
    """
    Calls the LLM once on the full transcript to build a :class:`CharacterRoster`.

    Parameters
    ----------
    client:
        Any :class:`BaseLLMClient` (Gemini, OpenRouter, …).
    retry_attempts / retry_delay:
        Retry settings for transient API errors.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self._client = client
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay

    def extract(
        self,
        transcription: TranscriptionResult,
        config: SceneConfig,
        language: Optional[str] = None,
    ) -> CharacterRoster:
        """
        Analyse *transcription* and return a :class:`CharacterRoster`.

        Parameters
        ----------
        transcription:
            Full STT result.
        config:
            Scene config (provides model, temperature, etc.).
        language:
            Human-readable language name (e.g. ``"Japanese"``).
            Falls back to the language code in *transcription*.
        """
        from .prompt_generator import _language_code_to_name  # avoid circular import

        lang = language or _language_code_to_name(transcription.language)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(transcription, lang)},
        ]

        logger.info(
            "CharacterExtractor: analysing transcript (%d segments, lang=%s) …",
            len(transcription.segments),
            lang,
        )

        import time

        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                raw = self._client.chat(
                    messages=messages,
                    model=config.model,
                    temperature=0.3,        # lower temp for factual extraction
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                logger.debug("CharacterExtractor raw response:\n%s", raw)
                roster = _parse_roster(raw)
                logger.info(
                    "CharacterExtractor: found %d character(s): %s",
                    len(roster.characters),
                    [c.label for c in roster.characters],
                )
                return roster

            except LLMError as exc:
                last_exc = exc
                wait = self._retry_delay * (2 ** (attempt - 1))
                if exc.status_code == 429:
                    wait = max(wait, 30.0)
                logger.warning(
                    "Character extraction failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, self._retry_attempts, exc, wait,
                )
                if attempt < self._retry_attempts:
                    time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Character extraction failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, self._retry_attempts, exc, wait,
                )
                if attempt < self._retry_attempts:
                    time.sleep(wait)

        logger.error("Character extraction gave up after %d attempts.", self._retry_attempts)
        raise RuntimeError("Character extraction failed.") from last_exc
