"""
PromptGenerator: calls the OpenRouter LLM to convert a MergedSegment's
transcript text into a structured ImagePrompt dict.

The LLM is instructed to return strict JSON so parsing is reliable.
A fallback parser handles cases where the model wraps JSON in markdown fences.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, List, Optional

from .models import (
    CharacterRoster,
    ImagePrompt,
    MergedSegment,
    SceneConfig,
    ScenePipelineResult,
    SceneResult,
)
from .base_client import BaseLLMClient, CancelledError, LLMError, interruptible_sleep
from stt.models import TranscriptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a creative director and visual artist who translates spoken narrative \
into detailed image-generation prompts.

GLOBAL STYLE DIRECTIVE: Every image MUST be rendered in this visual style — \
do not deviate:
  "{global_style}"

{character_block}\
Given a segment of transcribed audio, output ONLY a JSON object with exactly \
these keys (no extra keys, no markdown, no explanation):

{{
  "scene":      "<what is happening and where — be specific>",
  "characters": "<see CHARACTER RULES below>",
  "style":      "<must reflect the global style: {global_style}>",
  "lighting":   "<lighting description suited to the global style>",
  "colors":     "<dominant color palette, e.g. muted blues and warm amber accents>",
  "mood":       "<emotional atmosphere, e.g. tense and desperate>",
  "camera":     "<shot type and angle, e.g. close-up, eye level, shallow depth of field>"
}}

CHARACTER RULES for the "characters" field:
{character_rules}

General rules:
- Each JSON value must be a single descriptive phrase (no nested objects).
- The "style" field MUST always reflect "{global_style}".
- Stay faithful to the transcript; do not invent unrelated elements.
- If the transcript segment includes multiple beats, prioritize the first scene/event that appears when writing the "scene" field.
- Write all values in English regardless of the source language.
- No mentioning the word like celebrities, or a famous person in real life.
- Don't mention any character that is not visible in the scene. Just describe who should be visible in the scene.
- Output ONLY the JSON object — no markdown fences, no preamble.\
"""

_CHARACTER_BLOCK_WITH_ROSTER = """\
CHARACTER ROSTER (main characters with fixed designs):
{roster_lines}

"""

_CHARACTER_RULES_WITH_ROSTER = """\
Build the value as an ordered list separated by " | ":

  1. MAIN characters (from the roster above) who appear in this scene:
     Format → "Name - {<their roster description>, <pose/expression for THIS scene>}"
     Use the roster description as the base, but adapt it to match the scene's
     context (e.g., older/younger, injured, wearing different era-appropriate
     clothing). Keep the same core identity and recognizable traits.
     Include only those who would plausibly be present.

  2. NON-MAIN / background characters (anyone NOT in the roster):
     Format → " <brief visual description of the group or individual>"
     You can describe the appearance of the character for not mistaken it with the main characters.
     Example → "three suited office workers standing in the doorway, shocked expressions"
     Omit this part entirely if there are no background figures.
"""

_CHARACTER_RULES_NO_ROSTER = (
    "- Describe who is present, their appearance, emotion, and body language."
)


def _build_system_prompt(
    global_style: str,
    roster: "CharacterRoster | None" = None,
) -> str:
    if roster and roster.characters:
        character_block = _CHARACTER_BLOCK_WITH_ROSTER.format(
            roster_lines=roster.to_roster_block()
        )
        character_rules = _CHARACTER_RULES_WITH_ROSTER
    else:
        character_block = ""
        character_rules = _CHARACTER_RULES_NO_ROSTER

    return _SYSTEM_PROMPT_TEMPLATE.format(
        global_style=global_style,
        character_block=character_block,
        character_rules=character_rules,
    )


def _make_user_message(segment: MergedSegment, language: str) -> str:
    return (
        f"Source language: {language}\n"
        f"Timestamp: {segment.format_timestamp()}\n"
        f"Transcript:\n{segment.text.strip()}"
    )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(raw: Optional[str]) -> str:
    """Strip markdown fences if present, then return the JSON substring."""
    if not raw:
        return ""
    fence_match = _JSON_FENCE_RE.search(raw)
    if fence_match:
        return fence_match.group(1)
    # Try to find the first { … } block directly
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _parse_image_prompt(raw_content: Optional[str]) -> ImagePrompt:
    """Parse the LLM response into an ImagePrompt, with graceful fallback."""
    json_str = _extract_json(raw_content)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (%s); using placeholder values.", exc)
        data = {}

    return ImagePrompt(
        scene=data.get("scene", "Unknown scene"),
        characters=data.get("characters", ""),
        style=data.get("style", "cinematic realism"),
        lighting=data.get("lighting", "natural light"),
        colors=data.get("colors", "neutral tones"),
        mood=data.get("mood", "neutral"),
        camera=data.get("camera", "medium shot"),
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class PromptGenerator:
    """
    Generates :class:`ImagePrompt` objects for each :class:`MergedSegment`
    by querying the configured LLM provider.

    Parameters
    ----------
    client:
        A pre-constructed :class:`BaseLLMClient` (Gemini, OpenAI-compatible,
        OpenRouter, …).  The generator does NOT close it — the caller owns
        the lifecycle.
    retry_attempts:
        Number of retries on transient API errors before giving up.
    retry_delay:
        Seconds to wait between retries (simple linear backoff).
    """

    def __init__(
        self,
        client: BaseLLMClient,
        retry_attempts: int = 5,
        retry_delay: float = 3.0,
    ) -> None:
        self._client = client
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_for_segment(
        self,
        segment: MergedSegment,
        config: SceneConfig,
        language: str = "unknown",
        roster: "CharacterRoster | None" = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> tuple[ImagePrompt, str]:
        """
        Generate an :class:`ImagePrompt` for a single merged segment.

        Returns
        -------
        (ImagePrompt, raw_llm_response)
        """
        print("--------------------------------")
        print("Get prompt for segment: ", segment.group_id)
        messages = [
            {
                "role": "system",
                "content": _build_system_prompt(config.global_style, roster),
            },
            {"role": "user", "content": _make_user_message(segment, language)},
        ]

        logger.debug(
            "Generating group %d  style=%r  characters=%s",
            segment.group_id,
            config.global_style,
            [c.name for c in roster.characters] if roster else [],
        )

        raw = self._call_with_retry(messages, config, cancel_check=cancel_check)

        logger.debug(
            "LLM raw response for group %d:\n%s",
            segment.group_id,
            raw,
        )

        prompt = _parse_image_prompt(raw)
        return prompt, raw or ""

    def generate_all(
        self,
        transcription: TranscriptionResult,
        merged_segments: List[MergedSegment],
        config: SceneConfig,
        roster: "CharacterRoster | None" = None,
    ) -> ScenePipelineResult:
        """
        Process every merged segment and return a full :class:`ScenePipelineResult`.

        Parameters
        ----------
        roster:
            Optional :class:`CharacterRoster` produced by
            :class:`CharacterExtractor`.  When provided, every scene prompt
            includes the full character list so the LLM picks who appears.
        """
        language = config.language or _language_code_to_name(transcription.language)
        roster = roster or CharacterRoster()

        scenes: List[SceneResult] = []
        total = len(merged_segments)

        for idx, seg in enumerate(merged_segments, 1):
            logger.info(
                "Generating prompt %d/%d  %s  (%.1fs)",
                idx, total, seg.format_timestamp(), seg.duration,
            )
            image_prompt, raw = self.generate_for_segment(
                seg, config, language, roster
            )
            scenes.append(
                SceneResult(
                    group_id=seg.group_id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    source_segment_ids=seg.source_segment_ids,
                    image_prompt=image_prompt,
                    raw_llm_response=raw,
                )
            )

        return ScenePipelineResult(
            audio_path=transcription.audio_path,
            language=transcription.language,
            audio_duration=transcription.duration,
            config=config,
            character_roster=roster,
            scenes=scenes,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        messages: list,
        config: SceneConfig,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._retry_attempts + 1):
            if cancel_check is not None and cancel_check():
                raise CancelledError()
            try:
                return self._client.chat(
                    messages=messages,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    response_format={"type": "json_object"},
                    cancel_check=cancel_check,
                )
            except CancelledError:
                raise
            except LLMError as exc:
                last_exc = exc
                # Exponential backoff; extra pause on rate-limit (429).
                wait = self._retry_delay * (2 ** (attempt - 1))
                if exc.status_code == 429:
                    wait = max(wait, 30.0)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, self._retry_attempts, exc, wait,
                )
                if attempt < self._retry_attempts:
                    interruptible_sleep(wait, cancel_check)
            except Exception as exc:
                last_exc = exc
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, self._retry_attempts, exc, wait,
                )
                if attempt < self._retry_attempts:
                    interruptible_sleep(wait, cancel_check)

        raise RuntimeError(
            f"All {self._retry_attempts} LLM attempts failed."
        ) from last_exc


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_LANGUAGE_NAMES: dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}


def _language_code_to_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code.lower(), code)
