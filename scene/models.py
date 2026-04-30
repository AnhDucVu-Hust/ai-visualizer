"""
Pydantic models for the scene generation pipeline.

Flow:
  TranscriptionResult
      → Merger → List[MergedSegment]
      → PromptGenerator → List[SceneResult]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SceneConfig(BaseModel):
    """
    Controls both the segment merging step and the LLM prompt generation step.

    Merging thresholds
    ------------------
    min_duration:
        Keep accumulating consecutive segments until the group reaches at
        least this many seconds.  Short fragments are always merged together.
    max_duration:
        Hard ceiling — a group is finalized before adding the next segment
        if doing so would exceed this limit, even if min_duration hasn't
        been reached yet.

    LLM settings
    ------------
    model:
        OpenRouter model slug.  Prefer free tiers, e.g.
        ``"z-ai/glm-4.5-air:free"``.
    language:
        Language hint forwarded to the LLM so it can interpret the source
        text correctly (e.g. ``"Japanese"``).
    """

    # Merging
    min_duration: float = Field(
        10.0,
        ge=0.0,
        description="Minimum group duration in seconds before starting a new group",
    )
    max_duration: float = Field(
        30.0,
        ge=1.0,
        description="Maximum group duration in seconds (hard ceiling)",
    )

    # LLM
    model: str = Field(
        "z-ai/glm-4.5-air:free",
        description="OpenRouter model slug",
    )
    language: Optional[str] = Field(
        None,
        description="Source language hint for the LLM (e.g. 'Japanese'). "
                    "None = infer from TranscriptionResult.language",
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=64)

    # Visual style applied globally to every generated image prompt
    global_style: str = Field(
        default_factory=lambda: os.environ.get(
            "GLOBAL_STYLE", "2D animated anime style"
        ),
        description=(
            "Art/visual style enforced for every scene. "
            "Loaded from GLOBAL_STYLE in .env when not set explicitly."
        ),
    )

    # Concurrency
    max_concurrent_requests: int = Field(
        4,
        ge=1,
        description="Max parallel LLM calls when processing many scenes",
    )


# ---------------------------------------------------------------------------
# Merger output
# ---------------------------------------------------------------------------


class MergedSegment(BaseModel):
    """
    One group of consecutive transcription segments combined into a single unit
    ready for scene-prompt generation.
    """

    group_id: int
    start: float = Field(..., description="Start time of the first source segment (s)")
    end: float = Field(..., description="End time of the last source segment (s)")
    text: str = Field(..., description="Concatenated transcript text")
    source_segment_ids: List[int] = Field(
        default_factory=list,
        description="IDs of the original Segment objects that make up this group",
    )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timestamp(self) -> str:
        return f"[{self.start:.2f}s -> {self.end:.2f}s]"

    def __str__(self) -> str:
        return f"{self.format_timestamp()} {self.text.strip()}"


# ---------------------------------------------------------------------------
# Character models
# ---------------------------------------------------------------------------


class Character(BaseModel):
    """
    A character extracted from the full transcript.

    ``name`` is optional — some characters are identified only by role or
    visual appearance (e.g. "office workers", "the interviewer").
    When ``name`` is absent, ``role`` is used as the display label.
    At least one of ``name`` or ``role`` must be provided.
    """

    name: Optional[str] = Field(
        None,
        description="Character's proper name, if known (e.g. 'Tanaka')",
    )
    role: Optional[str] = Field(
        None,
        description=(
            "Role or group label when no name is known "
            "(e.g. 'office workers', 'the narrator', 'interviewer')"
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Visual description for image generation: age, build, hair, "
            "clothing, distinctive features"
        ),
    )

    @property
    def label(self) -> str:
        """Best available display label: name → role → 'unnamed character'."""
        return self.name or self.role or "unnamed character"

    @property
    def is_named(self) -> bool:
        return bool(self.name)

    def to_roster_line(self) -> str:
        return f"- {self.label}: {self.description}"

    def to_prompt_str(self) -> str:
        """Format used inside an image prompt: 'Label - description'."""
        return f"{self.label} - {self.description}"


class CharacterRoster(BaseModel):
    """All characters extracted from the full transcript."""

    characters: List[Character] = Field(default_factory=list)

    @property
    def named(self) -> List[Character]:
        """Characters with a proper name."""
        return [c for c in self.characters if c.is_named]

    @property
    def unnamed(self) -> List[Character]:
        """Characters identified only by role/description."""
        return [c for c in self.characters if not c.is_named]

    def to_roster_block(self) -> str:
        """Multi-line block listing every character for the LLM system prompt."""
        if not self.characters:
            return "(no characters identified)"
        lines: List[str] = []
        if self.named:
            lines.append("Named characters:")
            lines.extend(f"  {c.to_roster_line()}" for c in self.named)
        if self.unnamed:
            lines.append("Unnamed / recurring background:")
            lines.extend(f"  {c.to_roster_line()}" for c in self.unnamed)
        return "\n".join(lines)

    def find(self, name: str) -> Optional[Character]:
        """Case-insensitive lookup by name or role."""
        target = name.lower()
        for c in self.characters:
            if (c.name or "").lower() == target:
                return c
            if (c.role or "").lower() == target:
                return c
        return None

    def __bool__(self) -> bool:
        return bool(self.characters)


# ---------------------------------------------------------------------------
# Scene-level parsed character
# ---------------------------------------------------------------------------


class SceneCharacter(BaseModel):
    """
    One character entry parsed out of the ``ImagePrompt.characters`` string
    for a single scene.

    The raw LLM string looks like (pipe-separated):
      "Tanaka - slumped at desk; mid-30s salaryman, dark suit | [non-main] three office workers"

    Parsing produces one SceneCharacter per pipe-segment.
    """

    label: str = Field(
        ...,
        description="Display label: roster name/role, or '[non-main]' for unlisted figures",
    )
    pose: str = Field(
        "",
        description="Pose / expression for this specific scene (main characters only)",
    )
    description: str = Field(
        ...,
        description=(
            "Visual description — taken from the roster for main characters, "
            "or freely described for non-main ones"
        ),
    )
    is_main: bool = Field(
        ...,
        description="True when the character appears in the CharacterRoster",
    )
    roster_entry: Optional["Character"] = Field(
        None,
        description="The matching Character from the roster, if any",
    )

    def to_prompt_str(self) -> str:
        """Single-line string suitable for an image generation prompt."""
        if self.is_main:
            parts = filter(None, [self.pose, self.description])
            return f"{self.label} - {'; '.join(parts)}"
        return f"[non-main] {self.description}"


def parse_scene_characters(
    characters_str: str,
    roster: "CharacterRoster | None" = None,
) -> List["SceneCharacter"]:
    """
    Parse the ``ImagePrompt.characters`` pipe-separated string into a
    structured list of :class:`SceneCharacter` objects.

    Format produced by the LLM (each pipe segment is one entry):

    Main character entry::

        Name - <pose>; <roster description>

    Non-main / background entry::

        [non-main] <free visual description>

    Parameters
    ----------
    characters_str:
        Raw value of ``ImagePrompt.characters``.
    roster:
        Optional roster for cross-referencing main characters.
        When provided, ``SceneCharacter.roster_entry`` is populated and
        ``SceneCharacter.description`` falls back to the roster description
        if the LLM omitted it.

    Returns
    -------
    List[SceneCharacter]
    """
    import re as _re

    roster = roster or CharacterRoster()
    results: List[SceneCharacter] = []

    for raw_entry in characters_str.split("|"):
        entry = raw_entry.strip()
        if not entry:
            continue

        # ── Non-main background character ────────────────────────────────────
        if entry.lower().startswith("[non-main]"):
            desc = _re.sub(r"^\[non-main\]\s*", "", entry, flags=_re.IGNORECASE).strip()
            results.append(
                SceneCharacter(
                    label="[non-main]",
                    pose="",
                    description=desc or "background figure",
                    is_main=False,
                    roster_entry=None,
                )
            )
            continue

        # ── Main character: "Label - pose; description" ───────────────────────
        # Split on the first " - " to separate label from the rest.
        if " - " in entry:
            label, rest = entry.split(" - ", 1)
            label = label.strip()

            # The rest may be "pose; description" or just "description".
            if ";" in rest:
                pose, desc = rest.split(";", 1)
                pose = pose.strip()
                desc = desc.strip()
            else:
                pose = rest.strip()
                desc = ""
        else:
            # No " - " separator — treat the whole entry as the label.
            label = entry.strip()
            pose = ""
            desc = ""

        # Cross-reference with the roster.
        roster_entry = roster.find(label)
        if not desc and roster_entry:
            desc = roster_entry.description

        results.append(
            SceneCharacter(
                label=label,
                pose=pose,
                description=desc or label,
                is_main=True,
                roster_entry=roster_entry,
            )
        )

    return results


# ---------------------------------------------------------------------------
# LLM / image-prompt output
# ---------------------------------------------------------------------------


class ImagePrompt(BaseModel):
    """
    Structured image-generation prompt produced by the LLM for one scene.

    Every field is a short, descriptive phrase suitable for passing directly
    to a text-to-image model (Stable Diffusion, Midjourney, DALL-E, etc.).
    """

    scene: str = Field(
        ...,
        description="Main scene description — what is happening and where",
    )
    characters: str = Field(
        ...,
        description="Characters present, their appearance, pose, and expression",
    )
    style: str = Field(
        ...,
        description="Visual / art style (e.g. 'cinematic realism', 'anime', 'watercolor')",
    )
    lighting: str = Field(
        ...,
        description="Lighting conditions (e.g. 'soft golden hour', 'harsh fluorescent office')",
    )
    colors: str = Field(
        ...,
        description="Dominant color palette or mood (e.g. 'muted blues and greys')",
    )
    mood: str = Field(
        ...,
        description="Emotional atmosphere (e.g. 'tense and desperate', 'serene and hopeful')",
    )
    camera: str = Field(
        ...,
        description="Camera angle and shot type (e.g. 'close-up, eye level', 'wide establishing shot')",
    )
    def to_positive_prompt(self) -> str:
        """Flatten all fields into a comma-separated string for image APIs."""
        parts = [
            self.scene,
            self.characters,
            self.style,
            self.lighting,
            self.colors,
            self.mood,
            self.camera,
        ]
        return ", ".join(p.strip() for p in parts if p.strip())

    def to_text(self) -> str:
        """Human-readable multi-line labeled block."""
        return (
            f"Scene      : {self.scene}\n"
            f"Characters : {self.characters}\n"
            f"Style      : {self.style}\n"
            f"Lighting   : {self.lighting}\n"
            f"Colors     : {self.colors}\n"
            f"Mood       : {self.mood}\n"
            f"Camera     : {self.camera}"
        )

    def to_single_line(self) -> str:
        """All fields on one line, semicolon-separated. Good for prompts.txt."""
        return (
            f"Scene: {self.scene}; "
            f"Characters: {self.characters}; "
            f"Style: {self.style}; "
            f"Lighting: {self.lighting}; "
            f"Colors: {self.colors}; "
            f"Mood: {self.mood}; "
            f"Camera: {self.camera}"
        )


class SceneResult(BaseModel):
    """
    Final output for a single scene: the merged transcript + its image prompt.
    """

    group_id: int
    start: float
    end: float
    text: str
    source_segment_ids: List[int]
    image_prompt: ImagePrompt
    raw_llm_response: str = Field(
        "",
        description="Raw JSON string returned by the LLM (for debugging)",
    )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timestamp(self) -> str:
        return f"[{self.start:.2f}s -> {self.end:.2f}s]"

    def to_text(self) -> str:
        """Full human-readable block for this scene."""
        return (
            f"{self.format_timestamp()}\n"
            f"\n"
            f"{self.image_prompt.to_text()}"
        )


class ScenePipelineResult(BaseModel):
    """Top-level result of the full scene generation pipeline."""

    audio_path: str
    language: str
    audio_duration: float
    config: SceneConfig
    character_roster: CharacterRoster = Field(default_factory=CharacterRoster)
    scenes: List[SceneResult] = Field(default_factory=list)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def to_text(self) -> str:
        """Full human-readable transcript of all scenes."""
        lines: List[str] = [
            f"Audio    : {self.audio_path}",
            f"Language : {self.language}",
            f"Duration : {self.audio_duration:.1f}s",
            f"Scenes   : {self.scene_count}",
        ]

        if self.character_roster:
            lines += ["", "── Characters ──────────────────────────────"]
            for c in self.character_roster.characters:
                lines.append(f"  {c.name}: {c.description}")

        lines += ["", "── Scenes ──────────────────────────────────"]
        for scene in self.scenes:
            lines += ["", scene.to_text()]

        return "\n".join(lines)
