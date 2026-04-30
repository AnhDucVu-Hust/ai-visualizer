"""
Pydantic data models shared across all STT engines.

Keeping models engine-agnostic lets us swap or combine engines without
changing downstream consumers.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
    MPS = "mps"


class ComputeType(str, Enum):
    """CTranslate2 / faster-whisper quantisation levels."""

    DEFAULT = "default"
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class WordTimestamp(BaseModel):
    """A single word with its start / end timestamps and confidence."""

    word: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Token probability (0–1)"
    )

    def format_timestamp(self) -> str:
        """Return a human-readable '[start -> end] word' string."""
        return f"[{self.start:.2f}s -> {self.end:.2f}s] {self.word}"


class Segment(BaseModel):
    """A contiguous speech segment produced by a transcription engine."""

    id: int
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str
    words: List[WordTimestamp] = Field(
        default_factory=list,
        description="Word-level timestamps; populated when word_timestamps=True",
    )
    avg_logprob: float = Field(
        0.0, description="Average log-probability of tokens in the segment"
    )
    no_speech_prob: float = Field(
        0.0, ge=0.0, le=1.0, description="Probability that the segment contains no speech"
    )
    compression_ratio: float = Field(
        0.0, description="Compression ratio (gzip) of the segment text"
    )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timestamp(self) -> str:
        return f"[{self.start:.2f}s -> {self.end:.2f}s]"

    def __str__(self) -> str:
        return f"{self.format_timestamp()} {self.text.strip()}"


class TranscriptionResult(BaseModel):
    """Complete result returned by any STT engine."""

    audio_path: str
    language: str = Field(..., description="Detected or forced language code, e.g. 'en'")
    language_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of the detected language"
    )
    duration: float = Field(..., description="Total audio duration in seconds")
    segments: List[Segment] = Field(default_factory=list)
    engine_name: str = Field(..., description="Name of the engine that produced this result")
    model_name: str = Field(..., description="Model identifier used for transcription")

    @property
    def full_text(self) -> str:
        """Concatenated transcript text."""
        return " ".join(s.text.strip() for s in self.segments)

    def format_with_timestamps(self, word_level: bool = False) -> str:
        """Return a human-readable timestamped transcript."""
        lines: List[str] = [
            f"Language : {self.language} (prob={self.language_probability:.2%})",
            f"Duration : {self.duration:.2f}s",
            f"Engine   : {self.engine_name} / {self.model_name}",
            "",
        ]
        for seg in self.segments:
            lines.append(str(seg))
            if word_level and seg.words:
                for w in seg.words:
                    lines.append(f"    {w.format_timestamp()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class TranscriptionConfig(BaseModel):
    """
    Engine-agnostic transcription configuration.

    Engines receive this object and map supported fields to their own API.
    Unsupported fields are silently ignored, so a single config object can
    be passed to any engine.
    """

    # Model selection
    model_size: str = Field("base", description="Whisper model size or HuggingFace path")
    device: Device = Device.AUTO
    compute_type: ComputeType = ComputeType.DEFAULT
    num_workers: int = Field(1, ge=1, description="Number of parallel CTranslate2 workers")

    # Transcription behaviour
    language: Optional[str] = Field(
        None,
        description="Force a language code (e.g. 'en'). None = auto-detect",
    )
    task: str = Field("transcribe", description="'transcribe' or 'translate'")
    beam_size: int = Field(5, ge=1)
    best_of: int = Field(5, ge=1)
    patience: float = Field(1.0, ge=0.0)
    temperature: float | List[float] = 0.0
    log_progress: bool = Field(
        True,
        description="Show faster-whisper progress bars during transcription",
    )

    # Timestamp options
    word_timestamps: bool = Field(
        False, description="Enable word-level timestamps"
    )

    # VAD
    vad_filter: bool = Field(
        False,
        description="Use Silero VAD to strip non-speech segments before transcription",
    )
    vad_min_silence_ms: int = Field(
        2000,
        ge=0,
        description="Minimum silence duration (ms) filtered by VAD",
    )

    # Batched inference (faster-whisper)
    batch_size: Optional[int] = Field(
        None,
        description="Batch size for BatchedInferencePipeline. None = sequential mode",
    )

    # Output
    condition_on_previous_text: bool = True
