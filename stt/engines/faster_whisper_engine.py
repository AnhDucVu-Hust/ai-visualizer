"""
FasterWhisperEngine — concrete STT engine backed by faster-whisper.

Supports:
  - Standard sequential transcription (WhisperModel)
  - Batched transcription (BatchedInferencePipeline) when config.batch_size is set
  - Segment-level and word-level timestamps
  - VAD filtering via Silero VAD
  - All faster-whisper model sizes and quantisation levels
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .base import BaseSTTEngine
from stt.models import (
    ComputeType,
    Device,
    Segment,
    TranscriptionConfig,
    TranscriptionResult,
    WordTimestamp,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FasterWhisperEngine(BaseSTTEngine):
    """
    STT engine that wraps the faster-whisper library.

    Usage
    -----
    >>> from stt import FasterWhisperEngine, TranscriptionConfig
    >>> cfg = TranscriptionConfig(model_size="base", device="cpu", word_timestamps=True)
    >>> engine = FasterWhisperEngine()
    >>> engine.load(cfg)
    >>> result = engine.transcribe("audio/sample.mp3", cfg)
    >>> print(result.format_with_timestamps(word_level=True))
    """

    _ENGINE_NAME = "faster-whisper"

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._pipeline: Optional[object] = None
        self._current_config: Optional[TranscriptionConfig] = None
        self._model_name: str = "unloaded"

    # ------------------------------------------------------------------
    # BaseSTTEngine identity
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return self._ENGINE_NAME

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, config: TranscriptionConfig) -> None:
        """Load (or reload) the WhisperModel according to *config*."""
        if self._is_same_config(config):
            logger.debug(
                "FasterWhisperEngine: model %r already loaded with matching config — skipping.",
                config.model_size,
            )
            return

        logger.info(
            "FasterWhisperEngine: loading model=%r device=%r compute_type=%r",
            config.model_size,
            config.device.value,
            config.compute_type.value,
        )

        try:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            ) from exc

        device = self._resolve_device(config.device)
        compute_type = config.compute_type.value

        self._model = WhisperModel(
            config.model_size,
            device=device,
            compute_type=compute_type,
            num_workers=config.num_workers,
        )

        # Build batched pipeline only when a batch_size is requested.
        if config.batch_size is not None:
            self._pipeline = BatchedInferencePipeline(model=self._model)
        else:
            self._pipeline = None

        self._model_name = config.model_size
        self._current_config = config
        logger.info("FasterWhisperEngine: model loaded successfully.")

    def unload(self) -> None:
        self._model = None
        self._pipeline = None
        self._current_config = None
        self._model_name = "unloaded"
        logger.info("FasterWhisperEngine: model unloaded.")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str | Path,
        config: TranscriptionConfig,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe *audio_path* and return a populated TranscriptionResult.

        The engine auto-loads (or reloads) the model if the config changed.
        """
        path = self._validate_audio_path(audio_path)

        # Auto-load if needed.
        if not self.is_loaded or not self._is_same_config(config):
            self.load(config)

        assert self._model is not None  # guaranteed by load()

        logger.info(
            "FasterWhisperEngine: transcribing %r (batch_size=%s word_timestamps=%s vad=%s log_progress=%s)",
            str(path),
            config.batch_size,
            config.word_timestamps,
            config.vad_filter,
            config.log_progress,
        )

        segments_raw, info = self._run_inference(path, config)
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Transcription cancelled by user")

        segments = self._build_segments(
            segments_raw,
            config.word_timestamps,
            cancel_check=cancel_check,
        )

        return TranscriptionResult(
            audio_path=str(path),
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
            engine_name=self.engine_name,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, path: Path, config: TranscriptionConfig):
        """
        Dispatch to either the batched pipeline or the standard model and
        return the raw (segments_generator, TranscriptionInfo) tuple.
        """
        common_kwargs = dict(
            language=config.language,
            task=config.task,
            beam_size=config.beam_size,
            word_timestamps=config.word_timestamps,
            condition_on_previous_text=config.condition_on_previous_text,
            vad_filter=config.vad_filter,
            vad_parameters={"min_silence_duration_ms": config.vad_min_silence_ms},
            temperature=config.temperature,
            log_progress=config.log_progress,
        )

        if self._pipeline is not None and config.batch_size is not None:
            # BatchedInferencePipeline has a slightly different signature.
            segments_raw, info = self._pipeline.transcribe(
                str(path),
                batch_size=config.batch_size,
                **common_kwargs,
            )
        else:
            segments_raw, info = self._model.transcribe(  # type: ignore[union-attr]
                str(path),
                best_of=config.best_of,
                patience=config.patience,
                **common_kwargs,
            )

        return segments_raw, info

    @staticmethod
    def _build_segments(
        segments_raw,
        word_timestamps_enabled: bool,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> list[Segment]:
        """Convert faster-whisper namedtuples to our Pydantic Segment models."""
        segments: list[Segment] = []

        for idx, seg in enumerate(segments_raw):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Transcription cancelled by user")
            words: list[WordTimestamp] = []
            if word_timestamps_enabled and seg.words:
                for w in seg.words:
                    words.append(
                        WordTimestamp(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

            segments.append(
                Segment(
                    id=idx,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=words,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    compression_ratio=seg.compression_ratio,
                )
            )

        return segments

    def _is_same_config(self, config: TranscriptionConfig) -> bool:
        """Check whether the currently loaded model matches *config*."""
        if self._current_config is None:
            return False
        return (
            self._current_config.model_size == config.model_size
            and self._current_config.device == config.device
            and self._current_config.compute_type == config.compute_type
            and self._current_config.num_workers == config.num_workers
            and self._current_config.batch_size == config.batch_size
        )

    @staticmethod
    def _resolve_device(device: Device) -> str:
        """Resolve the AUTO device to 'cuda' or 'cpu' based on availability."""
        if device != Device.AUTO:
            return device.value

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

        try:
            import ctranslate2  # type: ignore

            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                return "cuda"
        except Exception:
            pass

        return "cpu"
