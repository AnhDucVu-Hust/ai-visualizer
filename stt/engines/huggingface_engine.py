"""
HuggingFaceEngine — STT engine backed by transformers ASR pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import BaseSTTEngine
from stt.models import Segment, TranscriptionConfig, TranscriptionResult

logger = logging.getLogger(__name__)


class HuggingFaceEngine(BaseSTTEngine):
    _ENGINE_NAME = "huggingface"

    def __init__(self) -> None:
        self._pipeline: Optional[object] = None
        self._current_config: Optional[TranscriptionConfig] = None
        self._model_name: str = "unloaded"

    @property
    def engine_name(self) -> str:
        return self._ENGINE_NAME

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, config: TranscriptionConfig) -> None:
        if self._is_same_config(config):
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            ) from exc

        device = self._resolve_device(config.device.value)
        logger.info(
            "HuggingFaceEngine: loading model=%r device=%s",
            config.model_size,
            "cpu" if device == -1 else f"cuda:{device}",
        )
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=config.model_size,
            device=device,
        )
        self._current_config = config
        self._model_name = config.model_size

    def unload(self) -> None:
        self._pipeline = None
        self._current_config = None
        self._model_name = "unloaded"

    def transcribe(self, audio_path: str | Path, config: TranscriptionConfig) -> TranscriptionResult:
        path = self._validate_audio_path(audio_path)
        if not self.is_loaded or not self._is_same_config(config):
            self.load(config)

        assert self._pipeline is not None
        generate_kwargs = {}
        if config.language:
            generate_kwargs["language"] = config.language
        if config.task:
            generate_kwargs["task"] = config.task

        # Always use segment-level timestamps.
        result = self._pipeline(
            str(path),
            return_timestamps=True,
            return_language=True,
            generate_kwargs=generate_kwargs,
        )

        segments = self._build_segments(result)

        duration = max((s.end for s in segments), default=0.0)
        language = (config.language or result.get("language") or "unknown").lower()
        language_prob = 1.0 if config.language else 0.0

        return TranscriptionResult(
            audio_path=str(path),
            language=language,
            language_probability=language_prob,
            duration=duration,
            segments=segments,
            engine_name=self.engine_name,
            model_name=self.model_name,
        )

    def _is_same_config(self, config: TranscriptionConfig) -> bool:
        if self._current_config is None:
            return False
        return (
            self._current_config.model_size == config.model_size
            and self._current_config.device == config.device
        )

    @staticmethod
    def _build_segments(result: dict) -> list[Segment]:
        chunks = result.get("chunks") or []
        full_text = (result.get("text") or "").strip()

        if not chunks:
            return [Segment(id=0, start=0.0, end=0.0, text=full_text)]

        segments: list[Segment] = []
        for idx, chunk in enumerate(chunks):
            start, end = chunk.get("timestamp") or (None, None)
            if start is None or end is None:
                continue
            segments.append(
                Segment(
                    id=idx,
                    start=float(start),
                    end=float(end),
                    text=(chunk.get("text") or "").strip(),
                )
            )

        if not segments:
            return [Segment(id=0, start=0.0, end=0.0, text=full_text)]
        return segments

    @staticmethod
    def _resolve_device(device: str) -> int:
        if device == "cpu":
            return -1
        if device == "cuda":
            return 0

        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
