"""
High-level Transcriber façade.

Transcriber owns a registry of engines and exposes a single
`transcribe()` entry point.  Callers never need to touch the engines
directly, but can still access them via `transcriber.engine`.

Example
-------
>>> from stt import Transcriber, TranscriptionConfig
>>> t = Transcriber()                          # defaults to FasterWhisper
>>> cfg = TranscriptionConfig(
...     model_size="base",
...     device="cpu",
...     word_timestamps=True,
...     vad_filter=True,
... )
>>> result = t.transcribe("audio/interview.mp3", cfg)
>>> print(result.format_with_timestamps(word_level=True))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Type

from .engines.base import BaseSTTEngine
from .engines.faster_whisper_engine import FasterWhisperEngine
from .engines.huggingface_engine import HuggingFaceEngine
from .models import TranscriptionConfig, TranscriptionResult

logger = logging.getLogger(__name__)

# Registry: name -> engine class
_ENGINE_REGISTRY: Dict[str, Type[BaseSTTEngine]] = {
    "faster-whisper": FasterWhisperEngine,
    "whisper": FasterWhisperEngine,
    "huggingface": HuggingFaceEngine,
}


def register_engine(name: str, engine_cls: Type[BaseSTTEngine]) -> None:
    """
    Register a custom engine so Transcriber can resolve it by name.

    Parameters
    ----------
    name:
        Unique string key, e.g. ``"whisperx"``.
    engine_cls:
        A class (not instance) that subclasses :class:`BaseSTTEngine`.
    """
    if not issubclass(engine_cls, BaseSTTEngine):
        raise TypeError(f"{engine_cls} must subclass BaseSTTEngine")
    _ENGINE_REGISTRY[name] = engine_cls
    logger.debug("Registered STT engine: %r", name)


class Transcriber:
    """
    Façade that wires a :class:`BaseSTTEngine` to a :class:`TranscriptionConfig`.

    Parameters
    ----------
    engine_name:
        Key in the engine registry.  Defaults to ``"faster-whisper"``.
    engine:
        Pass a pre-constructed engine instance to skip the registry lookup.
    """

    def __init__(
        self,
        engine_name: str = "faster-whisper",
        engine: Optional[BaseSTTEngine] = None,
    ) -> None:
        if engine is not None:
            self._engine = engine
        elif engine_name in _ENGINE_REGISTRY:
            self._engine = _ENGINE_REGISTRY[engine_name]()
        else:
            available = ", ".join(sorted(_ENGINE_REGISTRY))
            raise ValueError(
                f"Unknown engine {engine_name!r}. Available: {available}"
            )

        logger.debug("Transcriber using engine: %s", self._engine.engine_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def engine(self) -> BaseSTTEngine:
        """Direct access to the underlying engine (for advanced use)."""
        return self._engine

    def transcribe(
        self,
        audio_path: str | Path,
        config: Optional[TranscriptionConfig] = None,
    ) -> TranscriptionResult:
        """
        Transcribe *audio_path* and return a :class:`TranscriptionResult`.

        A default :class:`TranscriptionConfig` is used when none is provided.
        """
        if config is None:
            config = TranscriptionConfig()

        return self._engine.transcribe(audio_path, config)

    def unload(self) -> None:
        """Release model weights held by the underlying engine."""
        self._engine.unload()
