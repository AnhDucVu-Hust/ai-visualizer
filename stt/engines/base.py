"""
Abstract base class for every speech-to-text engine.

To add a new engine:
  1. Subclass BaseSTTEngine.
  2. Implement the abstract methods.
  3. Register it in stt/engines/__init__.py.

The rest of the codebase depends only on this interface, so engines are
fully interchangeable.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from stt.models import TranscriptionConfig, TranscriptionResult

logger = logging.getLogger(__name__)


class BaseSTTEngine(abc.ABC):
    """
    Interface that every STT engine must satisfy.

    Engines are responsible for:
      - Loading and caching their model.
      - Accepting a TranscriptionConfig and an audio file path.
      - Returning a fully-populated TranscriptionResult.
    """

    # ------------------------------------------------------------------
    # Identity helpers (override in subclasses)
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine identifier, e.g. 'faster-whisper'."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Active model identifier, e.g. 'large-v3'."""

    @property
    def is_loaded(self) -> bool:
        """Return True once the model is ready to accept inference calls."""
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self, config: "TranscriptionConfig") -> None:
        """
        Load (or reload) the model using the given config.

        Implementations should be idempotent — calling load() a second time
        with the same config should be a no-op.
        """

    def unload(self) -> None:
        """
        Release model weights and free memory.

        Override when your engine holds GPU/CPU tensors that need explicit
        cleanup (e.g. del self._model; torch.cuda.empty_cache()).
        """
        logger.debug("%s: unload() not implemented — skipping.", self.engine_name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        config: "TranscriptionConfig",
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> "TranscriptionResult":
        """
        Transcribe *audio_path* according to *config*.

        Parameters
        ----------
        audio_path:
            Path to the audio file.  Engines should accept any format
            supported by their underlying decoder.
        config:
            Engine-agnostic configuration.  Engines silently ignore any
            fields they do not support.
        cancel_check:
            Optional callback returning True when the caller requested
            cancellation. Engines should check this cooperatively and
            abort as soon as practical.

        Returns
        -------
        TranscriptionResult
            Fully populated result including at minimum language detection
            info, audio duration, and a list of Segments.
        """

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _validate_audio_path(self, audio_path: str | Path) -> Path:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded else "not loaded"
        return f"<{self.__class__.__name__} engine={self.engine_name!r} model={self.model_name!r} {loaded}>"
