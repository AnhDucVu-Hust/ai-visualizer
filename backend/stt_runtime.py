"""Shared STT runtime for reusing one loaded transcriber/model."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from config import load_config
from stt import Transcriber, TranscriptionConfig
from stt.models import ComputeType, Device, TranscriptionResult

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "config.yaml"

_TRANSCIBER_LOCK = threading.Lock()
_RUNTIME_LOCK = threading.Lock()
_SHARED_ENGINE_NAME: str | None = None
_SHARED_TRANSCRIBER: Transcriber | None = None


def _resolve_shared_transcriber(engine_name: str) -> Transcriber:
    global _SHARED_ENGINE_NAME, _SHARED_TRANSCRIBER
    with _RUNTIME_LOCK:
        if _SHARED_TRANSCRIBER is None or _SHARED_ENGINE_NAME != engine_name:
            _SHARED_TRANSCRIBER = Transcriber(engine_name=engine_name)
            _SHARED_ENGINE_NAME = engine_name
            logger.info("Created shared STT transcriber for engine=%s", engine_name)
        return _SHARED_TRANSCRIBER


def transcribe_shared(
    *,
    engine_name: str,
    audio_path: str | Path,
    config: TranscriptionConfig,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> TranscriptionResult:
    """Run STT via the shared transcriber. Calls are serialized with a lock."""
    transcriber = _resolve_shared_transcriber(engine_name)
    with _TRANSCIBER_LOCK:
        return transcriber.transcribe(audio_path, config, cancel_check=cancel_check)


def preload_default_stt_model() -> None:
    """Best-effort preload of default STT model from config.yaml."""
    enabled = os.getenv("PRELOAD_STT_MODEL", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        logger.info("STT preload disabled by PRELOAD_STT_MODEL.")
        return

    try:
        cfg = load_config(yaml_path=_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None)
        t_cfg = TranscriptionConfig(
            model_size=cfg.stt_model,
            device=Device(cfg.device),
            compute_type=ComputeType(cfg.compute_type),
            log_progress=cfg.log_progress,
        )
        transcriber = _resolve_shared_transcriber(cfg.stt_engine)
        with _TRANSCIBER_LOCK:
            transcriber.engine.load(t_cfg)
        logger.info(
            "Preloaded STT model engine=%s model=%s device=%s compute_type=%s",
            cfg.stt_engine,
            cfg.stt_model,
            cfg.device,
            cfg.compute_type,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to preload STT model: %s", exc)
