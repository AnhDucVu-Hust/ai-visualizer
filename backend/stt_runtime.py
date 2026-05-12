"""Shared STT runtime for reusing one loaded transcriber/model."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from config import load_config, resolve_app_config_yaml_path
from stt import GroqEngine, Transcriber, TranscriptionConfig
from stt.models import ComputeType, Device, TranscriptionResult

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

_TRANSCIBER_LOCK = threading.Lock()
_RUNTIME_LOCK = threading.Lock()
_SHARED_ENGINE_NAME: str | None = None
_SHARED_TRANSCRIBER: Transcriber | None = None


def _resolve_groq_api_keys() -> list[str]:
    yaml_path = resolve_app_config_yaml_path(_ROOT)
    cfg = load_config(yaml_path=yaml_path if yaml_path else None)
    keys: list[str] = []
    if cfg.api_keys:
        keys = [str(k).strip() for k in cfg.api_keys if str(k).strip()]
    elif cfg.api_key:
        keys = [k.strip() for k in str(cfg.api_key).split(",") if k.strip()]
    if not keys:
        raw = (
            os.environ.get("LLM_API_KEYS")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError(
            "No Groq API keys found. Set llm.api_keys / llm.api_key in config, "
            "or OPENAI_API_KEY (comma-separated) in the environment / .env."
        )
    return keys


def _resolve_shared_transcriber(engine_name: str) -> Transcriber:
    global _SHARED_ENGINE_NAME, _SHARED_TRANSCRIBER
    with _RUNTIME_LOCK:
        if _SHARED_TRANSCRIBER is None or _SHARED_ENGINE_NAME != engine_name:
            if engine_name == "groq":
                _SHARED_TRANSCRIBER = Transcriber(engine=GroqEngine(api_keys=_resolve_groq_api_keys()))
            else:
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
    """Run STT via the shared transcriber. Local engines are serialized with a lock."""
    transcriber = _resolve_shared_transcriber(engine_name)
    if engine_name == "groq":
        return transcriber.transcribe(audio_path, config, cancel_check=cancel_check)
    with _TRANSCIBER_LOCK:
        return transcriber.transcribe(audio_path, config, cancel_check=cancel_check)


def preload_default_stt_model() -> None:
    """Best-effort preload of default STT model from config.yaml."""
    enabled = os.getenv("PRELOAD_STT_MODEL", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        logger.info("STT preload disabled by PRELOAD_STT_MODEL.")
        return

    try:
        yaml_path = resolve_app_config_yaml_path(_ROOT)
        cfg = load_config(yaml_path=yaml_path if yaml_path else None)
        if cfg.stt_engine == "groq":
            logger.info("Skipping STT preload for engine=groq (remote model).")
            return
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
