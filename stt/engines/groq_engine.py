"""
GroqEngine — STT engine backed by Groq's audio transcription API.

API reference:
  POST https://api.groq.com/openai/v1/audio/transcriptions

This engine returns the project's standard TranscriptionResult/Segment schema.
If an input audio file is larger than Groq's 25MB limit, it is chunked into
10-minute segments using ffmpeg. Each chunk is transcribed and merged back
with timestamp offsets applied.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from .base import BaseSTTEngine
from stt.models import Segment, TranscriptionConfig, TranscriptionResult

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_TRANSCRIPT_PATH = "/audio/transcriptions"

# Groq docs state 25MB limit; keep a safety margin for multipart overhead.
_MAX_FILE_BYTES = 24 * 1024 * 1024
_CHUNK_SECONDS = 600  # 10 minutes

_DEFAULT_MODEL = "whisper-large-v3-turbo"

_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_QUOTA_RETRY_WAIT_SECONDS = 60.0
_DEFAULT_MAX_QUOTA_RETRIES = 3


class GroqEngine(BaseSTTEngine):
    _ENGINE_NAME = "groq"

    def __init__(
        self,
        *,
        api_keys: Optional[List[str]] = None,
        base_url: str = _GROQ_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        quota_retry_wait_seconds: float = _DEFAULT_QUOTA_RETRY_WAIT_SECONDS,
        max_quota_retries: int = _DEFAULT_MAX_QUOTA_RETRIES,
    ) -> None:
        self._api_keys: List[str] = [k.strip() for k in (api_keys or []) if str(k).strip()]
        self._key_idx: int = 0
        self._base_url = str(base_url).rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._quota_retry_wait_seconds = float(quota_retry_wait_seconds)
        self._max_quota_retries = int(max_quota_retries)
        self._http: Optional[httpx.Client] = None
        self._model_name: str = _DEFAULT_MODEL

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
        return bool(self._api_keys) and self._http is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, config: TranscriptionConfig) -> None:
        # Model is remote; "load" just validates config + ensures HTTP client exists.
        if not self._api_keys:
            raise ValueError(
                "GroqEngine requires API keys. Provide them via PipelineConfig.llm.api_keys "
                "and wire them into GroqEngine(api_keys=...)."
            )
        self._model_name = (config.model_size or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
        if self._http is None:
            self._http = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            )

    def unload(self) -> None:
        if self._http is not None:
            try:
                self._http.close()
            finally:
                self._http = None

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str | Path,
        config: TranscriptionConfig,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TranscriptionResult:
        path = self._validate_audio_path(audio_path)
        if not self.is_loaded:
            self.load(config)
        assert self._http is not None

        if cancel_check is not None and cancel_check():
            raise RuntimeError("Transcription cancelled by user")

        if path.stat().st_size <= _MAX_FILE_BYTES:
            return self._transcribe_single_file(path, config, cancel_check=cancel_check)

        with tempfile.TemporaryDirectory(prefix="groq_chunks_") as tmp_dir:
            chunk_dir = Path(tmp_dir)
            chunks = self._split_audio_ffmpeg(path, chunk_dir, cancel_check=cancel_check)
            if not chunks:
                raise RuntimeError("ffmpeg produced no chunks for oversized audio.")

            all_segments: list[Segment] = []
            global_id = 0
            time_offset = 0.0
            language: Optional[str] = None
            total_duration = 0.0

            for idx, chunk_path in enumerate(chunks, 1):
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("Transcription cancelled by user")

                # If chunk is still too large (copy-split can keep big codecs), re-encode.
                chunk_for_api = chunk_path
                if chunk_for_api.stat().st_size > _MAX_FILE_BYTES:
                    chunk_for_api = self._reencode_small_mp3(chunk_for_api, chunk_dir)

                result_payload = self._call_groq_transcribe(
                    file_path=chunk_for_api,
                    config=config,
                    cancel_check=cancel_check,
                )

                chunk_language = str(result_payload.get("language") or "").strip()
                if language is None and chunk_language:
                    language = chunk_language

                chunk_duration = self._safe_float(result_payload.get("duration")) or self._probe_duration_ffprobe(
                    chunk_for_api
                )

                raw_segments = result_payload.get("segments") or []
                if not isinstance(raw_segments, list):
                    raw_segments = []

                for seg in raw_segments:
                    if not isinstance(seg, dict):
                        continue
                    start = self._safe_float(seg.get("start")) or 0.0
                    end = self._safe_float(seg.get("end")) or start
                    text = str(seg.get("text") or "")
                    all_segments.append(
                        Segment(
                            id=global_id,
                            start=start + time_offset,
                            end=end + time_offset,
                            text=text,
                        )
                    )
                    global_id += 1

                # Advance offset for next chunk.
                time_offset += float(chunk_duration or 0.0)
                total_duration = time_offset
                logger.info("GroqEngine: merged chunk %d/%d duration=%.2fs", idx, len(chunks), float(chunk_duration or 0.0))

            return TranscriptionResult(
                audio_path=str(path),
                language=(language or "unknown"),
                language_probability=1.0,
                duration=float(total_duration),
                segments=all_segments,
                engine_name=self.engine_name,
                model_name=self.model_name,
            )

    # ------------------------------------------------------------------
    # Key rotation helpers
    # ------------------------------------------------------------------

    @property
    def _current_key(self) -> str:
        return self._api_keys[self._key_idx]

    def _rotate_key(self) -> None:
        self._key_idx = (self._key_idx + 1) % len(self._api_keys)

    # ------------------------------------------------------------------
    # Groq API calls
    # ------------------------------------------------------------------

    def _call_groq_transcribe(
        self,
        *,
        file_path: Path,
        config: TranscriptionConfig,
        cancel_check: Optional[Callable[[], bool]],
    ) -> Dict[str, object]:
        assert self._http is not None
        model = (config.model_size or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

        data: Dict[str, str] = {
            "model": model,
            "response_format": "verbose_json",
        }

        # Groq expects ISO-639-1 code to help; keep as-is if provided.
        if config.language:
            data["language"] = str(config.language).strip()

        # Groq supports temperature 0..1; keep within bounds.
        temperature = config.temperature if isinstance(config.temperature, (int, float)) else 0.0
        try:
            temperature_f = float(temperature)
        except Exception:
            temperature_f = 0.0
        if temperature_f < 0:
            temperature_f = 0.0
        if temperature_f > 1:
            temperature_f = 1.0
        data["temperature"] = f"{temperature_f:.3f}"

        # Request segment timestamps to align with existing Segment schema.
        # Per docs: timestamp_granularities[] requires response_format=verbose_json.
        data["timestamp_granularities[]"] = "segment"

        last_error: Optional[RuntimeError] = None
        total_cycles = self._max_quota_retries + 1

        for cycle in range(total_cycles):
            saw_quota_error = False
            for _ in range(len(self._api_keys)):
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("Transcription cancelled by user")
                key = self._current_key
                try:
                    with file_path.open("rb") as f:
                        files = {"file": (file_path.name, f.read())}
                        resp = self._http.post(
                            _GROQ_TRANSCRIPT_PATH,
                            headers={"Authorization": f"Bearer {key}"},
                            data=data,
                            files=files,
                        )
                except httpx.HTTPError as exc:
                    last_error = RuntimeError(f"Groq transport error: {exc}")
                    logger.warning(
                        "GroqEngine: transport error on key #%d: %s (rotating)",
                        self._key_idx + 1,
                        exc,
                    )
                    self._rotate_key()
                    continue

                if resp.status_code == 429:
                    saw_quota_error = True
                    detail = self._safe_error_detail(resp)
                    last_error = RuntimeError(f"Groq rate limit/quota (429): {detail}")
                    logger.warning(
                        "GroqEngine: 429 on key #%d: %s (rotating)",
                        self._key_idx + 1,
                        detail,
                    )
                    self._rotate_key()
                    continue

                if 500 <= resp.status_code <= 599:
                    detail = self._safe_error_detail(resp)
                    last_error = RuntimeError(f"Groq server error ({resp.status_code}): {detail}")
                    logger.warning(
                        "GroqEngine: %d on key #%d: %s (rotating)",
                        resp.status_code,
                        self._key_idx + 1,
                        detail,
                    )
                    self._rotate_key()
                    continue

                if resp.status_code != 200:
                    detail = self._safe_error_detail(resp)
                    raise RuntimeError(f"Groq transcription failed ({resp.status_code}): {detail}")

                try:
                    payload = resp.json()
                except json.JSONDecodeError:
                    raise RuntimeError(f"Groq transcription returned non-JSON: {resp.text[:2000]}")

                if not isinstance(payload, dict):
                    raise RuntimeError(f"Groq transcription response malformed: {payload!r}")
                return payload  # type: ignore[return-value]

            if saw_quota_error and cycle < total_cycles - 1:
                logger.warning(
                    "GroqEngine: all %d key(s) rate-limited; sleeping %.1fs before retry cycle %d/%d",
                    len(self._api_keys),
                    self._quota_retry_wait_seconds,
                    cycle + 2,
                    total_cycles,
                )
                self._interruptible_sleep(self._quota_retry_wait_seconds, cancel_check)
                self._key_idx = 0
                continue

            if last_error is not None:
                raise last_error
            raise RuntimeError("Groq transcription failed without a detailed error.")

        raise RuntimeError("Groq transcription exhausted retries unexpectedly.")

    def _transcribe_single_file(
        self,
        path: Path,
        config: TranscriptionConfig,
        cancel_check: Optional[Callable[[], bool]],
    ) -> TranscriptionResult:
        payload = self._call_groq_transcribe(file_path=path, config=config, cancel_check=cancel_check)
        language = str(payload.get("language") or "unknown")
        duration = self._safe_float(payload.get("duration")) or self._probe_duration_ffprobe(path) or 0.0
        raw_segments = payload.get("segments") or []
        if not isinstance(raw_segments, list):
            raw_segments = []

        segments: list[Segment] = []
        for idx, seg in enumerate(raw_segments):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Transcription cancelled by user")
            if not isinstance(seg, dict):
                continue
            start = self._safe_float(seg.get("start")) or 0.0
            end = self._safe_float(seg.get("end")) or start
            text = str(seg.get("text") or "")
            segments.append(Segment(id=idx, start=start, end=end, text=text))

        return TranscriptionResult(
            audio_path=str(path),
            language=language,
            language_probability=1.0,
            duration=float(duration),
            segments=segments,
            engine_name=self.engine_name,
            model_name=(config.model_size or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL,
        )

    # ------------------------------------------------------------------
    # ffmpeg helpers
    # ------------------------------------------------------------------

    def _split_audio_ffmpeg(
        self,
        src: Path,
        out_dir: Path,
        cancel_check: Optional[Callable[[], bool]],
    ) -> List[Path]:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Transcription cancelled by user")

        out_dir.mkdir(parents=True, exist_ok=True)
        ext = src.suffix.lstrip(".").lower() or "mp3"
        template = out_dir / f"chunk_%03d.{ext}"

        # Segment using stream copy for speed (may still exceed size on some codecs).
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            str(_CHUNK_SECONDS),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            str(template),
        ]

        logger.info("GroqEngine: splitting oversized audio into 10-min chunks via ffmpeg.")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg chunking failed: {stderr[:4000]}") from exc

        chunks = sorted(out_dir.glob(f"chunk_*.{ext}"))
        return [p for p in chunks if p.is_file() and p.stat().st_size > 0]

    def _reencode_small_mp3(self, src: Path, out_dir: Path) -> Path:
        out_path = out_dir / f"{src.stem}_reenc_{uuid.uuid4().hex[:8]}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            str(out_path),
        ]
        logger.info("GroqEngine: re-encoding chunk to mp3 for size safety: %s", src.name)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg re-encode failed: {stderr[:4000]}") from exc
        if not out_path.is_file() or out_path.stat().st_size <= 0:
            raise RuntimeError("ffmpeg re-encode produced no output file.")
        return out_path

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_error_detail(resp: httpx.Response) -> str:
        try:
            j = resp.json()
            if isinstance(j, dict):
                err = j.get("error")
                if isinstance(err, dict) and err.get("message"):
                    return str(err["message"])
            return resp.text
        except Exception:
            return resp.text

    @staticmethod
    def _interruptible_sleep(seconds: float, cancel_check: Optional[Callable[[], bool]]) -> None:
        # Simple cooperative sleep; no dependency on scene.base_client.
        end_at = time.time() + max(0.0, float(seconds))
        while time.time() < end_at:
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Transcription cancelled by user")
            time.sleep(0.2)

    @staticmethod
    def _probe_duration_ffprobe(path: Path) -> Optional[float]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="replace").strip()
            if not out:
                return None
            return float(out)
        except Exception:
            return None

