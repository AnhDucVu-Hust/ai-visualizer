"""Tiny JSON-backed store for UI state that should survive app restarts."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
_STATE_PATH = _ROOT / ".app_state.json"

_DEFAULTS: Dict[str, Any] = {
    "last_audio_path": None,
    "last_output_dir": "results",
    "last_images_dir": None,
    "last_scenes_path": "results/scenes.json",
    "last_video_output": "results/video.mp4",
    "last_min_duration": 7.0,
    "last_max_duration": 20.0,
    "current_job_id": None,
}

_lock = Lock()


def _read_raw() -> Dict[str, Any]:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8")) or {}
    except json.JSONDecodeError:
        return {}


def _normalize_client_key(client_key: Optional[str]) -> Optional[str]:
    if client_key is None:
        return None
    key = str(client_key).strip()
    return key or None


def _client_bucket(raw: Dict[str, Any], client_key: Optional[str]) -> Dict[str, Any]:
    key = _normalize_client_key(client_key)
    by_client = raw.get("by_client")
    if not key or not isinstance(by_client, dict):
        return {}
    bucket = by_client.get(key)
    return bucket if isinstance(bucket, dict) else {}


def _global_bucket(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: raw.get(k, default) for k, default in _DEFAULTS.items()}


def load_state(client_key: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        raw = _read_raw()
        if client_key is None:
            return _global_bucket(raw)
        # Fallback to global values for first-time client buckets.
        return {**_global_bucket(raw), **_client_bucket(raw, client_key)}


def save_state(patch: Dict[str, Any], client_key: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        raw = _read_raw()
        key = _normalize_client_key(client_key)
        if key is None:
            current = _global_bucket(raw)
        else:
            # Seed new client bucket from existing global state.
            current = {**_global_bucket(raw), **_client_bucket(raw, key)}

        for k, v in patch.items():
            if k in _DEFAULTS:
                current[k] = v

        if key is None:
            persisted = dict(raw)
            persisted.update(current)
        else:
            persisted = dict(raw)
            by_client = persisted.get("by_client")
            if not isinstance(by_client, dict):
                by_client = {}
            by_client[key] = current
            persisted["by_client"] = by_client

        _STATE_PATH.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
        return current


def clear_current_job(client_key: Optional[str], job_id: str) -> Dict[str, Any]:
    with _lock:
        raw = _read_raw()
        key = _normalize_client_key(client_key)
        if key is None:
            current = _global_bucket(raw)
            if current.get("current_job_id") == job_id:
                current["current_job_id"] = None
                persisted = dict(raw)
                persisted.update(current)
                _STATE_PATH.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
            return current

        current = {**_global_bucket(raw), **_client_bucket(raw, key)}
        if current.get("current_job_id") == job_id:
            current["current_job_id"] = None
            persisted = dict(raw)
            by_client = persisted.get("by_client")
            if not isinstance(by_client, dict):
                by_client = {}
            by_client[key] = current
            persisted["by_client"] = by_client
            _STATE_PATH.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
        return current
