"""Tiny JSON-backed store for UI state that should survive app restarts."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict

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
}

_lock = Lock()


def _read_raw() -> Dict[str, Any]:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8")) or {}
    except json.JSONDecodeError:
        return {}


def load_state() -> Dict[str, Any]:
    with _lock:
        data = {**_DEFAULTS, **_read_raw()}
        return data


def save_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        current = {**_DEFAULTS, **_read_raw()}
        for k, v in patch.items():
            if k in _DEFAULTS:
                current[k] = v
        _STATE_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return current
