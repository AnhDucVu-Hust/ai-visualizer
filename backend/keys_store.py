"""Simple JSON-backed API key activation store."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
_KEYS_PATH = _ROOT / "backend" / "keys.json"

_lock = Lock()


def _read_raw() -> Dict[str, Dict[str, Any]]:
    if not _KEYS_PATH.exists():
        return {}
    try:
        payload = json.loads(_KEYS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _write_raw(data: Dict[str, Dict[str, Any]]) -> None:
    _KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _KEYS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def activate_key(key: str) -> Dict[str, Any]:
    normalized = key.strip()
    if not normalized:
        return {"success": False, "error": "KEY_NOT_FOUND"}

    with _lock:
        keys = _read_raw()
        entry = keys.get(normalized)
        if entry is None:
            return {"success": False, "error": "KEY_NOT_FOUND"}
        if bool(entry.get("activated")):
            return {"success": False, "error": "KEY_ALREADY_ACTIVATED"}

        entry["activated"] = True
        keys[normalized] = entry
        _write_raw(keys)
        return {"success": True}


def verify_key(key: str) -> Dict[str, Any]:
    normalized = key.strip()
    if not normalized:
        return {"success": False, "error": "KEY_NOT_FOUND"}

    with _lock:
        keys = _read_raw()
        entry = keys.get(normalized)
        if entry is None:
            return {"success": False, "error": "KEY_NOT_FOUND"}
        if not bool(entry.get("activated")):
            return {"success": False, "error": "KEY_NOT_ACTIVATED"}
        return {"success": True}
