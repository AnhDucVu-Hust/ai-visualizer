"""Merge ``video_combine_config.yaml`` with environment variables (same idea as ``config.py`` + ``.env``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

from video_combine.combine import load_yaml_config, parse_resolution

VIDEO_COMBINE_CONFIG_ENV = "VIDEO_COMBINE_CONFIG_PATH"


def resolve_video_combine_config_path(repo_root: Path) -> Optional[Path]:
    """``VIDEO_COMBINE_CONFIG_PATH`` if set and exists, else ``<repo_root>/video_combine_config.yaml``."""
    root = repo_root.resolve()
    raw = os.environ.get(VIDEO_COMBINE_CONFIG_ENV)
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{VIDEO_COMBINE_CONFIG_ENV} not found: {path}")
        return path
    default = root / "video_combine_config.yaml"
    if default.is_file():
        return default
    return None


def _truthy_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def apply_video_combine_env_overrides(defaults: dict[str, Any]) -> None:
    """
    Apply encoding-related env vars into ``defaults`` (same keys as ``load_yaml_config``).

    Paths (images, scenes, …) can be set for CLI workflows; backend upload APIs usually ignore them.
    """
    # --- input / output paths (optional; relative paths are resolved by callers vs repo root) ---
    path_pairs = [
        ("VIDEO_INPUT_IMAGES", "images"),
        ("VIDEO_INPUT_SCENES", "scenes"),
        ("VIDEO_INPUT_SUBTITLES_JSON", "subtitles_json"),
        ("VIDEO_AUDIO_NARRATION", "audio"),
        ("VIDEO_OUTPUT", "output"),
    ]
    for env_k, dict_k in path_pairs:
        raw = os.environ.get(env_k)
        if raw is not None and str(raw).strip() != "":
            defaults[dict_k] = str(raw).strip()

    if _truthy_env("VIDEO_AUDIO_DISABLED"):
        defaults["no_audio"] = True

    nv = os.environ.get("VIDEO_NARRATION_VOLUME")
    if nv is not None and str(nv).strip() != "":
        try:
            defaults["narration_volume"] = float(str(nv).strip())
        except ValueError:
            pass

    res = os.environ.get("VIDEO_RESOLUTION")
    if res is not None and str(res).strip() != "":
        try:
            defaults["resolution"] = parse_resolution(str(res).strip())
        except Exception:  # noqa: BLE001
            pass

    int_pairs = [
        ("VIDEO_FPS", "fps"),
        ("VIDEO_THREADS", "threads"),
        ("VIDEO_SUBTITLE_FONT_SIZE", "subtitle_font_size"),
        ("VIDEO_SUBTITLE_BOTTOM_MARGIN", "subtitle_bottom_margin"),
        ("VIDEO_SUBTITLE_MAX_LINES", "subtitle_max_lines"),
        ("VIDEO_SUBTITLE_MAX_CHARS", "subtitle_max_chars"),
        ("VIDEO_SUBTITLE_STROKE_WIDTH", "subtitle_stroke_width"),
        ("VIDEO_CRF", "crf"),
        ("VIDEO_PRE_SCALE", "pre_scale"),
    ]
    for env_k, dict_k in int_pairs:
        raw = os.environ.get(env_k)
        if raw is not None and str(raw).strip() != "":
            try:
                defaults[dict_k] = int(str(raw).strip())
            except ValueError:
                pass

    if os.environ.get("VIDEO_PRESET", "").strip():
        defaults["preset"] = os.environ.get("VIDEO_PRESET", "").strip()

    if os.environ.get("VIDEO_SUBTITLE_FONT_NAME", "").strip():
        defaults["subtitle_font_name"] = os.environ.get("VIDEO_SUBTITLE_FONT_NAME", "").strip()

    tb = _truthy_env("VIDEO_SUBTITLE_SPLIT_BY_SPACE")
    if tb is not None:
        defaults["subtitle_split_by_space"] = tb
    bb = _truthy_env("VIDEO_SUBTITLE_BLACK_BACKGROUND")
    if bb is not None:
        defaults["subtitle_black_background"] = bb
    bs = _truthy_env("VIDEO_BURN_SUBTITLES")
    if bs is not None:
        defaults["burn_subtitles"] = bs


def load_merged_video_combine_defaults(repo_root: Path) -> Tuple[dict[str, Any], list]:
    """
    Load YAML if present, then apply env overrides (env wins over YAML for set variables).
    Returns (defaults_dict, music_specs).
    """
    path = resolve_video_combine_config_path(repo_root)
    if path is not None:
        defaults, music = load_yaml_config(path)
    else:
        defaults, music = {}, []
    apply_video_combine_env_overrides(defaults)
    return defaults, music
