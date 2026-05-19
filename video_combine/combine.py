"""Combine a folder of images with scenes.json timestamps and an audio track.

Images are matched to scenes one-to-one, ordered by numeric filenames
(e.g. ``001.png``, ``002.png`` ...). Each image is shown
for its scene's ``end - start`` duration with a Ken Burns style zoom/pan.
The full audio track is then muxed over the resulting video.

Optional background music tracks can be layered on top of the narration at
specific times with per-track volumes. Specify them either inline via
``--music`` (repeatable), in bulk through ``--music-config music.json``, or
directly in the YAML config passed to ``--config``.

Examples:
    # Run everything from a YAML config
    python -m video_combine.combine --config video_combine_config.yaml

    # Override a single value from the config
    python -m video_combine.combine --config video_combine_config.yaml --fps 60

    # Pure CLI (no config file)
    python -m video_combine.combine \\
        --images path/to/images \\
        --scenes results/scenes.json \\
        --audio  audio/script.wav \\
        --output results/video.mp4

    # Narration + two background music cues
    python -m video_combine.combine \\
        --images path/to/images \\
        --audio  audio/script.wav \\
        --music  "file=music/intro.mp3,start=0,end=12,volume=0.3" \\
        --music  "file=music/outro.mp3,start=55,volume=0.25" \\
        --output results/video.mp4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont

# Scene images must be named ``<digits>.png`` only (e.g. ``001.png``, ``12.png``).
_IMAGE_DIR_RE = re.compile(r"^images(?:_(\d+))?$")


def _prefix_index(path: Path) -> int | None:
    """Return the scene index from ``NNN.png``, or ``None`` if the name is invalid."""
    if path.suffix.lower() != ".png":
        return None
    stem = path.stem.strip()
    if not stem.isdigit():
        return None
    return int(stem)


def collect_images(folder: Path) -> list[Path]:
    """Return ``NNN.png`` files in ``folder`` sorted by scene number.

    Only ``<digits>.png`` is accepted (e.g. ``001.png``, ``002.png``). Other
    names are skipped with a warning. If two files share the same index, the
    first one (alphabetical tiebreak) wins.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    all_images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if not all_images:
        raise FileNotFoundError(f"No .png images found in {folder}")

    indexed: list[tuple[int, Path]] = []
    skipped: list[Path] = []
    for p in all_images:
        idx = _prefix_index(p)
        if idx is None:
            skipped.append(p)
        else:
            indexed.append((idx, p))

    if skipped:
        names = ", ".join(sorted(p.name for p in skipped))
        print(
            f"[warn] Ignoring {len(skipped)} file(s) not named <digits>.png: {names}",
            file=sys.stderr,
        )

    if not indexed:
        raise FileNotFoundError(
            f"No <digits>.png images found in {folder} (e.g. 001.png, 002.png)"
        )

    indexed.sort(key=lambda t: (t[0], t[1].name))

    seen: dict[int, Path] = {}
    for idx, p in indexed:
        if idx in seen:
            print(
                f"[warn] Duplicate prefix {idx:02d}: {seen[idx].name} and {p.name} "
                f"(keeping {seen[idx].name})",
                file=sys.stderr,
            )
        else:
            seen[idx] = p

    return [seen[k] for k in sorted(seen)]


def discover_image_dirs(scene_root: Path) -> list[Path]:
    """Return ``images``, ``images_1``, ``images_2``, … under *scene_root*, sorted.

    Folder ``images`` sorts before ``images_1``, ``images_2``, etc. When no
    matching subfolder exists, returns ``[scene_root]`` so numbered files in
    the scene root still work.
    """
    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {scene_root}")

    found: list[tuple[int, Path]] = []
    for child in scene_root.iterdir():
        if not child.is_dir():
            continue
        m = _IMAGE_DIR_RE.match(child.name)
        if m:
            suffix = m.group(1)
            order = 0 if suffix is None else int(suffix)
            found.append((order, child))

    if found:
        found.sort(key=lambda t: t[0])
        return [p for _, p in found]

    return [scene_root]


def collect_images_from_dirs(folders: list[Path]) -> list[Path]:
    """Collect images from one or more folders in order.

    Each folder uses the same ``NNN.png`` naming as :func:`collect_images`.
    Later folders continue the global sequence (e.g. ``001``–``100`` in
    ``images_1``, then ``001``–``099`` in ``images_2`` map to scenes 1–199).
    """
    if not folders:
        raise FileNotFoundError("No image folders provided")

    if len(folders) == 1:
        return collect_images(folders[0])

    all_images: list[Path] = []
    for folder in folders:
        chunk = collect_images(folder)
        if not chunk:
            raise FileNotFoundError(f"No images found in {folder}")
        if all_images:
            start = len(all_images) + 1
            end = len(all_images) + len(chunk)
            print(
                f"[info] {folder.name}: {len(chunk)} images (scenes {start}–{end})",
                file=sys.stderr,
            )
        all_images.extend(chunk)
    return all_images


def parse_music_spec(s: str) -> dict:
    """Parse a single ``--music`` spec string into a dict.

    Format: comma-separated ``key=value`` pairs. The first bare segment
    (no ``=``) is treated as ``file=...``. Recognised keys:

    - ``file`` / ``path`` (required): path to the music file
    - ``start`` (float seconds, default 0.0): when on the video timeline the
      track should start playing
    - ``end`` (float seconds, optional): when to stop. If omitted, plays to
      the natural end of the clip (or the end of the video, whichever comes
      first).
    - ``duration`` / ``dur`` (float seconds, optional): alternative to
      ``end``. Interpreted as ``end = start + duration``.
    - ``offset`` (float seconds, default 0.0): skip this much at the head of
      the music file before it starts playing.
    - ``volume`` / ``vol`` (float, default 1.0): linear volume multiplier
      (e.g. 0.3 = 30% loudness).

    Examples:
        "file=music/intro.mp3,start=0,end=12,volume=0.3"
        "music/outro.mp3,start=55,volume=0.25"
        "bed.mp3,volume=0.15"  # plays from 0 to end of video
    """
    spec: dict = {
        "file": None,
        "start": 0.0,
        "end": None,
        "duration": None,
        "offset": 0.0,
        "volume": 1.0,
    }
    segments = [seg.strip() for seg in s.split(",") if seg.strip()]
    for seg in segments:
        if "=" not in seg:
            if spec["file"] is None:
                spec["file"] = seg
                continue
            raise argparse.ArgumentTypeError(
                f"Invalid music spec segment {seg!r} in {s!r}: expected key=value."
            )
        key, value = seg.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in ("file", "path"):
            spec["file"] = value
        elif key == "start":
            spec["start"] = float(value)
        elif key == "end":
            spec["end"] = float(value)
        elif key in ("duration", "dur"):
            spec["duration"] = float(value)
        elif key == "offset":
            spec["offset"] = float(value)
        elif key in ("volume", "vol"):
            spec["volume"] = float(value)
        else:
            raise argparse.ArgumentTypeError(
                f"Unknown music field {key!r} in {s!r}. "
                "Allowed: file, start, end, duration, offset, volume."
            )
    if not spec["file"]:
        raise argparse.ArgumentTypeError(f"Music spec missing file= : {s!r}")
    if spec["end"] is None and spec["duration"] is not None:
        spec["end"] = spec["start"] + spec["duration"]
    if spec["end"] is not None and spec["end"] <= spec["start"]:
        raise argparse.ArgumentTypeError(
            f"Music spec has end <= start: {s!r}"
        )
    if spec["volume"] < 0:
        raise argparse.ArgumentTypeError(f"Music volume must be >= 0: {s!r}")
    spec.pop("duration", None)
    return spec


def _normalize_music_item(item: dict, source: str, index: int) -> dict:
    """Validate and canonicalise a single music-spec dict.

    ``source`` is a human-readable description of where the item came from
    (used in error messages, e.g. ``"config.yaml"``).
    """
    if not isinstance(item, dict):
        raise ValueError(f"{source} music entry #{index} is not a mapping: {item!r}")
    if "file" not in item and "path" not in item:
        raise ValueError(f"{source} music entry #{index} missing 'file'/'path'.")
    spec = {
        "file": item.get("file") or item.get("path"),
        "start": float(item.get("start", 0.0)),
        "end": (float(item["end"]) if item.get("end") is not None else None),
        "offset": float(item.get("offset", 0.0)),
        "volume": float(item.get("volume", item.get("vol", 1.0))),
    }
    if spec["end"] is None and item.get("duration") is not None:
        spec["end"] = spec["start"] + float(item["duration"])
    if spec["end"] is not None and spec["end"] <= spec["start"]:
        raise ValueError(f"{source} music entry #{index} has end <= start.")
    if spec["volume"] < 0:
        raise ValueError(f"{source} music entry #{index} has negative volume.")
    return spec


def load_music_config(config_path: Path) -> list[dict]:
    """Load music specs from a JSON file.

    The JSON may be a list of objects, or an object with a ``"music"`` key
    whose value is a list of objects. Each object accepts the same fields as
    ``parse_music_spec`` (``file``, ``start``, ``end``, ``duration``,
    ``offset``, ``volume``).
    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    items = data.get("music", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(
            f"Music config {config_path} must be a list (or an object with 'music' list)."
        )
    return [_normalize_music_item(item, str(config_path), i) for i, item in enumerate(items)]


def load_yaml_config(config_path: Path) -> tuple[dict, list[dict]]:
    """Load a YAML config file and return ``(argparse_defaults, music_specs)``.

    ``argparse_defaults`` is a dict whose keys match the argparse ``dest``
    names in :func:`main`; it can be passed to ``parser.set_defaults(**d)``.
    ``music_specs`` is the list of music entries from the ``music:`` section.

    Relative paths are resolved against the config file's parent directory so
    the YAML can live next to the project root without surprises.
    """
    try:
        import yaml  # local import so the module works even without pyyaml
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "pyyaml is required to load --config YAML files. "
            "Install it with: pip install pyyaml"
        ) from exc

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must be a YAML mapping at the top level.")

    base_dir = config_path.resolve().parent
    defaults: dict = {}

    def _as_path(value) -> Path:
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (base_dir / p)

    inp = raw.get("input") or {}
    if not isinstance(inp, dict):
        raise ValueError(f"{config_path}: 'input' must be a mapping.")
    if inp.get("images") is not None:
        defaults["images"] = _as_path(inp["images"])
    if inp.get("scenes") is not None:
        defaults["scenes"] = _as_path(inp["scenes"])
    if inp.get("subtitles_json") is not None:
        defaults["subtitles_json"] = _as_path(inp["subtitles_json"])

    aud = raw.get("audio") or {}
    if not isinstance(aud, dict):
        raise ValueError(f"{config_path}: 'audio' must be a mapping.")
    if aud.get("narration") is not None:
        defaults["audio"] = _as_path(aud["narration"])
    if aud.get("enabled") is False:
        defaults["no_audio"] = True
    if aud.get("narration_volume") is not None:
        defaults["narration_volume"] = float(aud["narration_volume"])

    vid = raw.get("video") or {}
    if not isinstance(vid, dict):
        raise ValueError(f"{config_path}: 'video' must be a mapping.")
    if vid.get("resolution") is not None:
        defaults["resolution"] = parse_resolution(str(vid["resolution"]))
    if vid.get("fps") is not None:
        defaults["fps"] = int(vid["fps"])
    if vid.get("threads") is not None:
        defaults["threads"] = int(vid["threads"])
    if vid.get("preset") is not None:
        defaults["preset"] = str(vid["preset"])
    if vid.get("subtitle_font_size") is not None:
        defaults["subtitle_font_size"] = int(vid["subtitle_font_size"])
    if vid.get("subtitle_bottom_margin") is not None:
        defaults["subtitle_bottom_margin"] = int(vid["subtitle_bottom_margin"])
    if vid.get("subtitle_max_lines") is not None:
        defaults["subtitle_max_lines"] = int(vid["subtitle_max_lines"])
    if vid.get("subtitle_max_chars") is not None:
        defaults["subtitle_max_chars"] = int(vid["subtitle_max_chars"])
    if vid.get("subtitle_split_by_space") is not None:
        defaults["subtitle_split_by_space"] = bool(vid["subtitle_split_by_space"])
    if vid.get("subtitle_black_background") is not None:
        defaults["subtitle_black_background"] = bool(vid["subtitle_black_background"])
    if vid.get("subtitle_stroke_width") is not None:
        defaults["subtitle_stroke_width"] = int(vid["subtitle_stroke_width"])
    if vid.get("burn_subtitles") is not None:
        defaults["burn_subtitles"] = bool(vid["burn_subtitles"])
    if vid.get("subtitle_font_name") is not None:
        defaults["subtitle_font_name"] = str(vid["subtitle_font_name"])

    if raw.get("output") is not None:
        defaults["output"] = _as_path(raw["output"])

    music_raw = raw.get("music") or []
    if not isinstance(music_raw, list):
        raise ValueError(f"{config_path}: 'music' must be a list.")
    music_specs = [
        _normalize_music_item(item, str(config_path), i)
        for i, item in enumerate(music_raw)
    ]

    return defaults, music_specs


def _resolve_music_path(spec: dict, base_dir: Path) -> Path:
    p = Path(spec["file"])
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Music file not found: {p}")
    return p


def build_music_clips(
    music_specs: list[dict],
    video_duration: float,
    base_dir: Path,
) -> list:
    """Turn music specs into positioned, trimmed, volume-scaled AudioClips.

    Each returned clip is already placed on the video timeline via
    ``with_start`` so it can be fed directly into ``CompositeAudioClip``.
    """
    clips = []
    for i, spec in enumerate(music_specs):
        path = _resolve_music_path(spec, base_dir)
        start = max(0.0, float(spec["start"]))
        if start >= video_duration:
            print(
                f"[warn] Music track #{i+1} ({path.name}) starts at {start:.2f}s "
                f"which is beyond the video duration ({video_duration:.2f}s); skipping.",
                file=sys.stderr,
            )
            continue

        clip = AudioFileClip(str(path))
        src_dur = float(clip.duration)
        offset = max(0.0, float(spec.get("offset", 0.0)))
        if offset >= src_dur:
            print(
                f"[warn] Music track #{i+1} ({path.name}) offset={offset:.2f}s "
                f"is past its duration ({src_dur:.2f}s); skipping.",
                file=sys.stderr,
            )
            clip.close()
            continue

        requested_end = spec["end"] if spec["end"] is not None else video_duration
        end_on_timeline = min(float(requested_end), video_duration)
        wanted_dur = end_on_timeline - start
        available_dur = src_dur - offset
        dur = min(wanted_dur, available_dur)
        if dur <= 0:
            clip.close()
            continue

        clip = clip.subclipped(offset, offset + dur)
        volume = float(spec.get("volume", 1.0))
        if volume != 1.0:
            clip = clip.with_volume_scaled(volume)
        clip = clip.with_start(start)

        print(
            f"  [music {i+1}] {path.name}  "
            f"{start:.2f}s \u2192 {start + dur:.2f}s  "
            f"(vol {volume:.2f}, offset {offset:.2f}s)"
        )
        clips.append(clip)

    return clips


def load_scenes(scenes_path: Path) -> list[dict]:
    """Load the ``scenes`` list from a ``scenes.json`` (or compatible) file."""
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes", data if isinstance(data, list) else None)
    if not scenes:
        raise ValueError(f"Could not find scenes in {scenes_path}")
    for i, s in enumerate(scenes):
        if "start" not in s or "end" not in s:
            raise ValueError(f"Scene {i} is missing start/end timestamps")
    return scenes


def load_subtitle_segments(out_json_path: Path) -> list[dict]:
    """Load subtitle cues from an STT ``out.json`` file's ``segments`` list."""
    data = json.loads(out_json_path.read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Could not find a 'segments' list in {out_json_path}")

    cues: list[dict] = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        if "start" not in seg or "end" not in seg:
            raise ValueError(f"Subtitle segment {i} in {out_json_path} is missing start/end.")
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        cues.append({"start": start, "end": end, "text": text})
    return cues


def _wrap_subtitle_text(
    text: str,
    max_chars_per_line: int,
    *,
    split_by_space: bool = True,
) -> str:
    max_chars_per_line = max(1, int(max_chars_per_line))
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    if not split_by_space:
        return "\n".join(
            normalized[i: i + max_chars_per_line]
            for i in range(0, len(normalized), max_chars_per_line)
        )

    words = normalized.split(" ")
    if not words:
        return ""
    # Languages like Japanese often have no spaces; split by characters then.
    if len(words) == 1:
        token = words[0]
        return "\n".join(
            token[i: i + max_chars_per_line]
            for i in range(0, len(token), max_chars_per_line)
        )
    lines: list[str] = []
    line = words[0]
    for word in words[1:]:
        if len(word) > max_chars_per_line:
            if line:
                lines.append(line)
            chunks = [
                word[i: i + max_chars_per_line]
                for i in range(0, len(word), max_chars_per_line)
            ]
            lines.extend(chunks[:-1])
            line = chunks[-1]
            continue
        candidate = f"{line} {word}" if line else word
        if len(candidate) <= max_chars_per_line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return "\n".join(lines)


def _subtitle_image_clip(
    text: str,
    resolution: tuple[int, int],
    *,
    font_size: int,
    bottom_margin: int,
    max_lines: int,
    max_chars_per_line: int,
    split_by_space: bool,
    black_background: bool,
    stroke_width: int,
) -> ImageClip:
    """Render a subtitle cue as an RGBA image clip."""
    width, height = resolution
    max_lines = min(4, max(1, int(max_lines)))
    if "\n" in text:
        wrapped = [line for line in text.splitlines() if line.strip()]
    else:
        auto_max_chars = max(12, int(width / max(font_size * 0.62, 1)))
        max_chars = min(auto_max_chars, max(1, max_chars_per_line))
        wrapped = _wrap_subtitle_text(
            text,
            max_chars,
            split_by_space=split_by_space,
        ).splitlines()
    if max_lines > 0:
        wrapped = wrapped[:max_lines]
    wrapped_text = "\n".join(wrapped)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial Unicode.ttf", font_size)
    except Exception:  # noqa: BLE001
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()

    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        spacing=int(font_size * 0.25),
        stroke_width=stroke_width,
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = max(0, height - bottom_margin - text_h)

    if black_background:
        pad_x = max(10, int(font_size * 0.45))
        pad_y = max(6, int(font_size * 0.25))
        bg = [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y]
        draw.rounded_rectangle(bg, radius=max(8, font_size // 3), fill=(0, 0, 0, 180))

    draw.multiline_text(
        (x, y),
        wrapped_text,
        font=font,
        fill=(255, 255, 255, 255),
        align="center",
        spacing=int(font_size * 0.25),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 230),
    )
    return ImageClip(np.array(canvas))


def build_subtitle_clips(
    subtitles: list[dict],
    *,
    resolution: tuple[int, int],
    video_duration: float,
    font_size: int,
    bottom_margin: int,
    max_lines: int,
    max_chars_per_line: int,
    split_by_space: bool,
    black_background: bool,
    stroke_width: int,
) -> list[ImageClip]:
    """Create timed subtitle overlays from STT segments."""
    clips: list[ImageClip] = []
    max_lines = min(4, max(1, int(max_lines)))
    for i, cue in enumerate(subtitles, 1):
        start = max(0.0, float(cue["start"]))
        end = min(float(cue["end"]), video_duration)
        if end <= start:
            continue
        text = str(cue["text"]).strip()
        if not text:
            continue

        wrapped_lines = _wrap_subtitle_text(
            text,
            max(1, int(max_chars_per_line)),
            split_by_space=split_by_space,
        ).splitlines()
        if not wrapped_lines:
            continue
        chunks = [wrapped_lines[j: j + max_lines] for j in range(0, len(wrapped_lines), max_lines)]
        chunk_texts = ["\n".join(chunk).strip() for chunk in chunks]
        chunk_weights = [max(1, len(t.replace("\n", ""))) for t in chunk_texts]
        total_weight = sum(chunk_weights)
        consumed_weight = 0

        for idx, chunk_text in enumerate(chunk_texts):
            if not chunk_text:
                consumed_weight += chunk_weights[idx]
                continue
            part_start = start + (end - start) * (consumed_weight / total_weight)
            consumed_weight += chunk_weights[idx]
            if idx == len(chunk_texts) - 1:
                part_end = end
            else:
                part_end = start + (end - start) * (consumed_weight / total_weight)
            if part_end <= part_start:
                continue
            clip = _subtitle_image_clip(
                text=chunk_text,
                resolution=resolution,
                font_size=font_size,
                bottom_margin=bottom_margin,
                max_lines=max_lines,
                max_chars_per_line=max_chars_per_line,
                split_by_space=split_by_space,
                black_background=black_background,
                stroke_width=stroke_width,
            ).with_start(part_start).with_duration(part_end - part_start)
            clips.append(clip)
        if i % 50 == 0:
            print(f"  [subs] built {i}/{len(subtitles)} subtitle cues")
    return clips


def ken_burns_clip(
    image_path: Path,
    duration: float,
    target_size: tuple[int, int],
    variant: int,
) -> CompositeVideoClip:
    """Create a Ken-Burns styled clip of ``duration`` seconds at ``target_size``.

    ``variant`` cycles through four motion flavours so adjacent scenes don't
    look identical: zoom-in center, zoom-out center, zoom-in pan right,
    zoom-in pan left.
    """
    tw, th = target_size

    base = ImageClip(str(image_path))
    iw, ih = base.size

    # Scale so the image fully covers the target (no black bars), with headroom
    # for zooming up to ~1.08x.
    cover = max(tw / iw, th / ih) * 1.10
    new_w, new_h = int(iw * cover) + 2, int(ih * cover) + 2
    base = base.resized((new_w, new_h)).with_duration(duration)

    zoom_amt = 0.08

    def z_in(t: float) -> float:
        return 1.0 + zoom_amt * (t / duration)

    def z_out(t: float) -> float:
        return 1.0 + zoom_amt * (1.0 - t / duration)

    pan_px = 30  # total pan offset across the clip

    if variant % 4 == 0:  # zoom in, centered
        base = base.resized(z_in).with_position(("center", "center"))
    elif variant % 4 == 1:  # zoom out, centered
        base = base.resized(z_out).with_position(("center", "center"))
    elif variant % 4 == 2:  # zoom in + pan right

        def pos_right(t: float):
            return ("center", "center") if duration <= 0 else (
                ((tw - new_w) // 2) - int(pan_px * (t / duration)),
                "center",
            )

        base = base.resized(z_in).with_position(pos_right)
    else:  # zoom in + pan left

        def pos_left(t: float):
            return (
                ((tw - new_w) // 2) + int(pan_px * (t / duration)),
                "center",
            )

        base = base.resized(z_in).with_position(pos_left)

    bg = ColorClip(size=target_size, color=(0, 0, 0), duration=duration)
    return CompositeVideoClip([bg, base], size=target_size).with_duration(duration)


def movement_clip(
    image_path: Path,
    duration: float,
    target_size: tuple[int, int],
    direction: int,
) -> CompositeVideoClip:
    """Create a directional movement clip (left/right/up/down, no zoom)."""
    tw, th = target_size

    base = ImageClip(str(image_path))
    iw, ih = base.size

    # Keep the same cover scale strategy as Ken Burns for consistent framing.
    cover = max(tw / iw, th / ih) * 1.10
    new_w, new_h = int(iw * cover) + 2, int(ih * cover) + 2
    base = base.resized((new_w, new_h)).with_duration(duration)

    move_px = 36
    safe_duration = max(duration, 1e-6)

    def pos_move_right(t: float):
        return (
            ((tw - new_w) // 2) + int(move_px * (t / safe_duration)),
            "center",
        )

    def pos_move_left(t: float):
        return (
            ((tw - new_w) // 2) - int(move_px * (t / safe_duration)),
            "center",
        )

    def pos_move_up(t: float):
        return (
            "center",
            ((th - new_h) // 2) - int(move_px * (t / safe_duration)),
        )

    def pos_move_down(t: float):
        return (
            "center",
            ((th - new_h) // 2) + int(move_px * (t / safe_duration)),
        )

    direction_mod = direction % 4
    if direction_mod == 0:
        base = base.with_position(pos_move_right)
    elif direction_mod == 1:
        base = base.with_position(pos_move_left)
    elif direction_mod == 2:
        base = base.with_position(pos_move_up)
    else:
        base = base.with_position(pos_move_down)

    bg = ColorClip(size=target_size, color=(0, 0, 0), duration=duration)
    return CompositeVideoClip([bg, base], size=target_size).with_duration(duration)


def build_video(
    images: list[Path],
    scenes: list[dict],
    audio_path: Path | None,
    output_path: Path,
    resolution: tuple[int, int],
    fps: int,
    threads: int,
    preset: str,
    music_specs: list[dict] | None = None,
    narration_volume: float = 1.0,
    music_base_dir: Path | None = None,
    subtitle_segments: list[dict] | None = None,
    subtitle_font_size: int = 44,
    subtitle_bottom_margin: int = 72,
    subtitle_max_lines: int = 2,
    subtitle_max_chars: int = 20,
    subtitle_split_by_space: bool = True,
    subtitle_black_background: bool = True,
    subtitle_stroke_width: int = 3,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    n = min(len(images), len(scenes))
    if len(images) != len(scenes):
        print(
            f"[warn] Image/scene count mismatch: {len(images)} images vs "
            f"{len(scenes)} scenes \u2014 using first {n} of each.",
            file=sys.stderr,
        )

    clips = []
    for i in range(n):
        if cancel_check and cancel_check():
            raise RuntimeError("Rendering cancelled by user.")
        scene = scenes[i]
        duration = float(scene["end"]) - float(scene["start"])
        if duration <= 0:
            print(f"[warn] Scene {i} has non-positive duration; skipping.", file=sys.stderr)
            continue
        img = images[i]
        effect_name = random.choice(("ken_burns", "movement"))
        print(
            f"  [{i+1:>3}/{n}] {img.name}  "
            f"{scene['start']:.2f}s \u2192 {scene['end']:.2f}s  "
            f"({duration:.2f}s) [{effect_name}]"
        )
        if effect_name == "ken_burns":
            clips.append(ken_burns_clip(img, duration, resolution, variant=i))
        else:
            clips.append(movement_clip(img, duration, resolution, direction=random.randint(0, 3)))

    if not clips:
        raise RuntimeError("No valid clips were generated.")

    video = concatenate_videoclips(clips, method="chain")

    narration = None
    if audio_path is not None:
        narration = AudioFileClip(str(audio_path))
        # Trim audio to video length (or vice versa) so muxing never fails.
        if narration.duration > video.duration:
            narration = narration.subclipped(0, video.duration)
        else:
            video = video.with_duration(narration.duration)
        if narration_volume != 1.0:
            narration = narration.with_volume_scaled(narration_volume)

    music_clips: list = []
    if music_specs:
        base_dir = music_base_dir or output_path.parent
        music_clips = build_music_clips(
            music_specs=music_specs,
            video_duration=float(video.duration),
            base_dir=base_dir,
        )

    if narration is not None and music_clips:
        mixed = CompositeAudioClip([narration, *music_clips])
        # CompositeAudioClip does not always inherit a duration; pin it so
        # ffmpeg writes a clean audio stream that matches the video length.
        mixed = mixed.with_duration(video.duration)
        video = video.with_audio(mixed)
    elif narration is not None:
        video = video.with_audio(narration)
    elif music_clips:
        mixed = CompositeAudioClip(music_clips).with_duration(video.duration)
        video = video.with_audio(mixed)

    if subtitle_segments:
        print(f"Building subtitles from {len(subtitle_segments)} segment cues...")
        subtitle_clips = build_subtitle_clips(
            subtitle_segments,
            resolution=resolution,
            video_duration=float(video.duration),
            font_size=subtitle_font_size,
            bottom_margin=subtitle_bottom_margin,
            max_lines=subtitle_max_lines,
            max_chars_per_line=subtitle_max_chars,
            split_by_space=subtitle_split_by_space,
            black_background=subtitle_black_background,
            stroke_width=subtitle_stroke_width,
        )
        if subtitle_clips:
            original_audio = video.audio
            video = CompositeVideoClip([video, *subtitle_clips], size=resolution).with_duration(
                video.duration
            )
            if original_audio is not None:
                video = video.with_audio(original_audio)
            print(f"Subtitles enabled: {len(subtitle_clips)} cues")
        else:
            print("[warn] No valid subtitle cues after trimming; subtitles skipped.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {output_path}  ({video.duration:.1f}s @ {fps}fps, {resolution[0]}x{resolution[1]})")
    if cancel_check and cancel_check():
        raise RuntimeError("Rendering cancelled by user.")
    video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset=preset,
        threads=threads,
    )


def parse_resolution(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"Expected WIDTHxHEIGHT (e.g. 1920x1080), got {s!r}"
        ) from exc


def detect_resolution(image_path: Path) -> tuple[int, int]:
    """Return the image's pixel size, rounded to even numbers.

    H.264 requires even width and height, so we floor-to-even here.
    """
    from PIL import Image  # lazy import; Pillow is already a project dep

    with Image.open(image_path) as im:
        w, h = im.size
    return (w - (w % 2), h - (h % 2))


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    argv_list = list(argv) if argv is not None else None

    # Pre-parse --config so we can use it as argparse defaults for everything
    # else. This lets CLI flags naturally override YAML values.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv_list)

    config_defaults: dict = {}
    config_music: list[dict] = []
    config_base_dir: Path | None = None
    if pre_args.config is not None:
        config_defaults, config_music = load_yaml_config(pre_args.config)
        config_base_dir = pre_args.config.resolve().parent

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=pre_args.config, help="YAML config file. CLI flags override values from the config.")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help=(
            "Scene root or image folder. If the path contains images/, images_1/, "
            "images_2/, … those folders are merged in order (each batch numbered from 001)."
        ),
    )
    parser.add_argument("--scenes", type=Path, default=repo_root / "results" / "scenes.json", help="Path to scenes.json (default: results/scenes.json).")
    parser.add_argument(
        "--subtitles-json",
        type=Path,
        default=None,
        help="Optional STT out.json path for segment-timed subtitles (independent of merged scenes).",
    )
    parser.add_argument("--audio", type=Path, default=repo_root / "audio" / "script.wav", help="Audio file to mux (default: audio/script.wav).")
    parser.add_argument("--output", type=Path, default=repo_root / "results" / "video.mp4", help="Output video file (default: results/video.mp4).")
    parser.add_argument("--resolution", type=parse_resolution, default=None, help="Output resolution, e.g. 1920x1080. If omitted, auto-detected from the first image.")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second (default: 30).")
    parser.add_argument("--no-audio", action="store_true", help="Do not mux the narration audio into the output.")
    parser.add_argument(
        "--narration-volume",
        type=float,
        default=1.0,
        help="Linear volume multiplier applied to the narration track (default: 1.0).",
    )
    parser.add_argument(
        "--music",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Background music track (repeatable). Format: "
            "'file=PATH,start=SEC,end=SEC,volume=0-1' (end/duration/offset optional). "
            "Example: --music 'file=music/bed.mp3,start=0,end=30,volume=0.3'"
        ),
    )
    parser.add_argument(
        "--music-config",
        type=Path,
        default=None,
        help="JSON file with a list of music specs (merged with --music). See module docstring for format.",
    )
    parser.add_argument("--threads", type=int, default=4, help="ffmpeg encoder threads (default: 4).")
    parser.add_argument("--preset", default="medium", help="libx264 preset (ultrafast..veryslow, default: medium).")
    parser.add_argument("--subtitle-font-size", type=int, default=44, help="Subtitle font size in pixels (default: 44).")
    parser.add_argument("--subtitle-bottom-margin", type=int, default=72, help="Bottom margin in pixels for subtitle placement (default: 72).")
    parser.add_argument("--subtitle-max-lines", type=int, default=2, help="Maximum wrapped subtitle lines (default: 2).")
    parser.add_argument("--subtitle-max-chars", type=int, default=20, help="Max characters per subtitle line (default: 20).")
    parser.add_argument("--subtitle-split-by-space", action=argparse.BooleanOptionalAction, default=True, help="Wrap subtitles by word boundaries when possible (default: true).")
    parser.add_argument("--subtitle-black-background", action=argparse.BooleanOptionalAction, default=True, help="Draw a black rounded subtitle background box (default: true).")
    parser.add_argument("--subtitle-stroke-width", type=int, default=3, help="Subtitle outline width (default: 3).")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(argv_list)

    if args.images is None:
        parser.error(
            "--images is required (either via CLI or set input.images in the --config file)."
        )

    image_dirs = discover_image_dirs(args.images)
    images = collect_images_from_dirs(image_dirs)
    scenes = load_scenes(args.scenes)
    audio = None if args.no_audio else args.audio
    if audio is not None and not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    if args.resolution is None:
        args.resolution = detect_resolution(images[0])
        print(
            f"Auto-detected resolution from {images[0].name}: "
            f"{args.resolution[0]}x{args.resolution[1]}"
        )

    subtitle_segments: list[dict] | None = None
    if args.subtitles_json is not None:
        if not args.subtitles_json.is_file():
            raise FileNotFoundError(f"Subtitles JSON not found: {args.subtitles_json}")
        subtitle_segments = load_subtitle_segments(args.subtitles_json)

    # Music precedence: config YAML first, then --music-config JSON, then
    # inline --music specs (so CLI entries always come last / on top).
    music_specs: list[dict] = list(config_music)
    if args.music_config is not None:
        if not args.music_config.is_file():
            raise FileNotFoundError(f"Music config not found: {args.music_config}")
        music_specs.extend(load_music_config(args.music_config))
    music_specs.extend(parse_music_spec(s) for s in args.music)

    music_base_dir = config_base_dir or repo_root

    if args.config is not None:
        print(f"Loaded config: {args.config}")
    if len(image_dirs) == 1:
        print(f"Found {len(images)} images in {image_dirs[0]}")
    else:
        names = ", ".join(d.name for d in image_dirs)
        print(f"Found {len(images)} images across {len(image_dirs)} folders ({names})")
    print(f"Found {len(scenes)} scenes in {args.scenes}")
    if audio is not None:
        print(f"Narration track: {audio} (volume {args.narration_volume:.2f})")
    if subtitle_segments is not None:
        print(f"Subtitle cues: {len(subtitle_segments)} from {args.subtitles_json}")
    if music_specs:
        print(f"Music tracks: {len(music_specs)}")
    print()

    build_video(
        images=images,
        scenes=scenes,
        audio_path=audio,
        output_path=args.output,
        resolution=args.resolution,
        fps=args.fps,
        threads=args.threads,
        preset=args.preset,
        music_specs=music_specs,
        narration_volume=args.narration_volume,
        music_base_dir=music_base_dir,
        subtitle_segments=subtitle_segments,
        subtitle_font_size=args.subtitle_font_size,
        subtitle_bottom_margin=args.subtitle_bottom_margin,
        subtitle_max_lines=args.subtitle_max_lines,
        subtitle_max_chars=args.subtitle_max_chars,
        subtitle_split_by_space=args.subtitle_split_by_space,
        subtitle_black_background=args.subtitle_black_background,
        subtitle_stroke_width=args.subtitle_stroke_width,
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
