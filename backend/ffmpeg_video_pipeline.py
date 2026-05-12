"""Run the ffmpeg-based image-folder + audio + scenes → MP4 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from video_combine.combine import collect_images, load_scenes, load_subtitle_segments
from video_combine.ffmpeg_combine import build_video

from .jobs import Job
from .video_combine_settings import load_merged_video_combine_defaults

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _pick_opt(merged: dict[str, Any], key: str, explicit: Optional[Any], fallback: Any) -> Any:
    """Prefer explicit API value; otherwise YAML/env merged value; otherwise code default."""
    if explicit is not None:
        return explicit
    if key in merged and merged[key] is not None:
        return merged[key]
    return fallback


def run_ffmpeg_video_pipeline(
    job: Job,
    *,
    images_dir: Path,
    scenes_path: Path,
    subtitles_path: Optional[Path],
    audio_path: Optional[Path],
    output_path: Path,
    resolution: tuple[int, int] = (1920, 1080),
    fps: Optional[int] = None,
    threads: Optional[int] = None,
    preset: Optional[str] = None,
    narration_volume: Optional[float] = None,
    music_specs: Optional[List[Dict[str, Any]]] = None,
    music_base_dir: Optional[Path] = None,
    crf: Optional[int] = None,
    subtitle_font_name: Optional[str] = None,
    subtitle_font_size: Optional[int] = None,
    subtitle_bottom_margin: Optional[int] = None,
    subtitle_max_lines: Optional[int] = None,
    subtitle_max_chars: Optional[int] = None,
    subtitle_split_by_space: Optional[bool] = None,
    subtitle_black_background: Optional[bool] = None,
    subtitle_stroke_width: Optional[int] = None,
    pre_scale: Optional[int] = None,
    burn_subtitles: Optional[bool] = None,
) -> Dict[str, Any]:
    merged, _ = load_merged_video_combine_defaults(_REPO_ROOT)

    fps_i = int(_pick_opt(merged, "fps", fps, 30))
    threads_i = int(_pick_opt(merged, "threads", threads, 4))
    narration_volume_f = float(_pick_opt(merged, "narration_volume", narration_volume, 1.0))
    crf_i = int(_pick_opt(merged, "crf", crf, 20))
    subtitle_font_name_s = str(_pick_opt(merged, "subtitle_font_name", subtitle_font_name, "Arial"))
    subtitle_font_size_i = int(_pick_opt(merged, "subtitle_font_size", subtitle_font_size, 44))
    subtitle_bottom_margin_i = int(_pick_opt(merged, "subtitle_bottom_margin", subtitle_bottom_margin, 72))
    subtitle_max_lines_i = int(_pick_opt(merged, "subtitle_max_lines", subtitle_max_lines, 2))
    subtitle_max_chars_i = int(_pick_opt(merged, "subtitle_max_chars", subtitle_max_chars, 20))
    subtitle_split_b = bool(_pick_opt(merged, "subtitle_split_by_space", subtitle_split_by_space, True))
    subtitle_black_b = bool(_pick_opt(merged, "subtitle_black_background", subtitle_black_background, True))
    subtitle_stroke_i = int(_pick_opt(merged, "subtitle_stroke_width", subtitle_stroke_width, 3))
    pre_scale_i = int(_pick_opt(merged, "pre_scale", pre_scale, 4))
    burn_b = bool(_pick_opt(merged, "burn_subtitles", burn_subtitles, True))

    preset_raw = _pick_opt(merged, "preset", preset, None)
    effective_preset = str(preset_raw) if preset_raw else "medium"

    job.raise_if_cancelled()
    job.update(message="Collecting ffmpeg inputs…")
    images = collect_images(images_dir)
    scenes = load_scenes(scenes_path)
    subtitles = (
        load_subtitle_segments(subtitles_path)
        if (burn_b and subtitles_path)
        else None
    )
    job.append_log(f"Found {len(images)} image files")
    job.append_log(f"Found {len(scenes)} scene entries")
    if burn_b:
        job.append_log(f"Loaded {len(subtitles or [])} subtitle segments")
    else:
        job.append_log("Subtitles: disabled (clean video, no burn-in)")

    n = min(len(images), len(scenes))
    job.update(total=n, current=0, message=f"Preparing {n} ffmpeg clips…")
    for i in range(n):
        job.raise_if_cancelled()
        scene = scenes[i]
        duration = float(scene["end"]) - float(scene["start"])
        image_name = images[i].name
        job.update(current=i + 1, total=n, message=f"Prepared scene {i + 1}/{n} — {image_name}")
        job.append_log(
            f"Scene {i + 1}/{n}: {image_name} | "
            f"{float(scene['start']):.2f}s -> {float(scene['end']):.2f}s ({duration:.2f}s)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    job.update(current=n, total=n, message="Rendering with ffmpeg…")
    job.append_log(
        f"Render settings: {resolution[0]}x{resolution[1]} @ {fps_i}fps, "
        f"preset={effective_preset}, crf={crf_i}, threads={threads_i}"
    )
    last_render_log = {"sec": -1.0}

    def _on_ffmpeg_progress(done_seconds: float, total_seconds: float) -> None:
        pct = max(0.0, min(100.0, (done_seconds / max(0.001, total_seconds)) * 100.0))
        job.update(
            current=n,
            total=n,
            message=f"Rendering with ffmpeg… {pct:.1f}% ({done_seconds:.1f}/{total_seconds:.1f}s)",
        )
        if done_seconds - last_render_log["sec"] >= 1.0:
            job.append_log(
                f"ffmpeg progress: {pct:.1f}% ({done_seconds:.1f}/{total_seconds:.1f}s)"
            )
            last_render_log["sec"] = done_seconds

    build_video(
        images=images,
        scenes=scenes,
        subtitle_segments=subtitles,
        audio_path=audio_path,
        output_path=output_path,
        resolution=resolution,
        fps=fps_i,
        threads=threads_i,
        preset=effective_preset,
        crf=crf_i,
        music_specs=music_specs or [],
        narration_volume=narration_volume_f,
        music_base_dir=music_base_dir,
        subtitle_font_name=subtitle_font_name_s,
        subtitle_font_size=subtitle_font_size_i,
        subtitle_bottom_margin=subtitle_bottom_margin_i,
        subtitle_max_lines=subtitle_max_lines_i,
        subtitle_max_chars=subtitle_max_chars_i,
        subtitle_split_by_space=subtitle_split_b,
        subtitle_black_background=subtitle_black_b,
        subtitle_stroke_width=subtitle_stroke_i,
        pre_scale=pre_scale_i,
        burn_subtitles=burn_b,
        cancel_check=job.is_cancel_requested,
        progress_callback=_on_ffmpeg_progress,
    )

    total_duration = max(float(s["end"]) for s in scenes[:n]) if n else 0.0
    return {
        "output_path": str(output_path),
        "duration": total_duration,
        "clip_count": n,
        "engine": "ffmpeg",
        "scenes_path": str(scenes_path),
        "subtitles_path": str(subtitles_path) if subtitles_path else None,
        "burn_subtitles": burn_b,
    }
