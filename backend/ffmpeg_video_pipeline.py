"""Run the ffmpeg-based image-folder + audio + scenes → MP4 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from video_combine.combine import collect_images, load_scenes, load_subtitle_segments
from video_combine.ffmpeg_combine import build_video

from .jobs import Job


def run_ffmpeg_video_pipeline(
    job: Job,
    *,
    images_dir: Path,
    scenes_path: Path,
    subtitles_path: Optional[Path],
    audio_path: Optional[Path],
    output_path: Path,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    threads: int = 4,
    preset: str = "medium",
    narration_volume: float = 1.0,
    music_specs: Optional[List[Dict[str, Any]]] = None,
    music_base_dir: Optional[Path] = None,
    crf: int = 20,
    subtitle_font_name: str = "Arial",
    subtitle_font_size: int = 44,
    subtitle_bottom_margin: int = 72,
    subtitle_max_lines: int = 2,
    subtitle_max_chars: int = 20,
    subtitle_split_by_space: bool = True,
    subtitle_black_background: bool = True,
    subtitle_stroke_width: int = 3,
    pre_scale: int = 4,
) -> Dict[str, Any]:
    job.raise_if_cancelled()
    job.update(message="Collecting ffmpeg inputs…")
    images = collect_images(images_dir)
    scenes = load_scenes(scenes_path)
    subtitles = load_subtitle_segments(subtitles_path) if subtitles_path else None
    job.append_log(f"Found {len(images)} image files")
    job.append_log(f"Found {len(scenes)} scene entries")
    job.append_log(f"Loaded {len(subtitles or [])} subtitle segments")

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
        f"Render settings: {resolution[0]}x{resolution[1]} @ {fps}fps, "
        f"preset={preset}, crf={crf}, threads={threads}"
    )
    last_render_log = {"sec": -1.0}

    def _on_ffmpeg_progress(done_seconds: float, total_seconds: float) -> None:
        pct = max(0.0, min(100.0, (done_seconds / max(0.001, total_seconds)) * 100.0))
        job.update(
            current=n,
            total=n,
            message=f"Rendering with ffmpeg… {pct:.1f}% ({done_seconds:.1f}/{total_seconds:.1f}s)",
        )
        # Throttle log lines to roughly 1-second granularity.
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
        fps=fps,
        threads=threads,
        preset=preset,
        crf=crf,
        music_specs=music_specs or [],
        narration_volume=narration_volume,
        music_base_dir=music_base_dir,
        subtitle_font_name=subtitle_font_name,
        subtitle_font_size=subtitle_font_size,
        subtitle_bottom_margin=subtitle_bottom_margin,
        subtitle_max_lines=subtitle_max_lines,
        subtitle_max_chars=subtitle_max_chars,
        subtitle_split_by_space=subtitle_split_by_space,
        subtitle_black_background=subtitle_black_background,
        subtitle_stroke_width=subtitle_stroke_width,
        pre_scale=pre_scale,
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
    }
