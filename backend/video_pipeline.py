"""Run the image-folder + audio + scenes → MP4 pipeline with progress reporting.

Wraps ``video_combine.combine`` so the UI can show a per-clip counter instead
of the CLI's stdout stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

from video_combine.combine import (
    build_subtitle_clips,
    build_music_clips,
    collect_images_from_dirs,
    ken_burns_clip,
    load_scenes,
    load_subtitle_segments,
)

from .jobs import Job


def run_video_pipeline(
    job: Job,
    *,
    image_dirs: list[Path],
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
    subtitle_font_size: int = 44,
    subtitle_bottom_margin: int = 72,
    subtitle_max_lines: int = 2,
    subtitle_max_chars: int = 20,
    subtitle_split_by_space: bool = True,
    subtitle_black_background: bool = True,
    subtitle_stroke_width: int = 3,
) -> Dict[str, Any]:
    job.raise_if_cancelled()
    job.update(message="Collecting images…")
    images = collect_images_from_dirs(image_dirs)
    scenes = load_scenes(scenes_path)
    subtitles = load_subtitle_segments(subtitles_path) if subtitles_path else None

    if audio_path is not None and not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    n = min(len(images), len(scenes))
    job.update(total=n, current=0, message=f"Building {n} clips…")

    clips = []
    for i in range(n):
        job.raise_if_cancelled()
        scene = scenes[i]
        duration = float(scene["end"]) - float(scene["start"])
        if duration <= 0:
            continue
        img = images[i]
        clips.append(ken_burns_clip(img, duration, resolution, variant=i))
        job.update(
            current=i + 1,
            message=f"Building clip {i + 1}/{n} — {img.name}",
        )

    if not clips:
        raise RuntimeError("No valid clips were generated.")

    job.raise_if_cancelled()
    job.update(message="Concatenating clips…")
    video = concatenate_videoclips(clips, method="chain")

    if subtitles:
        job.raise_if_cancelled()
        job.update(message=f"Building subtitles ({len(subtitles)} cues)…")
        subtitle_clips = build_subtitle_clips(
            subtitles,
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
            video = CompositeVideoClip([video, *subtitle_clips], size=resolution).with_duration(
                video.duration
            )

    narration = None
    if audio_path is not None:
        narration = AudioFileClip(str(audio_path))
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
        mixed = CompositeAudioClip([narration, *music_clips]).with_duration(video.duration)
        video = video.with_audio(mixed)
    elif narration is not None:
        video = video.with_audio(narration)
    elif music_clips:
        mixed = CompositeAudioClip(music_clips).with_duration(video.duration)
        video = video.with_audio(mixed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    job.update(
        current=n,
        total=n,
        message=f"Rendering video ({video.duration:.1f}s @ {fps}fps)…",
    )
    job.raise_if_cancelled()
    video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset=preset,
        threads=threads,
    )

    return {
        "output_path": str(output_path),
        "duration": float(video.duration),
        "clip_count": n,
        "engine": "moviepy",
        "scenes_path": str(scenes_path),
        "subtitles_path": str(subtitles_path) if subtitles_path else None,
    }
