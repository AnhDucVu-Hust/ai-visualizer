"""Fast ffmpeg-based image + audio + subtitles combiner.

Drop-in replacement for :mod:`video_combine.combine` that builds the whole
video in a single ffmpeg process using native C filters (``zoompan``,
``concat``, ``amix``, ``subtitles``). Dramatically faster than the MoviePy
implementation because there is no per-frame Python loop.

Same CLI surface and YAML config (`video_combine_config.yaml`) as
:mod:`video_combine.combine`, so you can swap commands:

    python -m video_combine.ffmpeg_combine --config video_combine_config.yaml
    python -m video_combine.ffmpeg_combine --config video_combine_config.yaml --fps 30

Pipeline:
    PNG   -> scale(*pre_scale) -> zoompan (Ken Burns) -> concat
                                                       -> subtitles (ASS)
                                                       -> libx264
    WAV   -> volume -> amix with music -> aac

Requirements:
    pip install ffmpeg-python
    # and the ffmpeg CLI must be on PATH with libass enabled (for subtitles):
    #   brew install ffmpeg       (macOS)
    #   apt install ffmpeg        (Debian/Ubuntu)

Uses `ffmpeg-python` — https://github.com/kkroening/ffmpeg-python
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable
import time
import shutil
import ffmpeg

from .combine import (
    collect_images_from_dirs,
    discover_image_dirs,
    detect_resolution,
    load_music_config,
    load_scenes,
    load_subtitle_segments,
    load_yaml_config,
    parse_music_spec,
    parse_resolution,
)


# ---------------------------------------------------------------------------
# Ken Burns via zoompan
# ---------------------------------------------------------------------------


def _zoompan_expressions(
    variant: int,
    duration_frames: int,
    pre_scale: int,
    max_zoom: float = 1.6,
    pan_px: int = 110,
) -> tuple[str, str, str]:
    """Return ``(z_expr, x_expr, y_expr)`` for a Ken-Burns variant.

    ``variant % 4``:
        0 — zoom in, centered
        1 — zoom out, centered
        2 — zoom in + pan right
        3 — zoom in + pan left
    """
    # Per-frame zoom step so we reach ``max_zoom`` at the last output frame.
    zoom_step = (max_zoom - 1.0) / max(duration_frames - 1, 1)

    v = variant % 4
    if v == 0:
        z = f"min(zoom+{zoom_step:.6f},{max_zoom:.3f})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif v == 1:
        # Start zoomed in, zoom out toward 1.0
        z = (
            f"if(eq(on,0),{max_zoom:.3f},"
            f"max(zoom-{zoom_step:.6f},1.0))"
        )
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif v == 2:
        z = f"min(zoom+{zoom_step:.6f},{max_zoom:.3f})"
        x = (
            f"iw/2-(iw/zoom/2)"
            f"+(on/{duration_frames})*{pan_px * pre_scale}"
        )
        y = "ih/2-(ih/zoom/2)"
    else:
        z = f"min(zoom+{zoom_step:.6f},{max_zoom:.3f})"
        x = (
            f"iw/2-(iw/zoom/2)"
            f"-(on/{duration_frames})*{pan_px * pre_scale}"
        )
        y = "ih/2-(ih/zoom/2)"
    return z, x, y


def _movement_expressions(
    variant: int,
    duration_frames: int,
    move_ratio: float = 0.25,
    movement_zoom: float = 1.25
) -> tuple[str, str, str]:
    """Return (z_expr, x_expr, y_expr) for visible directional pan.

    Movement needs zoom > 1.0. Otherwise zoompan has no cropped window
    to move across, so x/y changes are not visible.
    """
    frames = max(duration_frames - 1, 1)
    progress = f"(on/{frames})"

    z = f"{movement_zoom:.3f}"

    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"

    max_x = f"(iw-iw/zoom)"
    max_y = f"(ih-ih/zoom)"

    delta_x = f"({max_x}*{move_ratio:.3f})"
    delta_y = f"({max_y}*{move_ratio:.3f})"

    v = variant % 4

    if v == 0:  # move right
        x = f"{center_x}-{delta_x}/2+{progress}*{delta_x}"
        y = center_y
    elif v == 1:  # move left
        x = f"{center_x}+{delta_x}/2-{progress}*{delta_x}"
        y = center_y
    elif v == 2:  # move up
        x = center_x
        y = f"{center_y}+{delta_y}/2-{progress}*{delta_y}"
    else:  # move down
        x = center_x
        y = f"{center_y}-{delta_y}/2+{progress}*{delta_y}"

    return z, x, y


def build_ken_burns_stream(
    image_path: Path,
    duration: float,
    resolution: tuple[int, int],
    fps: int,
    variant: int,
    pre_scale: int = 4,
):
    """Return an ffmpeg video stream for one animated image clip.

    The input PNG is read as a single frame; ``zoompan`` then synthesises
    ``duration*fps`` output frames with the Ken-Burns motion. Pre-scaling the
    image before ``zoompan`` greatly reduces the filter's characteristic
    jitter on small step values.
    """
    width, height = resolution
    duration_frames = max(1, int(round(duration * fps)))

    z, x, y = _zoompan_expressions(variant, duration_frames, pre_scale)

    # Read the still image as a single frame. `zoompan` then synthesizes
    # exactly `duration_frames` output frames for that one frame.
    # Using loop+t+fps here can multiply frames per input frame and make the
    # first scene consume nearly the whole timeline.
    stream = ffmpeg.input(str(image_path))
    stream = stream.filter(
        "scale",
        f"iw*{pre_scale}",
        f"ih*{pre_scale}",
        flags="lanczos",
    )
    stream = stream.filter(
        "zoompan",
        z=z,
        x=x,
        y=y,
        d=duration_frames,
        s=f"{width}x{height}",
        fps=fps,
    )
    stream = stream.filter("setsar", "1")
    return stream


def build_movement_stream(
    image_path: Path,
    duration: float,
    resolution: tuple[int, int],
    fps: int,
    variant: int,
    pre_scale: int = 4,
):
    """Return an ffmpeg video stream for directional movement (no zoom)."""
    width, height = resolution
    duration_frames = max(1, int(round(duration * fps)))
    z, x, y = _movement_expressions(variant, duration_frames)

    stream = ffmpeg.input(str(image_path))
    stream = stream.filter(
        "scale",
        f"iw*{pre_scale}",
        f"ih*{pre_scale}",
        flags="lanczos",
    )
    stream = stream.filter(
        "zoompan",
        z=z,
        x=x,
        y=y,
        d=duration_frames,
        s=f"{width}x{height}",
        fps=fps,
    )
    stream = stream.filter("setsar", "1")
    return stream


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------


def _ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    # ASS times use centiseconds, not milliseconds
    return f"{h}:{m:02d}:{s:05.2f}"


def _ass_escape_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", "\\N")
    )


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


def write_ass_subtitles(
    cues: list[dict],
    ass_path: Path,
    resolution: tuple[int, int],
    *,
    font_name: str = "Arial",
    font_size: int = 44,
    bottom_margin: int = 72,
    max_lines: int = 2,
    max_chars_per_line: int = 20,
    split_by_space: bool = True,
    black_background: bool = True,
    stroke_width: int = 3,
    shadow: int = 2,
) -> None:
    """Write an ASS v4+ subtitle file, bottom-centered, white with black stroke."""
    w, h = resolution
    if black_background:
        styles = (
            # Box style: opaque-box layer behind text.
            # libass can be picky here, so use a non-zero Outline with
            # BorderStyle=3 to force a visible rectangular background.
            f"Style: Box,{font_name},{font_size},"
            "&H20000000,&H000000FF,&H20000000,&H20000000,"
            "0,0,0,0,100,100,0,0,"
            f"3,12,0,2,40,40,{bottom_margin},1\n"
            # Text style: normal white text + black outline/shadow.
            f"Style: Text,{font_name},{font_size},"
            "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,"
            f"1,{stroke_width},{shadow},2,40,40,{bottom_margin},1\n"
        )
    else:
        styles = (
            f"Style: Text,{font_name},{font_size},"
            "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,"
            f"1,{stroke_width},{shadow},2,40,40,{bottom_margin},1\n"
        )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\n"
        f"PlayResY: {h}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Alignment=2 (bottom center).
        # Colors are ASS-format &HAABBGGRR (alpha 00 = fully opaque).
        f"{styles}"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )

    lines: list[str] = [header]
    # Hard cap: never render more than 4 lines at once.
    max_lines = 4
    for cue in cues:
        cue_start = float(cue["start"])
        cue_end = float(cue["end"])
        cue_duration = cue_end - cue_start
        if cue_duration <= 0:
            continue
        raw_text = str(cue["text"]).strip()
        if not raw_text:
            continue
        wrapped = _wrap_subtitle_text(
            raw_text,
            max(1, max_chars_per_line),
            split_by_space=split_by_space,
        ).splitlines()
        if not wrapped:
            continue

        # If there are too many wrapped lines, split into multiple timed
        # sub-cues. Each sub-cue duration is proportional to its char count.
        chunks = [wrapped[i: i + max_lines] for i in range(0, len(wrapped), max_lines)]
        chunk_texts = ["\n".join(chunk).strip() for chunk in chunks]
        chunk_weights = [max(1, len(t.replace("\n", ""))) for t in chunk_texts]
        total_weight = sum(chunk_weights)
        consumed_weight = 0

        for idx, text in enumerate(chunk_texts):
            if not text:
                consumed_weight += chunk_weights[idx]
                continue
            part_start = cue_start + cue_duration * (consumed_weight / total_weight)
            consumed_weight += chunk_weights[idx]
            if idx == len(chunk_texts) - 1:
                part_end = cue_end
            else:
                part_end = cue_start + cue_duration * (consumed_weight / total_weight)
            if part_end <= part_start:
                continue

            part_start_ass = _ass_time(part_start)
            part_end_ass = _ass_time(part_end)
            escaped = _ass_escape_text(text)
            if black_background:
                lines.append(
                    f"Dialogue: 0,{part_start_ass},{part_end_ass},Box,,0,0,0,,{escaped}\n"
                )
            lines.append(
                f"Dialogue: 1,{part_start_ass},{part_end_ass},Text,,0,0,0,,{escaped}\n"
            )

    ass_path.write_text("".join(lines), encoding="utf-8")


def _escape_subtitles_path(path: Path) -> str:
    """Escape a filesystem path for ffmpeg's ``subtitles`` filter.

    We pass this as ``filename=...`` (named option) because ffmpeg's filter
    parser can misread bare absolute paths in complex graphs.
    """
    subtitle_path = Path(path).resolve().as_posix()
    return subtitle_path


def _resolve_ffmpeg_binaries() -> tuple[str, str]:
    """Resolve ffmpeg/ffprobe executables across dev and bundled builds.

    Resolution order:
      1) Explicit env vars (FFMPEG_BINARY / FFPROBE_BINARY)
      2) Executables available on PATH
      3) imageio_ffmpeg bundled binary (and sibling ffprobe if present)
      4) Fallback names (ffmpeg / ffprobe)
    """
    ffmpeg_cmd = (
        os.getenv("FFMPEG_BINARY")
        or shutil.which("ffmpeg")
        or "ffmpeg"
    )
    ffprobe_cmd = (
        os.getenv("FFPROBE_BINARY")
        or shutil.which("ffprobe")
        or "ffprobe"
    )

    try:
        import imageio_ffmpeg  # type: ignore

        imageio_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency/runtime
        imageio_ffmpeg_exe = None

    if imageio_ffmpeg_exe:
        ffmpeg_exe = Path(imageio_ffmpeg_exe)
        if not Path(ffmpeg_cmd).is_file() and shutil.which(ffmpeg_cmd) is None:
            ffmpeg_cmd = str(ffmpeg_exe)

        suffix = ".exe" if ffmpeg_exe.suffix.lower() == ".exe" else ""
        ffprobe_candidate = ffmpeg_exe.with_name(f"ffprobe{suffix}")
        if ffprobe_candidate.is_file():
            if not Path(ffprobe_cmd).is_file() and shutil.which(ffprobe_cmd) is None:
                ffprobe_cmd = str(ffprobe_candidate)

    return ffmpeg_cmd, ffprobe_cmd


def _ffmpeg_has_filter(filter_name: str, ffmpeg_cmd: str = "ffmpeg") -> bool:
    """Return True if the local ffmpeg binary exposes ``filter_name``."""
    try:
        proc = subprocess.run(
            [ffmpeg_cmd, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    # Example row: " ... subtitles         V->V       Render text subtitles ..."
    return f" {filter_name} " in proc.stdout


# ---------------------------------------------------------------------------
# Audio mix
# ---------------------------------------------------------------------------


def build_audio_stream(
    audio_path: Path | None,
    narration_volume: float,
    music_specs: list[dict],
    music_base_dir: Path,
    video_duration: float,
):
    """Return a single mixed audio stream, or ``None`` if there is no audio."""
    streams = []

    if audio_path is not None:
        narration = ffmpeg.input(str(audio_path)).audio
        if narration_volume != 1.0:
            narration = narration.filter("volume", narration_volume)
        streams.append(narration)

    for i, spec in enumerate(music_specs):
        p = Path(spec["file"])
        if not p.is_absolute():
            p = (music_base_dir / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Music file not found: {p}")

        start = max(0.0, float(spec.get("start", 0.0)))
        if start >= video_duration:
            print(
                f"[warn] Music track #{i+1} ({p.name}) starts at {start:.2f}s, "
                f"beyond the video duration ({video_duration:.2f}s); skipping.",
                file=sys.stderr,
            )
            continue

        end = spec.get("end")
        end_on_timeline = min(float(end), video_duration) if end is not None else video_duration
        dur = end_on_timeline - start
        if dur <= 0:
            continue

        offset = max(0.0, float(spec.get("offset", 0.0)))
        volume = float(spec.get("volume", 1.0))

        # -ss offset -t dur on the INPUT side = hardware-accelerated seek + trim
        stream = ffmpeg.input(str(p), ss=offset, t=dur).audio
        if volume != 1.0:
            stream = stream.filter("volume", volume)
        if start > 0:
            delay_ms = int(round(start * 1000))
            stream = stream.filter("adelay", delays=f"{delay_ms}|{delay_ms}", all=1)
        streams.append(stream)

        print(
            f"  [music {i+1}] {p.name}  {start:.2f}s -> {end_on_timeline:.2f}s  "
            f"(vol {volume:.2f}, offset {offset:.2f}s)"
        )

    if not streams:
        return None
    if len(streams) == 1:
        return streams[0]
    return ffmpeg.filter(
        streams,
        "amix",
        inputs=len(streams),
        duration="longest",
        normalize=0,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_video(
    images: list[Path],
    scenes: list[dict],
    audio_path: Path | None,
    output_path: Path,
    resolution: tuple[int, int],
    fps: int,
    threads: int,
    preset: str,
    crf: int = 20,
    music_specs: list[dict] | None = None,
    narration_volume: float = 1.0,
    music_base_dir: Path | None = None,
    subtitle_segments: list[dict] | None = None,
    subtitle_font_name: str = "Arial",
    subtitle_font_size: int = 44,
    subtitle_bottom_margin: int = 72,
    subtitle_max_lines: int = 2,
    subtitle_max_chars: int = 20,
    subtitle_split_by_space: bool = True,
    subtitle_black_background: bool = True,
    subtitle_stroke_width: int = 3,
    pre_scale: int = 4,
    burn_subtitles: bool = True,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
    dry_run: bool = False,
) -> None:
    """Build the final video with a single ffmpeg invocation."""
    music_specs = music_specs or []
    music_base_dir = music_base_dir or output_path.parent
    ffmpeg_cmd, ffprobe_cmd = _resolve_ffmpeg_binaries()

    n = min(len(images), len(scenes))
    if len(images) != len(scenes):
        print(
            f"[warn] Image/scene count mismatch: {len(images)} images vs "
            f"{len(scenes)} scenes \u2014 using first {n} of each.",
            file=sys.stderr,
        )

    # One animated video stream per scene: random Ken Burns or movement.
    video_streams = []
    total_duration = 0.0
    for i in range(n):
        scene = scenes[i]
        duration = float(scene["end"]) - float(scene["start"])
        if duration <= 0:
            print(f"[warn] Scene {i} has non-positive duration; skipping.", file=sys.stderr)
            continue
        img = images[i]
        effect_name = random.choice(("ken_burns", "movement"))
        print(
            f"  [{i+1:>3}/{n}] {img.name}  "
            f"{scene['start']:.2f}s \u2192 {scene['end']:.2f}s "
            f"({duration:.2f}s) [{effect_name}]"
        )
        if effect_name == "ken_burns":
            video_streams.append(
                build_ken_burns_stream(
                    image_path=img,
                    duration=duration,
                    resolution=resolution,
                    fps=fps,
                    variant=i,
                    pre_scale=pre_scale,
                )
            )
        else:
            video_streams.append(
                build_movement_stream(
                    image_path=img,
                    duration=duration,
                    resolution=resolution,
                    fps=fps,
                    variant=random.randint(0, 3),
                    pre_scale=pre_scale,
                )
            )
        total_duration += duration

    if not video_streams:
        raise RuntimeError("No valid video clips were generated.")

    print(f"\nTotal video duration: {total_duration:.2f}s")
    if progress_callback is not None:
        progress_callback(0.0, max(total_duration, 0.001))
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Render cancelled.")

    # Concatenate video streams.
    if len(video_streams) > 1:
        video = ffmpeg.concat(*video_streams, v=1, a=0)
    else:
        video = video_streams[0]

    if not burn_subtitles and subtitle_segments:
        print(
            f"Subtitles: {len(subtitle_segments)} cues available but "
            f"burn_subtitles=False (clean video, no ASS burn-in).",
            file=sys.stderr,
        )

    # Match combine.py behaviour: if narration is shorter than the image
    # sequence, clip the video down to the narration length.
    final_duration = total_duration
    if audio_path is not None:
        try:
            probe = ffmpeg.probe(str(audio_path), cmd=ffprobe_cmd)
            audio_duration = float(probe["format"]["duration"])
            if audio_duration < total_duration:
                print(
                    f"[info] Narration ({audio_duration:.2f}s) shorter than video "
                    f"({total_duration:.2f}s); trimming output to {audio_duration:.2f}s."
                )
                final_duration = audio_duration
        except (ffmpeg.Error, FileNotFoundError, OSError) as exc:  # noqa: PERF203
            print(f"[warn] ffprobe failed on {audio_path}: {exc}", file=sys.stderr)

    # Burn-in subtitles via libass (ASS file written to a temp dir).
    tmp_dir: tempfile.TemporaryDirectory | None = None
    if burn_subtitles and subtitle_segments:
        if not _ffmpeg_has_filter("subtitles", ffmpeg_cmd=ffmpeg_cmd):
            raise RuntimeError(
                "Your ffmpeg build does not include the 'subtitles' filter "
                "(libass missing). Install/reinstall ffmpeg with libass support, "
                "or run without --subtitles-json."
            )
        tmp_dir = tempfile.TemporaryDirectory(prefix="ffmpeg_combine_")
        ass_path = Path(tmp_dir.name) / "subtitles.ass"
        write_ass_subtitles(
            subtitle_segments,
            ass_path,
            resolution=resolution,
            font_name=subtitle_font_name,
            font_size=subtitle_font_size,
            bottom_margin=subtitle_bottom_margin,
            max_lines=subtitle_max_lines,
            max_chars_per_line=subtitle_max_chars,
            split_by_space=subtitle_split_by_space,
            black_background=subtitle_black_background,
            stroke_width=subtitle_stroke_width,
        )
        video = video.filter(
            "subtitles",
            filename=_escape_subtitles_path(ass_path),
        )
        print(
            f"Subtitles: {len(subtitle_segments)} cues \u2192 "
            f"{ass_path}  (libass)"
        )

    # Audio stream (may be None for silent output)
    audio_stream = build_audio_stream(
        audio_path=audio_path,
        narration_volume=narration_volume,
        music_specs=music_specs,
        music_base_dir=music_base_dir,
        video_duration=final_duration,
    )

    # Output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_kwargs: dict = dict(
        vcodec="libx264",
        preset=preset,
        crf=crf,
        pix_fmt="yuv420p",
        t=final_duration,
        r=fps,
        threads=threads,
        movflags="+faststart",
    )
    if audio_stream is not None:
        out_kwargs["acodec"] = "aac"
        out_kwargs["audio_bitrate"] = "192k"
        out = ffmpeg.output(video, audio_stream, str(output_path), **out_kwargs)
    else:
        out = ffmpeg.output(video, str(output_path), **out_kwargs)

    out = out.overwrite_output()

    cmd_preview = " ".join(out.compile())
    print(f"\nffmpeg command:\n  {cmd_preview}\n")

    print(
        f"Writing {output_path}  "
        f"({final_duration:.1f}s @ {fps}fps, {resolution[0]}x{resolution[1]})"
    )

    progress_time_re = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
    try:
        if dry_run:
            print("[dry-run] Skipping ffmpeg execution.")
        else:
            process = out.run_async(cmd=ffmpeg_cmd, pipe_stdout=True, pipe_stderr=True)
            cancelled = False
            latest_done_seconds = 0.0
            while True:
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    process.terminate()
                    break

                line = process.stderr.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                match = progress_time_re.search(text)
                if match and progress_callback is not None:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = float(match.group(3))
                    latest_done_seconds = (hours * 3600) + (minutes * 60) + seconds
                    progress_callback(min(latest_done_seconds, final_duration), max(final_duration, 0.001))

            return_code = process.wait()
            if cancelled:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise RuntimeError("Render cancelled.")
            if return_code != 0:
                raise ffmpeg.Error("ffmpeg", b"", b"ffmpeg process failed")

            if progress_callback is not None:
                progress_callback(max(latest_done_seconds, final_duration), max(final_duration, 0.001))
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


# ---------------------------------------------------------------------------
# CLI (mirrors combine.main so config/flags are interchangeable)
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    argv_list = list(argv) if argv is not None else None

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv_list)
    if pre_args.config is None:
        default_cfg = repo_root / "video_combine_config.yaml"
        if default_cfg.is_file():
            pre_args.config = default_cfg

    config_defaults: dict = {}
    config_music: list[dict] = []
    config_base_dir: Path | None = None
    if pre_args.config is not None:
        config_defaults, config_music = load_yaml_config(pre_args.config)
        config_base_dir = pre_args.config.resolve().parent

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
        help="YAML config file (same format as combine.py). CLI flags override it.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help=(
            "Scene root or image folder. Auto-discovers images/, images_1/, images_2/, … "
            "and merges batches in order."
        ),
    )
    parser.add_argument("--scenes", type=Path, default=repo_root / "results" / "scenes.json")
    parser.add_argument("--subtitles-json", type=Path, default=None,
                        help="Optional STT out.json for segment-timed subtitles.")
    parser.add_argument("--audio", type=Path, default=repo_root / "audio" / "script.wav")
    parser.add_argument("--output", type=Path, default=repo_root / "results" / "video.mp4")
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=None,
        help="Output resolution (e.g. 1920x1080). Auto-detected from the first image when omitted.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--narration-volume", type=float, default=1.0)
    parser.add_argument(
        "--music",
        action="append",
        default=[],
        metavar="SPEC",
        help="Repeatable 'file=PATH,start=SEC,end=SEC,volume=0-1' spec.",
    )
    parser.add_argument("--music-config", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preset", default="slower", help="libx264 preset (ultrafast..veryslow).")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF (lower = better quality, bigger file).")
    parser.add_argument("--subtitle-font-size", type=int, default=44)
    parser.add_argument("--subtitle-bottom-margin", type=int, default=72)
    parser.add_argument("--subtitle-max-lines", type=int, default=2)
    parser.add_argument("--subtitle-max-chars", type=int, default=20)
    parser.add_argument("--subtitle-split-by-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtitle-black-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtitle-stroke-width", type=int, default=3)
    parser.add_argument(
        "--burn-subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Burn STT subtitles into the video (libass). --no-burn-subtitles = clean video.",
    )
    parser.add_argument("--subtitle-font-name", default="Arial")
    parser.add_argument(
        "--pre-scale",
        type=int,
        default=4,
        help="Image pre-scale multiplier before zoompan (reduces jitter). Default: 4.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the ffmpeg command but do not run it.")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(argv_list)

    if args.images is None:
        parser.error("--images is required (CLI or input.images in --config).")

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

    # Music precedence: YAML → --music-config JSON → inline --music (CLI wins).
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
    start_time = time.time()
    build_video(
        images=images,
        scenes=scenes,
        audio_path=audio,
        output_path=args.output,
        resolution=args.resolution,
        fps=args.fps,
        threads=args.threads,
        preset=args.preset,
        crf=args.crf,
        music_specs=music_specs,
        narration_volume=args.narration_volume,
        music_base_dir=music_base_dir,
        subtitle_segments=subtitle_segments,
        subtitle_font_name=args.subtitle_font_name,
        subtitle_font_size=args.subtitle_font_size,
        subtitle_bottom_margin=args.subtitle_bottom_margin,
        subtitle_max_lines=args.subtitle_max_lines,
        subtitle_max_chars=args.subtitle_max_chars,
        subtitle_split_by_space=args.subtitle_split_by_space,
        subtitle_black_background=args.subtitle_black_background,
        subtitle_stroke_width=args.subtitle_stroke_width,
        pre_scale=args.pre_scale,
        burn_subtitles=args.burn_subtitles,
        dry_run=args.dry_run,
    )
    end_time = time.time()
    print(f"\nDone in {end_time - start_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
