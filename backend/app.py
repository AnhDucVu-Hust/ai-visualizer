"""FastAPI app exposing Prompts + Video pipelines to the React frontend."""

from __future__ import annotations

import os
import json
import shutil
import tempfile
import zipfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .jobs import registry
from .prompts_pipeline import run_prompts_pipeline, scenes_info
from .state import load_state, save_state
from .ffmpeg_video_pipeline import run_ffmpeg_video_pipeline
from .stt_runtime import preload_default_stt_model
from .video_pipeline import run_video_pipeline

_ROOT = Path(__file__).resolve().parent.parent
_TEMP_ROOT = _ROOT / "temp"
_AUDIO_DIR = _TEMP_ROOT / "audio"
_MUSIC_DIR = _TEMP_ROOT / "music"
_PROMPTS_DIR = _TEMP_ROOT / "prompts"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_MUSIC_DIR.mkdir(parents=True, exist_ok=True)
_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
_UPLOAD_JOB_TTL_SECONDS = int(os.getenv("UPLOAD_JOB_TTL_SECONDS", "1200"))
_VIDEO_JOB_TTL_SECONDS = int(os.getenv("VIDEO_JOB_TTL_SECONDS", str(_UPLOAD_JOB_TTL_SECONDS)))

app = FastAPI(title="AI Visualizer Studio", version="0.1.0")


@app.on_event("startup")
def _startup_preload_models() -> None:
    preload_default_stt_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_path(p: str | None) -> Optional[Path]:
    if not p:
        return None
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (_ROOT / path).resolve()
    return path


def _job_payload(job) -> Dict[str, Any]:
    return job.to_dict()


def _is_under_temp(path: Path) -> bool:
    try:
        path.resolve().relative_to(_TEMP_ROOT.resolve())
        return True
    except Exception:  # noqa: BLE001
        return False


def _safe_cleanup_paths(*paths: Path | None) -> list[str]:
    clean: list[str] = []
    for p in paths:
        if p is None:
            continue
        rp = Path(p).resolve()
        if _is_under_temp(rp):
            clean.append(str(rp))
    return clean


def _build_prompts_bundle(job_id: str, result: Dict[str, Any]) -> Path:
    scenes_path = _resolve_path(str(result.get("scenes_path") or ""))
    prompts_path = _resolve_path(str(result.get("prompts_path") or ""))
    transcription_path = _resolve_path(str(result.get("transcription_path") or ""))
    if not scenes_path or not scenes_path.is_file():
        raise HTTPException(status_code=404, detail="scenes.json not found for this prompts job.")
    if not prompts_path or not prompts_path.is_file():
        raise HTTPException(status_code=404, detail="prompts.txt not found for this prompts job.")
    if not transcription_path or not transcription_path.is_file():
        raise HTTPException(status_code=404, detail="out.json not found for this prompts job.")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"prompts_bundle_{job_id}_", dir=str(_TEMP_ROOT)))
    bundle_path = temp_dir / "prompts_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(scenes_path, arcname="scenes.json")
        zf.write(transcription_path, arcname="out.json")
        zf.write(prompts_path, arcname="prompts.txt")
    return bundle_path


def _schedule_upload_cleanup(path: Path) -> None:
    """Delete a standalone uploaded file after the upload TTL window."""
    registry.schedule_cleanup(
        job_id=f"upload_{uuid.uuid4().hex}",
        delay_seconds=_UPLOAD_JOB_TTL_SECONDS,
        paths=_safe_cleanup_paths(path),
        delete_job=False,
    )


# ---------------------------------------------------------------------------
# State endpoints
# ---------------------------------------------------------------------------


class StatePatch(BaseModel):
    last_audio_path: Optional[str] = None
    last_output_dir: Optional[str] = None
    last_images_dir: Optional[str] = None
    last_scenes_path: Optional[str] = None
    last_video_output: Optional[str] = None
    last_min_duration: Optional[float] = None
    last_max_duration: Optional[float] = None


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return load_state()


@app.put("/api/state")
def put_state(patch: StatePatch) -> Dict[str, Any]:
    return save_state({k: v for k, v in patch.model_dump().items() if v is not None})


# ---------------------------------------------------------------------------
# Upload endpoints
# ---------------------------------------------------------------------------


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    filename = os.path.basename(file.filename or "upload.bin")
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no name.")
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    unique_name = f"{stem}_{os.urandom(4).hex()}{suffix}"
    dest = dest_dir / unique_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


@app.post("/api/upload/audio")
def upload_audio(file: UploadFile = File(...)) -> Dict[str, str]:
    dest = _save_upload(file, _AUDIO_DIR)
    _schedule_upload_cleanup(dest)
    save_state({"last_audio_path": str(dest)})
    return {"path": str(dest), "name": dest.name}


@app.post("/api/upload/music")
def upload_music(file: UploadFile = File(...)) -> Dict[str, str]:
    dest = _save_upload(file, _MUSIC_DIR)
    _schedule_upload_cleanup(dest)
    return {"path": str(dest), "name": dest.name}


# ---------------------------------------------------------------------------
# Scenes info endpoint (used by the video panel to set music defaults)
# ---------------------------------------------------------------------------


@app.get("/api/scenes-info")
def get_scenes_info(path: str) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p or not p.is_file():
        raise HTTPException(status_code=404, detail=f"scenes.json not found: {path}")
    try:
        return scenes_info(p)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Prompts pipeline
# ---------------------------------------------------------------------------


class PromptsJobRequest(BaseModel):
    audio_path: str = Field(..., description="Absolute or workspace-relative path to the audio file.")
    min_duration: float = 7.0
    max_duration: float = 20.0
    output_dir: str = "results"


@app.post("/api/prompts/jobs")
def start_prompts_job(req: PromptsJobRequest) -> Dict[str, Any]:
    audio_path = _resolve_path(req.audio_path)
    if not audio_path or not audio_path.is_file():
        raise HTTPException(status_code=400, detail=f"Audio file not found: {req.audio_path}")
    job = registry.create("prompts")
    output_dir = _PROMPTS_DIR / job.id

    save_state({
        "last_audio_path": str(audio_path),
        "last_output_dir": str(output_dir),
        "last_min_duration": req.min_duration,
        "last_max_duration": req.max_duration,
    })

    def _target(job):
        result = run_prompts_pipeline(
            job,
            audio_path=audio_path,
            min_duration=req.min_duration,
            max_duration=req.max_duration,
            output_dir=output_dir,
        )
        save_state({
            "last_scenes_path": result["scenes_path"],
            "last_output_dir": result["output_dir"],
        })
        return result

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_UPLOAD_JOB_TTL_SECONDS,
            paths=_safe_cleanup_paths(audio_path, output_dir),
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------


class MusicSpec(BaseModel):
    file: str
    start: float = 0.0
    end: Optional[float] = None
    offset: float = 0.0
    volume: float = 1.0


class VideoJobRequest(BaseModel):
    images_dir: Optional[str] = None
    scenes_path: Optional[str] = None
    scene_dir: Optional[str] = None
    subtitles_path: Optional[str] = None
    audio_path: Optional[str] = None
    output_path: str = "results/video.mp4"
    resolution: str = "1920x1080"
    fps: int = 30
    narration_volume: float = 1.0
    include_narration: bool = True
    music: List[MusicSpec] = Field(default_factory=list)
    preset: str = "medium"
    threads: int = 4


class FfmpegVideoJobRequest(VideoJobRequest):
    crf: int = 20
    subtitle_font_name: str = "Arial"
    subtitle_font_size: int = 44
    subtitle_bottom_margin: int = 72
    subtitle_max_lines: int = 2
    subtitle_max_chars: int = 20
    subtitle_split_by_space: bool = True
    subtitle_black_background: bool = True
    subtitle_stroke_width: int = 3
    pre_scale: int = 4


def _parse_resolution(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad resolution: {s!r}") from exc


def _normalize_music_specs(raw_music: List[MusicSpec], base_dir: Path) -> List[Dict[str, Any]]:
    music_specs: List[Dict[str, Any]] = []
    for m in raw_music:
        p = _resolve_path(m.file)
        if p is None:
            p = (base_dir / m.file).resolve()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"Music file not found: {m.file}")

        start = float(m.start)
        end = float(m.end) if m.end is not None else None
        offset = float(m.offset)
        volume = float(m.volume)

        if start < 0:
            raise HTTPException(status_code=400, detail=f"Music start must be >= 0: {m.file}")
        if end is not None and end < start:
            raise HTTPException(
                status_code=400,
                detail=f"Music end must be >= start for file: {m.file}",
            )
        if offset < 0:
            raise HTTPException(status_code=400, detail=f"Music offset must be >= 0: {m.file}")
        if volume < 0:
            raise HTTPException(status_code=400, detail=f"Music volume must be >= 0: {m.file}")

        music_specs.append({
            "file": str(p),
            "start": start,
            "end": end,
            "offset": offset,
            "volume": volume,
        })
    return music_specs


def _parse_music_specs_form(music: Optional[str], base_dir: Path) -> List[Dict[str, Any]]:
    if not music:
        return []
    try:
        payload = json.loads(music)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid music JSON payload: {exc}") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Music payload must be a JSON array.")
    try:
        parsed = [MusicSpec.model_validate(item) for item in payload]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid music spec payload: {exc}") from exc
    return _normalize_music_specs(parsed, base_dir)


def _extract_scene_bundle(scene_bundle: UploadFile) -> Path:
    if not scene_bundle.filename:
        raise HTTPException(status_code=400, detail="scene_bundle file name is required.")
    if not scene_bundle.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="scene_bundle must be a .zip file.")

    temp_root = Path(tempfile.mkdtemp(prefix="scene_bundle_", dir=str(_TEMP_ROOT)))
    archive_path = temp_root / "scene_bundle.zip"
    extract_dir = temp_root / "scene"

    with archive_path.open("wb") as f:
        shutil.copyfileobj(scene_bundle.file, f)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid zip file for scene_bundle.") from exc

    return extract_dir


def _extract_scene_files(scene_files: List[UploadFile]) -> Path:
    if not scene_files:
        raise HTTPException(status_code=400, detail="scene_files is required.")

    temp_root = Path(tempfile.mkdtemp(prefix="scene_folder_", dir=str(_TEMP_ROOT)))
    extract_dir = temp_root / "scene"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for upload in scene_files:
        raw_name = (upload.filename or "").replace("\\", "/").strip("/")
        if not raw_name:
            continue
        rel_path = Path(raw_name)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise HTTPException(status_code=400, detail=f"Unsafe uploaded path: {upload.filename}")
        dest = extract_dir.joinpath(*rel_path.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

    return extract_dir


def _detect_scene_root(extract_dir: Path) -> Path:
    direct_scenes = extract_dir / "scenes.json"
    direct_out = extract_dir / "out.json"
    if direct_scenes.is_file() and direct_out.is_file():
        return extract_dir

    for child in extract_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "scenes.json").is_file() and (child / "out.json").is_file():
            return child

    raise HTTPException(
        status_code=400,
        detail=(
            "scene_bundle must contain a folder with both scenes.json and out.json "
            "(either at zip root or one top-level subfolder)."
        ),
    )


def _detect_images_dir(scene_root: Path) -> Path:
    images_dir = scene_root / "images"
    if images_dir.is_dir():
        return images_dir
    return scene_root


def _resolve_scene_inputs(
    req: VideoJobRequest,
    *,
    require_subtitles: bool = False,
) -> tuple[Path, Path, Optional[Path]]:
    scene_dir = _resolve_path(req.scene_dir)

    images_dir = _resolve_path(req.images_dir)
    scenes_path = _resolve_path(req.scenes_path)
    subtitles_path = _resolve_path(req.subtitles_path)

    if scene_dir:
        if not scene_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Scene folder not found: {req.scene_dir}")

        if images_dir is None:
            images_candidate = scene_dir / "images"
            images_dir = images_candidate if images_candidate.is_dir() else scene_dir
        if scenes_path is None:
            scenes_path = scene_dir / "scenes.json"
        if subtitles_path is None:
            subtitle_candidate = scene_dir / "out.json"
            subtitles_path = subtitle_candidate if subtitle_candidate.is_file() else None

    if not images_dir or not images_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "Images folder not found. Provide `images_dir` directly, or set "
                "`scene_dir` containing either an `images/` subfolder or numbered images."
            ),
        )
    if not scenes_path or not scenes_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                "scenes.json not found. Provide `scenes_path` directly, or set "
                "`scene_dir` containing `scenes.json`."
            ),
        )
    if subtitles_path is not None and not subtitles_path.is_file():
        raise HTTPException(status_code=400, detail=f"Subtitles JSON not found: {req.subtitles_path}")
    if require_subtitles and subtitles_path is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "out.json not found. For ffmpeg render, provide `scene_dir` containing "
                "both `scenes.json` and `out.json`, or set `subtitles_path` explicitly."
            ),
        )

    return images_dir, scenes_path, subtitles_path


@app.post("/api/video/jobs")
def start_video_job(req: VideoJobRequest) -> Dict[str, Any]:
    images_dir, scenes_path, subtitles_path = _resolve_scene_inputs(req, require_subtitles=True)
    job = registry.create("video")

    audio_path = _resolve_path(req.audio_path) if req.include_narration else None
    if req.include_narration and (audio_path is None or not audio_path.is_file()):
        raise HTTPException(status_code=400, detail=f"Audio file not found: {req.audio_path}")

    output_path = _TEMP_ROOT / "video" / f"{job.id}.mp4"
    resolution = _parse_resolution(req.resolution)

    music_specs = _normalize_music_specs(req.music, _ROOT)

    save_state({
        "last_images_dir": str(images_dir),
        "last_scenes_path": str(scenes_path),
        "last_audio_path": str(audio_path) if audio_path else None,
        "last_video_output": str(output_path),
    })

    def _target(job):
        return run_video_pipeline(
            job,
            images_dir=images_dir,
            scenes_path=scenes_path,
            subtitles_path=subtitles_path,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution,
            fps=req.fps,
            threads=req.threads,
            preset=req.preset,
            narration_volume=req.narration_volume,
            music_specs=music_specs,
            music_base_dir=_ROOT,
            subtitle_font_size=44,
            subtitle_bottom_margin=72,
            subtitle_max_lines=2,
            subtitle_max_chars=20,
            subtitle_split_by_space=True,
            subtitle_black_background=True,
            subtitle_stroke_width=3,
        )

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_VIDEO_JOB_TTL_SECONDS,
            paths=_safe_cleanup_paths(output_path, audio_path, *[Path(m["file"]) for m in music_specs]),
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


@app.post("/api/video/jobs/upload")
def start_video_job_upload(
    scene_bundle: UploadFile = File(..., description="Zip containing scenes.json, out.json, and images."),
    include_narration: bool = Form(True),
    narration_file: Optional[UploadFile] = File(default=None),
    output_filename: str = Form("video.mp4"),
    resolution: str = Form("1920x1080"),
    fps: int = Form(30),
    narration_volume: float = Form(1.0),
    music: Optional[str] = Form(default=None),
    preset: str = Form("medium"),
    threads: int = Form(4),
    subtitle_font_name: str = Form("Arial"),
    subtitle_font_size: int = Form(44),
    subtitle_bottom_margin: int = Form(72),
    subtitle_max_lines: int = Form(2),
    subtitle_max_chars: int = Form(20),
    subtitle_split_by_space: bool = Form(True),
    subtitle_black_background: bool = Form(True),
    subtitle_stroke_width: int = Form(3),
) -> Dict[str, Any]:
    extract_dir = _extract_scene_bundle(scene_bundle)
    scene_root = _detect_scene_root(extract_dir)
    images_dir = _detect_images_dir(scene_root)
    scenes_path = scene_root / "scenes.json"
    subtitles_path = scene_root / "out.json"
    if not subtitles_path.is_file():
        raise HTTPException(status_code=400, detail="out.json not found in uploaded scene bundle.")
    resolution_tuple = _parse_resolution(resolution)

    safe_output_name = os.path.basename(output_filename.strip() or "video.mp4")
    if not safe_output_name.lower().endswith(".mp4"):
        safe_output_name = f"{safe_output_name}.mp4"
    output_path = extract_dir / "output" / safe_output_name

    audio_path: Optional[Path] = None
    if include_narration:
        if narration_file is None:
            raise HTTPException(status_code=400, detail="narration_file is required when include_narration=true.")
        original_name = os.path.basename(narration_file.filename or "narration.wav")
        audio_path = extract_dir / "audio" / original_name
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with audio_path.open("wb") as f:
            shutil.copyfileobj(narration_file.file, f)

    music_specs = _parse_music_specs_form(music, extract_dir)
    job = registry.create("video")

    def _target(job):
        return run_video_pipeline(
            job,
            images_dir=images_dir,
            scenes_path=scenes_path,
            subtitles_path=subtitles_path,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution_tuple,
            fps=fps,
            threads=threads,
            preset=preset,
            narration_volume=narration_volume,
            music_specs=music_specs,
            music_base_dir=extract_dir,
            subtitle_font_size=subtitle_font_size,
            subtitle_bottom_margin=subtitle_bottom_margin,
            subtitle_max_lines=subtitle_max_lines,
            subtitle_max_chars=subtitle_max_chars,
            subtitle_split_by_space=subtitle_split_by_space,
            subtitle_black_background=subtitle_black_background,
            subtitle_stroke_width=subtitle_stroke_width,
        )

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_UPLOAD_JOB_TTL_SECONDS,
            paths=_safe_cleanup_paths(
                extract_dir.parent,
                audio_path,
                *[Path(m["file"]) for m in music_specs],
            ),
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


@app.post("/api/video/jobs/upload-folder")
def start_video_job_upload_folder(
    scene_files: List[UploadFile] = File(..., description="Folder files containing scenes.json, out.json, and images."),
    include_narration: bool = Form(True),
    narration_file: Optional[UploadFile] = File(default=None),
    narration_path: Optional[str] = Form(default=None),
    output_filename: str = Form("video.mp4"),
    resolution: str = Form("1920x1080"),
    fps: int = Form(30),
    narration_volume: float = Form(1.0),
    music: Optional[str] = Form(default=None),
    preset: str = Form("medium"),
    threads: int = Form(4),
    subtitle_font_name: str = Form("Arial"),
    subtitle_font_size: int = Form(44),
    subtitle_bottom_margin: int = Form(72),
    subtitle_max_lines: int = Form(2),
    subtitle_max_chars: int = Form(20),
    subtitle_split_by_space: bool = Form(True),
    subtitle_black_background: bool = Form(True),
    subtitle_stroke_width: int = Form(3),
) -> Dict[str, Any]:
    extract_dir = _extract_scene_files(scene_files)
    scene_root = _detect_scene_root(extract_dir)
    images_dir = _detect_images_dir(scene_root)
    scenes_path = scene_root / "scenes.json"
    subtitles_path = scene_root / "out.json"
    if not subtitles_path.is_file():
        raise HTTPException(status_code=400, detail="out.json not found in uploaded scene folder.")
    resolution_tuple = _parse_resolution(resolution)

    safe_output_name = os.path.basename(output_filename.strip() or "video.mp4")
    if not safe_output_name.lower().endswith(".mp4"):
        safe_output_name = f"{safe_output_name}.mp4"
    output_path = extract_dir / "output" / safe_output_name

    audio_path: Optional[Path] = None
    if include_narration:
        if narration_file is not None:
            original_name = os.path.basename(narration_file.filename or "narration.wav")
            audio_path = extract_dir / "audio" / original_name
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            with audio_path.open("wb") as f:
                shutil.copyfileobj(narration_file.file, f)
        elif narration_path:
            resolved = _resolve_path(narration_path)
            if resolved is None or not resolved.is_file():
                raise HTTPException(status_code=400, detail=f"Audio file not found: {narration_path}")
            audio_path = resolved
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide narration_file or narration_path when include_narration=true.",
            )

    music_specs = _parse_music_specs_form(music, extract_dir)
    job = registry.create("video")

    def _target(job):
        return run_video_pipeline(
            job,
            images_dir=images_dir,
            scenes_path=scenes_path,
            subtitles_path=subtitles_path,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution_tuple,
            fps=fps,
            threads=threads,
            preset=preset,
            narration_volume=narration_volume,
            music_specs=music_specs,
            music_base_dir=extract_dir,
            subtitle_font_size=subtitle_font_size,
            subtitle_bottom_margin=subtitle_bottom_margin,
            subtitle_max_lines=subtitle_max_lines,
            subtitle_max_chars=subtitle_max_chars,
            subtitle_split_by_space=subtitle_split_by_space,
            subtitle_black_background=subtitle_black_background,
            subtitle_stroke_width=subtitle_stroke_width,
        )

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_UPLOAD_JOB_TTL_SECONDS,
            paths=_safe_cleanup_paths(
                extract_dir.parent,
                audio_path,
                *[Path(m["file"]) for m in music_specs],
            ),
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


@app.post("/api/video/ffmpeg/jobs")
def start_ffmpeg_video_job(req: FfmpegVideoJobRequest) -> Dict[str, Any]:
    images_dir, scenes_path, subtitles_path = _resolve_scene_inputs(req, require_subtitles=True)

    audio_path = _resolve_path(req.audio_path) if req.include_narration else None
    if req.include_narration and (audio_path is None or not audio_path.is_file()):
        raise HTTPException(status_code=400, detail=f"Audio file not found: {req.audio_path}")

    job = registry.create("video_ffmpeg")
    output_path = _TEMP_ROOT / "video" / f"{job.id}.mp4"
    resolution = _parse_resolution(req.resolution)

    music_specs = _normalize_music_specs(req.music, _ROOT)

    save_state({
        "last_images_dir": str(images_dir),
        "last_scenes_path": str(scenes_path),
        "last_audio_path": str(audio_path) if audio_path else None,
        "last_video_output": str(output_path),
    })

    def _target(job):
        return run_ffmpeg_video_pipeline(
            job,
            images_dir=images_dir,
            scenes_path=scenes_path,
            subtitles_path=subtitles_path,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution,
            fps=req.fps,
            threads=req.threads,
            preset=req.preset,
            crf=req.crf,
            narration_volume=req.narration_volume,
            music_specs=music_specs,
            music_base_dir=_ROOT,
            subtitle_font_name=req.subtitle_font_name,
            subtitle_font_size=req.subtitle_font_size,
            subtitle_bottom_margin=req.subtitle_bottom_margin,
            subtitle_max_lines=req.subtitle_max_lines,
            subtitle_max_chars=req.subtitle_max_chars,
            subtitle_split_by_space=req.subtitle_split_by_space,
            subtitle_black_background=req.subtitle_black_background,
            subtitle_stroke_width=req.subtitle_stroke_width,
            pre_scale=req.pre_scale,
        )

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_VIDEO_JOB_TTL_SECONDS,
            paths=_safe_cleanup_paths(output_path, audio_path, *[Path(m["file"]) for m in music_specs]),
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


@app.post("/api/video/ffmpeg/jobs/upload")
def start_ffmpeg_video_job_upload(
    scene_bundle: UploadFile = File(..., description="Zip containing scenes.json, out.json, and images."),
    include_narration: bool = Form(True),
    narration_file: Optional[UploadFile] = File(default=None),
    output_filename: str = Form("video.mp4"),
    resolution: str = Form("1920x1080"),
    fps: int = Form(30),
    narration_volume: float = Form(1.0),
    preset: str = Form("medium"),
    threads: int = Form(4),
    crf: int = Form(20),
    subtitle_font_name: str = Form("Arial"),
    subtitle_font_size: int = Form(44),
    subtitle_bottom_margin: int = Form(72),
    subtitle_max_lines: int = Form(2),
    subtitle_max_chars: int = Form(20),
    subtitle_split_by_space: bool = Form(True),
    subtitle_black_background: bool = Form(True),
    subtitle_stroke_width: int = Form(3),
    pre_scale: int = Form(4),
) -> Dict[str, Any]:
    extract_dir = _extract_scene_bundle(scene_bundle)
    scene_root = _detect_scene_root(extract_dir)
    images_dir = _detect_images_dir(scene_root)
    scenes_path = scene_root / "scenes.json"
    subtitles_path = scene_root / "out.json"
    resolution_tuple = _parse_resolution(resolution)

    safe_output_name = os.path.basename(output_filename.strip() or "video.mp4")
    if not safe_output_name.lower().endswith(".mp4"):
        safe_output_name = f"{safe_output_name}.mp4"
    output_path = extract_dir / "output" / safe_output_name

    audio_path: Optional[Path] = None
    if include_narration:
        if narration_file is None:
            raise HTTPException(status_code=400, detail="narration_file is required when include_narration=true.")
        original_name = os.path.basename(narration_file.filename or "narration.wav")
        audio_path = extract_dir / "audio" / original_name
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with audio_path.open("wb") as f:
            shutil.copyfileobj(narration_file.file, f)

    job = registry.create("video_ffmpeg")

    def _target(job):
        return run_ffmpeg_video_pipeline(
            job,
            images_dir=images_dir,
            scenes_path=scenes_path,
            subtitles_path=subtitles_path,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution_tuple,
            fps=fps,
            threads=threads,
            preset=preset,
            crf=crf,
            narration_volume=narration_volume,
            music_specs=[],
            music_base_dir=extract_dir,
            subtitle_font_name=subtitle_font_name,
            subtitle_font_size=subtitle_font_size,
            subtitle_bottom_margin=subtitle_bottom_margin,
            subtitle_max_lines=subtitle_max_lines,
            subtitle_max_chars=subtitle_max_chars,
            subtitle_split_by_space=subtitle_split_by_space,
            subtitle_black_background=subtitle_black_background,
            subtitle_stroke_width=subtitle_stroke_width,
            pre_scale=pre_scale,
        )

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_UPLOAD_JOB_TTL_SECONDS,
            paths=[str(extract_dir.parent)],
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


@app.post("/api/video/ffmpeg/jobs/upload-folder")
def start_ffmpeg_video_job_upload_folder(
    scene_files: List[UploadFile] = File(..., description="Folder files containing scenes.json, out.json, and images."),
    include_narration: bool = Form(True),
    narration_file: Optional[UploadFile] = File(default=None),
    narration_path: Optional[str] = Form(default=None),
    output_filename: str = Form("video.mp4"),
    resolution: str = Form("1920x1080"),
    fps: int = Form(30),
    narration_volume: float = Form(1.0),
    music: Optional[str] = Form(default=None),
    preset: str = Form("medium"),
    threads: int = Form(4),
    crf: int = Form(20),
    subtitle_font_name: str = Form("Arial"),
    subtitle_font_size: int = Form(44),
    subtitle_bottom_margin: int = Form(72),
    subtitle_max_lines: int = Form(2),
    subtitle_max_chars: int = Form(20),
    subtitle_split_by_space: bool = Form(True),
    subtitle_black_background: bool = Form(True),
    subtitle_stroke_width: int = Form(3),
    pre_scale: int = Form(4),
) -> Dict[str, Any]:
    extract_dir = _extract_scene_files(scene_files)
    scene_root = _detect_scene_root(extract_dir)
    images_dir = _detect_images_dir(scene_root)
    scenes_path = scene_root / "scenes.json"
    subtitles_path = scene_root / "out.json"
    resolution_tuple = _parse_resolution(resolution)

    safe_output_name = os.path.basename(output_filename.strip() or "video.mp4")
    if not safe_output_name.lower().endswith(".mp4"):
        safe_output_name = f"{safe_output_name}.mp4"
    output_path = extract_dir / "output" / safe_output_name

    audio_path: Optional[Path] = None
    if include_narration:
        if narration_file is not None:
            original_name = os.path.basename(narration_file.filename or "narration.wav")
            audio_path = extract_dir / "audio" / original_name
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            with audio_path.open("wb") as f:
                shutil.copyfileobj(narration_file.file, f)
        elif narration_path:
            resolved = _resolve_path(narration_path)
            if resolved is None or not resolved.is_file():
                raise HTTPException(status_code=400, detail=f"Audio file not found: {narration_path}")
            audio_path = resolved
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide narration_file or narration_path when include_narration=true.",
            )

    music_specs = _parse_music_specs_form(music, extract_dir)
    job = registry.create("video_ffmpeg")

    def _target(job):
        return run_ffmpeg_video_pipeline(
            job,
            images_dir=images_dir,
            scenes_path=scenes_path,
            subtitles_path=subtitles_path,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution_tuple,
            fps=fps,
            threads=threads,
            preset=preset,
            crf=crf,
            narration_volume=narration_volume,
            music_specs=music_specs,
            music_base_dir=extract_dir,
            subtitle_font_name=subtitle_font_name,
            subtitle_font_size=subtitle_font_size,
            subtitle_bottom_margin=subtitle_bottom_margin,
            subtitle_max_lines=subtitle_max_lines,
            subtitle_max_chars=subtitle_max_chars,
            subtitle_split_by_space=subtitle_split_by_space,
            subtitle_black_background=subtitle_black_background,
            subtitle_stroke_width=subtitle_stroke_width,
            pre_scale=pre_scale,
        )

    def _on_finished(_job):
        registry.schedule_cleanup(
            job_id=job.id,
            delay_seconds=_UPLOAD_JOB_TTL_SECONDS,
            paths=_safe_cleanup_paths(
                extract_dir.parent,
                audio_path,
                *[Path(m["file"]) for m in music_specs],
            ),
            delete_job=True,
        )

    registry.run(job, _target, on_finished=_on_finished)
    return _job_payload(job)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return _job_payload(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if job.progress.stage in {"done", "error", "cancelled"}:
        return _job_payload(job)
    job.request_cancel()
    job.update(message="Cancelling…")
    return _job_payload(job)


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    lines, next_offset = job.logs_since(offset)
    return {"lines": lines, "next_offset": next_offset}


@app.get("/api/jobs/{job_id}/artifact")
def download_job_artifact(job_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    payload = job.to_dict()
    if payload["stage"] != "done":
        raise HTTPException(status_code=409, detail="Job is not done yet.")
    result = payload.get("result") or {}
    output_path = result.get("output_path")
    if output_path:
        p = Path(str(output_path))
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"Output file not found: {output_path}")
        download_name = p.name
        if payload.get("kind") in {"video", "video_ffmpeg"}:
            download_name = f"{job_id}.mp4"
        return FileResponse(path=str(p), filename=download_name, media_type="video/mp4")

    if payload.get("kind") == "prompts":
        bundle_path = _build_prompts_bundle(job_id, result)
        background_tasks.add_task(shutil.rmtree, str(bundle_path.parent), True)
        return FileResponse(
            path=str(bundle_path),
            filename=f"prompts_{job_id}.zip",
            media_type="application/zip",
        )

    raise HTTPException(status_code=404, detail="No output artifact for this job.")


# ---------------------------------------------------------------------------
# Static file serving (production build)
# ---------------------------------------------------------------------------


_DIST = _ROOT / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
