"""Run the audio → scenes → prompts pipeline with progress reporting.

This mirrors ``generate_scenes.main()`` but reports granular progress to a
``Job`` so the UI can show "Scene X/Y" while the LLM is working.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from config import PipelineConfig, load_config
from scene import (
    BaseLLMClient,
    CharacterExtractor,
    GeminiClient,
    OpenAICompatibleClient,
    OpenRouterClient,
    PromptGenerator,
    SceneConfig,
    SceneResult,
    ScenePipelineResult,
    merge_segments,
)
from stt.models import TranscriptionResult

from .jobs import Job
from .stt_runtime import transcribe_shared

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "config.yaml"
_CONFIG_ENV = "APP_CONFIG_PATH"


def _resolve_config_path() -> Optional[Path]:
    from_env = os.getenv(_CONFIG_ENV)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file from {_CONFIG_ENV} not found: {path}")
        return path
    if _DEFAULT_CONFIG.exists():
        return _DEFAULT_CONFIG
    return None


def _load_cfg(
    audio_path: Path,
    min_duration: float,
    max_duration: float,
    output_dir: Path,
) -> PipelineConfig:
    overrides: Dict[str, Any] = {
        "audio": str(audio_path),
        "from_json": None,
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "output_dir": str(output_dir),
    }
    yaml_path = _resolve_config_path()
    return load_config(yaml_path=yaml_path, overrides=overrides)


def _build_client(cfg: PipelineConfig) -> BaseLLMClient:
    keys = cfg.api_keys or cfg.api_key
    if cfg.llm_client == "gemini":
        return GeminiClient(api_key=keys)
    if cfg.llm_client == "openai":
        if not cfg.base_url:
            raise ValueError(
                "llm_client='openai' in config.yaml requires llm.base_url to be set "
                "(e.g. https://api.groq.com/openai/v1)."
            )
        return OpenAICompatibleClient(api_key=keys, base_url=cfg.base_url)
    return OpenRouterClient(api_key=keys)


def _run_transcription(cfg: PipelineConfig, job: Job) -> TranscriptionResult:
    from stt import TranscriptionConfig
    from stt.models import ComputeType, Device

    job.update(message="Transcribing audio (this can take a while)…")
    t_cfg = TranscriptionConfig(
        model_size=cfg.stt_model,
        device=Device(cfg.device),
        compute_type=ComputeType(cfg.compute_type),
        log_progress=cfg.log_progress,
    )
    job.raise_if_cancelled()
    return transcribe_shared(
        engine_name=cfg.stt_engine,
        audio_path=cfg.audio,
        config=t_cfg,
        cancel_check=job.is_cancel_requested,
    )


def run_prompts_pipeline(
    job: Job,
    *,
    audio_path: Path,
    min_duration: float,
    max_duration: float,
    output_dir: Path,
) -> Dict[str, Any]:
    cfg = _load_cfg(audio_path, min_duration, max_duration, output_dir)
    job.raise_if_cancelled()
    cfg.output_path.mkdir(parents=True, exist_ok=True)

    # Step 1 — transcribe
    transcription = _run_transcription(cfg, job)
    job.raise_if_cancelled()

    # Step 2 — merge
    job.update(message="Grouping transcript into scenes…")
    scene_cfg = SceneConfig(
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        model=cfg.default_model,
        language=cfg.language,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        **({"global_style": cfg.global_style} if cfg.global_style else {}),
    )
    merged = merge_segments(transcription, scene_cfg)
    job.raise_if_cancelled()
    total = len(merged)
    job.update(total=total, current=0, message=f"Merged into {total} scenes.")

    # Persist raw transcription next to scenes.json for later "from_json" reuse
    transcription_payload = transcription.model_dump()
    transcription_payload.pop("engine_name", None)
    transcription_payload.pop("model_name", None)
    (cfg.output_path / "out.json").write_text(
        json.dumps(transcription_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Step 3 — LLM prompts
    client = _build_client(cfg)
    with client:
        roster = None
        if not cfg.skip_characters:
            job.update(message="Extracting characters…")
            roster = CharacterExtractor(client).extract(transcription, scene_cfg)

        generator = PromptGenerator(client)
        language = scene_cfg.language or _language_name(transcription.language)

        scenes = []
        for idx, seg in enumerate(merged, 1):
            job.raise_if_cancelled()
            job.update(
                current=idx - 1,
                message=f"Generating prompt {idx}/{total}…",
            )
            image_prompt, raw = generator.generate_for_segment(
                seg, scene_cfg, language, roster
            )
            scenes.append(
                SceneResult(
                    group_id=seg.group_id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    source_segment_ids=seg.source_segment_ids,
                    image_prompt=image_prompt,
                    raw_llm_response=raw,
                )
            )
            job.update(current=idx)

    pipeline_result = ScenePipelineResult(
        audio_path=transcription.audio_path,
        language=transcription.language,
        audio_duration=transcription.duration,
        config=scene_cfg,
        character_roster=roster,
        scenes=scenes,
    )

    # Step 4 — save outputs
    job.raise_if_cancelled()
    job.update(message="Writing files…")
    scene_payload = pipeline_result.model_dump()
    scene_payload.pop("config", None)
    cfg.scenes_json_path.write_text(
        json.dumps(scene_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    prompts_lines = [s.image_prompt.to_single_line() for s in pipeline_result.scenes]
    cfg.prompts_txt_path.write_text("\n\n".join(prompts_lines), encoding="utf-8")

    return {
        "scenes_path": str(cfg.scenes_json_path),
        "prompts_path": str(cfg.prompts_txt_path),
        "transcription_path": str(cfg.output_path / "out.json"),
        "scene_count": total,
        "audio_path": str(audio_path),
        "output_dir": str(cfg.output_path),
    }


def scenes_info(scenes_path: Path) -> Dict[str, Any]:
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes", data if isinstance(data, list) else [])
    if not scenes:
        raise ValueError(f"No scenes found in {scenes_path}")
    total_duration = max(float(s["end"]) for s in scenes)
    return {
        "scene_count": len(scenes),
        "total_duration": total_duration,
        "audio_path": data.get("audio_path"),
    }


_LANG = {
    "ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
}


def _language_name(code: Optional[str]) -> str:
    if not code:
        return "unknown"
    return _LANG.get(code.lower(), code)
