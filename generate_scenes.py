"""
Scene generation pipeline CLI.

Quickstart — run from a config file (recommended):
    python generate_scenes.py --config config.yaml

Override individual settings on the fly:
    python generate_scenes.py --config config.yaml --min-duration 5 --verbose

Or pass everything via flags without a config file:
    python generate_scenes.py --from-json results/out.json --llm-client gemini
    python generate_scenes.py audio/script.wav --stt-model small
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from config import PipelineConfig, load_config
from scene import (
    BaseLLMClient,
    CharacterRoster,
    GeminiClient,
    ImagePrompt,
    OpenAICompatibleClient,
    OpenRouterClient,
    PromptGenerator,
    SceneConfig,
    ScenePipelineResult,
    SceneResult,
    merge_segments,
)
from scene.character_extractor import CharacterExtractor
from scene.prompt_generator import _language_code_to_name
from stt.models import TranscriptionResult

logger = logging.getLogger(__name__)


def _groq_stt_api_keys(cfg: PipelineConfig) -> list[str]:
    """Groq audio transcription uses the same keys as llm.api_keys / llm.api_key."""
    keys: list[str] = []
    if cfg.api_keys:
        keys = [str(k).strip() for k in cfg.api_keys if str(k).strip()]
    elif cfg.api_key:
        keys = [k.strip() for k in str(cfg.api_key).split(",") if k.strip()]
    return keys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate image-prompt scenes from a transcription.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file (e.g. config.yaml). "
             "CLI flags override values in the file.",
    )

    # Input source (mutually exclusive)
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to an audio file (transcription runs automatically).",
    )
    source.add_argument(
        "--from-json",
        metavar="PATH",
        default=None,
        dest="from_json",
        help="Path to an existing transcription JSON (skips transcription step).",
    )

    # Transcription options
    t = p.add_argument_group("Transcription  (ignored with --from-json)")
    t.add_argument(
        "--stt-engine",
        default=None,
        choices=["whisper", "huggingface"],
        dest="stt_engine",
        help="STT backend. whisper=faster-whisper, huggingface=Transformers pipeline.",
    )
    t.add_argument(
        "--stt-model",
        default=None,
        metavar="SIZE",
        dest="stt_model",
        help="Whisper size or Hugging Face model id (depends on --stt-engine).",
    )
    t.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Compute device. Default: auto.",
    )
    t.add_argument(
        "--compute-type",
        default=None,
        dest="compute_type",
        choices=["default", "int8", "int8_float16", "float16", "float32"],
        help="Quantisation type. Default: default.",
    )
    t.add_argument(
        "--log-progress",
        action="store_true",
        dest="log_progress",
        help="Show faster-whisper progress bars during transcription.",
    )

    # Merging
    m = p.add_argument_group("Segment merging")
    m.add_argument(
        "--min-duration",
        type=float,
        default=None,
        dest="min_duration",
        metavar="S",
        help="Minimum group duration in seconds. Default: 7.0.",
    )
    m.add_argument(
        "--max-duration",
        type=float,
        default=None,
        dest="max_duration",
        metavar="S",
        help="Hard ceiling for group duration in seconds. Default: 20.0.",
    )

    # LLM
    l = p.add_argument_group("LLM")
    l.add_argument(
        "--llm-client",
        default=None,
        choices=["gemini", "openai", "openrouter"],
        dest="llm_client",
        help="Which LLM provider to use. Default: gemini.",
    )
    l.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        metavar="URL",
        help="Base URL for OpenAI-compatible endpoints (only used with --llm-client openai). "
             "E.g. https://api.groq.com/openai/v1",
    )
    l.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help="API key for the selected provider. Falls back to .env keys.",
    )
    l.add_argument(
        "--model",
        default=None,
        help="Model identifier. "
             "Defaults: gemini=gemini-2.0-flash  openai=llama-3.3-70b-versatile  "
             "openrouter=deepseek/deepseek-chat-v3-0324:free.",
    )
    l.add_argument(
        "--language",
        default=None,
        help="Source language hint for the LLM (e.g. 'Japanese'). Auto-detected when absent.",
    )
    l.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Default: 0.7.",
    )
    l.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        help="Maximum tokens per LLM response. Default: 512.",
    )
    l.add_argument(
        "--skip-characters",
        action="store_true",
        dest="skip_characters",
        help="Skip character extraction (faster, less character consistency).",
    )
    l.add_argument(
        "--style",
        default=None,
        dest="global_style",
        metavar="STYLE",
        help="Global visual style for every image prompt. Falls back to GLOBAL_STYLE in .env.",
    )

    # Output
    o = p.add_argument_group("Output")
    o.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        metavar="DIR",
        help="Folder where all output files are written. Default: results/.",
    )
    o.add_argument(
        "--prompt-batch-size",
        type=int,
        default=None,
        dest="prompt_batch_size",
        metavar="N",
        help="Split prompts into files of N each (prompt_batch_1.txt, …). "
             "Default: single prompts.txt.",
    )

    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")

    return p


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Return only the values that were explicitly set on the command line."""
    overrides: dict = {}
    # Scalar args: include when not None
    scalar_keys = [
        "audio", "from_json", "stt_engine", "stt_model", "device", "compute_type",
        "min_duration", "max_duration", "llm_client", "api_key", "model",
        "language", "temperature", "max_tokens", "global_style",
        "output_dir", "prompt_batch_size",
    ]
    for key in scalar_keys:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    # Boolean store_true flags: include only when True (explicitly passed)
    if args.verbose:
        overrides["verbose"] = True
    if args.skip_characters:
        overrides["skip_characters"] = True
    if args.log_progress:
        overrides["log_progress"] = True

    return overrides


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_transcription_from_json(path: Path) -> TranscriptionResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TranscriptionResult(**data)


def save_transcription_to_json(transcription: TranscriptionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcription.model_dump_json(indent=2), encoding="utf-8")


def load_character_roster(path: Path) -> CharacterRoster | None:
    if not path.exists():
        return None
    try:
        return CharacterRoster.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to load character roster from %s: %s", path, exc)
        return None


def save_character_roster(roster: CharacterRoster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(roster.model_dump_json(indent=2), encoding="utf-8")


def load_existing_pipeline_result(path: Path) -> ScenePipelineResult | None:
    if not path.exists():
        return None
    try:
        return ScenePipelineResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to load existing scenes from %s: %s", path, exc)
        return None


_PROMPT_FIELDS = {
    "Scene": "scene",
    "Characters": "characters",
    "Style": "style",
    "Lighting": "lighting",
    "Colors": "colors",
    "Mood": "mood",
    "Camera": "camera",
}


def parse_prompt_line(line: str) -> ImagePrompt:
    flattened = " ".join(line.splitlines()).strip()
    data: dict[str, str] = {}
    for part in (p.strip() for p in flattened.split(";") if p.strip()):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        field = _PROMPT_FIELDS.get(key)
        if field:
            data[field] = value.strip()

    return ImagePrompt(
        scene=data.get("scene", "Unknown scene"),
        characters=data.get("characters", ""),
        style=data.get("style", "cinematic realism"),
        lighting=data.get("lighting", "natural light"),
        colors=data.get("colors", "neutral tones"),
        mood=data.get("mood", "neutral"),
        camera=data.get("camera", "medium shot"),
    )


def load_existing_prompts(path: Path) -> list[ImagePrompt]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    prompts: list[ImagePrompt] = []
    for block in blocks:
        try:
            prompts.append(parse_prompt_line(block))
        except Exception as exc:
            logging.warning("Failed to parse prompt block; skipping. %s", exc)
    return prompts


def build_resume_scenes(
    merged_segments,
    prompts: list[ImagePrompt],
) -> list[SceneResult]:
    scenes: list[SceneResult] = []
    for seg, prompt in zip(merged_segments, prompts):
        scenes.append(
            SceneResult(
                group_id=seg.group_id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                source_segment_ids=seg.source_segment_ids,
                image_prompt=prompt,
                raw_llm_response="",
            )
        )
    return scenes


def write_prompts_file(path: Path, scenes: list[SceneResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [scene.image_prompt.to_single_line() for scene in scenes]
    path.write_text("\n\n".join(lines), encoding="utf-8")


def _prompt_separator(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        offset = min(4, size)
        handle.seek(-offset, os.SEEK_END)
        tail = handle.read(offset)
    if tail.endswith(b"\n\n"):
        return ""
    if tail.endswith(b"\n"):
        return "\n"
    return "\n\n"


def append_prompt_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = _prompt_separator(path)
    with path.open("a", encoding="utf-8") as handle:
        if separator:
            handle.write(separator)
        handle.write(line.strip())
        handle.write("\n\n")


def run_transcription(cfg: PipelineConfig) -> TranscriptionResult:
    from stt import GroqEngine, Transcriber, TranscriptionConfig
    from stt.models import Device, ComputeType

    config = TranscriptionConfig(
        model_size=cfg.stt_model,
        device=Device(cfg.device),
        compute_type=ComputeType(cfg.compute_type),
        log_progress=cfg.log_progress,
    )
    if cfg.stt_engine == "groq":
        groq_keys = _groq_stt_api_keys(cfg)
        if not groq_keys:
            raise ValueError(
                "Groq STT requires llm.api_keys or llm.api_key in config.yaml "
                "(same Groq API keys as chat)."
            )
        transcriber = Transcriber(engine=GroqEngine(api_keys=groq_keys))
    else:
        transcriber = Transcriber(engine_name=cfg.stt_engine)
    return transcriber.transcribe(cfg.audio, config)


def save_prompts(pipeline_result, cfg: PipelineConfig) -> None:
    lines = [s.image_prompt.to_single_line() for s in pipeline_result.scenes]
    batch_size = cfg.prompt_batch_size

    if not batch_size:
        prompts_path = cfg.prompts_txt_path
        prompts_path.write_text("\n\n".join(lines), encoding="utf-8")
        print(f"Saved prompts: {prompts_path}  ({len(lines)} prompts)", file=sys.stderr)
        return

    batch_dir = cfg.prompts_batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)

    for old in batch_dir.glob("prompt_batch_*.txt"):
        old.unlink()

    total_batches = (len(lines) + batch_size - 1) // batch_size
    for i in range(total_batches):
        chunk = lines[i * batch_size : (i + 1) * batch_size]
        (batch_dir / f"prompt_batch_{i + 1}.txt").write_text("\n\n".join(chunk), encoding="utf-8")

    print(
        f"Saved prompts: {batch_dir}/  "
        f"({len(lines)} prompts → {total_batches} files of ≤{batch_size})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Build config: YAML file → overridden by explicit CLI flags
    try:
        cfg = load_config(yaml_path=args.config, overrides=_cli_overrides(args))
    except FileNotFoundError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if cfg.verbose else logging.INFO,
    )

    if not cfg.audio and not cfg.from_json:
        parser.error(
            "Provide an input source via --audio / --from-json, "
            "or set input.audio / input.from_json in your config.yaml."
        )

    cfg.output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: obtain TranscriptionResult ───────────────────────────────
    if cfg.from_json:
        json_path = Path(cfg.from_json)
        if not json_path.exists():
            print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading transcription from {json_path} …", file=sys.stderr)
        transcription = load_transcription_from_json(json_path)
    else:
        audio_path = Path(cfg.audio)
        if not audio_path.exists():
            print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Transcribing {audio_path} …", file=sys.stderr)
        transcription = run_transcription(cfg)
        out_json_path = cfg.output_path / "out.json"
        save_transcription_to_json(transcription, out_json_path)
        print(
            f"Transcription done — {len(transcription.segments)} segments, "
            f"language={transcription.language}",
            file=sys.stderr,
        )
        print(f"Saved STT JSON: {out_json_path}", file=sys.stderr)

    # ── Step 2: merge segments ────────────────────────────────────────────
    model = cfg.default_model
    scene_config = SceneConfig(
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        model=model,
        language=cfg.language,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        **({"global_style": cfg.global_style} if cfg.global_style else {}),
    )
    print(f"Global style : {scene_config.global_style!r}", file=sys.stderr)

    merged = merge_segments(transcription, scene_config)
    print(
        f"Merged {len(transcription.segments)} segments → {len(merged)} groups "
        f"(min={cfg.min_duration}s  max={cfg.max_duration}s)",
        file=sys.stderr,
    )

    # ── Step 2a: resume from previous output if available ──────────────────
    resume_scenes: list[SceneResult] = []
    resume_count = 0
    existing_result = load_existing_pipeline_result(cfg.scenes_json_path)
    if existing_result and existing_result.scenes:
        resume_count = min(len(existing_result.scenes), len(merged))
        resume_scenes = existing_result.scenes[:resume_count]
        print(
            f"Resuming from {resume_count} scene(s) in {cfg.scenes_json_path}",
            file=sys.stderr,
        )
    else:
        existing_prompts = load_existing_prompts(cfg.prompts_txt_path)
        if existing_prompts:
            resume_count = min(len(existing_prompts), len(merged))
            resume_scenes = build_resume_scenes(merged, existing_prompts[:resume_count])
            print(
                f"Resuming from {resume_count} prompt(s) in {cfg.prompts_txt_path}",
                file=sys.stderr,
            )

    if resume_count == 0 and cfg.prompts_txt_path.exists():
        cfg.prompts_txt_path.unlink()
    elif resume_count > 0:
        write_prompts_file(cfg.prompts_txt_path, resume_scenes)

    # ── Step 3: build LLM client ──────────────────────────────────────────
    t0 = time.perf_counter()
    client: BaseLLMClient
    keys = cfg.api_keys or cfg.api_key
    key_count = len(cfg.api_keys) if cfg.api_keys else 1

    if cfg.llm_client == "gemini":
        client = GeminiClient(api_key=keys)
        print(
            f"LLM client   : Gemini ({model})  [{key_count} API key(s)]",
            file=sys.stderr,
        )
    elif cfg.llm_client == "openai":
        if not cfg.base_url:
            raise ValueError(
                "llm_client='openai' requires llm.base_url to be set "
                "(e.g. https://api.groq.com/openai/v1 for Groq)."
            )
        client = OpenAICompatibleClient(api_key=keys, base_url=cfg.base_url)
        print(
            f"LLM client   : OpenAI-compatible ({model})  "
            f"[base_url={cfg.base_url}  {key_count} API key(s)]",
            file=sys.stderr,
        )
    else:  # openrouter
        client = OpenRouterClient(api_key=keys)
        print(
            f"LLM client   : OpenRouter ({model})  [{key_count} API key(s)]",
            file=sys.stderr,
        )

    with client:
        # ── Step 3a: character extraction ─────────────────────────────────
        roster = None
        if not cfg.skip_characters:
            roster = load_character_roster(cfg.characters_json_path)
            if roster is None and existing_result is not None:
                roster = existing_result.character_roster
                save_character_roster(roster, cfg.characters_json_path)

            if roster is not None:
                print(
                    f"Loaded {len(roster.characters)} character(s) from "
                    f"{cfg.characters_json_path}",
                    file=sys.stderr,
                )
            else:
                print("Extracting characters from transcript …", file=sys.stderr)
                extractor = CharacterExtractor(client)
                roster = extractor.extract(transcription, scene_config)
                save_character_roster(roster, cfg.characters_json_path)
                print(
                    f"Saved {len(roster.characters)} character(s) to "
                    f"{cfg.characters_json_path}",
                    file=sys.stderr,
                )
                if roster.characters:
                    for c in roster.characters:
                        print(f"  • {c.label}: {c.description}", file=sys.stderr)
                else:
                    print("No named characters found.", file=sys.stderr)
        else:
            print("Character extraction skipped (--skip-characters).", file=sys.stderr)

        # ── Step 3b: generate image prompts ───────────────────────────────
        generator = PromptGenerator(client)
        scenes = list(resume_scenes)
        start_idx = len(scenes)
        total = len(merged)
        language = scene_config.language or _language_code_to_name(transcription.language)
        roster = roster or CharacterRoster()

        if start_idx:
            print(
                f"Continuing prompt generation at {start_idx + 1}/{total} …",
                file=sys.stderr,
            )

        for idx, seg in enumerate(merged[start_idx:], start_idx + 1):
            logger.info(
                "Generating prompt %d/%d  %s  (%.1fs)",
                idx,
                total,
                seg.format_timestamp(),
                seg.duration,
            )
            image_prompt, raw = generator.generate_for_segment(
                seg, scene_config, language, roster
            )
            scene = SceneResult(
                group_id=seg.group_id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                source_segment_ids=seg.source_segment_ids,
                image_prompt=image_prompt,
                raw_llm_response=raw,
            )
            scenes.append(scene)
            append_prompt_line(cfg.prompts_txt_path, image_prompt.to_single_line())

            partial = ScenePipelineResult(
                audio_path=transcription.audio_path,
                language=transcription.language,
                audio_duration=transcription.duration,
                config=scene_config,
                character_roster=roster,
                scenes=scenes,
            )
            cfg.scenes_json_path.write_text(
                partial.model_dump_json(indent=2), encoding="utf-8"
            )

        pipeline_result = ScenePipelineResult(
            audio_path=transcription.audio_path,
            language=transcription.language,
            audio_duration=transcription.duration,
            config=scene_config,
            character_roster=roster,
            scenes=scenes,
        )

    elapsed = time.perf_counter() - t0
    print(f"LLM prompts generated in {elapsed:.1f}s", file=sys.stderr)

    # ── Step 4: save output ───────────────────────────────────────────────
    cfg.scenes_json_path.write_text(pipeline_result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved JSON   : {cfg.scenes_json_path}", file=sys.stderr)

    save_prompts(pipeline_result, cfg)

    print(pipeline_result.to_text())


if __name__ == "__main__":
    main()
