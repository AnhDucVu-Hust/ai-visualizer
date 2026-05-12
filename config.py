"""
Pipeline configuration — single source of truth for all settings.

Loading priority (highest → lowest):
  1. CLI arguments that are explicitly provided
  2. Values in config.yaml (or whatever YAML file is given via --config)
  3. Pydantic field defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    # ── Input ─────────────────────────────────────────────────────────────
    audio: Optional[str] = Field(
        None,
        description="Path to an audio file. Transcription runs automatically.",
    )
    from_json: Optional[str] = Field(
        None,
        description="Path to an existing transcription JSON (skips transcription step).",
    )

    # ── Transcription ──────────────────────────────────────────────────────
    stt_engine: Literal["whisper", "huggingface", "groq"] = Field(
        "whisper",
        description="STT backend engine. whisper=faster-whisper, huggingface=Transformers ASR pipeline.",
    )
    stt_model: str = Field("base", description="Whisper model size (tiny/base/small/medium/large-v3).")
    device: Literal["cpu", "cuda", "auto","mps"] = Field("auto", description="Compute device.")
    compute_type: Literal["default", "int8", "int8_float16", "float16", "float32"] = Field(
        "default", description="Quantisation / compute type for faster-whisper."
    )
    log_progress: bool = Field(
        True,
        description="Show faster-whisper progress bar during transcription.",
    )

    # ── Segment merging ────────────────────────────────────────────────────
    min_duration: float = Field(7.0, description="Keep accumulating until group reaches this many seconds.")
    max_duration: float = Field(20.0, description="Hard ceiling — finalize group before exceeding this.")

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_client: Literal["gemini", "openai", "openrouter"] = Field(
        "gemini",
        description=(
            "Which LLM provider to use. "
            "'openai' = any OpenAI-compatible endpoint (Groq, Together, DeepSeek, etc. "
            "— set base_url accordingly). "
            "'openrouter' is kept as a convenience alias for OpenRouter."
        ),
    )
    base_url: Optional[str] = Field(
        None,
        description=(
            "Base URL for the OpenAI-compatible endpoint (only used when "
            "llm_client='openai'). Examples:\n"
            "  Groq      https://api.groq.com/openai/v1\n"
            "  OpenAI    https://api.openai.com/v1\n"
            "  Together  https://api.together.xyz/v1\n"
            "  DeepSeek  https://api.deepseek.com/v1"
        ),
    )
    model: Optional[str] = Field(
        None,
        description=(
            "Model identifier. Defaults: gemini=gemini-2.0-flash, "
            "openrouter=deepseek/deepseek-chat-v3-0324:free, "
            "openai=llama-3.3-70b-versatile (Groq)."
        ),
    )
    api_key: Optional[str] = Field(
        None,
        description=(
            "Single API key (or comma-separated list for automatic rotation). "
            "Falls back to GEMINI_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY in .env."
        ),
    )
    api_keys: Optional[List[str]] = Field(
        None,
        description=(
            "List of API keys for automatic rotation on quota/rate-limit (429). "
            "Takes precedence over api_key when set. "
            "Works for Gemini and any OpenAI-compatible provider (Groq, OpenRouter, …). "
            "In YAML: use a sequence under llm.api_keys."
        ),
    )
    language: Optional[str] = Field(
        None,
        description="Source language hint for the LLM (e.g. 'Japanese'). Auto-detected when null.",
    )
    temperature: float = Field(0.7, description="LLM sampling temperature.")
    max_tokens: int = Field(512, description="Maximum tokens per LLM response.")
    skip_characters: bool = Field(
        False,
        description="Skip the character extraction step (faster, less character consistency).",
    )
    global_style: Optional[str] = Field(
        None,
        description=(
            "Visual style applied to every image prompt. "
            "Falls back to GLOBAL_STYLE in .env, then '2D animated anime style'."
        ),
    )

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = Field(
        "results",
        description="Directory where all output files are written (scenes.json, prompts.txt, …).",
    )
    prompt_batch_size: Optional[int] = Field(
        None,
        description=(
            "Split prompts into files of this many prompts each, saved as "
            "prompt_batch_1.txt, … inside a 'prompts/' sub-folder of output_dir. "
            "Null → single prompts.txt."
        ),
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    verbose: bool = Field(False, description="Enable DEBUG logging.")

    # ── Convenience helpers ────────────────────────────────────────────────
    @property
    def default_model(self) -> str:
        _defaults = {
            "gemini": "gemini-2.0-flash",
            "openai": "llama-3.3-70b-versatile",
            "openrouter": "deepseek/deepseek-chat-v3-0324:free",
        }
        return self.model or _defaults[self.llm_client]

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def scenes_json_path(self) -> Path:
        return self.output_path / "scenes.json"

    @property
    def prompts_txt_path(self) -> Path:
        return self.output_path / "prompts.txt"

    @property
    def prompts_batch_dir(self) -> Path:
        return self.output_path / "prompts"

    @property
    def characters_json_path(self) -> Path:
        return self.output_path / "characters.json"


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _flatten_yaml(data: dict) -> dict:
    """Convert nested YAML sections into a flat dict matching PipelineConfig fields."""
    flat: dict = {}

    def _pull(section: str, mapping: dict[str, str]) -> None:
        block = data.get(section) or {}
        for yaml_key, field_name in mapping.items():
            if yaml_key in block and block[yaml_key] is not None:
                flat[field_name] = block[yaml_key]

    _pull("input", {"audio": "audio", "from_json": "from_json"})
    _pull(
        "transcription",
        {
            "engine": "stt_engine",
            "stt_model": "stt_model",
            "device": "device",
            "compute_type": "compute_type",
            "log_progress": "log_progress",
        },
    )
    _pull("merging", {"min_duration": "min_duration", "max_duration": "max_duration"})
    _pull("llm", {
        "client": "llm_client",
        "base_url": "base_url",
        "model": "model",
        "api_key": "api_key",
        "api_keys": "api_keys",
        "language": "language",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "skip_characters": "skip_characters",
        "global_style": "global_style",
    })

    out = data.get("output") or {}
    if "dir" in out and out["dir"] is not None:
        flat["output_dir"] = out["dir"]
    if "prompt_batch_size" in out and out["prompt_batch_size"] is not None:
        flat["prompt_batch_size"] = out["prompt_batch_size"]

    if "verbose" in data and data["verbose"] is not None:
        flat["verbose"] = data["verbose"]

    return flat


def load_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[dict] = None,
) -> PipelineConfig:
    """
    Build a PipelineConfig from an optional YAML file plus explicit CLI overrides.

    Priority (highest → lowest):
      1. overrides  — CLI args that were explicitly set by the user
      2. yaml_path  — values from the config YAML file
      3. Pydantic defaults defined in PipelineConfig
    """
    base: dict = {}

    if yaml_path:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = _flatten_yaml(raw)

    if overrides:
        base.update(overrides)

    return PipelineConfig(**base)
