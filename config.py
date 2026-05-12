"""
Pipeline configuration — single source of truth for all settings.

Loading priority (highest → lowest):
  1. CLI arguments that are explicitly provided
  2. Values in config.yaml (or whatever YAML file is given via --config)
  3. Pydantic field defaults
"""

from __future__ import annotations

import os
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


def _optional_bool_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _apply_env_pipeline_overrides(base: dict) -> None:
    """
    Optional overrides from environment (Render / shell). Only sets a field when
    the corresponding env var is non-empty.

    Matches ``config.yaml`` sections ``transcription`` and ``llm`` plus ``verbose``.
    """
    str_pairs = [
        ("TRANSCRIPTION_ENGINE", "stt_engine"),
        ("STT_ENGINE", "stt_engine"),
        ("STT_MODEL", "stt_model"),
        ("STT_DEVICE", "device"),
        ("STT_COMPUTE_TYPE", "compute_type"),
        ("LLM_CLIENT", "llm_client"),
        ("LLM_MODEL", "model"),
        ("LLM_LANGUAGE", "language"),
        ("GLOBAL_STYLE", "global_style"),
    ]
    for env_k, field_k in str_pairs:
        raw = os.environ.get(env_k)
        if raw is not None and str(raw).strip() != "":
            base[field_k] = str(raw).strip()

    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url is not None and str(base_url).strip() != "":
        base["base_url"] = str(base_url).strip()

    temp = os.environ.get("LLM_TEMPERATURE")
    if temp is not None and str(temp).strip() != "":
        try:
            base["temperature"] = float(str(temp).strip())
        except ValueError:
            pass
    mt = os.environ.get("LLM_MAX_TOKENS")
    if mt is not None and str(mt).strip() != "":
        try:
            base["max_tokens"] = int(str(mt).strip())
        except ValueError:
            pass

    lp = _optional_bool_env("STT_LOG_PROGRESS")
    if lp is not None:
        base["log_progress"] = lp
    sk = _optional_bool_env("SKIP_CHARACTERS")
    if sk is not None:
        base["skip_characters"] = sk
    vb = _optional_bool_env("VERBOSE")
    if vb is None:
        vb = _optional_bool_env("APP_VERBOSE")
    if vb is not None:
        base["verbose"] = vb


def _apply_env_api_keys(base: dict) -> None:
    """
    If YAML + overrides left no keys, fill ``api_keys`` from the env var that
    matches ``llm_client`` (comma-separated list for rotation).

    - gemini     → GEMINI_API_KEY
    - openai     → OPENAI_API_KEY  (Groq, OpenAI, Together, …)
    - openrouter → OPENROUTER_API_KEY
    """
    existing = base.get("api_keys")
    if isinstance(existing, list) and len(existing) > 0:
        return
    single = base.get("api_key")
    if single is not None and str(single).strip():
        return

    client = base.get("llm_client") or "gemini"
    env_name = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(client, "GEMINI_API_KEY")
    # Alias LLM_API_KEYS / GEMINI_API_KEYS — same meaning as llm.api_keys in YAML (comma-separated).
    raw = ""
    if client == "openai":
        raw = os.environ.get("LLM_API_KEYS") or os.environ.get(env_name, "") or ""
    elif client == "gemini":
        raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get(env_name, "") or ""
    elif client == "openrouter":
        raw = os.environ.get("LLM_API_KEYS") or os.environ.get(env_name, "") or ""
    else:
        raw = os.environ.get(env_name, "") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if keys:
        base["api_keys"] = keys


def load_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[dict] = None,
) -> PipelineConfig:
    """
    Build a PipelineConfig from an optional YAML file plus explicit CLI overrides.

    Priority (highest → lowest):
      1. overrides  — CLI args that were explicitly set by the user
      2. yaml_path  — values from the config YAML file
      3. Same keys from environment variables (see ``_apply_env_pipeline_overrides``)
      4. Comma-separated API keys from env (see ``_apply_env_api_keys``) when
         ``api_key`` / ``api_keys`` are still unset
      5. Pydantic defaults defined in PipelineConfig
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

    _apply_env_pipeline_overrides(base)
    _apply_env_api_keys(base)

    return PipelineConfig(**base)


_APP_CONFIG_ENV = "APP_CONFIG_PATH"


def resolve_app_config_yaml_path(repo_root: Path | str) -> Optional[Path]:
    """
    Backend YAML path: ``APP_CONFIG_PATH``, else ``<repo_root>/config.yaml`` if it exists,
    else ``None`` (defaults only; API keys can come from env).
    """
    root = Path(repo_root).resolve()
    from_env = os.getenv(_APP_CONFIG_ENV)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file from {_APP_CONFIG_ENV} not found: {path}")
        return path
    default = root / "config.yaml"
    if default.exists():
        return default
    return None
