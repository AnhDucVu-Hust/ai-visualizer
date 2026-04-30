"""
CLI runner / demo script for the AI Visualizer STT system.

Usage
-----
# Basic (auto-detect language, CPU, base model):
    python transcribe.py audio/sample.mp3

# Word-level timestamps, GPU, large model:
    python transcribe.py audio/sample.mp3 --model large-v3 --device cuda --word-timestamps

# Batched inference for long files:
    python transcribe.py audio/long_interview.mp3 --batch-size 8 --vad

# Force Spanish, translate to English:
    python transcribe.py audio/es_clip.mp3 --language es --task translate

# Save output to file:
    python transcribe.py audio/sample.mp3 --output results/transcript.txt

Run `python transcribe.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from stt import Transcriber, TranscriptionConfig
from stt.models import ComputeType, Device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Speech-to-text with faster-whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional
    p.add_argument("audio", help="Path to the audio file to transcribe")

    # Model
    p.add_argument(
        "--model",
        default="base",
        metavar="SIZE",
        help=(
            "Whisper model size or HuggingFace path. "
            "Options: tiny, base, small, medium, large-v2, large-v3, turbo, "
            "distil-large-v3, or a custom HF repo."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=[d.value for d in Device],
        help="Inference device",
    )
    p.add_argument(
        "--compute-type",
        default="default",
        choices=[c.value for c in ComputeType],
        dest="compute_type",
        help="Quantisation level",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        dest="num_workers",
        help="CTranslate2 parallel workers",
    )

    # Transcription
    p.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Force language code, e.g. 'en', 'fr'. Default = auto-detect",
    )
    p.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'translate' outputs English regardless of source language",
    )
    p.add_argument("--beam-size", type=int, default=5, dest="beam_size")
    p.add_argument("--best-of", type=int, default=5, dest="best_of")

    # Timestamps
    p.add_argument(
        "--word-timestamps",
        action="store_true",
        dest="word_timestamps",
        help="Enable word-level timestamps",
    )

    # VAD
    p.add_argument(
        "--vad",
        action="store_true",
        dest="vad_filter",
        help="Enable Silero VAD filtering",
    )
    p.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=2000,
        dest="vad_min_silence_ms",
        help="Minimum silence duration (ms) for VAD to remove",
    )

    # Batched inference
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        metavar="N",
        help="Enable batched inference with batch size N (GPU recommended)",
    )

    # Output
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write transcript to this file instead of stdout",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON (useful for downstream processing)",
    )

    # Logging
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    config = TranscriptionConfig(
        model_size=args.model,
        device=Device(args.device),
        compute_type=ComputeType(args.compute_type),
        num_workers=args.num_workers,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        best_of=args.best_of,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        vad_min_silence_ms=args.vad_min_silence_ms,
        batch_size=args.batch_size,
    )

    transcriber = Transcriber(engine_name="faster-whisper")

    print(f"Transcribing: {audio_path}", file=sys.stderr)
    print(
        f"Model: {args.model} | Device: {args.device} | Compute: {args.compute_type}",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    result = transcriber.transcribe(audio_path, config)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.2f}s  (audio duration: {result.duration:.2f}s)", file=sys.stderr)

    # ----- Format output -----
    if args.output_json:
        output_text = result.model_dump_json(indent=2)
    else:
        output_text = result.format_with_timestamps(word_level=args.word_timestamps)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Transcript saved to: {out_path}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
