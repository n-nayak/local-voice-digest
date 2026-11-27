"""Command-line interface for Local Audio Transcription & Acoustic Analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from voice_digest.dictation import run_dictation_daemon
from voice_digest.dual_channel import DualChannelConfig, run_dual_channel_pipeline
from voice_digest.formatter import VALID_METADATA_MODES
from voice_digest.models import (
    DEFAULT_WHISPER_MODEL,
    SUPPORTED_WHISPER_MODELS,
    download_all_models,
    download_dictation_models,
)
from voice_digest.offline import enforce_offline_environment
from voice_digest.pipeline import PipelineConfig, run_transcription_pipeline

DEFAULT_SPEAKERS = 2


def default_output_path(input_file: Path) -> Path:
    """Compute default transcript path: <input_dir>/<input_stem>_transcript.txt."""
    resolved = input_file.expanduser().resolve()
    return resolved.with_name(f"{resolved.stem}_transcript.txt")


def prompt_speaker_count(
    default: int = DEFAULT_SPEAKERS,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Prompt the user via stdin for expected speaker count, defaulting to 2 on Enter."""
    while True:
        try:
            raw = input_fn(f"Number of speakers [{default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        try:
            val = int(raw)
            if val >= 1:
                return val
        except ValueError:
            pass
        print("Please enter a positive integer (or press Enter for default).", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe.py",
        description=(
            "100% local, offline-capable CLI for transcribing long-form .m4a recordings, "
            "identifying individual speakers, and extracting acoustic/prosodic metadata."
        ),
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Path to the .m4a audio file (required unless --download-models is set).",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="One-time online setup: download and cache all required models locally.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Local directory for cached model weights (default: ./models).",
    )
    parser.add_argument(
        "-s",
        "--speakers",
        type=int,
        default=None,
        help="Number of expected speakers. If omitted, prompts interactively (default: 2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .txt file path (default: <input_filename>_transcript.txt).",
    )
    parser.add_argument(
        "-m",
        "--model-size",
        choices=SUPPORTED_WHISPER_MODELS,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL}).",
    )
    parser.add_argument(
        "--prosody",
        action="store_true",
        help="Enable local acoustic analysis (pace, disfluencies/blocks, stress markers).",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=VALID_METADATA_MODES,
        default="both",
        help="Prosody output mode when --prosody is enabled (default: both).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=2.0,
        help="Silence gap in seconds that splits consecutive turns by the same speaker (default: 2.0).",
    )
    parser.add_argument(
        "--diarizer",
        choices=("pyannote", "sherpa-onnx"),
        default="pyannote",
        help="Speaker diarization backend (default: pyannote).",
    )
    parser.add_argument(
        "--asr-backend",
        choices=("auto", "mlx", "whisperx"),
        default="auto",
        help="ASR backend: 'mlx' (Apple Silicon GPU), 'whisperx' (faster-whisper), or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--align",
        choices=("auto", "wav2vec2", "none"),
        default="auto",
        help="Word alignment mode (default: auto; native timestamps for mlx, wav2vec2 for whisperx).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache intermediate ASR & diarization JSON for fast re-runs.",
    )
    parser.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Disable FFmpeg EBU R128 loudness normalization during audio preprocessing.",
    )
    parser.add_argument(
        "--loudness-target",
        type=float,
        default=-16.0,
        help="Target integrated loudness in LUFS when normalization is enabled (range -30 to -5, default: -16.0).",
    )
    parser.add_argument(
        "--llm-repair",
        action="store_true",
        help="Run a local MLX LLM (Qwen3-4B-Instruct-4bit) pass to repair conversational speaker-label swaps.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate a local AI summary (Qwen3-4B-Instruct-4bit) at the top of the output transcript.",
    )
    parser.add_argument(
        "--summary-focus",
        type=str,
        default=None,
        help="Optional focus instruction (up to 160 chars) for --summary, e.g. 'Key takeaways and TODOs'.",
    )
    parser.add_argument(
        "--dictation-models",
        action="store_true",
        help="With --download-models: download only the public system-wide dictation models (Parakeet v3 + Qwen3-1.7B).",
    )
    parser.add_argument(
        "--include-dictation",
        action="store_true",
        help="With --download-models: also download the system-wide dictation models alongside transcription/LLM models.",
    )
    parser.add_argument(
        "--dictation-daemon",
        action="store_true",
        help="Run a warm persistent JSON-RPC stdin/stdout daemon for <500ms system-wide dictation.",
    )
    parser.add_argument(
        "--no-dictation-llm",
        action="store_true",
        help="With --dictation-daemon or --dictation-models: skip loading/downloading the Qwen3-1.7B cleanup LLM.",
    )
    parser.add_argument(
        "--json-progress",
        action="store_true",
        help="Emit machine-readable '@@PROGRESS@@ {...}' JSON lines for GUI frontends.",
    )
    parser.add_argument(
        "--mic-file",
        type=Path,
        default=None,
        help="Dual-channel mode: path to isolated microphone WAV ('Me').",
    )
    parser.add_argument(
        "--system-file",
        type=Path,
        default=None,
        help="Dual-channel mode: path to isolated system audio WAV ('Other').",
    )
    parser.add_argument(
        "--remote-speakers",
        type=int,
        default=1,
        help="Dual-channel mode: expected remote speakers on system audio (1 = exact 'Other', >=2 = diarize remote).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Dual-channel live mode: append transcribed chunk lines to output file instead of overwriting.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO-639-1 language code (e.g. 'en'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override ASR batch size (default: 16 on CUDA, 4 on CPU).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Limit CPU threads used by ASR (helpful to reduce heat on fanless laptops, e.g. --threads 4).",
    )
    parser.add_argument(
        "--no-ser",
        action="store_true",
        help="Skip neural SpeechBrain SER during --prosody (uses fast openSMILE/DSP stress features only).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token used only during --download-models for gated pyannote models.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.download_models:
        try:
            if args.dictation_models:
                download_dictation_models(
                    model_dir=args.model_dir,
                    include_llm=not args.no_dictation_llm,
                )
            else:
                download_all_models(
                    model_dir=args.model_dir,
                    model_size=args.model_size,
                    language=args.language or "en",
                    diarizer=args.diarizer,
                    include_prosody_ser=args.prosody,
                    hf_token=args.hf_token,
                    asr_backend=args.asr_backend,
                    include_llm_repair=args.llm_repair or args.summary,
                    include_dictation=args.include_dictation,
                )
            return 0
        except Exception as exc:
            print(f"Error downloading models: {exc}", file=sys.stderr)
            return 1

    # Enforce strict offline mode before running any inference
    enforce_offline_environment(args.model_dir)

    if args.dictation_daemon:
        return run_dictation_daemon(
            model_dir=args.model_dir,
            load_llm=not args.no_dictation_llm,
        )

    # Dual-channel live call mode (--mic-file and/or --system-file)
    if args.mic_file is not None or args.system_file is not None:
        if args.output is None:
            parser.error("--output is required when using --mic-file / --system-file")
        dual_cfg = DualChannelConfig(
            mic_file=args.mic_file or Path("/dev/null"),
            system_file=args.system_file or Path("/dev/null"),
            output_file=args.output,
            remote_speakers=max(1, args.remote_speakers),
            model_dir=args.model_dir,
            model_size=args.model_size,
            device=args.device,
            prosody=args.prosody,
            metadata_mode=args.metadata_mode,
            pause_threshold=args.pause_threshold,
            diarizer=args.diarizer,
            language=args.language,
            batch_size=args.batch_size,
            threads=args.threads,
            use_ser=not args.no_ser,
            asr_backend=args.asr_backend,
            align_mode=args.align,
            normalize_loudness=not args.no_loudnorm,
            loudness_target_lufs=args.loudness_target,
            publish_summary=args.summary,
            summary_focus=args.summary_focus,
            json_progress=args.json_progress,
            append_mode=args.append,
        )
        try:
            written = run_dual_channel_pipeline(dual_cfg)
            print(f"Transcript saved to: {written}")
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.input_file is None:
        parser.error("the following arguments are required: input_file")

    if args.speakers is not None:
        if args.speakers < 1:
            parser.error("--speakers must be >= 1")
        num_speakers = args.speakers
    else:
        num_speakers = prompt_speaker_count(default=DEFAULT_SPEAKERS, input_fn=input_fn)

    output_path = args.output if args.output is not None else default_output_path(args.input_file)

    config = PipelineConfig(
        input_file=args.input_file,
        output_file=output_path,
        num_speakers=num_speakers,
        model_dir=args.model_dir,
        model_size=args.model_size,
        device=args.device,
        prosody=args.prosody,
        metadata_mode=args.metadata_mode,
        pause_threshold=args.pause_threshold,
        diarizer=args.diarizer,
        language=args.language,
        batch_size=args.batch_size,
        threads=args.threads,
        use_ser=not args.no_ser,
        asr_backend=args.asr_backend,
        align_mode=args.align,
        cache_dir=args.cache_dir,
        normalize_loudness=not args.no_loudnorm,
        loudness_target_lufs=args.loudness_target,
        llm_repair=args.llm_repair,
        publish_summary=args.summary,
        summary_focus=args.summary_focus,
        json_progress=args.json_progress,
    )

    try:
        written = run_transcription_pipeline(config)
        print(f"Transcript saved to: {written}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
