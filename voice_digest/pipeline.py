"""End-to-end offline transcription, diarization, and prosody pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from voice_digest.alignment import align_and_merge_turns
from voice_digest.asr import resolve_asr_backend, transcribe_and_align
from voice_digest.audio import temporary_wav
from voice_digest.diarization import diarize_audio
from voice_digest.formatter import format_transcript
from voice_digest.llm_repair import repair_speaker_turns, summarize_transcript
from voice_digest.models import (
    DEFAULT_WHISPER_MODEL,
    release_memory,
    resolve_device_config,
)
from voice_digest.offline import enforce_offline_environment, offline_socket_guard
from voice_digest.power import format_elapsed, prevent_system_sleep, timed_stage
from voice_digest.prosody import analyze_session_prosody
from voice_digest.types import ASRSegment, DiarizationSegment, Word

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for an offline transcription run."""

    input_file: Path
    output_file: Path
    num_speakers: int = 2
    model_dir: Path = Path("./models")
    model_size: str = DEFAULT_WHISPER_MODEL
    device: str = "auto"
    prosody: bool = False
    metadata_mode: str = "both"
    pause_threshold: float = 2.0
    diarizer: str = "pyannote"
    language: Optional[str] = None
    batch_size: Optional[int] = None
    threads: Optional[int] = None
    use_ser: bool = True
    asr_backend: str = "auto"
    align_mode: str = "auto"
    cache_dir: Optional[Path] = None
    normalize_loudness: bool = True
    loudness_target_lufs: float = -16.0
    llm_repair: bool = False
    publish_summary: bool = False
    summary_focus: Optional[str] = None
    json_progress: bool = False


def emit_json_progress(
    percent: float,
    stage: str,
    detail: str = "",
    output: Optional[str] = None,
) -> None:
    """Emit a single-line JSON progress event for GUI frontends."""
    payload: dict[str, object] = {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "stage": stage,
        "detail": detail,
    }
    if output is not None:
        payload["output"] = output
    print(f"@@PROGRESS@@ {json.dumps(payload)}", flush=True)


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.name}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _save_asr_cache(path: Path, segments: list[ASRSegment], language: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"language": language, "segments": [asdict(s) for s in segments]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_asr_cache(path: Path) -> Optional[tuple[list[ASRSegment], str]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            ASRSegment(
                text=s["text"],
                start=float(s["start"]),
                end=float(s["end"]),
                words=[
                    Word(
                        text=w["text"],
                        start=float(w["start"]),
                        end=float(w["end"]),
                        score=float(w.get("score", 1.0)),
                    )
                    for w in s.get("words", [])
                ],
            )
            for s in payload.get("segments", [])
        ]
        return segments, str(payload.get("language", "en"))
    except Exception:
        return None


def _save_diar_cache(path: Path, segments: list[DiarizationSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in segments]), encoding="utf-8")


def _load_diar_cache(path: Path) -> Optional[list[DiarizationSegment]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            DiarizationSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                speaker=str(s["speaker"]),
            )
            for s in raw
        ]
    except Exception:
        return None


def run_transcription_pipeline(config: PipelineConfig) -> Path:
    """Execute the full offline transcription pipeline and write the output .txt file."""
    t_total = time.monotonic()
    model_dir = config.model_dir.expanduser().resolve()
    enforce_offline_environment(model_dir)

    cache_root = config.cache_dir.expanduser().resolve() if config.cache_dir else None

    with prevent_system_sleep(), offline_socket_guard():
        device_config = resolve_device_config(
            requested_device=config.device,
            batch_size=config.batch_size,
        )
        effective_asr = resolve_asr_backend(config.asr_backend, model_dir, config.model_size)
        logger.info(
            "Starting offline transcription: input=%s, speakers=%d, asr_backend=%s, model=%s, diarizer=%s",
            config.input_file,
            config.num_speakers,
            effective_asr,
            config.model_size,
            config.diarizer,
        )

        fp = _file_fingerprint(config.input_file.expanduser().resolve()) if config.input_file.expanduser().exists() else "unknown"
        asr_cache_path = (
            cache_root / f"asr_{fp}_{effective_asr}_{config.model_size}_{config.language or 'auto'}_{config.align_mode}.json"
            if cache_root
            else None
        )
        diar_cache_path = (
            cache_root / f"diar_{fp}_{config.diarizer}_{config.num_speakers}spk.json"
            if cache_root
            else None
        )

        def _report(pct: float, stage: str, detail: str = "", output: Optional[str] = None) -> None:
            if config.json_progress:
                emit_json_progress(pct, stage, detail, output=output)

        _report(2.0, "Preparing audio (FFmpeg)", f"Decoding & normalizing ({config.loudness_target_lufs:g} LUFS)" if config.normalize_loudness else "Decoding to 16 kHz WAV")
        with timed_stage("[Stage 1/5] FFmpeg transcode & loudness normalization"):
            wav_ctx = temporary_wav(
                config.input_file,
                normalize_loudness=config.normalize_loudness,
                loudness_target_lufs=config.loudness_target_lufs,
            )
            wav_path, duration_seconds = wav_ctx.__enter__()
        _report(12.0, "Audio ready", f"Duration: {format_elapsed(duration_seconds)}")

        try:
            # Stage 2: Batched ASR & optional forced alignment
            cached_asr = _load_asr_cache(asr_cache_path) if asr_cache_path else None
            if cached_asr is not None:
                asr_segments, detected_lang = cached_asr
                print(
                    f"\n==> [Stage 2/5] Loaded cached ASR ({len(asr_segments)} segments, lang={detected_lang})",
                    flush=True,
                )
            else:
                _report(15.0, f"Transcribing audio ({effective_asr} / {config.model_size})", "Running speech recognition")
                with timed_stage(
                    f"[Stage 2/5] ASR ({effective_asr} / {config.model_size}, align={config.align_mode})"
                ):
                    asr_segments, detected_lang = transcribe_and_align(
                        wav_path=wav_path,
                        model_dir=model_dir,
                        model_size=config.model_size,
                        device_config=device_config,
                        language=config.language,
                        threads=config.threads,
                        asr_backend=effective_asr,
                        align_mode=config.align_mode,
                    )
                if asr_cache_path is not None:
                    _save_asr_cache(asr_cache_path, asr_segments, detected_lang)
            _report(50.0, "Transcription complete", f"{len(asr_segments)} segments ({detected_lang})")
            logger.info(
                "ASR complete: %d segments (language=%s)", len(asr_segments), detected_lang
            )
            release_memory()

            # Stage 3: Speaker diarization
            cached_diar = _load_diar_cache(diar_cache_path) if diar_cache_path else None
            if cached_diar is not None:
                diarization_segments = cached_diar
                print(
                    f"\n==> [Stage 3/5] Loaded cached diarization ({len(diarization_segments)} spans)",
                    flush=True,
                )
            else:
                _report(52.0, f"Identifying speakers ({config.diarizer})", f"Target: {config.num_speakers} speakers")
                with timed_stage(
                    f"[Stage 3/5] Speaker diarization ({config.diarizer}, {config.num_speakers} speakers)"
                ):
                    diarization_segments = diarize_audio(
                        wav_path=wav_path,
                        num_speakers=config.num_speakers,
                        model_dir=model_dir,
                        device_config=device_config,
                        diarizer=config.diarizer,
                        threads=config.threads,
                        progress_cb=(
                            lambda frac, detail: _report(
                                52.0 + 30.0 * frac,
                                f"Identifying speakers ({config.diarizer})",
                                detail,
                            )
                        )
                        if config.json_progress
                        else None,
                    )
                if diar_cache_path is not None:
                    _save_diar_cache(diar_cache_path, diarization_segments)
            _report(83.0, "Speaker diarization complete", f"{len(diarization_segments)} spans")
            logger.info("Diarization complete: %d spans", len(diarization_segments))
            release_memory()

            # Stage 4: Speaker-word alignment, turn merging & optional local LLM speaker repair
            turns = align_and_merge_turns(
                asr_segments=asr_segments,
                diarization_segments=diarization_segments,
                pause_threshold=config.pause_threshold,
            )
            logger.info("Turn merging complete: %d turns", len(turns))

            if config.llm_repair and config.num_speakers >= 2:
                _report(84.0, "Repairing speaker labels (Local Qwen3.5 4B)", f"{len(turns)} turns")
                with timed_stage(
                    f"[Stage 3b/5] Local LLM speaker repair ({len(turns)} turns)"
                ):
                    turns = repair_speaker_turns(
                        turns=turns,
                        num_speakers=config.num_speakers,
                        model_dir=model_dir,
                        pause_threshold=config.pause_threshold,
                        progress_cb=(
                            lambda frac, detail: _report(
                                84.0 + 6.0 * frac,
                                "Repairing speaker labels (Local Qwen3.5 4B)",
                                detail,
                            )
                        )
                        if config.json_progress
                        else None,
                    )
                logger.info("LLM speaker repair complete: %d turns", len(turns))

            summary = None
            if config.prosody:
                _report(90.0, "Analyzing speech & prosody", f"{len(turns)} turns")
                with timed_stage(
                    f"[Stage 4/5] Acoustic & prosody analysis ({len(turns)} turns, ser={'on' if config.use_ser else 'off'})"
                ):
                    summary = analyze_session_prosody(
                        turns=turns,
                        wav_path=wav_path,
                        duration_seconds=duration_seconds,
                        model_dir=model_dir,
                        device_config=device_config,
                        use_ser=config.use_ser,
                        progress_cb=(
                            lambda frac, detail: _report(
                                90.0 + 7.0 * frac,
                                "Analyzing speech & prosody",
                                detail,
                            )
                        )
                        if config.json_progress
                        else None,
                    )
                    release_memory()
            else:
                print("\n==> [Stage 4/5] Prosody analysis skipped (pass --prosody to enable)", flush=True)

            ai_summary_text: Optional[str] = None
            if config.publish_summary:
                _report(95.0, "Generating AI summary (Local Qwen3.5 4B)", "Summarizing transcript")
                with timed_stage("[Stage 4b/5] Local AI summary (Qwen3.5 4B)"):
                    ai_summary_text = summarize_transcript(
                        turns=turns,
                        model_dir=model_dir,
                        focus_prompt=config.summary_focus,
                        progress_cb=(
                            lambda frac, detail: _report(
                                95.0 + 3.0 * frac,
                                "Generating AI summary (Local Qwen3.5 4B)",
                                detail,
                            )
                        )
                        if config.json_progress
                        else None,
                    )

            # Stage 5: Format and save transcript
            _report(98.5, "Formatting & saving transcript", "")
            with timed_stage("[Stage 5/5] Formatting & writing transcript"):
                rendered = format_transcript(
                    turns=turns,
                    prosody=config.prosody,
                    metadata_mode=config.metadata_mode,
                    summary=summary,
                    ai_summary=ai_summary_text,
                )
        finally:
            wav_ctx.__exit__(None, None, None)

    output_path = config.output_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    _report(100.0, "Complete", f"Finished in {format_elapsed(time.monotonic() - t_total)}", output=str(output_path))
    print(f"\nTotal runtime: {format_elapsed(time.monotonic() - t_total)}", flush=True)
    logger.info("Transcript written to %s", output_path)
    return output_path
