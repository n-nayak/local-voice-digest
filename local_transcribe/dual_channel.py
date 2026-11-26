"""Dual-channel (Mic = 'Me' + System Audio = 'Other' / 1:N) offline transcription pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from local_transcribe.alignment import align_and_merge_turns
from local_transcribe.asr import resolve_asr_backend, transcribe_and_align
from local_transcribe.audio import get_wav_rms, temporary_wav
from local_transcribe.diarization import diarize_audio
from local_transcribe.formatter import format_transcript
from local_transcribe.llm_repair import summarize_transcript
from local_transcribe.models import (
    DEFAULT_WHISPER_MODEL,
    release_memory,
    resolve_device_config,
)
from local_transcribe.offline import enforce_offline_environment, offline_socket_guard
from local_transcribe.pipeline import emit_json_progress
from local_transcribe.power import format_elapsed, prevent_system_sleep, timed_stage
from local_transcribe.prosody import analyze_session_prosody
from local_transcribe.types import ASRSegment, DiarizationSegment, SessionSummary, Turn

logger = logging.getLogger(__name__)


@dataclass
class DualChannelConfig:
    """Configuration for dual-stream (Mic + System Audio) transcription."""

    mic_file: Path
    system_file: Path
    output_file: Path
    remote_speakers: int = 1  # 1 = exact 'Other' (zero diarization); >=2 = 'Other 1'..'Other N'
    model_dir: Path = Path("./models")
    model_size: str = DEFAULT_WHISPER_MODEL
    device: str = "auto"
    prosody: bool = False
    metadata_mode: str = "both"
    pause_threshold: float = 2.0
    diarizer: str = "sherpa-onnx"
    language: Optional[str] = None
    batch_size: Optional[int] = None
    threads: Optional[int] = None
    use_ser: bool = False
    asr_backend: str = "auto"
    align_mode: str = "auto"
    normalize_loudness: bool = True
    loudness_target_lufs: float = -16.0
    publish_summary: bool = False
    summary_focus: Optional[str] = None
    json_progress: bool = False
    append_mode: bool = False  # True for rolling 60s live chunks


def _segments_to_single_speaker_turns(
    segments: list[ASRSegment],
    speaker_label: str,
    pause_threshold: float,
) -> list[Turn]:
    """Convert ASR segments from a single isolated hardware channel into merged Turns."""
    if not segments:
        return []
    max_end = max(s.end for s in segments) + 1.0
    dummy_diar = [DiarizationSegment(start=0.0, end=max_end, speaker="SINGLE")]
    turns = align_and_merge_turns(segments, dummy_diar, pause_threshold=pause_threshold)
    for t in turns:
        t.speaker = speaker_label
        for w in t.words:
            w.speaker = speaker_label
    return turns


def merge_dual_channel_turns(
    mic_turns: list[Turn],
    remote_turns: list[Turn],
    pause_threshold: float = 2.0,
) -> list[Turn]:
    """Chronologically interleave 'Me' turns and 'Other' turns, merging consecutive same-speaker turns."""
    combined = sorted(mic_turns + remote_turns, key=lambda t: (t.start, t.end))
    if not combined:
        return []

    merged: list[Turn] = [combined[0]]
    for turn in combined[1:]:
        prev = merged[-1]
        gap = turn.start - prev.end
        if turn.speaker == prev.speaker and gap <= pause_threshold:
            prev.end = max(prev.end, turn.end)
            prev.text = (prev.text.rstrip() + " " + turn.text.lstrip()).strip()
            prev.words = list(prev.words) + list(turn.words)
        else:
            merged.append(turn)
    return merged


def run_dual_channel_pipeline(config: DualChannelConfig) -> Path:
    """Transcribe isolated Mic ('Me') and System ('Other' / 1:N) audio streams offline."""
    t_total = time.monotonic()
    model_dir = config.model_dir.expanduser().resolve()
    enforce_offline_environment(model_dir)

    def _report(pct: float, stage: str, detail: str = "", output: Optional[str] = None) -> None:
        if config.json_progress:
            emit_json_progress(pct, stage, detail, output=output)

    with prevent_system_sleep(), offline_socket_guard():
        device_config = resolve_device_config(
            requested_device=config.device,
            batch_size=config.batch_size,
        )
        effective_asr = resolve_asr_backend(config.asr_backend, model_dir, config.model_size)

        _report(5.0, "Preparing dual-channel audio", "Checking & normalizing Mic & System tracks")
        # Silence gate: skip channels with no audible speech (RMS < 0.0005 ~= -66 dBFS)
        # before loudnorm, preventing Whisper from hallucinating "Thank you." on silence.
        min_speech_rms = 0.0005
        mic_rms = get_wav_rms(config.mic_file) if (config.mic_file.exists() and config.mic_file.stat().st_size > 44) else 0.0
        sys_rms = get_wav_rms(config.system_file) if (config.system_file.exists() and config.system_file.stat().st_size > 44) else 0.0
        logger.info("Dual-channel RMS: mic=%.5f, system=%.5f", mic_rms, sys_rms)
        has_mic = mic_rms >= min_speech_rms
        has_sys = sys_rms >= min_speech_rms
        if config.mic_file.exists() and not has_mic:
            logger.warning("Mic track is silent (RMS < %.4f); skipping 'Me' channel.", min_speech_rms)
        if config.system_file.exists() and not has_sys:
            logger.warning("System track is silent (RMS < %.4f); skipping 'Other' channel.", min_speech_rms)

        mic_ctx = (
            temporary_wav(
                config.mic_file,
                normalize_loudness=config.normalize_loudness,
                loudness_target_lufs=config.loudness_target_lufs,
            )
            if has_mic
            else None
        )
        sys_ctx = (
            temporary_wav(
                config.system_file,
                normalize_loudness=config.normalize_loudness,
                loudness_target_lufs=config.loudness_target_lufs,
            )
            if has_sys
            else None
        )

        mic_wav: Optional[Path] = None
        sys_wav: Optional[Path] = None
        mic_dur = 0.0
        sys_dur = 0.0

        try:
            if mic_ctx is not None:
                mic_wav, mic_dur = mic_ctx.__enter__()
            if sys_ctx is not None:
                sys_wav, sys_dur = sys_ctx.__enter__()

            duration_seconds = max(mic_dur, sys_dur)

            # Stage 1: Transcribe Mic ('Me')
            mic_turns: list[Turn] = []
            if mic_wav is not None and mic_dur > 0.2:
                _report(15.0, f"Transcribing your mic ({config.model_size})", "Channel 1: Me")
                with timed_stage(f"[Dual 1/4] ASR Mic ('Me') — {effective_asr}/{config.model_size}"):
                    mic_segs, _ = transcribe_and_align(
                        wav_path=mic_wav,
                        model_dir=model_dir,
                        model_size=config.model_size,
                        device_config=device_config,
                        language=config.language,
                        threads=config.threads,
                        asr_backend=effective_asr,
                        align_mode=config.align_mode,
                    )
                mic_turns = _segments_to_single_speaker_turns(
                    mic_segs, "Me", pause_threshold=config.pause_threshold
                )
                release_memory()

            # Stage 2: Transcribe System Audio ('Other' or 'Other 1..N')
            remote_turns: list[Turn] = []
            if sys_wav is not None and sys_dur > 0.2:
                _report(45.0, f"Transcribing remote audio ({config.model_size})", "Channel 2: Other")
                with timed_stage(f"[Dual 2/4] ASR System ('Other') — {effective_asr}/{config.model_size}"):
                    sys_segs, _ = transcribe_and_align(
                        wav_path=sys_wav,
                        model_dir=model_dir,
                        model_size=config.model_size,
                        device_config=device_config,
                        language=config.language,
                        threads=config.threads,
                        asr_backend=effective_asr,
                        align_mode=config.align_mode,
                    )
                release_memory()

                if config.remote_speakers >= 2 and sys_segs:
                    _report(70.0, "Separating remote speakers", f"Target: {config.remote_speakers} remote")
                    with timed_stage(f"[Dual 2b/4] Diarizing remote channel ({config.remote_speakers} speakers)"):
                        diar_segs = diarize_audio(
                            wav_path=sys_wav,
                            num_speakers=config.remote_speakers,
                            model_dir=model_dir,
                            device_config=device_config,
                            diarizer=config.diarizer,
                            threads=config.threads,
                        )
                    remote_turns = align_and_merge_turns(
                        sys_segs, diar_segs, pause_threshold=config.pause_threshold
                    )
                    # Rename 'speaker 1' -> 'Other 1', etc.
                    for t in remote_turns:
                        if t.speaker.startswith("speaker "):
                            t.speaker = t.speaker.replace("speaker ", "Other ", 1)
                            for w in t.words:
                                w.speaker = t.speaker
                    release_memory()
                else:
                    remote_turns = _segments_to_single_speaker_turns(
                        sys_segs, "Other", pause_threshold=config.pause_threshold
                    )

            # Stage 3: Interleave chronologically
            turns = merge_dual_channel_turns(
                mic_turns, remote_turns, pause_threshold=config.pause_threshold
            )

            # Stage 4: Optional per-channel prosody (analyzed on each speaker's isolated clean WAV!)
            summary: Optional[SessionSummary] = None
            if config.prosody and turns and not config.append_mode:
                _report(82.0, "Analyzing speech & prosody", f"{len(turns)} turns")
                with timed_stage(f"[Dual 3/4] Acoustic & prosody analysis ({len(turns)} turns)"):
                    me_list = [t for t in turns if t.speaker == "Me"]
                    other_list = [t for t in turns if t.speaker != "Me"]
                    summaries = []
                    if me_list:
                        s_me = analyze_session_prosody(
                            me_list,
                            wav_path=mic_wav,
                            duration_seconds=duration_seconds,
                            model_dir=model_dir,
                            device_config=device_config,
                            use_ser=config.use_ser,
                        )
                        summaries.extend(s_me.speakers)
                    if other_list:
                        s_other = analyze_session_prosody(
                            other_list,
                            wav_path=sys_wav,
                            duration_seconds=duration_seconds,
                            model_dir=model_dir,
                            device_config=device_config,
                            use_ser=config.use_ser,
                        )
                        summaries.extend(s_other.speakers)
                    summary = SessionSummary(
                        duration_seconds=duration_seconds,
                        speaker_count=len(summaries),
                        speakers=summaries,
                    )
                    release_memory()

            # Stage 5: Optional AI Summary
            ai_summary_text: Optional[str] = None
            if config.publish_summary and turns and not config.append_mode:
                _report(92.0, "Generating AI summary (Local Qwen 7B)", "Summarizing call")
                with timed_stage("[Dual 4/4] Local AI summary (Qwen 7B)"):
                    ai_summary_text = summarize_transcript(
                        turns=turns,
                        model_dir=model_dir,
                        focus_prompt=config.summary_focus,
                    )

            rendered = format_transcript(
                turns=turns,
                prosody=config.prosody and not config.append_mode,
                metadata_mode=config.metadata_mode,
                summary=summary,
                ai_summary=ai_summary_text,
            )
        finally:
            if mic_ctx is not None:
                mic_ctx.__exit__(None, None, None)
            if sys_ctx is not None:
                sys_ctx.__exit__(None, None, None)

    output_path = config.output_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.append_mode:
        if rendered.strip():
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(rendered)
    else:
        output_path.write_text(rendered, encoding="utf-8")

    _report(
        100.0,
        "Complete",
        f"Finished in {format_elapsed(time.monotonic() - t_total)}",
        output=str(output_path),
    )
    return output_path
