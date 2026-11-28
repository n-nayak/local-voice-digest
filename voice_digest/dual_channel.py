"""Dual-channel (Mic = 'Me' + System Audio = 'Other' / 1:N) offline transcription pipeline."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from voice_digest.alignment import _join_word_texts, align_and_merge_turns
from voice_digest.asr import resolve_asr_backend, transcribe_and_align
from voice_digest.audio import get_wav_rms, temporary_wav
from voice_digest.diarization import diarize_audio
from voice_digest.formatter import format_transcript
from voice_digest.llm_repair import summarize_transcript
from voice_digest.models import (
    DEFAULT_WHISPER_MODEL,
    release_memory,
    resolve_device_config,
)
from voice_digest.offline import enforce_offline_environment, offline_socket_guard
from voice_digest.pipeline import emit_json_progress
from voice_digest.power import format_elapsed, prevent_system_sleep, timed_stage
from voice_digest.prosody import analyze_session_prosody
from voice_digest.types import ASRSegment, DiarizationSegment, SessionSummary, Turn, Word

logger = logging.getLogger(__name__)

_ECHO_NORM_RE = re.compile(r"[^a-z0-9']+")


def _norm_echo_token(text: str) -> str:
    return _ECHO_NORM_RE.sub("", text.lower()).strip("'")


def _flatten_segments_to_words(segments: list[ASRSegment]) -> list[Word]:
    """Extract word-level timestamps from ASR segments, interpolating if a segment lacks words."""
    words: list[Word] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
            continue
        tokens = seg.text.strip().split()
        if not tokens:
            continue
        dur = max(0.05, seg.end - seg.start)
        step = dur / len(tokens)
        for idx, tok in enumerate(tokens):
            words.append(
                Word(
                    text=tok,
                    start=seg.start + idx * step,
                    end=seg.start + (idx + 1) * step,
                )
            )
    return words


def suppress_mic_echo_segments(
    mic_segs: list[ASRSegment],
    sys_segs: list[ASRSegment],
    max_lead_sec: float = 0.35,
    max_lag_sec: float = 0.65,
) -> list[ASRSegment]:
    """Remove speaker-to-mic acoustic bleed (echo) from Mic ('Me') segments using System ('Other') words.

    Acoustic bleed from laptop speakers arrives at the microphone causally (at or slightly after
    the system audio timestamp, with small ASR timestamp jitter). Matching word runs within
    `[sys_start - max_lead_sec, sys_end + max_lag_sec]` are excised from `mic_segs` so `Other`
    speech is never duplicated under `Me`, and surviving `Me` utterances split cleanly around
    each remote turn.
    """
    if not mic_segs or not sys_segs:
        return mic_segs

    mic_words = _flatten_segments_to_words(mic_segs)
    sys_words = _flatten_segments_to_words(sys_segs)
    if not mic_words or not sys_words:
        return mic_segs

    sys_index: list[tuple[str, float, float]] = []
    for sw in sys_words:
        tok = _norm_echo_token(sw.text)
        if tok:
            sys_index.append((tok, sw.start, sw.end))
    if not sys_index:
        return mic_segs

    n = len(mic_words)
    mic_norms = [_norm_echo_token(w.text) for w in mic_words]

    # Step 1: Flag every mic word that matches a system word in the causal echo window
    direct_match = [False] * n
    for i, (w, tok) in enumerate(zip(mic_words, mic_norms)):
        if not tok:
            continue
        for sys_tok, s_start, s_end in sys_index:
            if s_end + max_lag_sec < w.start:
                continue
            if s_start - max_lead_sec > w.end:
                break
            if sys_tok == tok:
                direct_match[i] = True
                break

    # Step 2: Bridge 1-word ASR mismatch gaps inside an echo phrase
    bridged = list(direct_match)
    for i in range(1, n - 1):
        if not bridged[i] and direct_match[i - 1] and direct_match[i + 1]:
            gap_prev = mic_words[i].start - mic_words[i - 1].end
            gap_next = mic_words[i + 1].start - mic_words[i].end
            if gap_prev <= 0.8 and gap_next <= 0.8:
                bridged[i] = True

    # Step 3: Drop contiguous matched runs of >= 2 words (preserves genuine 1-word backchannels)
    drop = [False] * n
    i = 0
    while i < n:
        if not bridged[i]:
            i += 1
            continue
        j = i + 1
        while j < n and bridged[j] and (mic_words[j].start - mic_words[j - 1].end) <= 1.2:
            j += 1
        run_len = j - i
        if run_len >= 2:
            for k in range(i, j):
                drop[k] = True
        i = j

    # Step 3b: Phrase-level n-gram fallback (>= 4 consecutive matching words within ±30s)
    # Catches speaker bleed even if two recordings have slight clock drift between mic and system.
    sys_norms = [t for t, _, _ in sys_index]
    m_sys = len(sys_norms)
    if n >= 4 and m_sys >= 4:
        # Index 4-grams in system words -> list of start indices
        ngram_map: dict[tuple[str, str, str, str], list[int]] = {}
        for s_idx in range(m_sys - 3):
            key = (sys_norms[s_idx], sys_norms[s_idx + 1], sys_norms[s_idx + 2], sys_norms[s_idx + 3])
            if all(key):
                ngram_map.setdefault(key, []).append(s_idx)

        i = 0
        while i <= n - 4:
            if drop[i]:
                i += 1
                continue
            key = (mic_norms[i], mic_norms[i + 1], mic_norms[i + 2], mic_norms[i + 3])
            candidates = ngram_map.get(key)
            if not candidates:
                i += 1
                continue
            best_len = 0
            for s_idx in candidates:
                # Must be within ±30s of the system phrase
                if abs(mic_words[i].start - sys_index[s_idx][1]) > 30.0:
                    continue
                k = 4
                while (
                    i + k < n
                    and s_idx + k < m_sys
                    and mic_norms[i + k] == sys_norms[s_idx + k]
                ):
                    k += 1
                if k > best_len:
                    best_len = k
            if best_len >= 4:
                for k in range(i, i + best_len):
                    drop[k] = True
                i += best_len
            else:
                i += 1

    # Step 4: Drop isolated distinctive echo words (>= 6 chars) right next to a dropped echo run
    # (catches trailing Whisper echo repeats like "...in the inner city especially")
    extended_lag = max_lag_sec * 2.5
    for i in range(n):
        if drop[i] or len(mic_norms[i]) < 6:
            continue
        near_prev = i > 0 and drop[i - 1] and (mic_words[i].start - mic_words[i - 1].end) <= 1.2
        near_next = i + 1 < n and drop[i + 1] and (mic_words[i + 1].start - mic_words[i].end) <= 1.2
        if not (near_prev or near_next):
            continue
        if direct_match[i]:
            drop[i] = True
            continue
        tok = mic_norms[i]
        w = mic_words[i]
        for sys_tok, s_start, s_end in sys_index:
            if s_end + extended_lag < w.start:
                continue
            if s_start - max_lead_sec > w.end:
                break
            if sys_tok == tok:
                drop[i] = True
                break

    dropped_count = sum(1 for d in drop if d)
    if dropped_count == 0:
        return mic_segs

    logger.info(
        "Dual-channel echo cancellation removed %d / %d echoed speaker words from Mic ('Me').",
        dropped_count,
        n,
    )

    # Step 5: Rebuild ASRSegments from surviving mic words, splitting across excised echo gaps
    rebuilt: list[ASRSegment] = []
    buf: list[Word] = []
    for idx, w in enumerate(mic_words):
        if drop[idx]:
            if buf:
                rebuilt.append(
                    ASRSegment(
                        start=buf[0].start,
                        end=buf[-1].end,
                        text=_join_word_texts(buf),
                        words=list(buf),
                    )
                )
                buf = []
            continue
        if buf and (w.start - buf[-1].end) > 1.2:
            rebuilt.append(
                ASRSegment(
                    start=buf[0].start,
                    end=buf[-1].end,
                    text=_join_word_texts(buf),
                    words=list(buf),
                )
            )
            buf = []
        buf.append(w)

    if buf:
        rebuilt.append(
            ASRSegment(
                start=buf[0].start,
                end=buf[-1].end,
                text=_join_word_texts(buf),
                words=list(buf),
            )
        )
    return rebuilt


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


def _split_turn_at_timestamps(turn: Turn, cut_times: list[float]) -> list[Turn]:
    """Split a long Turn at sentence/word boundaries near any interjecting turn start times."""
    if not cut_times or len(turn.words) < 2:
        return [turn]

    words = list(turn.words)
    split_indices: list[int] = []
    for t_cut in sorted(cut_times):
        if t_cut <= words[0].start + 0.4 or t_cut >= words[-1].end - 0.4:
            continue
        # Find the first word starting at or after t_cut
        word_idx = next((idx for idx, w in enumerate(words) if w.start >= t_cut), None)
        if word_idx is None or word_idx <= 0:
            continue

        # Prefer a nearby sentence boundary (., ?, !) within ±1.5s so sentences stay intact
        best_idx = word_idx
        best_dist = 1.5
        for cand in range(max(1, word_idx - 8), min(len(words), word_idx + 8)):
            prev_text = words[cand - 1].text.strip()
            if prev_text.endswith((".", "?", "!")):
                dist = abs(words[cand].start - t_cut)
                if dist <= best_dist:
                    best_dist = dist
                    best_idx = cand
        if best_idx > 0 and best_idx < len(words):
            split_indices.append(best_idx)

    if not split_indices:
        return [turn]

    unique_splits = sorted(set(split_indices))
    pieces: list[Turn] = []
    prev_i = 0
    for s_i in unique_splits + [len(words)]:
        chunk = words[prev_i:s_i]
        if chunk:
            pieces.append(
                Turn(
                    speaker=turn.speaker,
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=_join_word_texts(chunk),
                    words=chunk,
                )
            )
        prev_i = s_i
    return pieces or [turn]


def merge_dual_channel_turns(
    mic_turns: list[Turn],
    remote_turns: list[Turn],
    pause_threshold: float = 2.0,
) -> list[Turn]:
    """Chronologically interleave 'Me' turns and 'Other' turns, splitting long turns around interjections."""
    mic_starts = [t.start for t in mic_turns]
    remote_starts = [t.start for t in remote_turns]

    split_remote: list[Turn] = []
    for rt in remote_turns:
        cuts = [ts for ts in mic_starts if rt.start + 0.4 < ts < rt.end - 0.4]
        split_remote.extend(_split_turn_at_timestamps(rt, cuts))

    split_mic: list[Turn] = []
    for mt in mic_turns:
        cuts = [ts for ts in remote_starts if mt.start + 0.4 < ts < mt.end - 0.4]
        split_mic.extend(_split_turn_at_timestamps(mt, cuts))

    combined = sorted(split_mic + split_remote, key=lambda t: (t.start, t.end))
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
            mic_segs: list[ASRSegment] = []
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
                release_memory()

            # Stage 2: Transcribe System Audio ('Other' or 'Other 1..N')
            sys_segs: list[ASRSegment] = []
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

            # Stage 2c: Suppress speaker-to-mic acoustic echo before building 'Me' turns
            if mic_segs and sys_segs:
                mic_segs = suppress_mic_echo_segments(mic_segs, sys_segs)

            mic_turns: list[Turn] = []
            if mic_segs:
                # Build separate turns per surviving segment so 'Other' turns interleave cleanly
                # even when the user replied immediately after remote speech.
                for seg in mic_segs:
                    seg_turns = _segments_to_single_speaker_turns(
                        [seg], "Me", pause_threshold=config.pause_threshold
                    )
                    mic_turns.extend(seg_turns)

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
                _report(92.0, "Generating AI summary (Local Qwen3.5 4B)", "Summarizing call")
                with timed_stage("[Dual 4/4] Local AI summary (Qwen3.5 4B)"):
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
