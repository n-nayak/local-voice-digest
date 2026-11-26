"""Local acoustic and prosodic analysis engine (strictly non-lexical biomarkers)."""

from __future__ import annotations

import logging
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from local_transcribe.audio import read_wav_segment
from local_transcribe.models import DeviceConfig, release_memory, ser_model_path
from local_transcribe.types import SessionSummary, SpeakerSummary, Turn, TurnProsody, Word

logger = logging.getLogger(__name__)

# Pace thresholds (Spec Section 5.1)
FAST_WPM_THRESHOLD = 175.0
SLOW_WPM_THRESHOLD = 110.0
FAST_ZSCORE_THRESHOLD = 1.5

# Disfluency thresholds (Spec Section 5.1)
BLOCK_GAP_MIN_SEC = 0.4
BLOCK_GAP_MAX_SEC = 1.5
PROLONGATION_MULTIPLIER = 3.5
PROLONGATION_MIN_SEC = 0.85

# SER windowing & duration limits (prevents O(T^2) wav2vec2 OOM on long turns)
SER_MIN_DURATION_SEC = 1.5
SER_WINDOW_SEC = 8.0
SER_MAX_WINDOWS_PER_TURN = 4

_CLAUSE_END_PUNCT = (".", "?", "!", ";", ":")
_WORD_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_VOWEL_GROUP_RE = re.compile(r"[aeiouyAEIOUY]+")


def _estimate_word_syllables(word_text: str) -> int:
    """Rough phonetic syllable count from vowel nuclei for articulation-rate normalization."""
    cleaned = re.sub(r"[^a-zA-Z]", "", word_text).lower()
    if not cleaned:
        return 0
    groups = _VOWEL_GROUP_RE.findall(cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1 and not cleaned.endswith("le"):
        count -= 1
    return max(1, count)


def compute_turn_pace(turn: Turn) -> tuple[float, float, float]:
    """Compute articulation rate (WPM, syllables/sec, active_duration_sec) excluding long silences."""
    spoken_words = [w for w in turn.words if _WORD_ALNUM_RE.search(w.text)]
    if not spoken_words:
        # Fallback if word timestamps are unavailable
        tokens = [t for t in turn.text.split() if _WORD_ALNUM_RE.search(t)]
        dur = max(0.25, turn.duration)
        wpm = (len(tokens) / dur) * 60.0
        syllables = sum(_estimate_word_syllables(t) for t in tokens)
        return wpm, syllables / dur, dur

    # Use total turn span minus long silent pauses (>= BLOCK_GAP_MIN_SEC).
    # This is far more robust to noisy per-word durations from Whisper DTW timestamps
    # than summing individual word durations.
    span_dur = max(0.0, spoken_words[-1].end - spoken_words[0].start)
    long_pauses = 0.0
    for prev, curr in zip(spoken_words, spoken_words[1:]):
        gap = curr.start - prev.end
        if gap >= BLOCK_GAP_MIN_SEC:
            long_pauses += gap

    active_dur = max(0.25, span_dur - long_pauses)
    wpm = (len(spoken_words) / active_dur) * 60.0
    total_syllables = sum(_estimate_word_syllables(w.text) for w in spoken_words)
    syllables_per_sec = total_syllables / active_dur
    return wpm, syllables_per_sec, active_dur


def detect_mid_clause_blocks(words: Sequence[Word]) -> int:
    """Count abnormal mid-clause silent gaps (0.4s-1.5s) from forced-alignment timestamps."""
    if len(words) < 2:
        return 0
    blocks = 0
    for prev, curr in zip(words, words[1:]):
        prev_text = prev.text.strip()
        if prev_text.endswith(_CLAUSE_END_PUNCT):
            continue
        gap = curr.start - prev.end
        if BLOCK_GAP_MIN_SEC <= gap <= BLOCK_GAP_MAX_SEC:
            blocks += 1
    return blocks


def detect_prolongations(
    words: Sequence[Word],
    median_sec_per_syllable: float = 0.22,
) -> int:
    """Detect abnormal phoneme/word duration spikes (> 3x expected duration)."""
    if not words:
        return 0
    baseline = max(0.14, min(0.35, median_sec_per_syllable))
    count = 0
    for word in words:
        if not _WORD_ALNUM_RE.search(word.text):
            continue
        syls = max(1, _estimate_word_syllables(word.text))
        expected = syls * baseline
        if word.duration >= PROLONGATION_MIN_SEC and word.duration > PROLONGATION_MULTIPLIER * expected:
            count += 1
    return count


def detect_syllable_repetitions(samples: Sequence[float], sample_rate: int) -> int:
    """Detect rapid repetitive energy bursts (80ms-250ms intervals) in the amplitude envelope."""
    if len(samples) < int(0.4 * sample_rate):
        return 0

    frame_len = max(1, int(0.020 * sample_rate))  # 20ms
    hop_len = max(1, int(0.010 * sample_rate))  # 10ms
    rms_env: list[float] = []
    for start in range(0, len(samples) - frame_len + 1, hop_len):
        frame = samples[start : start + frame_len]
        energy = sum(x * x for x in frame) / frame_len
        rms_env.append(math.sqrt(energy))

    if len(rms_env) < 10:
        return 0

    peak_max = max(rms_env)
    if peak_max < 0.02:
        return 0

    threshold = peak_max * 0.35
    peaks: list[tuple[float, float]] = []
    min_peak_sep_frames = max(1, int(0.06 / 0.01))  # 60ms minimum peak separation
    last_peak_idx = -min_peak_sep_frames

    for i in range(1, len(rms_env) - 1):
        val = rms_env[i]
        if val >= threshold and val >= rms_env[i - 1] and val > rms_env[i + 1]:
            if i - last_peak_idx >= min_peak_sep_frames:
                peaks.append((i * 0.01, val))
                last_peak_idx = i

    if len(peaks) < 3:
        return 0

    # Look for clusters of >= 3 rapid bursts with inter-peak intervals in [80ms, 250ms]
    # and consistent burst amplitudes (CV < 0.35).
    repetitions = 0
    i = 0
    while i <= len(peaks) - 3:
        dt1 = peaks[i + 1][0] - peaks[i][0]
        dt2 = peaks[i + 2][0] - peaks[i + 1][0]
        if 0.08 <= dt1 <= 0.25 and 0.08 <= dt2 <= 0.25:
            amps = [peaks[i][1], peaks[i + 1][1], peaks[i + 2][1]]
            mean_amp = sum(amps) / 3.0
            std_amp = math.sqrt(sum((a - mean_amp) ** 2 for a in amps) / 3.0)
            if mean_amp > 0 and (std_amp / mean_amp) < 0.35:
                repetitions += 1
                i += 3
                continue
        i += 1
    return repetitions


def _extract_dsp_features_pure(
    samples: Sequence[float], sample_rate: int
) -> dict[str, Optional[float]]:
    """Lightweight CPU DSP extractor (F0, jitter, spectral flux, breath intakes)."""
    if len(samples) < int(0.1 * sample_rate):
        return {
            "f0_mean": None,
            "jitter": None,
            "spectral_flux": None,
            "rapid_breath_intakes": 0.0,
        }

    frame_len = int(0.030 * sample_rate)  # 30ms
    hop_len = int(0.015 * sample_rate)  # 15ms
    min_lag = max(1, int(sample_rate / 400))  # 400 Hz
    max_lag = max(min_lag + 1, int(sample_rate / 75))  # 75 Hz

    periods: list[float] = []
    f0_values: list[float] = []
    frame_energies: list[float] = []
    frame_zcrs: list[float] = []

    for start in range(0, len(samples) - frame_len + 1, hop_len):
        frame = samples[start : start + frame_len]
        rms = math.sqrt(sum(x * x for x in frame) / frame_len)
        frame_energies.append(rms)

        # Zero-crossing rate
        crossings = sum(
            1 for a, b in zip(frame, frame[1:]) if (a >= 0.0) != (b >= 0.0)
        )
        zcr = crossings / float(frame_len)
        frame_zcrs.append(zcr)

        # Autocorrelation F0 on voiced frames (sufficient energy, low ZCR)
        if rms > 0.015 and zcr < 0.18 and max_lag < frame_len:
            r0 = sum(x * x for x in frame)
            if r0 > 1e-8:
                best_lag = 0
                best_corr = 0.0
                # Step by 2 samples for speed
                for lag in range(min_lag, min(max_lag, frame_len // 2), 2):
                    corr = sum(
                        frame[k] * frame[k + lag] for k in range(0, frame_len - lag, 2)
                    )
                    norm_corr = (2.0 * corr) / r0
                    if norm_corr > best_corr:
                        best_corr = norm_corr
                        best_lag = lag
                if best_corr > 0.45 and best_lag > 0:
                    period_sec = best_lag / float(sample_rate)
                    periods.append(period_sec)
                    f0_values.append(1.0 / period_sec)

    f0_mean = statistics.fmean(f0_values) if f0_values else None

    jitter: Optional[float] = None
    if len(periods) >= 3:
        mean_period = statistics.fmean(periods)
        if mean_period > 0:
            diffs = [abs(b - a) for a, b in zip(periods, periods[1:])]
            jitter = statistics.fmean(diffs) / mean_period

    # Proxy spectral flux from frame-to-frame RMS + ZCR envelope variation
    spectral_flux: Optional[float] = None
    if len(frame_energies) >= 2:
        flux_vals = [
            abs(e2 - e1) + 0.5 * abs(z2 - z1)
            for (e1, z1), (e2, z2) in zip(
                zip(frame_energies, frame_zcrs),
                zip(frame_energies[1:], frame_zcrs[1:]),
            )
        ]
        spectral_flux = statistics.fmean(flux_vals)

    # Rapid breath intakes: short unvoiced high-ZCR bursts (0.06s-0.30s) at moderate energy
    breath_intakes = 0
    in_breath = False
    breath_frames = 0
    for rms, zcr in zip(frame_energies, frame_zcrs):
        is_breath_frame = 0.006 <= rms <= 0.06 and zcr >= 0.25
        if is_breath_frame:
            if not in_breath:
                in_breath = True
                breath_frames = 1
            else:
                breath_frames += 1
        else:
            if in_breath and 4 <= breath_frames <= 20:
                breath_intakes += 1
            in_breath = False
            breath_frames = 0

    return {
        "f0_mean": f0_mean,
        "jitter": jitter,
        "spectral_flux": spectral_flux,
        "rapid_breath_intakes": float(breath_intakes),
    }


class _OpenSmileExtractor:
    """Wrapper around openSMILE eGeMAPSv02 when available."""

    def __init__(self) -> None:
        self._smile: Any = None
        try:
            import opensmile

            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except Exception:
            self._smile = None

    @property
    def available(self) -> bool:
        return self._smile is not None

    def extract(
        self, samples: Sequence[float], sample_rate: int
    ) -> dict[str, Optional[float]]:
        if not self._smile or not samples:
            return {}
        try:
            import numpy as np

            signal = np.asarray(samples, dtype=np.float32)
            df = self._smile.process_signal(signal, sample_rate)
            row = df.iloc[0]
            return {
                "f0_mean": float(row.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0.0)) or None,
                "jitter": float(row.get("jitterLocal_sma3nz_amean", 0.0)) or None,
                "spectral_flux": float(row.get("spectralFlux_sma3nz_amean", 0.0)) or None,
            }
        except Exception as exc:
            logger.debug("openSMILE extraction fallback triggered: %s", exc)
            return {}


class _LocalSERClassifier:
    """Optional local SpeechBrain SER classifier for high-arousal / anxious states."""

    HIGH_AROUSAL_LABELS = frozenset({"ang", "angry", "fea", "fear", "anxious", "tense"})

    def __init__(self, model_dir: Optional[Path], device: str) -> None:
        self._classifier: Any = None
        if model_dir is None:
            return
        local_dir = ser_model_path(model_dir)
        if not local_dir.exists() or not any(local_dir.iterdir()):
            return
        try:
            from speechbrain.inference.interfaces import foreign_class

            self._classifier = foreign_class(
                source=str(local_dir),
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=str(local_dir),
                run_opts={"device": device},
            )
        except Exception as exc:
            logger.warning("Local SpeechBrain SER unavailable (%s); using DSP stress model.", exc)
            self._classifier = None

    @property
    def available(self) -> bool:
        return self._classifier is not None

    def score_anxious(self, samples: Sequence[float], sample_rate: int) -> Optional[float]:
        min_samples = int(SER_MIN_DURATION_SEC * sample_rate)
        if not self._classifier or len(samples) < min_samples:
            return None
        try:
            import torch

            win_samples = int(SER_WINDOW_SEC * sample_rate)
            total = len(samples)
            if total <= win_samples:
                windows = [samples]
            else:
                # Sample up to SER_MAX_WINDOWS_PER_TURN evenly spaced 8s windows
                # to bound wav2vec2 self-attention memory on multi-minute turns.
                num_wins = min(SER_MAX_WINDOWS_PER_TURN, max(2, math.ceil(total / win_samples)))
                max_start = total - win_samples
                starts = [
                    int(round(i * max_start / (num_wins - 1))) for i in range(num_wins)
                ]
                windows = [samples[s : s + win_samples] for s in starts]

            scores: list[float] = []
            with torch.no_grad():
                for chunk in windows:
                    wav_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                    wav_lens = torch.tensor([1.0], dtype=torch.float32)
                    out_prob, _score, _index, text_lab = self._classifier.classify_batch(
                        wav_tensor, wav_lens
                    )
                    probs = torch.softmax(out_prob, dim=-1).squeeze(0).detach().cpu().tolist()
                    label = str(text_lab[0]).strip().lower() if text_lab else ""
                    max_prob = max(probs) if probs else 0.0
                    if label in self.HIGH_AROUSAL_LABELS:
                        scores.append(float(max_prob))
                    else:
                        scores.append(float(max(0.0, 1.0 - max_prob) * 0.35))
                    del wav_tensor, wav_lens, out_prob
            return statistics.fmean(scores) if scores else None
        except Exception as exc:
            logger.debug("SER inference failed on turn: %s", exc)
            return None

    def close(self) -> None:
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
            release_memory()


def _safe_zscore(value: Optional[float], values: Sequence[float]) -> float:
    if value is None or len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std < 1e-6:
        return 0.0
    return (value - mean) / std


def _classify_pace_label(avg_wpm: float) -> str:
    if avg_wpm > FAST_WPM_THRESHOLD:
        return "Fast"
    if avg_wpm < SLOW_WPM_THRESHOLD:
        return "Slow"
    return "Normal"


def analyze_session_prosody(
    turns: list[Turn],
    wav_path: Optional[Path],
    duration_seconds: float,
    model_dir: Optional[Path] = None,
    device_config: Optional[DeviceConfig] = None,
    use_ser: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> SessionSummary:
    """Compute acoustic & prosodic biomarkers for every turn and build a SessionSummary."""
    if not turns:
        return SessionSummary(
            duration_seconds=duration_seconds,
            speaker_count=0,
            speakers=[],
        )

    smile = _OpenSmileExtractor()
    torch_dev = device_config.torch_device if device_config else "cpu"
    ser = _LocalSERClassifier(model_dir=model_dir if use_ser else None, device=torch_dev)

    # Pass 1: Compute per-turn raw pace, disfluency, and acoustic features
    per_speaker_syl_durs: dict[str, list[float]] = {}
    for turn in turns:
        for w in turn.words:
            if _WORD_ALNUM_RE.search(w.text) and w.duration > 0.02:
                syls = max(1, _estimate_word_syllables(w.text))
                per_speaker_syl_durs.setdefault(turn.speaker, []).append(w.duration / syls)

    speaker_median_syl_dur = {
        spk: statistics.median(durs)
        for spk, durs in per_speaker_syl_durs.items()
        if durs
    }

    raw_records: list[dict[str, Any]] = []
    total_turns = len(turns)
    try:
        for idx, turn in enumerate(turns, start=1):
            if total_turns >= 1 and (idx == 1 or idx % 5 == 0 or idx == total_turns):
                pct = int(100 * idx / total_turns)
                if progress_cb is not None:
                    progress_cb(idx / float(total_turns), f"{idx}/{total_turns} turns")
                elif total_turns >= 5:
                    sys.stdout.write(f"\r    Prosody progress: {pct:3d}% ({idx}/{total_turns} turns)")
                    sys.stdout.flush()
            wpm, syl_per_sec, _active_dur = compute_turn_pace(turn)
            blocks = detect_mid_clause_blocks(turn.words)
            prolongations = detect_prolongations(
                turn.words,
                median_sec_per_syllable=speaker_median_syl_dur.get(turn.speaker, 0.22),
            )

            repetitions = 0
            dsp: dict[str, Optional[float]] = {
                "f0_mean": None,
                "jitter": None,
                "spectral_flux": None,
                "rapid_breath_intakes": 0.0,
            }
            ser_score: Optional[float] = None

            if wav_path is not None and wav_path.exists() and turn.duration > 0.05:
                samples, sr = read_wav_segment(wav_path, turn.start, turn.end)
                if samples:
                    repetitions = detect_syllable_repetitions(samples, sr)
                    dsp = _extract_dsp_features_pure(samples, sr)
                    if smile.available:
                        smile_feats = smile.extract(samples, sr)
                        for key, val in smile_feats.items():
                            if val is not None:
                                dsp[key] = val
                    if ser.available and turn.duration >= SER_MIN_DURATION_SEC:
                        ser_score = ser.score_anxious(samples, sr)

            raw_records.append(
                {
                    "turn": turn,
                    "wpm": wpm,
                    "syl_per_sec": syl_per_sec,
                    "blocks": blocks,
                    "prolongations": prolongations,
                    "repetitions": repetitions,
                    "f0_mean": dsp.get("f0_mean"),
                    "jitter": dsp.get("jitter"),
                    "spectral_flux": dsp.get("spectral_flux"),
                    "rapid_breath_intakes": int(dsp.get("rapid_breath_intakes") or 0),
                    "ser_score": ser_score,
                }
            )
    finally:
        if total_turns >= 5:
            sys.stdout.write("\n")
            sys.stdout.flush()
        ser.close()

    # Pass 2: Speaker-normalized baselines for pace (+1.5 sigma) and vocal stress
    speaker_wpms: dict[str, list[float]] = {}
    speaker_f0s: dict[str, list[float]] = {}
    speaker_jitters: dict[str, list[float]] = {}
    speaker_fluxes: dict[str, list[float]] = {}

    for rec in raw_records:
        spk = rec["turn"].speaker
        speaker_wpms.setdefault(spk, []).append(rec["wpm"])
        if rec["f0_mean"] is not None:
            speaker_f0s.setdefault(spk, []).append(rec["f0_mean"])
        if rec["jitter"] is not None:
            speaker_jitters.setdefault(spk, []).append(rec["jitter"])
        if rec["spectral_flux"] is not None:
            speaker_fluxes.setdefault(spk, []).append(rec["spectral_flux"])

    speaker_order: list[str] = []
    speaker_turns: dict[str, list[Turn]] = {}

    for rec in raw_records:
        turn: Turn = rec["turn"]
        spk = turn.speaker
        if spk not in speaker_turns:
            speaker_order.append(spk)
            speaker_turns[spk] = []

        wpm = rec["wpm"]
        wpms = speaker_wpms.get(spk, [])
        wpm_z = _safe_zscore(wpm, wpms) if len(wpms) >= 3 else 0.0
        is_fast = (wpm > FAST_WPM_THRESHOLD) or (
            len(wpms) >= 3 and statistics.pstdev(wpms) > 5.0 and wpm_z > FAST_ZSCORE_THRESHOLD
        )

        # Combine speaker-relative acoustic stress markers + optional SER score
        f0_z = max(0.0, _safe_zscore(rec["f0_mean"], speaker_f0s.get(spk, [])))
        jitter_z = max(0.0, _safe_zscore(rec["jitter"], speaker_jitters.get(spk, [])))
        flux_drop_z = max(0.0, -_safe_zscore(rec["spectral_flux"], speaker_fluxes.get(spk, [])))
        breath_bonus = min(0.25, 0.08 * rec["rapid_breath_intakes"])
        disfluency_bonus = min(0.20, 0.07 * (rec["blocks"] + rec["prolongations"] + rec["repetitions"]))
        pace_bonus = 0.15 if is_fast else 0.0

        dsp_stress = min(
            1.0,
            0.22 * min(f0_z, 3.0) / 3.0
            + 0.25 * min(jitter_z, 3.0) / 3.0
            + 0.13 * min(flux_drop_z, 3.0) / 3.0
            + breath_bonus
            + disfluency_bonus
            + pace_bonus,
        )

        if rec["ser_score"] is not None:
            stress_score = 0.55 * rec["ser_score"] + 0.45 * dsp_stress
        else:
            stress_score = dsp_stress

        if stress_score >= 0.60:
            state = "tense"
        elif stress_score >= 0.40:
            state = "elevated"
        else:
            state = "calm"

        turn.prosody = TurnProsody(
            wpm=wpm,
            syllables_per_sec=rec["syl_per_sec"],
            is_fast=is_fast,
            block_count=rec["blocks"],
            prolongation_count=rec["prolongations"],
            repetition_count=rec["repetitions"],
            stress_score=stress_score,
            state=state,
            ser_anxious_score=rec["ser_score"],
            f0_mean=rec["f0_mean"],
            jitter=rec["jitter"],
            spectral_flux=rec["spectral_flux"],
            rapid_breath_intakes=rec["rapid_breath_intakes"],
        )
        speaker_turns[spk].append(turn)

    # Pass 3: Build SessionSummary
    speaker_summaries: list[SpeakerSummary] = []
    for spk in speaker_order:
        sturns = speaker_turns[spk]
        prosodies = [t.prosody for t in sturns if t.prosody is not None]
        avg_wpm = statistics.fmean(p.wpm for p in prosodies) if prosodies else 0.0
        total_disf = sum(p.total_disfluencies for p in prosodies)
        elevated_cnt = sum(1 for p in prosodies if p.is_elevated_stress)
        elevated_pct = (100.0 * elevated_cnt / len(prosodies)) if prosodies else 0.0
        speaker_summaries.append(
            SpeakerSummary(
                speaker=spk,
                avg_wpm=avg_wpm,
                pace_label=_classify_pace_label(avg_wpm),
                total_disfluencies=total_disf,
                elevated_stress_pct=elevated_pct,
                turn_count=len(sturns),
            )
        )

    return SessionSummary(
        duration_seconds=duration_seconds,
        speaker_count=len(speaker_summaries),
        speakers=speaker_summaries,
    )
