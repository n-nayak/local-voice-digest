"""Tests for acoustic and prosodic analysis (pace, blocks, prolongations, repetitions, non-lexical rule)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from local_transcribe.audio import TARGET_SAMPLE_RATE
from local_transcribe.prosody import (
    analyze_session_prosody,
    compute_turn_pace,
    detect_mid_clause_blocks,
    detect_prolongations,
    detect_syllable_repetitions,
)
from local_transcribe.types import Turn, Word
from tests.test_audio import write_synthetic_wav


def _make_words(tokens: list[str], start: float, word_dur: float, gap: float) -> list[Word]:
    words: list[Word] = []
    t = start
    for tok in tokens:
        words.append(Word(text=tok, start=t, end=t + word_dur))
        t += word_dur + gap
    return words


class TestProsodyEngine(unittest.TestCase):
    def test_fast_speech_flagged_and_normal_not_flagged(self) -> None:
        # Normal speech: 10 words over ~4.4s (~136 WPM)
        normal_words = _make_words(
            ["Good", "morning", "thanks", "for", "making", "time", "to", "review", "this", "today."],
            start=0.0,
            word_dur=0.38,
            gap=0.06,
        )
        # Rapid speech: 12 words over ~3.5s (~205 WPM > 180 WPM)
        fast_words = _make_words(
            ["I", "have", "several", "urgent", "concerns", "about", "the", "deployment", "window", "we", "discussed", "yesterday."],
            start=5.0,
            word_dur=0.24,
            gap=0.05,
        )

        t1 = Turn("speaker 1", normal_words[0].start, normal_words[-1].end, " ".join(w.text for w in normal_words), normal_words)
        t2 = Turn("speaker 2", fast_words[0].start, fast_words[-1].end, " ".join(w.text for w in fast_words), fast_words)

        summary = analyze_session_prosody([t1, t2], wav_path=None, duration_seconds=10.0)
        assert t1.prosody is not None and t2.prosody is not None
        self.assertLess(t1.prosody.wpm, 160.0)
        self.assertFalse(t1.prosody.is_fast)
        self.assertGreater(t2.prosody.wpm, 180.0)
        self.assertTrue(t2.prosody.is_fast)
        self.assertEqual(summary.speakers[0].pace_label, "Normal")
        self.assertEqual(summary.speakers[1].pace_label, "Fast")

    def test_mid_clause_blocks_detected_and_lexical_fillers_ignored(self) -> None:
        # Mid-clause blocks: two 0.7s silent gaps inside a clause
        words_with_blocks = [
            Word("I", 0.0, 0.2),
            Word("think", 0.9, 1.2),  # 0.7s mid-clause block
            Word("we", 1.25, 1.45),
            Word("should", 2.15, 2.45),  # 0.7s mid-clause block
            Word("proceed.", 2.5, 2.9),
            Word("Next", 3.6, 3.9),  # 0.7s after period -> NOT a mid-clause block
            Word("item.", 3.95, 4.3),
        ]
        self.assertEqual(detect_mid_clause_blocks(words_with_blocks), 2)

        # Smooth speech with lexical filler words ("um", "uh", "like", "you", "know")
        # Must NOT be flagged as acoustic disfluencies!
        filler_words = _make_words(
            ["Um", "I", "uh", "like", "you", "know", "think", "it", "works", "fine."],
            start=0.0,
            word_dur=0.35,
            gap=0.06,
        )
        turn = Turn("speaker 1", filler_words[0].start, filler_words[-1].end, " ".join(w.text for w in filler_words), filler_words)
        analyze_session_prosody([turn], wav_path=None, duration_seconds=5.0)
        assert turn.prosody is not None
        self.assertEqual(turn.prosody.total_disfluencies, 0)

    def test_prolongation_detection(self) -> None:
        words = [
            Word("We", 0.0, 0.2),
            Word("ssssstart", 0.25, 1.35),  # 1.1s for 1 syllable (> 3x 0.22s)
            Word("now.", 1.4, 1.7),
        ]
        self.assertEqual(detect_prolongations(words, median_sec_per_syllable=0.22), 1)

    def test_syllable_repetition_bursts_from_audio(self) -> None:
        sr = TARGET_SAMPLE_RATE
        total = int(1.2 * sr)
        samples = [0.0] * total
        # 3 short bursts at 0.20s, 0.34s, 0.48s (140ms apart)
        for center_sec in (0.20, 0.34, 0.48):
            c = int(center_sec * sr)
            half = int(0.03 * sr)
            for i in range(max(0, c - half), min(total, c + half)):
                env = 0.5 * (1.0 - math.cos(2.0 * math.pi * (i - (c - half)) / (2 * half)))
                samples[i] = 0.7 * env * math.sin(2.0 * math.pi * 180.0 * i / sr)

        self.assertGreaterEqual(detect_syllable_repetitions(samples, sr), 1)

    def test_dsp_prosody_with_synthetic_wav(self) -> None:
        sr = TARGET_SAMPLE_RATE
        # 2s calm tone (140 Hz) followed by 2s high-pitched tense/jittered tone (260 Hz)
        calm_samples = [
            0.35 * math.sin(2.0 * math.pi * 140.0 * i / sr) for i in range(2 * sr)
        ]
        tense_samples = []
        phase = 0.0
        for i in range(2 * sr):
            freq = 260.0 + 35.0 * math.sin(2.0 * math.pi * 18.0 * i / sr)
            phase += 2.0 * math.pi * freq / sr
            tense_samples.append(0.45 * math.sin(phase))

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "session.wav"
            write_synthetic_wav(wav_path, calm_samples + tense_samples, sr)

            w1 = _make_words(["This", "is", "calm", "speech."], 0.0, 0.4, 0.05)
            w2 = _make_words(
                ["This", "is", "very", "fast", "and", "quite", "tense", "speech", "right", "now."],
                2.0,
                0.15,
                0.04,
            )
            turns = [
                Turn("speaker 1", 0.0, 2.0, "This is calm speech.", w1),
                Turn("speaker 1", 2.0, 4.0, "This is very fast and quite tense speech right now.", w2),
            ]
            summary = analyze_session_prosody(turns, wav_path=wav_path, duration_seconds=4.0)
            assert turns[0].prosody is not None and turns[1].prosody is not None
            self.assertGreater(turns[1].prosody.stress_score, turns[0].prosody.stress_score)
            self.assertEqual(summary.speaker_count, 1)


if __name__ == "__main__":
    unittest.main()
