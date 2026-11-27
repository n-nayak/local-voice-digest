"""Tests for dual-channel (Mic = 'Me' + System = 'Other' / 1:N) live call pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from voice_digest.audio import TARGET_SAMPLE_RATE
from voice_digest.cli import main
from voice_digest.dual_channel import merge_dual_channel_turns, suppress_mic_echo_segments
from voice_digest.types import ASRSegment, Turn, Word
from tests.test_audio import write_synthetic_wav


class TestDualChannelPipeline(unittest.TestCase):
    def test_merge_dual_channel_interleaves_chronologically(self) -> None:
        mic = [
            Turn("Me", 0.0, 1.2, "Hey how are you?", [Word("Hey", 0.0, 0.3), Word("how", 0.35, 0.6), Word("are", 0.65, 0.9), Word("you?", 0.95, 1.2)]),
            Turn("Me", 3.0, 4.0, "Awesome.", [Word("Awesome.", 3.0, 4.0)]),
        ]
        remote = [
            Turn("Other", 1.4, 2.6, "Doing great!", [Word("Doing", 1.4, 1.9), Word("great!", 2.0, 2.6)]),
        ]
        merged = merge_dual_channel_turns(mic, remote, pause_threshold=2.0)
        self.assertEqual(len(merged), 3)
        self.assertEqual([t.speaker for t in merged], ["Me", "Other", "Me"])
        self.assertEqual(merged[1].text, "Doing great!")

    def test_dual_channel_cli_end_to_end_exact_me_and_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mic_wav = Path(tmp) / "mic.wav"
            sys_wav = Path(tmp) / "system.wav"
            out_txt = Path(tmp) / "call_transcript.txt"
            write_synthetic_wav(mic_wav, [0.05] * (3 * TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)
            write_synthetic_wav(sys_wav, [0.05] * (3 * TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)

            def fake_transcode(in_path: Path, out_wav: Path, normalize_loudness: bool = True, loudness_target_lufs: float = -16.0) -> Path:
                out_wav.write_bytes(in_path.read_bytes())
                return out_wav

            call_idx = {"n": 0}

            def fake_asr(**_kwargs: object) -> tuple[list[ASRSegment], str]:
                call_idx["n"] += 1
                if call_idx["n"] == 1:
                    return ([ASRSegment("Hello from mic.", 0.0, 1.0, [Word("Hello", 0.0, 0.4), Word("from", 0.45, 0.7), Word("mic.", 0.75, 1.0)])], "en")
                return ([ASRSegment("Hello from remote.", 1.2, 2.4, [Word("Hello", 1.2, 1.6), Word("from", 1.65, 1.9), Word("remote.", 1.95, 2.4)])], "en")

            with (
                mock.patch("voice_digest.audio.transcode_to_wav", side_effect=fake_transcode),
                mock.patch("voice_digest.dual_channel.transcribe_and_align", side_effect=fake_asr),
            ):
                rc = main([
                    "--mic-file", str(mic_wav),
                    "--system-file", str(sys_wav),
                    "--remote-speakers", "1",
                    "--output", str(out_txt),
                ])

            self.assertEqual(rc, 0)
            content = out_txt.read_text(encoding="utf-8")
            self.assertIn("Me: Hello from mic.", content)
            self.assertIn("Other: Hello from remote.", content)

    def test_silent_dual_channel_skips_asr_and_prevents_hallucinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mic_wav = Path(tmp) / "mic.wav"
            sys_wav = Path(tmp) / "system.wav"
            out_txt = Path(tmp) / "silent_transcript.txt"
            write_synthetic_wav(mic_wav, [0.0] * (2 * TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)
            write_synthetic_wav(sys_wav, [0.0] * (2 * TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)

            with mock.patch("voice_digest.dual_channel.transcribe_and_align") as m_asr:
                rc = main([
                    "--mic-file", str(mic_wav),
                    "--system-file", str(sys_wav),
                    "--output", str(out_txt),
                ])

            self.assertEqual(rc, 0)
            self.assertEqual(m_asr.call_count, 0)
            self.assertEqual(out_txt.read_text(encoding="utf-8"), "")

    def test_suppress_mic_echo_removes_speaker_bleed_and_splits_turns(self) -> None:
        def _seg(text: str, start: float, step: float = 0.25) -> ASRSegment:
            toks = text.split()
            words = [
                Word(t, start + i * step, start + (i + 1) * step - 0.02)
                for i, t in enumerate(toks)
            ]
            return ASRSegment(text=text, start=words[0].start, end=words[-1].end, words=words)

        # Remote ('Other') plays through MacBook speakers in two chunks
        sys_segs = [
            _seg("Welcome back to the channel and welcome back to London And I have very very exciting news for you lot", 3.0),
            _seg("Property prices in London are falling, they are dropping, especially in the inner city", 11.0),
        ]

        # Mic ('Me') picks up user's real speech PLUS echoed speaker bleed at the same timestamps
        mic_segs = [
            _seg("Hello, can you hear what I am saying? Uh huh, yep.", 0.0),
            _seg("Welcome back to the channel and welcome back to London.", 3.1),
            _seg("I have very very exciting news for you lot.", 5.85),
            _seg("Ah, London, that's a nice place to live.", 8.2),
            _seg("Property prices in London are falling they are dropping especially in the inner city especially", 11.1),
            _seg("wow okay thank you very much", 15.0),
        ]
        # Sort mic words chronologically like a real single-track ASR output
        all_mic_words = sorted(
            [w for s in mic_segs for w in s.words],
            key=lambda w: w.start,
        )
        combined_mic = [ASRSegment(" ".join(w.text for w in all_mic_words), all_mic_words[0].start, all_mic_words[-1].end, all_mic_words)]

        cleaned = suppress_mic_echo_segments(combined_mic, sys_segs)
        mic_turns = [Turn("Me", s.start, s.end, s.text, s.words) for s in cleaned]
        remote_turns = [Turn("Other", s.start, s.end, s.text, s.words) for s in sys_segs]
        merged = merge_dual_channel_turns(mic_turns, remote_turns, pause_threshold=2.0)

        self.assertEqual([t.speaker for t in merged], ["Me", "Other", "Me", "Other", "Me"])
        self.assertNotIn("Welcome back to the channel", merged[0].text)
        self.assertIn("Uh huh, yep.", merged[0].text)
        self.assertIn("Ah, London, that's a nice place to live.", merged[2].text)
        self.assertNotIn("Property prices", merged[2].text)
        self.assertIn("wow okay thank you very much", merged[4].text)
        self.assertNotIn("especially", merged[4].text)

    def test_suppress_mic_echo_handles_clock_drift_for_multiword_phrases(self) -> None:
        def _seg(text: str, start: float, step: float = 0.25) -> ASRSegment:
            toks = text.split()
            words = [Word(t, start + i * step, start + (i + 1) * step - 0.02) for i, t in enumerate(toks)]
            return ASRSegment(text=text, start=words[0].start, end=words[-1].end, words=words)

        # Even if system.wav is shifted by 4 seconds relative to mic.wav, 4+ word echo phrases are still stripped
        sys_segs = [_seg("Welcome back to the channel and welcome back to London", 0.0)]
        mic_segs = [
            _seg("Hello my name is Alex.", 0.0),
            _seg("Welcome back to the channel and welcome back to London.", 4.5),
            _seg("That is awesome.", 8.0),
        ]
        all_mic_words = [w for s in mic_segs for w in s.words]
        combined_mic = [ASRSegment(" ".join(w.text for w in all_mic_words), all_mic_words[0].start, all_mic_words[-1].end, all_mic_words)]
        cleaned = suppress_mic_echo_segments(combined_mic, sys_segs)
        joined = " ".join(s.text for s in cleaned)
        self.assertIn("Hello my name is Alex.", joined)
        self.assertIn("That is awesome.", joined)
        self.assertNotIn("Welcome back to the channel", joined)

    def test_merge_splits_long_other_turn_around_mic_interjections(self) -> None:
        # One continuous 'Other' turn (4s..16s) with two 'Me' interjections inside it (at 8s and 13s)
        other_words = [
            Word("Great", 4.0, 4.5), Word("news", 4.6, 5.0), Word("for", 5.1, 5.4), Word("you.", 5.5, 6.0),
            Word("I", 8.2, 8.4), Word("have", 8.5, 8.8), Word("four", 8.9, 9.3), Word("properties.", 9.4, 10.2),
            Word("Prices", 13.2, 13.8), Word("are", 13.9, 14.2), Word("falling.", 14.3, 15.0),
        ]
        remote = [Turn("Other", 4.0, 15.0, "Great news for you. I have four properties. Prices are falling.", other_words)]
        mic = [
            Turn("Me", 0.0, 2.0, "Hello, what are you up to?", [Word("Hello,", 0.0, 0.5), Word("what", 0.6, 1.0), Word("are", 1.1, 1.4), Word("you", 1.5, 1.7), Word("up", 1.75, 1.85), Word("to?", 1.9, 2.0)]),
            Turn("Me", 7.8, 8.3, "What topic?", [Word("What", 7.8, 8.0), Word("topic?", 8.05, 8.3)]),
            Turn("Me", 12.8, 13.3, "Four properties!", [Word("Four", 12.8, 13.0), Word("properties!", 13.05, 13.3)]),
        ]
        merged = merge_dual_channel_turns(mic, remote, pause_threshold=2.0)
        self.assertEqual([t.speaker for t in merged], ["Me", "Other", "Me", "Other", "Me", "Other"])
        self.assertEqual(merged[1].text, "Great news for you.")
        self.assertEqual(merged[2].text, "What topic?")
        self.assertEqual(merged[3].text, "I have four properties.")
        self.assertEqual(merged[4].text, "Four properties!")
        self.assertEqual(merged[5].text, "Prices are falling.")


if __name__ == "__main__":
    unittest.main()
