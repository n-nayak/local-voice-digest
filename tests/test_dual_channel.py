"""Tests for dual-channel (Mic = 'Me' + System = 'Other' / 1:N) live call pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_transcribe.audio import TARGET_SAMPLE_RATE
from local_transcribe.cli import main
from local_transcribe.dual_channel import merge_dual_channel_turns
from local_transcribe.types import ASRSegment, Turn, Word
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
                mock.patch("local_transcribe.audio.transcode_to_wav", side_effect=fake_transcode),
                mock.patch("local_transcribe.dual_channel.transcribe_and_align", side_effect=fake_asr),
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

            with mock.patch("local_transcribe.dual_channel.transcribe_and_align") as m_asr:
                rc = main([
                    "--mic-file", str(mic_wav),
                    "--system-file", str(sys_wav),
                    "--output", str(out_txt),
                ])

            self.assertEqual(rc, 0)
            self.assertEqual(m_asr.call_count, 0)
            self.assertEqual(out_txt.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
