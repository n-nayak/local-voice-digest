"""Tests for CLI behavior, default speaker prompt, and end-to-end mocked pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from voice_digest.audio import TARGET_SAMPLE_RATE
from voice_digest.cli import default_output_path, main, prompt_speaker_count
from voice_digest.types import ASRSegment, DiarizationSegment, Word
from tests.test_audio import write_synthetic_wav


class TestCLI(unittest.TestCase):
    def test_prompt_speaker_count_enter_defaults_to_two(self) -> None:
        prompts_seen: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts_seen.append(prompt)
            return ""

        result = prompt_speaker_count(input_fn=fake_input)
        self.assertEqual(result, 2)
        self.assertEqual(prompts_seen, ["Number of speakers [2]: "])

    def test_prompt_speaker_count_custom_value(self) -> None:
        self.assertEqual(prompt_speaker_count(input_fn=lambda _: "3"), 3)

    def test_default_output_path(self) -> None:
        out = default_output_path(Path("/tmp/interview.m4a"))
        self.assertEqual(out.name, "interview_transcript.txt")

    def test_end_to_end_offline_cli_with_mocked_ml_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m4a = Path(tmp) / "interview.m4a"
            m4a.write_bytes(b"fake-m4a-data")
            out_txt = Path(tmp) / "transcript.txt"

            def fake_transcode(
                _in_path: Path,
                out_wav: Path,
                normalize_loudness: bool = True,
                loudness_target_lufs: float = -16.0,
            ) -> Path:
                write_synthetic_wav(out_wav, [0.0] * (5 * TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)
                return out_wav

            fake_asr = (
                [
                    ASRSegment(
                        text="Good morning.",
                        start=0.0,
                        end=1.0,
                        words=[Word("Good", 0.0, 0.4), Word("morning.", 0.5, 1.0)],
                    ),
                    ASRSegment(
                        text="Happy to be here.",
                        start=1.5,
                        end=2.6,
                        words=[
                            Word("Happy", 1.5, 1.75),
                            Word("to", 1.8, 1.95),
                            Word("be", 2.0, 2.2),
                            Word("here.", 2.25, 2.6),
                        ],
                    ),
                ],
                "en",
            )
            fake_diar = [
                DiarizationSegment(0.0, 1.1, "SPEAKER_01"),
                DiarizationSegment(1.4, 2.7, "SPEAKER_00"),
            ]

            with (
                mock.patch("voice_digest.audio.transcode_to_wav", side_effect=fake_transcode),
                mock.patch("voice_digest.pipeline.transcribe_and_align", return_value=fake_asr),
                mock.patch("voice_digest.pipeline.diarize_audio", return_value=fake_diar),
            ):
                # Press Enter at interactive speaker prompt -> defaults to 2
                rc = main(
                    [str(m4a), "--prosody", "--output", str(out_txt)],
                    input_fn=lambda _: "",
                )

            self.assertEqual(rc, 0)
            content = out_txt.read_text(encoding="utf-8")
            self.assertIn("=== SESSION ACOUSTIC SUMMARY ===", content)
            self.assertIn("speaker 1 [pace:", content)
            self.assertIn("Good morning.", content)
            self.assertIn("speaker 2 [pace:", content)
            self.assertIn("Happy to be here.", content)

    def test_stage_cache_reuses_asr_and_diarization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m4a = Path(tmp) / "interview.m4a"
            m4a.write_bytes(b"fake-m4a-data")
            out_txt = Path(tmp) / "transcript.txt"
            cache_dir = Path(tmp) / "cache"

            def fake_transcode(
                _in_path: Path,
                out_wav: Path,
                normalize_loudness: bool = True,
                loudness_target_lufs: float = -16.0,
            ) -> Path:
                write_synthetic_wav(out_wav, [0.0] * (3 * TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)
                return out_wav

            fake_asr = (
                [ASRSegment("Hello there.", 0.0, 1.0, [Word("Hello", 0.0, 0.4), Word("there.", 0.5, 1.0)])],
                "en",
            )
            fake_diar = [DiarizationSegment(0.0, 1.1, "SPEAKER_00")]

            with (
                mock.patch("voice_digest.audio.transcode_to_wav", side_effect=fake_transcode),
                mock.patch("voice_digest.pipeline.transcribe_and_align", return_value=fake_asr) as m_asr,
                mock.patch("voice_digest.pipeline.diarize_audio", return_value=fake_diar) as m_diar,
            ):
                rc1 = main([str(m4a), "--speakers", "2", "--cache-dir", str(cache_dir), "--output", str(out_txt)])
                rc2 = main([str(m4a), "--speakers", "2", "--cache-dir", str(cache_dir), "--prosody", "--no-ser", "--output", str(out_txt)])

            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)
            # Second run must reuse cached ASR & diarization without calling ML backends again
            self.assertEqual(m_asr.call_count, 1)
            self.assertEqual(m_diar.call_count, 1)
            self.assertIn("=== SESSION ACOUSTIC SUMMARY ===", out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
