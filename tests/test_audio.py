"""Tests for audio validation, WAV header reading, and segment slicing."""

from __future__ import annotations

import array
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from voice_digest.audio import (
    TARGET_SAMPLE_RATE,
    AudioValidationError,
    get_wav_duration,
    read_wav_segment,
    temporary_wav,
    validate_input_audio,
)


def write_synthetic_wav(path: Path, samples: list[float], sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    pcm = array.array("h", [max(-32768, min(32767, int(s * 32767.0))) for s in samples])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class TestAudioModule(unittest.TestCase):
    def test_validate_missing_and_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.m4a"
            with self.assertRaises(AudioValidationError):
                validate_input_audio(missing)

            empty = Path(tmp) / "empty.m4a"
            empty.write_bytes(b"")
            with self.assertRaises(AudioValidationError):
                validate_input_audio(empty)

            valid = Path(tmp) / "valid.m4a"
            valid.write_bytes(b"fake-m4a-bytes")
            self.assertEqual(validate_input_audio(valid), valid.resolve())

    def test_wav_duration_and_segment_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "tone.wav"
            sr = TARGET_SAMPLE_RATE
            duration = 2.0
            samples = [
                0.5 * math.sin(2.0 * math.pi * 220.0 * i / sr)
                for i in range(int(sr * duration))
            ]
            write_synthetic_wav(wav_path, samples, sr)

            self.assertAlmostEqual(get_wav_duration(wav_path), 2.0, places=3)
            seg, out_sr = read_wav_segment(wav_path, 0.5, 1.0)
            self.assertEqual(out_sr, sr)
            self.assertEqual(len(seg), int(0.5 * sr))

    def test_temporary_wav_cleans_up_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m4a_path = Path(tmp) / "sample.m4a"
            m4a_path.write_bytes(b"dummy")

            def fake_transcode(
                _in_path: Path,
                out_wav: Path,
                normalize_loudness: bool = True,
                loudness_target_lufs: float = -16.0,
            ) -> Path:
                write_synthetic_wav(out_wav, [0.0] * TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                return out_wav

            created_wav: Path | None = None
            with mock.patch("voice_digest.audio.transcode_to_wav", side_effect=fake_transcode):
                with temporary_wav(m4a_path) as (wav_path, duration):
                    created_wav = wav_path
                    self.assertTrue(wav_path.exists())
                    self.assertAlmostEqual(duration, 1.0, places=3)

            assert created_wav is not None
            self.assertFalse(created_wav.exists())


if __name__ == "__main__":
    unittest.main()
