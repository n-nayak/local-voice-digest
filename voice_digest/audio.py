"""Audio input validation, FFmpeg transcoding, and streaming WAV access."""

from __future__ import annotations

import array
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

SUPPORTED_MEDIA_EXTENSIONS = frozenset({
    ".m4a", ".mp3", ".wav", ".ogg", ".opus", ".flac",
    ".aac", ".aiff", ".aif", ".caf", ".wma",
    ".mp4", ".mov", ".mkv", ".webm",
})


class AudioValidationError(ValueError):
    """Raised when the input audio file is missing, invalid, or empty."""


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is unavailable or fails to decode the input audio."""


def validate_input_audio(input_path: Path | str) -> Path:
    """Validate that the input audio file exists and is non-empty."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise AudioValidationError(f"Input audio file not found: {path}")
    if not path.is_file():
        raise AudioValidationError(f"Input path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise AudioValidationError(f"Input audio file is empty (0 bytes): {path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        logger.warning(
            "Input file '%s' has an uncommon media extension; attempting FFmpeg decode anyway.",
            path.name,
        )
    return path


def check_ffmpeg_available() -> str:
    """Locate bundled or system ffmpeg executable, or raise a clear error."""
    env_ffmpeg = os.environ.get("LOCAL_TRANSCRIBE_FFMPEG")
    if env_ffmpeg and Path(env_ffmpeg).is_file():
        return env_ffmpeg

    # Check alongside PyInstaller / macOS .app bundle Resources/bin/ffmpeg
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (
            exe_dir / "ffmpeg",
            exe_dir.parent / "Resources" / "bin" / "ffmpeg",
            exe_dir.parent.parent / "bin" / "ffmpeg",
        ):
            if candidate.is_file():
                return str(candidate)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FFmpegError(
            "ffmpeg is required to decode .m4a audio but was not found on PATH. "
            "Install it with 'brew install ffmpeg' (macOS) or 'sudo apt install ffmpeg' (Linux)."
        )
    return ffmpeg_bin


def transcode_to_wav(
    input_path: Path,
    output_wav: Path,
    normalize_loudness: bool = True,
    loudness_target_lufs: float = -16.0,
) -> Path:
    """Transcode (and optionally loudness-normalize) input audio to 16 kHz mono PCM WAV."""
    ffmpeg_bin = check_ffmpeg_available()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
    ]
    if normalize_loudness:
        target = max(-30.0, min(-5.0, float(loudness_target_lufs)))
        cmd.extend(["-af", f"loudnorm=I={target:g}:TP=-1.5:LRA=11"])
    cmd.extend(["-c:a", "pcm_s16le", str(output_wav)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise FFmpegError(f"FFmpeg failed to transcode '{input_path}': {stderr}")
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise FFmpegError(f"FFmpeg produced an empty WAV file for '{input_path}'.")
    return output_wav


def get_wav_rms(wav_path: Path) -> float:
    """Compute overall RMS amplitude [0.0, 1.0] of a 16-bit PCM WAV file in chunks."""
    import math

    sum_sq = 0.0
    total_samples = 0
    chunk_frames = 16_000 * 10
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnframes() == 0:
            return 0.0
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            samples = array.array("h")
            samples.frombytes(raw)
            sum_sq += sum((s / 32768.0) ** 2 for s in samples)
            total_samples += len(samples)
    if total_samples == 0:
        return 0.0
    return math.sqrt(sum_sq / total_samples)


def get_wav_duration(wav_path: Path) -> float:
    """Read exact duration in seconds from a WAV header without loading audio samples."""
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def read_wav_segment(
    wav_path: Path, start_sec: float, end_sec: float
) -> tuple[list[float], int]:
    """Read a normalized float [-1.0, 1.0] mono slice directly from a 16-bit PCM WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        total_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
        if sampwidth != 2:
            raise AudioValidationError(
                f"Expected 16-bit PCM WAV, got {sampwidth * 8}-bit."
            )

        start_frame = max(0, min(total_frames, int(round(start_sec * rate))))
        end_frame = max(start_frame, min(total_frames, int(round(end_sec * rate))))
        num_frames = end_frame - start_frame
        if num_frames <= 0:
            return [], rate

        wf.setpos(start_frame)
        raw_bytes = wf.readframes(num_frames)

    samples_i16 = array.array("h")
    samples_i16.frombytes(raw_bytes)
    if nchannels > 1:
        # Downmix interleaved channels to mono if needed
        mono = [
            sum(samples_i16[i : i + nchannels]) / (nchannels * 32768.0)
            for i in range(0, len(samples_i16), nchannels)
        ]
        return mono, rate

    return [s / 32768.0 for s in samples_i16], rate


@contextmanager
def temporary_wav(
    input_path: Path | str,
    normalize_loudness: bool = True,
    loudness_target_lufs: float = -16.0,
) -> Iterator[tuple[Path, float]]:
    """Validate input, transcode to a temporary 16 kHz mono WAV, and auto-clean on exit."""
    validated = validate_input_audio(input_path)
    with tempfile.TemporaryDirectory(prefix="voice_digest_") as tmp_dir:
        wav_path = Path(tmp_dir) / "audio_16k_mono.wav"
        transcode_to_wav(
            validated,
            wav_path,
            normalize_loudness=normalize_loudness,
            loudness_target_lufs=loudness_target_lufs,
        )
        duration = get_wav_duration(wav_path)
        yield wav_path, duration
