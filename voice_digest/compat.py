"""Compatibility shims for PyTorch 2.6+ and torchaudio 2.9+ with pyannote.audio 3.x."""

from __future__ import annotations

import functools
import inspect
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PATCHED = False


@dataclass
class _CompatAudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def _compat_info(uri: Any, backend: Any = None) -> _CompatAudioMetaData:
    """Drop-in replacement for removed torchaudio.info using soundfile or stdlib wave."""
    try:
        import soundfile as sf

        sf_info = sf.info(uri)
        return _CompatAudioMetaData(
            sample_rate=int(sf_info.samplerate),
            num_frames=int(sf_info.frames),
            num_channels=int(sf_info.channels),
            bits_per_sample=16,
            encoding=str(sf_info.subtype or "PCM_S"),
        )
    except Exception:
        path = str(uri) if isinstance(uri, (str, Path)) else uri
        with wave.open(path, "rb") as wf:
            return _CompatAudioMetaData(
                sample_rate=wf.getframerate(),
                num_frames=wf.getnframes(),
                num_channels=wf.getnchannels(),
                bits_per_sample=wf.getsampwidth() * 8,
                encoding="PCM_S",
            )


def _wrap_hf_download_fn(fn: Any) -> Any:
    """Translate legacy `use_auth_token` kwarg to `token` for huggingface_hub >= 1.0."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn
    if "use_auth_token" in params:
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "use_auth_token" in kwargs:
            auth_val = kwargs.pop("use_auth_token")
            if "token" not in kwargs or kwargs["token"] is None:
                kwargs["token"] = auth_val
        return fn(*args, **kwargs)

    return wrapper


def _patch_huggingface_hub() -> None:
    try:
        import huggingface_hub
        import huggingface_hub.file_download as hf_fd
        import huggingface_hub._snapshot_download as hf_sd

        wrapped_dl = _wrap_hf_download_fn(huggingface_hub.hf_hub_download)
        huggingface_hub.hf_hub_download = wrapped_dl
        hf_fd.hf_hub_download = wrapped_dl

        wrapped_snap = _wrap_hf_download_fn(huggingface_hub.snapshot_download)
        huggingface_hub.snapshot_download = wrapped_snap
        hf_sd.snapshot_download = wrapped_snap
    except Exception:
        pass


def apply_torch_compat() -> None:
    """Apply PyTorch, torchaudio, and huggingface_hub shims for pyannote.audio 3.x."""
    global _PATCHED
    if _PATCHED:
        return

    # PyTorch 2.6+ defaults torch.load(..., weights_only=True), which breaks
    # pyannote.audio 3.1 / lightning checkpoints containing custom classes.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    _patch_huggingface_hub()

    try:
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            torchaudio.AudioMetaData = _CompatAudioMetaData  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda _backend: None  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "info"):
            torchaudio.info = _compat_info  # type: ignore[attr-defined]
    except ImportError:
        pass

    _PATCHED = True
