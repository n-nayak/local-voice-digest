"""Offline speaker diarization via pyannote.audio 3.1+ or Sherpa-ONNX."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

from local_transcribe.compat import apply_torch_compat
from local_transcribe.models import (
    PYANNOTE_DIARIZATION_MODEL,
    SHERPA_EMB_FILENAME,
    DeviceConfig,
    pyannote_model_path,
    release_memory,
    sherpa_model_path,
)
from local_transcribe.types import DiarizationSegment

logger = logging.getLogger(__name__)


def diarize_audio(
    wav_path: Path,
    num_speakers: int,
    model_dir: Path,
    device_config: DeviceConfig,
    diarizer: str = "pyannote",
    threads: Optional[int] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[DiarizationSegment]:
    """Run offline speaker diarization with a target speaker count."""
    if num_speakers < 1:
        raise ValueError(f"num_speakers must be >= 1, got {num_speakers}")

    if diarizer == "pyannote":
        return _diarize_pyannote(wav_path, num_speakers, model_dir, device_config)
    if diarizer == "sherpa-onnx":
        return _diarize_sherpa_onnx(
            wav_path, num_speakers, model_dir, threads=threads, progress_cb=progress_cb
        )
    raise ValueError(f"Unsupported diarizer '{diarizer}'. Choose 'pyannote' or 'sherpa-onnx'.")


def _load_wav_tensor(wav_path: Path) -> tuple[object, int]:
    """Load a 16-bit PCM WAV file into a (1, time) float32 torch Tensor without torchcodec."""
    import numpy as np
    import torch

    try:
        import soundfile as sf

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        return waveform, int(sr)
    except Exception:
        from local_transcribe.audio import read_wav_segment

        samples, sr = read_wav_segment(wav_path, 0.0, 1e9)
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        return waveform, int(sr)


def _diarize_pyannote(
    wav_path: Path,
    num_speakers: int,
    model_dir: Path,
    device_config: DeviceConfig,
) -> list[DiarizationSegment]:
    apply_torch_compat()
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run 'pip install -r requirements.txt' "
            "or use '--diarizer sherpa-onnx'."
        ) from exc

    # Pyannote 3.1 speaker embedding extraction can suffer numerical collapse on
    # Apple Silicon MPS, causing all speech to collapse into a single speaker.
    # Run diarization on CPU when MPS is selected; CUDA remains accelerated.
    diar_device = "cpu" if device_config.torch_device == "mps" else device_config.torch_device
    if device_config.torch_device == "mps":
        logger.info(
            "Using CPU for pyannote diarization (avoids Apple Silicon MPS speaker-clustering collapse)."
        )

    logger.info(
        "Loading pyannote diarization pipeline (%s) on %s for %d speakers",
        PYANNOTE_DIARIZATION_MODEL,
        diar_device,
        num_speakers,
    )
    pipeline = None
    try:
        cache_dir = pyannote_model_path(model_dir)
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_DIARIZATION_MODEL,
            cache_dir=str(cache_dir),
        )
        if pipeline is None:
            raise RuntimeError(
                f"Could not load local pyannote pipeline '{PYANNOTE_DIARIZATION_MODEL}'. "
                "Run 'python transcribe.py --download-models --model-dir ./models' first."
            )
        pipeline.to(torch.device(diar_device))
        waveform, sample_rate = _load_wav_tensor(wav_path)
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            with ProgressHook() as hook:
                annotation = pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    num_speakers=num_speakers,
                    hook=hook,
                )
        except Exception:
            annotation = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                num_speakers=num_speakers,
            )

        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
            )
        segments.sort(key=lambda s: (s.start, s.end))
        return segments
    finally:
        if pipeline is not None:
            del pipeline
        release_memory()


def _diarize_sherpa_onnx(
    wav_path: Path,
    num_speakers: int,
    model_dir: Path,
    threads: Optional[int] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[DiarizationSegment]:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa-onnx is not installed. Install it via 'uv pip install -e \".[mlx]\"' or 'pip install sherpa-onnx'."
        ) from exc

    base = sherpa_model_path(model_dir)
    seg_model = base / "segmentation.onnx"
    emb_model = base / SHERPA_EMB_FILENAME
    if not seg_model.exists() or not emb_model.exists():
        raise FileNotFoundError(
            f"Sherpa-ONNX models not found in '{base}'. "
            f"Run 'python transcribe.py --download-models --diarizer sherpa-onnx --model-dir {model_dir}' first."
        )

    num_threads = threads if (threads is not None and threads > 0) else 4
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model),
            num_threads=num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers),
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    try:
        import soundfile as sf

        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception:
        from local_transcribe.audio import read_wav_segment

        samples, sample_rate = read_wav_segment(wav_path, 0.0, 1e9)

    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"Sherpa-ONNX expects {sd.sample_rate} Hz audio, got {sample_rate} Hz."
        )

    last_pct = -1

    def _progress_cb(num_processed: int, num_total: int) -> int:
        nonlocal last_pct
        if num_total > 0:
            pct = int(100 * num_processed / num_total)
            if pct != last_pct and (pct % 5 == 0 or pct == 100):
                last_pct = pct
                detail = f"{pct}% ({num_processed}/{num_total} chunks)"
                if progress_cb is not None:
                    progress_cb(num_processed / float(num_total), detail)
                else:
                    sys.stdout.write(f"\r    Diarization progress: {pct:3d}% ({num_processed}/{num_total} chunks)")
                    sys.stdout.flush()
        return 0

    try:
        raw_segments = sd.process(samples, callback=_progress_cb).sort_by_start_time()
        if last_pct >= 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
    except TypeError:
        raw_segments = sd.process(samples).sort_by_start_time()

    return [
        DiarizationSegment(
            start=float(s.start),
            end=float(s.end),
            speaker=f"SPEAKER_{int(s.speaker):02d}",
        )
        for s in raw_segments
    ]
