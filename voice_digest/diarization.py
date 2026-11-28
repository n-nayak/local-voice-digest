"""Offline speaker diarization via pyannote.audio 3.1+ or Sherpa-ONNX."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

from voice_digest.compat import apply_torch_compat
from voice_digest.models import (
    PYANNOTE_DIARIZATION_MODEL,
    SHERPA_EMB_FILENAME,
    SORTFORMER_MAX_SPEAKERS,
    DeviceConfig,
    is_sortformer_available,
    pyannote_model_path,
    release_memory,
    sherpa_model_path,
    sortformer_model_path,
)
from voice_digest.types import DiarizationSegment

logger = logging.getLogger(__name__)


def resolve_diarizer(requested: str, num_speakers: int) -> str:
    """Resolve effective diarizer backend, auto-falling back to sherpa-onnx for >4 speakers."""
    if requested == "auto":
        if num_speakers <= SORTFORMER_MAX_SPEAKERS and is_sortformer_available():
            return "sortformer"
        return "sherpa-onnx"
    if requested == "sortformer" and num_speakers > SORTFORMER_MAX_SPEAKERS:
        logger.info(
            "Sortformer v2.1 supports up to %d speakers (requested %d); falling back to sherpa-onnx.",
            SORTFORMER_MAX_SPEAKERS,
            num_speakers,
        )
        return "sherpa-onnx"
    return requested


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

    effective = resolve_diarizer(diarizer, num_speakers)

    if effective == "sortformer":
        return _diarize_sortformer(
            wav_path, num_speakers, model_dir, progress_cb=progress_cb
        )
    if effective == "pyannote":
        return _diarize_pyannote(wav_path, num_speakers, model_dir, device_config)
    if effective == "sherpa-onnx":
        return _diarize_sherpa_onnx(
            wav_path, num_speakers, model_dir, threads=threads, progress_cb=progress_cb
        )
    raise ValueError(
        f"Unsupported diarizer '{diarizer}'. Choose 'auto', 'sortformer', 'sherpa-onnx', or 'pyannote'."
    )


def _clamp_to_top_speakers(
    segments: list[DiarizationSegment],
    num_speakers: int,
) -> list[DiarizationSegment]:
    """Ensure diarization output never exceeds `num_speakers` by remapping minor phantom slots."""
    if not segments or num_speakers < 1:
        return segments

    durations: dict[str, float] = {}
    for seg in segments:
        durations[seg.speaker] = durations.get(seg.speaker, 0.0) + max(0.0, seg.end - seg.start)

    if len(durations) <= num_speakers:
        return segments

    top_speakers = {
        spk
        for spk, _ in sorted(durations.items(), key=lambda kv: kv[1], reverse=True)[:num_speakers]
    }
    valid_spans = [seg for seg in segments if seg.speaker in top_speakers]
    if not valid_spans:
        return segments

    clamped: list[DiarizationSegment] = []
    for seg in segments:
        if seg.speaker in top_speakers:
            clamped.append(seg)
            continue
        mid = 0.5 * (seg.start + seg.end)
        nearest = min(
            valid_spans,
            key=lambda v: min(abs(mid - v.start), abs(mid - v.end)),
        )
        clamped.append(DiarizationSegment(start=seg.start, end=seg.end, speaker=nearest.speaker))
    return clamped


def _spherical_kmeans(embeddings: list[list[float]], k: int, max_iter: int = 25) -> list[int]:
    """Deterministic spherical K-means (cosine similarity) using furthest-point seeding."""
    import numpy as np

    mat = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-8)
    n = mat.shape[0]
    if n <= k:
        return list(range(n))

    # Deterministic furthest-point seeding (k-means++ style without randomness)
    mean_vec = mat.mean(axis=0)
    mean_vec /= max(float(np.linalg.norm(mean_vec)), 1e-8)
    first_idx = int(np.argmin(mat @ mean_vec))
    centers = [mat[first_idx].copy()]
    for _ in range(1, k):
        sims = np.max(np.stack([mat @ c for c in centers], axis=1), axis=1)
        next_idx = int(np.argmin(sims))
        centers.append(mat[next_idx].copy())
    centroids = np.stack(centers, axis=0)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        new_labels = np.argmax(mat @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k):
            mask = labels == c_idx
            if np.any(mask):
                vec = mat[mask].mean(axis=0)
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    centroids[c_idx] = vec / norm
    return [int(x) for x in labels.tolist()]


def _recluster_segments_with_wespeaker(
    segments: list[DiarizationSegment],
    samples: object,
    sample_rate: int,
    num_speakers: int,
    model_dir: Path,
) -> list[DiarizationSegment]:
    """Lock long-file speaker identity across all Sortformer segments using global WeSpeaker voiceprints.

    Prevents streaming speaker-cache drift during long (5+ minute) monologues while preserving
    Sortformer's sharp neural turn boundaries.
    """
    if num_speakers < 2 or len(segments) < num_speakers:
        return segments

    emb_model = sherpa_model_path(model_dir) / SHERPA_EMB_FILENAME
    if not emb_model.exists():
        return segments

    try:
        import numpy as np
        import sherpa_onnx
    except ImportError:
        return segments

    if not hasattr(sherpa_onnx, "SpeakerEmbeddingExtractor"):
        return segments

    try:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model),
            num_threads=4,
            debug=False,
            provider="cpu",
        )
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    except Exception as exc:
        logger.debug("Skipping WeSpeaker global re-clustering (%s).", exc)
        return segments

    audio = np.asarray(samples, dtype=np.float32)
    total_len = len(audio)
    embedded_indices: list[int] = []
    embeddings: list[list[float]] = []

    for idx, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur < 0.8:
            continue
        # For very long segments, sample up to 12 seconds from the middle for a clean voiceprint
        seg_start = seg.start
        seg_end = seg.end
        if dur > 12.0:
            mid = 0.5 * (seg_start + seg_end)
            seg_start = mid - 6.0
            seg_end = mid + 6.0
        s_i = max(0, min(total_len, int(seg_start * sample_rate)))
        e_i = max(s_i, min(total_len, int(seg_end * sample_rate)))
        if e_i - s_i < int(0.6 * sample_rate):
            continue
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate, audio[s_i:e_i].tolist())
        stream.input_finished()
        if extractor.is_ready(stream):
            vec = extractor.compute(stream)
            if vec:
                embedded_indices.append(idx)
                embeddings.append(list(vec))

    if len(embeddings) < num_speakers * 2:
        return segments

    cluster_ids = _spherical_kmeans(embeddings, k=num_speakers)
    assigned: dict[int, str] = {
        seg_idx: f"SPEAKER_{cid:02d}"
        for seg_idx, cid in zip(embedded_indices, cluster_ids)
    }

    logger.info(
        "Applied global WeSpeaker voiceprint locking across %d / %d Sortformer segments (%d speakers).",
        len(assigned),
        len(segments),
        num_speakers,
    )

    relabeled: list[DiarizationSegment] = []
    for idx, seg in enumerate(segments):
        if idx in assigned:
            spk = assigned[idx]
        else:
            # Inherit nearest embedded segment's global voiceprint
            nearest_idx = min(embedded_indices, key=lambda ei: abs(segments[ei].start - seg.start))
            spk = assigned[nearest_idx]
        relabeled.append(DiarizationSegment(start=seg.start, end=seg.end, speaker=spk))
    return relabeled


def _diarize_sortformer(
    wav_path: Path,
    num_speakers: int,
    model_dir: Path,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[DiarizationSegment]:
    """End-to-end neural speaker diarization on Apple Silicon MLX (NVIDIA Sortformer v2.1, up to 4 speakers)."""
    try:
        from mlx_audio.vad import load as load_vad_model
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is not installed. Run 'uv pip install mlx-audio' or use '--diarizer sherpa-onnx'."
        ) from exc

    local_dir = sortformer_model_path(model_dir)
    if not local_dir.exists() or not any(local_dir.iterdir()):
        raise FileNotFoundError(
            f"Sortformer v2.1 MLX model not found in '{local_dir}'. "
            f"Run 'python transcribe.py --download-models --diarizer sortformer --model-dir {model_dir}' first."
        )

    logger.info(
        "Loading NVIDIA Sortformer v2.1 MLX diarizer from %s (up to %d speakers)",
        local_dir,
        num_speakers,
    )
    if progress_cb is not None:
        progress_cb(0.15, "Loading NVIDIA Sortformer v2.1 (MLX GPU)")

    model = load_vad_model(str(local_dir))
    try:
        import numpy as np

        from voice_digest.audio import get_wav_duration, read_wav_segment

        total_dur = max(1.0, get_wav_duration(wav_path))
        # Use 15s offline chunks: 3x more acoustic context than 5s live streaming while keeping Metal RAM O(1)
        chunk_sec = 15.0
        est_chunks = max(1, int((total_dur + chunk_sec - 1) // chunk_sec))

        try:
            import soundfile as sf

            samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception:
            raw_samples, sample_rate = read_wav_segment(wav_path, 0.0, 1e9)
            samples = np.asarray(raw_samples, dtype=np.float32)

        if progress_cb is not None:
            progress_cb(0.15, f"Streaming Sortformer v2.1 ({est_chunks} chunks)")

        segments: list[DiarizationSegment] = []
        chunk_idx = 0
        # Use generate_stream (Arrival-Order Speaker Cache) so Metal GPU memory stays O(1)
        # even on 30-min to 3+ hour recordings (model.generate() is O(T^2) and OOMs Metal).
        for chunk_out in model.generate_stream(
            samples,
            sample_rate=int(sample_rate),
            chunk_duration=chunk_sec,
            threshold=0.5,
            merge_gap=0.2,
            verbose=False,
        ):
            chunk_idx += 1
            for seg in getattr(chunk_out, "segments", None) or []:
                spk_raw = getattr(seg, "speaker", 0)
                try:
                    spk_idx = int(spk_raw)
                    spk_label = f"SPEAKER_{spk_idx:02d}"
                except (ValueError, TypeError):
                    spk_label = str(spk_raw)
                segments.append(
                    DiarizationSegment(
                        start=float(getattr(seg, "start", 0.0)),
                        end=float(getattr(seg, "end", 0.0)),
                        speaker=spk_label,
                    )
                )
            if progress_cb is not None and (chunk_idx % 2 == 0 or chunk_idx >= est_chunks):
                frac = min(0.90, 0.15 + 0.75 * (chunk_idx / float(est_chunks)))
                progress_cb(frac, f"Chunk {chunk_idx}/{est_chunks}")

        segments.sort(key=lambda s: (s.start, s.end))

        # 1. Clamp strictly to top `num_speakers` slots (prevents phantom 3rd/4th slots on 2-speaker files)
        segments = _clamp_to_top_speakers(segments, num_speakers=num_speakers)

        # 2. Global WeSpeaker voiceprint locking (eliminates 5-min monologue drift across long files)
        if progress_cb is not None:
            progress_cb(0.94, "Locking global speaker voiceprints")
        segments = _recluster_segments_with_wespeaker(
            segments=segments,
            samples=samples,
            sample_rate=int(sample_rate),
            num_speakers=num_speakers,
            model_dir=model_dir,
        )

        # 3. Merge contiguous same-speaker spans split across chunk boundaries
        merged: list[DiarizationSegment] = []
        for seg in segments:
            if merged and merged[-1].speaker == seg.speaker and (seg.start - merged[-1].end) <= 0.35:
                merged[-1] = DiarizationSegment(
                    start=merged[-1].start,
                    end=max(merged[-1].end, seg.end),
                    speaker=seg.speaker,
                )
            else:
                merged.append(seg)

        if progress_cb is not None:
            progress_cb(1.0, f"Detected {len(merged)} speaker spans")
        return merged
    finally:
        del model
        release_memory()


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
        from voice_digest.audio import read_wav_segment

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
        from voice_digest.audio import read_wav_segment

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
