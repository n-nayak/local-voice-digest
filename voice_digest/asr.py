"""Batched ASR (WhisperX / faster-whisper) and wav2vec2 forced alignment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from voice_digest.compat import apply_torch_compat
from voice_digest.models import (
    DeviceConfig,
    is_mlx_available,
    mlx_whisper_model_path,
    release_memory,
    whisper_model_path,
)
from voice_digest.types import ASRSegment, Word

logger = logging.getLogger(__name__)


class MissingLocalModelError(FileNotFoundError):
    """Raised when required local model weights are missing in offline mode."""


def _resolve_local_whisper_source(model_dir: Path, model_size: str) -> str:
    local_path = whisper_model_path(model_dir, model_size)
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path)
    raise MissingLocalModelError(
        f"Local Whisper model '{model_size}' not found at '{local_path}'. "
        f"Run 'python transcribe.py --download-models --model-dir {model_dir} --model-size {model_size}' "
        "once while online to cache weights locally."
    )


def _resolve_local_mlx_source(model_dir: Path, model_size: str) -> str:
    local_path = mlx_whisper_model_path(model_dir, model_size)
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path)
    raise MissingLocalModelError(
        f"Local MLX Whisper model '{model_size}' not found at '{local_path}'. "
        f"Run 'python transcribe.py --download-models --asr-backend mlx --model-dir {model_dir} --model-size {model_size}' "
        "once while online to cache weights locally."
    )


def resolve_asr_backend(
    requested: str, model_dir: Path, model_size: str
) -> str:
    """Resolve effective ASR backend ('mlx' or 'whisperx')."""
    if requested == "mlx":
        return "mlx"
    if requested == "whisperx":
        return "whisperx"
    if requested != "auto":
        raise ValueError(f"Unsupported asr_backend '{requested}'. Choose auto, mlx, or whisperx.")

    mlx_path = mlx_whisper_model_path(model_dir, model_size)
    if is_mlx_available() and mlx_path.exists() and any(mlx_path.iterdir()):
        return "mlx"
    return "whisperx"


def _interpolate_missing_word_timestamps(
    raw_words: list[dict[str, Any]], seg_start: float, seg_end: float
) -> list[Word]:
    """Ensure every word has valid monotonic start/end timestamps."""
    if not raw_words:
        return []

    n = len(raw_words)
    starts: list[Optional[float]] = [
        float(w["start"]) if "start" in w and w["start"] is not None else None
        for w in raw_words
    ]
    ends: list[Optional[float]] = [
        float(w["end"]) if "end" in w and w["end"] is not None else None
        for w in raw_words
    ]

    for i in range(n):
        if starts[i] is None:
            prev_end = seg_start
            for j in range(i - 1, -1, -1):
                if ends[j] is not None:
                    prev_end = ends[j]  # type: ignore[assignment]
                    break
            starts[i] = prev_end
        if ends[i] is None:
            next_start = seg_end
            for j in range(i + 1, n):
                if starts[j] is not None:
                    next_start = starts[j]  # type: ignore[assignment]
                    break
            ends[i] = max(starts[i] or seg_start, next_start)

    words: list[Word] = []
    for i, w in enumerate(raw_words):
        text = str(w.get("word", w.get("text", ""))).strip()
        if not text:
            continue
        start = starts[i] if starts[i] is not None else seg_start
        end = ends[i] if ends[i] is not None else max(start, seg_end)
        if end < start:
            end = start
        score = float(w.get("score", w.get("probability", 1.0)) or 1.0)
        words.append(Word(text=text, start=start, end=end, score=score))
    return words


def transcribe_and_align(
    wav_path: Path,
    model_dir: Path,
    model_size: str,
    device_config: DeviceConfig,
    language: Optional[str] = None,
    threads: Optional[int] = None,
    asr_backend: str = "auto",
    align_mode: str = "auto",
) -> tuple[list[ASRSegment], str]:
    """Run batched ASR (MLX GPU or WhisperX/faster-whisper) with optional wav2vec2 alignment."""
    backend = resolve_asr_backend(asr_backend, model_dir, model_size)
    effective_align = (
        ("none" if backend == "mlx" else "wav2vec2")
        if align_mode == "auto"
        else align_mode
    )

    if backend == "mlx":
        model_source = _resolve_local_mlx_source(model_dir, model_size)
        return _run_mlx_whisper(
            wav_path=wav_path,
            model_source=model_source,
            device_config=device_config,
            language=language,
            run_wav2vec2=(effective_align == "wav2vec2"),
        )

    model_source = _resolve_local_whisper_source(model_dir, model_size)
    apply_torch_compat()

    try:
        import whisperx

        return _run_whisperx(
            whisperx=whisperx,
            wav_path=wav_path,
            model_source=model_source,
            device_config=device_config,
            language=language,
            threads=threads,
            run_wav2vec2=(effective_align == "wav2vec2"),
        )
    except ImportError:
        logger.warning(
            "whisperx is not installed; falling back to faster-whisper word timestamps."
        )
        return _run_faster_whisper_fallback(
            wav_path=wav_path,
            model_source=model_source,
            device_config=device_config,
            language=language,
            threads=threads,
        )


def _run_mlx_whisper(
    wav_path: Path,
    model_source: str,
    device_config: DeviceConfig,
    language: Optional[str],
    run_wav2vec2: bool = False,
) -> tuple[list[ASRSegment], str]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper is not installed. Install it via 'uv pip install -e \".[mlx]\"'."
        ) from exc

    logger.info("Running mlx-whisper on Apple Silicon GPU from %s", model_source)
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=model_source,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        verbose=False,
    )
    detected_lang = str(result.get("language") or language or "en")
    # Drop high no_speech_prob segments (eliminates Whisper silence hallucinations like "Thank you.")
    raw_segments = [
        seg
        for seg in result.get("segments", [])
        if float(seg.get("no_speech_prob", 0.0) or 0.0) <= 0.6
    ]
    release_memory()

    if run_wav2vec2 and raw_segments:
        try:
            apply_torch_compat()
            import whisperx

            audio = whisperx.load_audio(str(wav_path))
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=device_config.torch_device,
            )
            try:
                aligned = whisperx.align(
                    raw_segments,
                    align_model,
                    metadata,
                    audio,
                    device_config.torch_device,
                    return_char_alignments=False,
                )
                raw_segments = aligned.get("segments", raw_segments)
            finally:
                del align_model, audio
                release_memory()
        except Exception as exc:
            logger.warning(
                "wav2vec2 refinement unavailable (%s); using MLX word timestamps.", exc
            )

    segments: list[ASRSegment] = []
    for seg in raw_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = str(seg.get("text", "")).strip()
        words = _interpolate_missing_word_timestamps(
            seg.get("words", []), seg_start, seg_end
        )
        if not words and seg_text:
            tokens = seg_text.split()
            dur = max(0.01, seg_end - seg_start)
            step = dur / len(tokens)
            words = [
                Word(
                    text=tok,
                    start=seg_start + idx * step,
                    end=seg_start + (idx + 1) * step,
                )
                for idx, tok in enumerate(tokens)
            ]
        segments.append(
            ASRSegment(text=seg_text, start=seg_start, end=seg_end, words=words)
        )
    return segments, detected_lang


def _run_whisperx(
    whisperx: Any,
    wav_path: Path,
    model_source: str,
    device_config: DeviceConfig,
    language: Optional[str],
    threads: Optional[int] = None,
    run_wav2vec2: bool = True,
) -> tuple[list[ASRSegment], str]:
    audio = whisperx.load_audio(str(wav_path))

    # Stage 1a: Batched Whisper ASR
    logger.info(
        "Loading ASR model from %s (device=%s, compute=%s, batch=%d)",
        model_source,
        device_config.asr_device,
        device_config.asr_compute_type,
        device_config.asr_batch_size,
    )
    load_kwargs: dict[str, Any] = {
        "device": device_config.asr_device,
        "compute_type": device_config.asr_compute_type,
        "language": language,
        "local_files_only": True,
    }
    if threads is not None and threads > 0:
        load_kwargs["threads"] = threads
    asr_model = whisperx.load_model(model_source, **load_kwargs)
    try:
        asr_result = asr_model.transcribe(
            audio,
            batch_size=device_config.asr_batch_size,
            language=language,
        )
    finally:
        del asr_model
        release_memory()

    detected_lang = str(asr_result.get("language") or language or "en")

    # Stage 1b: wav2vec2 Forced Alignment
    aligned_segments_raw = asr_result.get("segments", [])
    if run_wav2vec2 and aligned_segments_raw:
        logger.info(
            "Loading wav2vec2 alignment model for '%s' on %s",
            detected_lang,
            device_config.torch_device,
        )
        align_model = None
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=device_config.torch_device,
            )
            aligned = whisperx.align(
                aligned_segments_raw,
                align_model,
                metadata,
                audio,
                device_config.torch_device,
                return_char_alignments=False,
            )
            aligned_segments_raw = aligned.get("segments", aligned_segments_raw)
        except Exception as exc:
            logger.warning(
                "wav2vec2 forced alignment unavailable locally for '%s' (%s); "
                "continuing with segment-level timestamps.",
                detected_lang,
                exc,
            )
        finally:
            if align_model is not None:
                del align_model
            release_memory()

    del audio
    release_memory()

    segments: list[ASRSegment] = []
    for seg in aligned_segments_raw:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = str(seg.get("text", "")).strip()
        words = _interpolate_missing_word_timestamps(
            seg.get("words", []), seg_start, seg_end
        )
        if not words and seg_text:
            tokens = seg_text.split()
            dur = max(0.01, seg_end - seg_start)
            step = dur / len(tokens)
            words = [
                Word(
                    text=tok,
                    start=seg_start + idx * step,
                    end=seg_start + (idx + 1) * step,
                )
                for idx, tok in enumerate(tokens)
            ]
        segments.append(
            ASRSegment(text=seg_text, start=seg_start, end=seg_end, words=words)
        )
    return segments, detected_lang


def _run_faster_whisper_fallback(
    wav_path: Path,
    model_source: str,
    device_config: DeviceConfig,
    language: Optional[str],
    threads: Optional[int] = None,
) -> tuple[list[ASRSegment], str]:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Neither whisperx nor faster-whisper is installed. "
            "Install dependencies via 'pip install -r requirements.txt'."
        ) from exc

    model = WhisperModel(
        model_source,
        device=device_config.asr_device,
        compute_type=device_config.asr_compute_type,
        cpu_threads=threads if (threads is not None and threads > 0) else 0,
        local_files_only=True,
    )
    try:
        batched = BatchedInferencePipeline(model=model)
        seg_iter, info = batched.transcribe(
            str(wav_path),
            batch_size=device_config.asr_batch_size,
            language=language,
            word_timestamps=True,
        )
        detected_lang = getattr(info, "language", None) or language or "en"
        segments: list[ASRSegment] = []
        for seg in seg_iter:
            raw_words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": getattr(w, "probability", 1.0),
                }
                for w in (seg.words or [])
            ]
            words = _interpolate_missing_word_timestamps(raw_words, seg.start, seg.end)
            segments.append(
                ASRSegment(
                    text=seg.text.strip(),
                    start=float(seg.start),
                    end=float(seg.end),
                    words=words,
                )
            )
    finally:
        del model
        release_memory()

    return segments, detected_lang
