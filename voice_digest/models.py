"""Model management, hardware device resolution, memory cleanup, and offline caching."""

from __future__ import annotations

import gc
import logging
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from local_transcribe.compat import apply_torch_compat
from local_transcribe.offline import configure_model_cache_env

logger = logging.getLogger(__name__)

SUPPORTED_WHISPER_MODELS = ("base", "small", "medium", "large-v3", "turbo")
DEFAULT_WHISPER_MODEL = "turbo"
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SER_MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

MLX_WHISPER_MODEL_MAP = {
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

SHERPA_SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SHERPA_EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)
SHERPA_EMB_FILENAME = "embedding_en_voxceleb.onnx"

LLM_REPAIR_REPO_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
LLM_REPAIR_DIRNAME = "qwen2.5-7b-4bit"


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved compute settings for CTranslate2 ASR and PyTorch stages."""

    asr_device: str  # "cuda" or "cpu" (CTranslate2 does not support MPS)
    asr_compute_type: str  # "float16" or "int8"
    asr_batch_size: int  # 16 on CUDA, 4 on CPU
    torch_device: str  # "cuda", "mps", or "cpu"


def resolve_device_config(
    requested_device: str = "auto",
    batch_size: Optional[int] = None,
) -> DeviceConfig:
    """Resolve optimal devices for ASR (CTranslate2) and PyTorch (alignment/diarization/SER)."""
    req = requested_device.lower()
    if req not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported device '{requested_device}'. Choose auto, cuda, mps, or cpu.")

    cuda_available = False
    mps_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    if req == "auto":
        if cuda_available:
            torch_device = "cuda"
        elif mps_available:
            torch_device = "mps"
        else:
            torch_device = "cpu"
    else:
        torch_device = req

    if torch_device == "cuda":
        asr_device = "cuda"
        asr_compute_type = "float16"
        default_batch = 16
    else:
        # Apple Silicon MPS and CPU both run faster-whisper via CTranslate2 CPU int8
        asr_device = "cpu"
        asr_compute_type = "int8"
        default_batch = 4

    resolved_batch = batch_size if batch_size is not None else default_batch
    return DeviceConfig(
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_batch_size=resolved_batch,
        torch_device=torch_device,
    )


def release_memory() -> None:
    """Force garbage collection and release GPU/MPS/MLX allocator caches between pipeline stages."""
    gc.collect()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except Exception:
            pass


def whisper_model_path(model_dir: Path, model_size: str) -> Path:
    return model_dir / "whisper" / model_size


def mlx_whisper_model_path(model_dir: Path, model_size: str) -> Path:
    return model_dir / "whisper-mlx" / model_size


def is_mlx_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    import importlib.util

    return importlib.util.find_spec("mlx_whisper") is not None


def pyannote_model_path(model_dir: Path) -> Path:
    return model_dir / "pyannote"


def ser_model_path(model_dir: Path) -> Path:
    return model_dir / "ser" / "emotion-recognition-wav2vec2-IEMOCAP"


def sherpa_model_path(model_dir: Path) -> Path:
    return model_dir / "sherpa-onnx"


def llm_repair_model_path(model_dir: Path) -> Path:
    return model_dir / "llm" / LLM_REPAIR_DIRNAME


def download_all_models(
    model_dir: Path | str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str = "en",
    diarizer: str = "pyannote",
    include_prosody_ser: bool = False,
    hf_token: Optional[str] = None,
    asr_backend: str = "auto",
    include_llm_repair: bool = False,
) -> None:
    """One-time online setup: download and cache all required weights into model_dir."""
    apply_torch_compat()
    root = configure_model_cache_env(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Explicitly ensure online mode during download
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    use_mlx = asr_backend == "mlx" or (asr_backend == "auto" and is_mlx_available())
    if use_mlx:
        print(f"[1/5] Downloading mlx-whisper GPU model '{model_size}' into {root}...")
        _download_mlx_whisper_model(root, model_size)
        print("[2/5] Skipping wav2vec2 alignment download (mlx-whisper provides native word timestamps).")
    else:
        print(f"[1/5] Downloading faster-whisper model '{model_size}' into {root}...")
        _download_whisper_model(root, model_size)
        print(f"[2/5] Downloading wav2vec2 forced-alignment model for language '{language}'...")
        _download_alignment_model(language)

    print(f"[3/5] Downloading speaker diarization model ({diarizer})...")
    if diarizer == "pyannote":
        _download_pyannote_model(root, token)
    elif diarizer == "sherpa-onnx":
        _download_sherpa_onnx_models(root)
    else:
        raise ValueError(f"Unsupported diarizer: {diarizer}")

    if include_prosody_ser:
        print("[4/5] Downloading local SpeechBrain SER model for --prosody...")
        _download_ser_model(root)
    else:
        print("[4/5] Skipping optional SER model (pass --prosody with --download-models to include it).")

    if include_llm_repair:
        print(f"[5/5] Downloading local MLX LLM ({LLM_REPAIR_REPO_ID}) for --llm-repair...")
        _download_llm_repair_model(root)
    else:
        print("[5/5] Skipping optional LLM speaker-repair model (pass --llm-repair to include it).")

    print(f"\nAll models cached in: {root}")
    print("You can now run offline transcription with zero network access.")


def _download_llm_repair_model(model_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    target = llm_repair_model_path(model_dir)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=LLM_REPAIR_REPO_ID, local_dir=str(target))
    return target


def _download_mlx_whisper_model(model_dir: Path, model_size: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    repo_id = MLX_WHISPER_MODEL_MAP[model_size]
    target = mlx_whisper_model_path(model_dir, model_size)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    return target


def _download_whisper_model(model_dir: Path, model_size: str) -> Path:
    try:
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    target = whisper_model_path(model_dir, model_size)
    target.mkdir(parents=True, exist_ok=True)
    download_model(model_size, output_dir=str(target), local_files_only=False)
    return target


def _download_alignment_model(language: str) -> None:
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    model, _metadata = whisperx.load_align_model(language_code=language, device="cpu")
    del model
    release_memory()


def _download_pyannote_model(model_dir: Path, token: Optional[str]) -> None:
    apply_torch_compat()
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    if not token:
        raise RuntimeError(
            f"Downloading gated model '{PYANNOTE_DIARIZATION_MODEL}' requires a Hugging Face token. "
            "Accept the model terms on Hugging Face, then pass --hf-token <TOKEN> or set HF_TOKEN, "
            "or use '--diarizer sherpa-onnx' for a zero-token alternative."
        )

    cache_dir = pyannote_model_path(model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline.from_pretrained(
        PYANNOTE_DIARIZATION_MODEL,
        use_auth_token=token,
        cache_dir=str(cache_dir),
    )
    del pipeline
    release_memory()


def _download_sherpa_onnx_models(model_dir: Path) -> Path:
    """Download official open-weight Sherpa-ONNX segmentation + English VoxCeleb embedding models."""
    target = sherpa_model_path(model_dir)
    target.mkdir(parents=True, exist_ok=True)
    seg_dest = target / "segmentation.onnx"
    emb_dest = target / SHERPA_EMB_FILENAME

    if not seg_dest.exists():
        print("  - Downloading pyannote-segmentation-3.0 ONNX (~6 MB)...")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "seg.tar.bz2"
            urllib.request.urlretrieve(SHERPA_SEG_URL, str(archive))
            with tarfile.open(archive, "r:bz2") as tf:
                onnx_member = next(
                    (m for m in tf.getmembers() if m.name.endswith(".onnx")),
                    None,
                )
                if onnx_member is None:
                    raise RuntimeError("Could not find .onnx file inside Sherpa segmentation archive.")
                src = tf.extractfile(onnx_member)
                if src is None:
                    raise RuntimeError("Failed to extract segmentation .onnx from archive.")
                with open(seg_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not emb_dest.exists():
        print("  - Downloading English WeSpeaker VoxCeleb ResNet34-LM ONNX (~26 MB)...")
        urllib.request.urlretrieve(SHERPA_EMB_URL, str(emb_dest))

    return target


def _download_ser_model(model_dir: Path) -> None:
    apply_torch_compat()
    try:
        from speechbrain.inference.interfaces import foreign_class
    except ImportError as exc:
        raise RuntimeError(
            "speechbrain is not installed. Install prosody extras: 'pip install \".[prosody]\"'."
        ) from exc

    savedir = ser_model_path(model_dir)
    savedir.mkdir(parents=True, exist_ok=True)
    classifier = foreign_class(
        source=SER_MODEL_ID,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=str(savedir),
        run_opts={"device": "cpu"},
    )
    del classifier
    release_memory()
