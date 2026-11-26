# Local Audio Transcribe (CLI + macOS SwiftUI App)

A 100% local, privacy-preserving application for transcribing long-form Apple MPEG-4 Audio (`.m4a`) recordings, identifying speakers, and extracting acoustic/prosodic cues with zero outbound network access during inference.

Includes both a **Python CLI** and a **native macOS SwiftUI App** (`LocalTranscribe.app` / `.dmg`).

---

## Features

- **Strict Offline / Air-Gap Inference:** Sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1` before importing ML libraries, plus a Python socket-level outbound network blocker during transcription.
- **Native macOS SwiftUI App:** Drag-and-drop `.m4a` files, choose model size, speaker count, language, loudness normalization, and prosody options, pick an output folder (defaults to `Desktop`), and track everything with a single unified progress bar.
- **Apple Silicon GPU Acceleration:** Runs Whisper natively on Apple Silicon GPUs via **`mlx-whisper`** (with `whisperx` / `faster-whisper` fallback on CUDA/CPU).
- **Zero-Token ONNX Diarization:** Uses **`sherpa-onnx`** (`pyannote-segmentation-3.0` + `3D-Speaker ERes2Net`, auto-downloaded from public GitHub Releases — no Hugging Face token required), with optional `pyannote.audio 3.1` support.
- **Long-Form Audio Resilience:** Loudness-normalizes quiet calls (`loudnorm` `-16 LUFS`, toggleable via `--no-loudnorm`), unloads models sequentially between stages (~2 GB peak RAM), prevents macOS idle sleep (`caffeinate`), and supports `--cache-dir` for instant re-runs.
- **Acoustic-First Prosody (`--prosody`):** Measures speaking pace (WPM & syllables/sec), mid-clause silent blocks (`0.4s–1.5s`), phoneme prolongations, syllable repetition bursts, and vocal stress (`openSMILE` eGeMAPS + DSP + optional windowed 8s SpeechBrain SER).

---

## Quick Start (CLI)

### 1. Prerequisites
- macOS (Apple Silicon recommended) or Linux, Python 3.11+
- `ffmpeg` (`brew install ffmpeg` on macOS)

### 2. Install Dependencies
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[prosody,mlx]"
```

### 3. One-Time Model Download (No HF Token Required for MLX + Sherpa-ONNX)
```bash
python transcribe.py --download-models --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --model-dir ./models
```

### 4. Run Offline Transcription
```bash
python transcribe.py call.m4a --speakers 2 --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --language en --prosody --no-ser --cache-dir ./.cache --output transcript.txt
```

---

## macOS SwiftUI App & `.dmg` Packaging

### Run the SwiftUI App Directly (Development)
```bash
swift run --package-path macos/LocalTranscribeApp
```

### Build `LocalTranscribe.app` & `LocalTranscribe.dmg`
```bash
chmod +x scripts/build_dmg.sh

# Lightweight build (uses your local .venv during development)
./scripts/build_dmg.sh

# Self-contained build (bundles standalone MLX/Sherpa-ONNX backend + ffmpeg inside the .app)
uv pip install pyinstaller
./scripts/build_dmg.sh --self-contained
```

The resulting drag-and-drop disk image is written to `dist/LocalTranscribe.dmg`. On first launch on a new Mac, the app automatically downloads the selected public Whisper MLX + Sherpa-ONNX models into `~/Library/Application Support/LocalTranscribe/models` (no token or Terminal required), and runs offline thereafter.

---

## Observed Benchmark (Apple M5 MacBook Air, 24 GB RAM)

On a **1h 03m `.m4a` (~61 MB)** with `mlx` `turbo` + `sherpa-onnx` + `--prosody`:
- **FFmpeg `loudnorm`:** ~1m 31s
- **MLX GPU ASR (`turbo`):** ~3m 19s
- **Sherpa-ONNX Diarization:** ~4m 15s
- **Prosody + chunked SER:** ~1m 51s *(~10s with `--no-ser`)*
- **Total:** **~10m 56s** (~9 min with `--no-ser`), peak RAM ~2 GB

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```
