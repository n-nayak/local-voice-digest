# VoiceDigest (Local Voice Digest — CLI + macOS SwiftUI App)

A 100% local, privacy-preserving application for transcribing long-form audio and video recordings (`.m4a`, `.mp3`, `.wav`, `.ogg`, `.flac`, `.mp4`, `.mov`, and more), capturing live calls from the macOS menu bar, identifying speakers, extracting acoustic/prosodic cues, and generating focused AI summaries — with zero outbound network access during inference.

Includes both a **Python CLI** and a **native macOS SwiftUI App** (`VoiceDigest.app` / `VoiceDigest.dmg`).

---

## Features

- **Strict Offline / Air-Gap Inference:** Sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1` before importing ML libraries, plus a Python socket-level outbound network blocker during transcription.
- **Broad Audio & Video Format Support:** Decodes `.m4a`, `.mp3`, `.wav`, `.ogg`/`.opus`, `.flac`, `.aac`, `.aiff`, `.caf`, `.wma`, and extracts audio tracks from `.mp4`, `.mov`, `.mkv`, and `.webm` via `ffmpeg`.
- **Menu Bar Live Call Capture (macOS 14.2+):** Menu bar popover with optional *Hide Dock Icon* mode and native dual-stream capture (Mic = `Me` + Core Audio Process Tap System Audio = `Other` / `1:N` — no BlackHole driver or Screen Recording permission required). Defaults to zero-overhead **Process on Stop** with optional **Live 60s Rolling** chunks and automatic temp cleanup.
- **Native macOS SwiftUI App:** Drag-and-drop audio/video files, choose model size, speaker count, language, configurable loudness normalization (`-24` to `-10 LUFS`), AI speaker repair, AI summary with custom 160-char focus + one-click presets, pick an output folder (defaults to `Desktop`), view the built-in `ⓘ` guide, and track everything with a single unified progress bar.
- **Apple Silicon GPU Acceleration:** Runs open-weight Whisper models natively on Apple Silicon GPUs via **`mlx-whisper`** (with `whisperx` / `faster-whisper` fallback on CUDA/CPU).
- **Zero-Token ONNX Diarization + Sentence Smoothing:** Uses **`sherpa-onnx`** (`pyannote-segmentation-3.0` + English `WeSpeaker VoxCeleb ResNet34-LM`, auto-downloaded from public GitHub Releases — no Hugging Face token required) with sentence-boundary majority smoothing, plus optional `pyannote.audio 3.1` support.
- **Local AI Speaker Repair & Summarization (`Qwen2.5-7B-Instruct-4bit`):** Optional offline LLM pass on Apple Silicon GPU to fix rapid Q&A/greeting speaker swaps (`--llm-repair`) and prepend a focused `=== AI SUMMARY ===` block (`--summary`, `--summary-focus`) using automatic map-reduce chunking so 1–3+ hour recordings never overflow the context window.
- **Acoustic-First Prosody (`--prosody`):** Measures speaking pace (WPM & syllables/sec), mid-clause silent blocks (`0.4s–1.5s`), prolongations, syllable repetition bursts, and vocal stress markers (`openSMILE` eGeMAPS + DSP + optional windowed 8s SpeechBrain SER).
- **Long-Form Resilience:** Unloads models sequentially between stages, prevents macOS idle sleep (`caffeinate`), and supports `--cache-dir` for fast re-runs.

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

### 3. One-Time Model Download (No HF Token Required)
```bash
python transcribe.py --download-models --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --llm-repair --model-dir ./models
```

### 4. Run Offline Transcription
```bash
python transcribe.py recording.m4a \
  --speakers 2 \
  --asr-backend mlx \
  --model-size turbo \
  --diarizer sherpa-onnx \
  --language en \
  --llm-repair \
  --summary --summary-focus "Key takeaways, decisions, and action items" \
  --prosody --no-ser \
  --cache-dir ./.cache \
  --output transcript.txt
```

---

## macOS SwiftUI App & `.dmg` Packaging

### Run the SwiftUI App Directly (Development)
```bash
swift run --package-path macos/LocalTranscribeApp
```

### Build `VoiceDigest.app` & `VoiceDigest.dmg`
```bash
chmod +x scripts/build_dmg.sh

# Lightweight build (uses your local .venv during development)
./scripts/build_dmg.sh

# Self-contained build (bundles standalone MLX/Sherpa-ONNX backend + ffmpeg inside the .app)
uv pip install pyinstaller
./scripts/build_dmg.sh --self-contained
```

The drag-and-drop disk image is written to `dist/VoiceDigest.dmg`. On first launch on a new Mac, the app automatically downloads public Whisper MLX + Sherpa-ONNX (+ optional Qwen 7B) weights into `~/Library/Application Support/VoiceDigest/models` (no token or Terminal required), and runs offline thereafter.

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
