# Technical Specification: Local Audio Transcription, Diarization, Acoustic Analysis & AI Summarization

## 1. Overview & Objectives
This specification defines a 100% local, privacy-preserving application (Python CLI + native macOS SwiftUI app) for transcribing long-form audio and video recordings (`.m4a`, `.mp3`, `.wav`, `.ogg`/`.opus`, `.flac`, `.mp4`, `.mov`, `.mkv`, `.webm`, etc.), identifying individual speakers, extracting paralinguistic (acoustic/prosodic) cues, and generating focused on-device AI summaries.

### Core Design Principles
* **Zero Cloud / Strict Air-Gap Capability:** Once open-weight models are cached locally, all decoding, inference, speaker repair, summarization, and acoustic extraction run strictly on-device with zero outbound network requests or telemetry (enforced via offline environment variables and a Python socket-level network blocker).
* **Long-Form Audio Resilience:** Engineered to process $\sim 1\text{--}3\text{-hour}$ recordings reliably without Out-Of-Memory (OOM) failures via sequential model unloading, 8-second windowed SER, and map-reduce LLM summarization.
* **FOSS-Only Stack:** Built exclusively on Free and Open-Source Software libraries and open-weight models (zero Hugging Face token required for the Apple Silicon MLX + Sherpa-ONNX stack).
* **Acoustic-First Metadata:** Paralinguistic analysis focuses exclusively on temporal and spectral audio properties (speaking pace, acoustic disfluencies/stammers, vocal stress markers) that would otherwise be lost in plain text, without flagging lexical filler words ("um", "uh").

---

## 2. System Architecture & Pipeline

```text
+-----------------------+     +-----------------------------------+     +-------------------------+
|  Input: Audio / Video | --> |  FFmpeg Preprocessor              | --> |  16kHz Mono PCM (.wav)  |
|  (.m4a, .mp3, .wav,   |     |  (Strip video + configurable      |     |  (~115 MB/hr temp file, |
|   .mp4, .mov, etc.)   |     |   EBU R128 loudnorm, -16 LUFS)    |     |   auto-cleaned on exit) |
+-----------------------+     +-----------------------------------+     +-------------------------+
                                                                                     |
                    +----------------------------------------------------------------+
                    |
                    v
+---------------------------------------+     +-----------------------------------+
|  1. ASR & Word Timestamps             |     |  2. Speaker Diarization           |
|  - mlx-whisper (Apple Silicon GPU)    |     |  - sherpa-onnx (pyannote seg +    |
|  - or WhisperX (faster-whisper +      |     |    English WeSpeaker VoxCeleb)    |
|    wav2vec2 forced alignment)         |     |  - or pyannote.audio 3.1+         |
+---------------------------------------+     +-----------------------------------+
                    |                                                |
                    +-----------------------+------------------------+
                                            |
                                            v
                    +-----------------------------------------------+
                    |  3. Sentence-Boundary Smoothing & Merging     |
                    |  - Majority-vote speaker per sentence         |
                    |  - Relabel speaker 1..N by first appearance   |
                    |  - Optional: Local Qwen 7B Speaker Repair     |
                    +-----------------------------------------------+
                                            |
                    +-----------------------+-----------------------+
                    | (Optional: --prosody / --summary)             |
                    v                                               |
+---------------------------------------+                           |
|  4. Prosody & Local AI Summary        |                           |
|  - Pace (WPM / Syllables/s)           |                           |
|  - Disfluencies (Gaps, Prolongations) |                           |
|  - Stress (openSMILE + optional SER)  |                           |
|  - Map-Reduce Summary (Qwen 7B)       |                           |
+---------------------------------------+                           |
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            v
                    +-----------------------------------------------+
                    |  5. Output Formatter (.txt)                   |
                    |  - AI Summary + Acoustic Summary + Transcript |
                    +-----------------------------------------------+
```

---

## 3. FOSS Tech Stack

| Component | Tool / Model | License | Rationale |
| :--- | :--- | :--- | :--- |
| **Runtime Language** | Python $3.11+$ & SwiftUI (macOS 13+) | PSF / MIT | Python ML backend with native macOS SwiftUI desktop front-end (`LocalTranscribe.app` / `.dmg`). |
| **Audio Decoding** | `ffmpeg` (`loudnorm` EBU R128) | LGPL / GPL | Decodes all major audio & video containers (stripping video via `-vn`), normalizes quiet recordings (configurable `-30` to `-5 LUFS`, default `-16`), and outputs $16\text{ kHz}$ mono WAV. |
| **ASR & Alignment** | **MLX-Whisper** (Apple Silicon GPU) or **WhisperX** (`faster-whisper` + `wav2vec2`) | MIT / BSD-2-Clause | Native Metal GPU inference via MLX on Apple Silicon (`word_timestamps=True`), or batched CTranslate2 + `wav2vec2` forced alignment on CUDA/CPU. |
| **Speaker Diarization** | **Sherpa-ONNX** (`pyannote-segmentation-3.0` + English `WeSpeaker VoxCeleb ResNet34-LM`) or **pyannote.audio 3.1** | Apache-2.0 / MIT | Zero-token, lightweight (~32 MB) English ONNX diarization with live chunk progress; `pyannote.audio` supported as an optional backend (routed to CPU on Apple Silicon to avoid MPS clustering collapse). |
| **Local LLM Repair & Summary** | **mlx-lm** (`Qwen2.5-7B-Instruct-4bit`) | Apache-2.0 / MIT | Optional on-device semantic speaker-attribution repair (sliding chunks with `[pause Xs]` cues) and map-reduce transcript summarization with custom focus prompts. |
| **Acoustic / Stress DSP** | **openSMILE** (`opensmile` Python) + DSP | BSD-3-Clause | Extracts eGeMAPS vocal features (pitch, pitch jitter, spectral flux) locally on CPU. |
| **Optional Neural SER** | **SpeechBrain** (`emotion-recognition-wav2vec2`) | Apache-2.0 | Local Speech Emotion Recognition evaluated in bounded 8-second windows (max 4 per turn) to prevent GPU OOM; skippable via `--no-ser`. |

### Hardware Acceleration Support
* **Apple Silicon (M1–M5):** `mlx-whisper` and `mlx-lm` run natively on the Apple GPU via Metal/MLX; PyTorch MPS is used for `wav2vec2` alignment and optional SER; diarization runs on CPU (`sherpa-onnx` or `pyannote`).
* **NVIDIA GPU:** CUDA $12.x$ with `float16` compute type (`batch_size=16`).
* **CPU-Only Fallback:** `int8` quantization via CTranslate2 (`faster-whisper`, `batch_size=4`).

---

## 4. Functional Requirements

### 4.1 Input Handling
* Accept audio and video files (`.m4a`, `.mp3`, `.wav`, `.ogg`/`.opus`, `.flac`, `.aac`, `.aiff`, `.caf`, `.wma`, `.mp4`, `.mov`, `.mkv`, `.webm`) via CLI argument or macOS SwiftUI drag-and-drop / file picker.
* Validate file existence and non-zero size before processing.
* Transcode to a temporary $16\text{ kHz}$, $16\text{-bit}$, single-channel PCM `.wav` file using `ffmpeg` (auto-cleaned on exit), with configurable EBU R128 loudness normalization (`--loudness-target`, `--no-loudnorm`).

### 4.2 Strict Offline Execution
* **Setup Phase (`--download-models`):** Explicit one-time download (or automatic first-run download in the macOS app into `~/Library/Application Support/VoiceDigest/models/`) caching all public weights locally.
* **Inference Phase (Default):** Enforce offline environment variables before importing ML libraries (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`) plus a socket-level guard that raises `OfflineNetworkError` on any outbound network or DNS attempt.

### 4.3 Speaker Identification, Smoothing & Semantic Repair
* Accept `--speakers N` (CLI) or interactive prompt (`Number of speakers [2]: `, defaulting to `2`).
* Smooth word-level speaker assignments to sentence boundaries via duration-weighted majority vote so sentences are never torn mid-clause across speakers.
* Assign consistent speaker labels (`speaker 1`, `speaker 2`, ..., `speaker N`) ordered by first appearance, and merge consecutive turns separated by $\le 2.0\text{s}$ (`--pause-threshold`).
* Optional `--llm-repair`: runs local `Qwen2.5-7B-Instruct-4bit` over sliding 24-sentence chunks (with 5-sentence overlap context and `[pause Xs]` cues) to fix conversational speaker swaps while preserving every transcribed word and timestamp verbatim.

### 4.4 Output Format & Local AI Summary
* **Base format:** Clean `speaker X: <text>` lines saved to `<input_stem>_transcript.txt` (or `--output path.txt`).
* **Optional AI Summary (`--summary`, `--summary-focus`):** Prepends a `=== AI SUMMARY ===` block at the top of the `.txt` file using `Qwen2.5-7B-Instruct-4bit`. Automatically uses single-pass summarization for short transcripts ($\le 12,000$ chars) and map-reduce chunking for long 1–3+ hour recordings so it never overflows the context window.

---

## 5. Paralinguistic & Acoustic Metadata (`--prosody`)

Strictly acoustic and temporal biomarkers extracted from the audio waveform and word timestamps (never flagging lexical filler words like "um", "uh", or "like"):

1. **Speaking Pace:** Articulation rate in $\text{WPM}$ and syllables/sec computed from turn span minus long silent pauses ($\ge 0.4\text{s}$). Turns exceeding $175\text{ WPM}$ or $> +1.5\sigma$ above that speaker's session baseline are flagged `fast`.
2. **Acoustic Disfluencies:**
   * **Mid-clause Blocks:** Abnormal silent gaps ($0.4\text{s}\text{--}1.5\text{s}$) mid-clause between consecutive words.
   * **Prolongations:** Abnormal word duration spikes ($\ge 0.85\text{s}$ and $> 3.5\times$ expected duration).
   * **Syllable Repetitions:** Rapid repetitive energy bursts ($80\text{--}250\text{ms}$ intervals) in the amplitude envelope.
3. **Vocal Stress Markers:** Speaker-normalized `openSMILE` eGeMAPS features ($F_0$ mean, pitch jitter, spectral flux, rapid breath intakes) plus optional windowed 8-second SpeechBrain SER (skippable via `--no-ser`).

---

## 6. Interfaces: CLI & macOS SwiftUI App

### CLI Examples
```bash
# 1. One-time online setup (Apple Silicon MLX GPU + Sherpa-ONNX + Qwen 7B, no HF token needed)
python transcribe.py --download-models --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --llm-repair --model-dir ./models

# 2. Offline execution with speaker repair, focused AI summary, prosody & stage caching
python transcribe.py recording.m4a \
  --speakers 2 --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --language en \
  --llm-repair --summary --summary-focus "Key takeaways and action items" \
  --prosody --no-ser --cache-dir ./.cache --output transcript.txt
```

### Key CLI Arguments
* `input_file`: Path to audio/video file (required unless `--download-models` is set).
* `--download-models`, `--model-dir`: Cache weights locally (default: `./models`).
* `-s, --speakers`: Expected speaker count (prompts interactively defaulting to `2` if omitted).
* `-o, --output`: Output `.txt` path (defaults to `<input_stem>_transcript.txt`).
* `-m, --model-size`: `base`, `small`, `medium`, `large-v3`, `turbo` (default: `turbo`).
* `--asr-backend`: `auto`, `mlx`, `whisperx` (default: `auto`).
* `--align`: `auto`, `wav2vec2`, `none` (default: `auto`).
* `--diarizer`: `pyannote`, `sherpa-onnx` (default: `pyannote`).
* `--language`: Optional ISO-639-1 code (e.g. `en`); auto-detected if omitted.
* `--loudness-target` / `--no-loudnorm`: Target EBU R128 loudness in LUFS (default: `-16.0`) or disable normalization.
* `--llm-repair`: Enable local Qwen 7B speaker-attribution repair.
* `--summary` / `--summary-focus`: Prepend local Qwen 7B summary with optional focus (up to 160 chars).
* `--prosody` / `--no-ser` / `--metadata-mode`: Enable acoustic metadata (`both`, `summary-only`, `inline-only`).
* `--device`, `--threads`, `--pause-threshold`, `--cache-dir`, `--json-progress`.

### macOS SwiftUI App (`VoiceDigest.app` / `macos/LocalTranscribeApp/`)
* Native macOS 14.2+ SwiftUI window with drag-and-drop, file picker, model size & language pickers, speaker stepper, configurable loudness slider (`-24` to `-10 LUFS`), **AI Speaker Repair**, **Publish AI Summary** (160-char focus box + 4 one-click presets), **Prosody** & **Fast DSP-only** toggles, output folder picker (defaults to `Desktop`), built-in `ⓘ` Info sheet, custom macOS Dock icon, and a single unified progress bar driven by `--json-progress`.
* **Menu Bar Popover Live Call Capture:** Menu bar popover with optional *Hide Dock Icon* mode and native dual-stream capture (`AVAudioEngine` mic = `Me` + Core Audio Process Tap system audio = `Other` / `1:N`) with unique per-call temp sessions under `~/Library/Caches/VoiceDigest/sessions/` and automatic cleanup.
* Packaged into `dist/VoiceDigest.app` and drag-and-drop `dist/VoiceDigest.dmg` via `scripts/build_dmg.sh` (supports `--self-contained` to bundle standalone PyInstaller backend + `ffmpeg`).

---

## 7. Performance, Memory & Verification
* **Memory Footprint:** Sequential stage unloading (`del`, `gc.collect()`, `mx.clear_cache()`, `torch.cuda.empty_cache()`, `torch.mps.empty_cache()`) keeps peak RAM around **~2 GB** (ASR + diarization + DSP prosody) or **~5 GB** when Qwen 7B speaker repair/summarization is active.
* **Reliability:** Live stage progress + wall-clock timers, automatic macOS `caffeinate -i` sleep prevention, and `--cache-dir` intermediate JSON caching.
* **Automated Verification (`python -m unittest discover -s tests -v`):** 23 unit/integration tests covering offline env & socket blocking, audio validation/slicing/cleanup, first-appearance relabeling, sentence-boundary smoothing, LLM JSON repair & map-reduce chunking, acoustic prosody (WPM, mid-clause blocks without lexical filler false positives, prolongations, repetitions), output formatting, CLI default prompts, and stage caching.
