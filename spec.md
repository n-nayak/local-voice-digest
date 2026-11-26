# Technical Specification: Local Audio Transcription, Diarization & Acoustic Analysis

## 1. Overview & Objectives
This specification defines a 100% local, privacy-preserving command-line application for transcribing long-form Apple MPEG-4 Audio (`.m4a`) recordings, identifying individual speakers, and extracting paralinguistic (acoustic/prosodic) cues. 

### Core Design Principles
* **Zero Cloud / Strict Air-Gap Capability:** Once model weights are cached locally, all inference, audio preprocessing, and metadata extraction must run strictly on-device with zero outbound network requests or telemetry.
* **Long-Form Audio Resilience:** Engineered to process $\sim 1\text{-hour}$ recordings ($100\text{--}200\text{ MB}$) reliably without Out-Of-Memory (OOM) failures.
* **FOSS-Only Stack:** Built exclusively on Free and Open-Source Software libraries and open-weight models.
* **Acoustic-First Metadata:** Paralinguistic analysis focuses exclusively on temporal and spectral audio properties (speaking pace, acoustic disfluencies/stammers, vocal stress/anxiety markers) that would otherwise be lost in plain text, leaving lexical filler-word detection (e.g., "um", "uh") to downstream text processing.

---

## 2. System Architecture & Pipeline

```text
+-------------------+     +-----------------------+     +-------------------------+
|  Input: .m4a      | --> |  FFmpeg Preprocessor  | --> |  16kHz Mono PCM (.wav)  |
|  (~1 hr, 150 MB)  |     |  (Normalize & Strip)  |     |  (~115 MB temp file)    |
+-------------------+     +-----------------------+     +-------------------------+
                                                                     |
                    +------------------------------------------------+
                    |
                    v
+---------------------------------------+     +-----------------------------------+
|  1. Batched ASR & Forced Alignment    |     |  2. Speaker Diarization           |
|  - faster-whisper (large-v3 / turbo)  |     |  - pyannote.audio 3.1+            |
|  - wav2vec2 phoneme/word alignment    |     |  - Target speakers: N (default 2) |
+---------------------------------------+     +-----------------------------------+
                    |                                                |
                    +-----------------------+------------------------+
                                            |
                                            v
                    +-----------------------------------------------+
                    |  3. Speaker-Word Alignment & Turn Merging     |
                    |  - Map word-level timestamps to speaker IDs   |
                    +-----------------------------------------------+
                                            |
                    +-----------------------+-----------------------+
                    | (Optional: --prosody flag)                    |
                    v                                               |
+---------------------------------------+                           |
|  4. Local Acoustic & Prosody Engine   |                           |
|  - Speaking Rate (WPM / Syllables/s)  |                           |
|  - Stammer / Hesitation (Gaps & F0)   |                           |
|  - Stress / Anxiety (openSMILE + SER) |                           |
+---------------------------------------+                           |
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            v
                    +-----------------------------------------------+
                    |  5. Output Formatter (.txt)                   |
                    |  - Overall Summary + Turn-by-Turn Transcript  |
                    +-----------------------------------------------+
```

---

## 3. Recommended FOSS Tech Stack

| Component | Recommended Tool / Model | License | Rationale |
| :--- | :--- | :--- | :--- |
| **Runtime Language** | Python $3.11+$ | PSF | Best native support for Whisper, PyTorch, and audio DSP libraries. |
| **Audio Decoding** | `ffmpeg` (`loudnorm` EBU R128) | LGPL / GPL | Handles Apple AAC/ALAC `.m4a` containers reliably; normalizes quiet recordings (`-16 LUFS`) and converts to $16\text{ kHz}$ mono WAV. |
| **ASR & Alignment** | **MLX-Whisper** (Apple Silicon GPU) or **WhisperX** (`faster-whisper` + `wav2vec2`) | MIT / BSD-2-Clause | Native Metal GPU inference via MLX on Apple Silicon (`word_timestamps=True`), or batched CTranslate2 + `wav2vec2` forced alignment on CUDA/CPU. |
| **Speaker Diarization** | **pyannote.audio** (`speaker-diarization-3.1`) | MIT | State-of-the-art FOSS diarization; natively accepts a fixed speaker count (`num_speakers=2`). Runs on CPU on Apple Silicon to avoid MPS clustering collapse. |
| **Alternative Diarization** | **Sherpa-ONNX** (`pyannote-segmentation-3.0` + 3D-Speaker ERes2Net) | Apache-2.0 | Zero-token, lightweight (~44 MB) ONNX alternative with live chunk progress and automatic `--download-models` support. |
| **Acoustic / Stress DSP** | **openSMILE** (`opensmile` Python) + `librosa` | BSD-3-Clause | Extracts eGeMAPS vocal features (pitch variance, jitter, shimmer, loudness) locally on CPU. |
| **Local Emotion/Anxiety** | **SpeechBrain** (`emotion-recognition-wav2vec2`) | Apache-2.0 / MIT | Local Speech Emotion Recognition (SER) evaluated in bounded 8-second windows (max 4 per turn) to detect high-arousal/anxious states without GPU OOM. |

### Hardware Acceleration Support
* **Apple Silicon (M1/M2/M3/M4/M5):** `mlx-whisper` runs ASR natively on the Apple GPU via Metal/MLX; PyTorch MPS is used for `wav2vec2` alignment and SER; diarization runs on CPU (`sherpa-onnx` or `pyannote`) to avoid PyTorch MPS embedding instability.
* **NVIDIA GPU:** CUDA $12.x$ with `float16` compute type (`batch_size=16`).
* **CPU-Only Fallback:** `int8` quantization via CTranslate2 (`faster-whisper`, `batch_size=4`).

---

## 4. Functional Requirements (Must-Haves)

### 4.1 Input Handling
* Accept `.m4a` file paths via CLI argument.
* Validate file existence and non-zero size before processing.
* Transcode `.m4a` to a temporary $16\text{ kHz}$, $16\text{-bit}$, single-channel PCM `.wav` file using `ffmpeg` (automatically cleaned up on exit).

### 4.2 Strict Offline Execution
* **Setup Phase (`--download-models`):** Explicit one-time command to download and cache all required Whisper, alignment, diarization, and acoustic models into a local `./models/` directory.
* **Inference Phase (Default):** Enforce offline environment variables at runtime before importing ML libraries:
  * `HF_HUB_OFFLINE=1`
  * `TRANSFORMERS_OFFLINE=1`
  * `HF_HUB_DISABLE_TELEMETRY=1`
* Any attempt to reach an external network socket during transcription must fail immediately rather than silently transmitting data.

### 4.3 Speaker Identification & Turn Merging
* Prompt the user interactively (or accept `--speakers N` via CLI), **defaulting to `2` speakers**.
* Assign consistent speaker labels (`speaker 1`, `speaker 2`, ..., `speaker N`) ordered by first appearance in the audio.
* Merge consecutive words/segments spoken by the same speaker into continuous turns unless separated by a pause exceeding a configurable threshold (e.g., $> 2.0\text{s}$).

### 4.4 Base Output Format
Generate a `.txt` file alongside the input file (or at `--output path.txt`) formatted cleanly as:

```text
speaker 1: Good morning, thanks for making time to go over the project timeline today.
speaker 2: Happy to be here. I have a few concerns about the deployment window we discussed last week.
speaker 1: Let's dive right into those blockers first.
```

---

## 5. Paralinguistic & Acoustic Metadata (Nice-to-Have)

Because downstream text tools can detect lexical fillers ("um", "like", "you know"), this module focuses strictly on **acoustic and temporal biomarkers** extracted from the raw audio waveform and word-level timestamps.

### 5.1 Target Acoustic Signals
1. **Speaking Pace ("Talk Too Fast"):**
   * **Metric:** Articulation rate in Words Per Minute ($\text{WPM}$) and syllables per second per turn, excluding long inter-turn silences.
   * **Thresholds:** Baseline conversational English is $\approx 120\text{--}150\text{ WPM}$. Turns exceeding $175\text{ WPM}$ (or $> +1.5\sigma$ above that specific speaker's session baseline) are flagged as `fast`.
2. **Stammering / Acoustic Disfluencies:**
   * **Intra-word / Inter-word Blocks:** Abnormal silent gaps ($0.4\text{s}\text{--}1.5\text{s}$) mid-clause detected via `wav2vec2` forced-alignment timestamp gaps.
   * **Phoneme Prolongations & Repetitions:** Abnormal phoneme duration spikes ($> 3\times$ expected duration) or rapid repetitive energy bursts (syllable repetition) detected in the mel-spectrogram envelope.
3. **Vocal Anxiety / Stress Markers:**
   * **Prosodic Features (`openSMILE` eGeMAPS):** Elevated fundamental frequency ($F_0$) mean and pitch jitter, reduced spectral flux, and rapid breathing intakes between phrases.
   * **Local SER Classifier:** Confidence score for `anxious / tense / high-arousal` states evaluated on audio segments $\ge 1.5\text{s}$.

### 5.2 Metadata Output Schema (Opt-In via `--prosody`)
When enabled, metadata is written to the same `.txt` output in a structured, human- and parser-friendly format:

```text
=== SESSION ACOUSTIC SUMMARY ===
Duration: 58m 14s | Speakers: 2
speaker 1: Avg Pace: 138 WPM (Normal) | Acoustic Disfluencies: 2 | Elevated Stress Turns: 5%
speaker 2: Avg Pace: 184 WPM (Fast)   | Acoustic Disfluencies: 11 | Elevated Stress Turns: 34%
================================

speaker 1 [pace: 135 WPM | state: calm]: Good morning, thanks for making time to go over the project timeline today.
speaker 2 [pace: 192 WPM (fast) | disfluency: 2 blocks | state: tense]: Happy to be here. I— I have a few concerns about the deployment window we discussed last week.
speaker 1 [pace: 141 WPM | state: calm]: Let's dive right into those blockers first.
```

---

## 6. CLI Interface & Configuration

```bash
# 1. One-time online setup to cache weights locally (Apple Silicon MLX GPU + Sherpa-ONNX)
python transcribe.py --download-models --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --model-dir ./models

# 2. Standard offline execution (prompts for speaker count, defaults to 2)
python transcribe.py interview.m4a

# 3. Non-interactive Apple Silicon GPU execution with prosody & stage caching
python transcribe.py interview.m4a --speakers 2 --asr-backend mlx --model-size turbo --diarizer sherpa-onnx --language en --prosody --cache-dir ./.cache --output transcript.txt
```

### CLI Arguments
* `input_file`: Path to the `.m4a` audio file (required unless `--download-models` is set).
* `--download-models`: One-time online setup to cache weights into `--model-dir` (default: `./models`).
* `-s, --speakers`: Integer number of expected speakers. If omitted, prompts via stdin (`Number of speakers [2]: `).
* `-o, --output`: Output `.txt` file path (defaults to `<input_filename>_transcript.txt`).
* `-m, --model-size`: Whisper model size (`base`, `small`, `medium`, `large-v3`, `turbo`; default: `turbo`).
* `--asr-backend`: ASR backend (`auto`, `mlx`, `whisperx`; default: `auto`).
* `--align`: Word alignment mode (`auto`, `wav2vec2`, `none`; default: `auto` — native word timestamps for `mlx`, `wav2vec2` for `whisperx`).
* `--diarizer`: Speaker diarization backend (`pyannote`, `sherpa-onnx`; default: `pyannote`).
* `--language`: Optional ISO-639-1 language code (e.g. `en`); auto-detected if omitted.
* `--prosody`: Enable local acoustic analysis (pace, stammers/blocks, stress markers).
* `--no-ser`: Skip neural SpeechBrain SER during `--prosody` (uses fast `openSMILE`/DSP stress features only).
* `--metadata-mode`: `both` (default), `summary-only`, or `inline-only`.
* `--device`: Compute device (`auto`, `cuda`, `mps`, `cpu`; default: `auto`).
* `--threads`: Limit CPU threads for ASR/ONNX (reduces heat on fanless laptops).
* `--pause-threshold`: Silence gap in seconds that splits consecutive turns by the same speaker (default: `2.0`).
* `--cache-dir`: Optional directory to cache intermediate ASR & diarization JSON for fast re-runs.

---

## 7. Performance & Memory Management ($1\text{-Hour}$ File Target)

* **Memory Footprint:** A $1\text{-hour}$ $16\text{ kHz}$ mono WAV file consumes $\approx 115\text{ MB}$ on disk. Sequential stage execution with explicit unloading (`del`, `gc.collect()`, `mx.clear_cache()`, `torch.cuda.empty_cache()`, `torch.mps.empty_cache()`) and bounded 8-second SER windowing keeps peak RAM/VRAM usage under **$2.5\text{--}4.5\text{ GB}$**.
* **UX & Reliability:** Prints live per-stage progress and wall-clock timers, automatically invokes macOS `caffeinate -i` to prevent idle sleep mid-run, and supports `--cache-dir` to reuse ASR/diarization outputs across re-runs.
* **Observed & Target Processing Time ($60\text{-minute}$ `.m4a`):**
  * Apple M5 MacBook Air (fanless, `mlx` `turbo` + `sherpa-onnx` + `--prosody` w/ SER): **~10–11 minutes** (~1.5m FFmpeg `loudnorm`, ~3.3m MLX GPU ASR, ~4.2m Sherpa-ONNX diarization, ~1.8m prosody/SER).
  * NVIDIA RTX 3060+ (`whisperx` CUDA `float16` + `pyannote`): **$3\text{--}6\text{ minutes}$**.
  * CPU-Only (`faster-whisper` `int8` + `pyannote`): **$15\text{--}25\text{ minutes}$**.

---

## 8. Acceptance Criteria & Verification

1. **Air-Gap Verification:** Run the script with Wi-Fi/Ethernet disabled (or inside a network-isolated container `docker run --network none`) on a $60\text{-minute}$, $150\text{ MB}$ `.m4a` file; execution must complete with exit code `0`.
2. **Format Compliance:** Output `.txt` matches `speaker X: <text>` line structure with clean speaker turn boundaries and no orphaned timestamps or JSON artifacts.
3. **Default Behavior:** Pressing `Enter` at the speaker count prompt cleanly defaults to `2` speakers without crashing.
4. **Acoustic Accuracy:** When `--prosody` is enabled on test audio containing rapid speech ($>180\text{ WPM}$) and simulated pauses/blocks, inline tags accurately reflect the pace increase and pause anomalies without flagging standard lexical filler words.