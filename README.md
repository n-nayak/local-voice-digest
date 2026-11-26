<p align="center">
  <img src="macos/VoiceDigestApp/Assets/AppIcon.png" alt="VoiceDigest Icon" width="110" />
</p>

<h1 align="center">VoiceDigest</h1>

<p align="center">
  <strong>100% offline macOS app & CLI for audio/video transcription, live call capture, acoustic prosody analysis, and on-device AI summaries — powered by Apple Silicon.</strong><br/>
  Zero cloud uploads. Zero Ollama setup. Zero Hugging Face tokens.
</p>

<p align="center">
  <a href="https://github.com/n-nayak/local-voice-digest/releases">
    <img src="https://img.shields.io/badge/Download-VoiceDigest.dmg%20(Apple%20Silicon)-0A84FF?style=for-the-badge&logo=apple&logoColor=white" alt="Download VoiceDigest.dmg" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/macOS-14.2%2B%20(Apple%20Silicon)-black?style=flat-square&logo=apple" alt="macOS 14.2+" />
  <img src="https://img.shields.io/badge/100%25-Offline%20Inference-2ea44f?style=flat-square" alt="100% Offline" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License" />
</p>

<p align="center">
  <img src="docs/screenshots/transcribe-file.png" alt="VoiceDigest — Transcribe File" width="48%" />
  &nbsp;
  <img src="docs/screenshots/live-call.png" alt="VoiceDigest — Live Call Capture" width="48%" />
</p>

---

## What Makes VoiceDigest Different?

Most local transcription tools stop at plain text — or require setting up Ollama, virtual audio drivers (BlackHole), or gated Hugging Face tokens. **VoiceDigest** ships as one self-contained Mac app that captures **what was said**, **how it was said**, and **what matters**:

- **Acoustic & Prosody Biomarkers (`--prosody`):** Measures speaking pace (WPM), mid-clause hesitation blocks, prolongations, and speaker-normalized vocal stress (`openSMILE` eGeMAPS + DSP) — built for interview prep, sales calls, and communication coaching.
- **Native Menu-Bar Live Call Capture (macOS 14.2+):** Captures your mic (`Me`) and system audio (`Other` / `1:N`) natively via **Core Audio Process Tap** — no meeting bots, no BlackHole driver, and no *Screen Recording* permission required.
- **All-in-One Apple Silicon MLX Stack:** Runs `mlx-whisper` (Metal GPU), `sherpa-onnx` (English WeSpeaker VoxCeleb diarization), and `Qwen2.5-7B-Instruct-4bit` (`mlx-lm` speaker-swap repair + map-reduce summarization) directly inside one `.app` with automatic anonymous model downloads.
- **Strict Air-Gap Enforcement:** Sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before ML imports and installs a Python socket-level outbound network blocker during inference.
- **Broad Audio & Video Support:** Decodes `.m4a`, `.mp3`, `.wav`, `.ogg`/`.opus`, `.flac`, `.aac`, `.aiff`, `.caf`, `.wma`, and extracts audio tracks from `.mp4`, `.mov`, `.mkv`, and `.webm` via bundled `ffmpeg` with configurable EBU R128 loudness normalization (`-24` to `-10 LUFS`).

---

## Sample Output

```text
=== AI SUMMARY ===
- Discussed deployment timeline and rollout milestones for Q3.
- Key concern: narrow release window requires earlier QA sign-off.
- Action item: Speaker 1 to share updated dependency checklist by Thursday.
==================

=== SESSION ACOUSTIC SUMMARY ===
Duration: 58m 14s | Speakers: 2
speaker 1: Avg Pace: 138 WPM (Normal) | Acoustic Disfluencies: 2  | Elevated Stress Turns: 5%
speaker 2: Avg Pace: 184 WPM (Fast)   | Acoustic Disfluencies: 11 | Elevated Stress Turns: 34%
================================

speaker 1 [pace: 135 WPM | state: calm]: Good morning, thanks for making time to go over the project timeline today.
speaker 2 [pace: 192 WPM (fast) | disfluency: 2 blocks | state: tense]: Happy to be here. I have a few concerns about the deployment window we discussed last week.
speaker 1 [pace: 141 WPM | state: calm]: Let's dive right into those blockers first.
```

---

## Install the Mac App (No Terminal Required)

1. Download **`VoiceDigest.dmg`** from the [**GitHub Releases**](https://github.com/n-nayak/local-voice-digest/releases) page.
2. Open the `.dmg` and drag **VoiceDigest** into **Applications**.
3. **First launch (macOS Gatekeeper):** Because this preview build is ad-hoc signed, **Right-click `VoiceDigest.app` → Open → Open** (or **System Settings → Privacy & Security → Open Anyway**) once — after that, it opens normally.
4. Drop an audio/video file (or click the menu bar waveform icon for live calls) and click **Start**. Public AI models download automatically on first run into `~/Library/Application Support/VoiceDigest/models/` (zero tokens or logins required).

---

## Quick Start (CLI / Developers)

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

## Build `VoiceDigest.app` & `VoiceDigest.dmg` from Source

```bash
chmod +x scripts/build_dmg.sh

# Lightweight dev build (uses your local .venv)
./scripts/build_dmg.sh

# Self-contained build (bundles standalone MLX/Sherpa-ONNX backend + ffmpeg inside the .app)
uv pip install pyinstaller
./scripts/build_dmg.sh --self-contained
```

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
