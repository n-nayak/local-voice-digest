import SwiftUI

struct InfoView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "info.circle.fill")
                    .font(.title2)
                    .foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text("About VoiceDigest")
                        .font(.title3.bold())
                    Text("Settings guide, supported formats, AI models & RAM usage")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding(18)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    settingsExplainedSection
                    supportedFormatsSection
                    modelsTableSection
                    enginesSection
                    privacySection
                }
                .padding(20)
            }
        }
        .frame(width: 600, height: 600)
    }

    private var settingsExplainedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Settings Explained")
                .font(.headline)

            infoBullet(
                "Model Size:",
                "Controls how large the Whisper speech-recognition model is. Larger models handle technical terms, fast speech, and accents better, at the cost of extra disk space and time. Turbo is the best default on Apple Silicon."
            )
            infoBullet(
                "Language:",
                "Choosing your recording's language (e.g. English) skips repeated language detection and improves accuracy. Use Auto-detect for multilingual recordings."
            )
            infoBullet(
                "Speakers:",
                "Expected number of distinct voices in the recording (defaults to 2). Speakers are labeled 'speaker 1', 'speaker 2', etc. in the order they first speak."
            )
            infoBullet(
                "Loudness Normalization (LUFS):",
                "Many phone, Zoom, and laptop recordings are very quiet (-40 dBFS), which makes speech and speaker detection less reliable. EBU R128 normalization boosts speech to a consistent level without amplifying silence. Recommended target: -16 LUFS for calls/podcasts (-14 LUFS louder, -20 LUFS softer)."
            )
            infoBullet(
                "AI Speaker Repair (Local Qwen 7B):",
                "Acoustic diarization alone can confuse short back-and-forth exchanges (like greetings or quick Q&A). Enabling this runs a local 7B LLM (Qwen2.5-7B-Instruct 4-bit, ~4.3 GB on disk, ~5 GB GPU RAM) strictly offline to reassign speaker labels using conversational logic—without altering a single transcribed word or timestamp."
            )
            infoBullet(
                "Publish AI Summary (Local Qwen 7B):",
                "Generates a concise AI summary at the top of your transcript file (=== AI SUMMARY ===) using the same local Qwen 7B model. Optionally enter up to 160 characters to guide the focus (e.g., 'Action items and TODOs', 'Key podcast insights', or 'Rate the interview'). Long recordings (1–3+ hours) are automatically summarized via map-reduce chunking so they never overflow the context window."
            )
            infoBullet(
                "Acoustic & Prosody Analysis:",
                "Adds a Session Acoustic Summary and inline tags ([pace: WPM | disfluency | state]) measuring speaking speed, mid-clause silent blocks (0.4s–1.5s), prolongations, and vocal stress markers—purely from audio acoustics, without flagging lexical filler words like 'um' or 'uh'."
            )
            infoBullet(
                "Fast DSP-only Prosody:",
                "Uses openSMILE + signal-processing stress features only (~5–10s) and skips the heavier neural SpeechBrain emotion classifier (~2 min). Recommended for long recordings."
            )
            infoBullet(
                "Output Folder:",
                "Where the final '<filename>_transcript.txt' file is saved (defaults to Desktop)."
            )
        }
    }

    private var supportedFormatsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Supported File Formats")
                .font(.headline)
            Text("FFmpeg automatically decodes almost any common audio or video recording into 16 kHz mono PCM audio before transcription:")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                infoBullet(
                    "Audio Formats:",
                    ".m4a (Apple Voice Memos, Zoom), .mp3, .wav, .ogg / .opus (WhatsApp, Discord), .flac, .aac, .aiff, .caf, .wma"
                )
                infoBullet(
                    "Video Formats (audio track extracted automatically):",
                    ".mp4, .mov, .mkv, .webm (Zoom, Google Meet, Loom, QuickTime screen recordings)"
                )
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var modelsTableSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Whisper Model Sizes & RAM Usage")
                .font(.headline)
            Text("Models are loaded one stage at a time and freed between stages, so peak memory stays low.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("Model").bold()
                    Text("Disk").bold()
                    Text("Peak RAM").bold()
                    Text("Best For").bold()
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                Divider()

                modelRow("Base", "~150 MB", "~0.6 GB", "Fastest drafts, clear audio")
                modelRow("Small", "~500 MB", "~1.0 GB", "Good balance of speed & accuracy")
                modelRow("Medium", "~1.5 GB", "~1.8 GB", "Strong accuracy on accents/jargon")
                modelRow("Turbo", "~1.6 GB", "~2.0 GB", "Recommended — fast + near large-v3 quality")
                modelRow("Large v3", "~3.0 GB", "~3.5 GB", "Highest accuracy, slowest")
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func modelRow(_ name: String, _ disk: String, _ ram: String, _ note: String) -> some View {
        GridRow {
            Text(name).fontWeight(.medium)
            Text(disk).monospacedDigit()
            Text(ram).monospacedDigit()
            Text(note).foregroundStyle(.secondary)
        }
        .font(.caption)
    }

    private var enginesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Under the Hood: AI Models & Engines")
                .font(.headline)

            infoBullet(
                "1. Audio Preprocessing — FFmpeg:",
                "Neural speech models cannot read compressed files (.m4a, .mp3, .mp4) directly—they expect raw 16,000 samples/sec mono sound waves. FFmpeg decodes and normalizes the audio once into a temporary WAV so all stages share an exact, drift-free timeline."
            )
            infoBullet(
                "2. Speech-to-Text (ASR) — MLX-Whisper:",
                "ASR stands for Automatic Speech Recognition: converting spoken audio into text and word-level timestamps. We use Apple's open-source MLX framework to run OpenAI's open-weight Whisper models natively on the Apple Silicon Metal GPU, which is 5–10× faster and cooler than running Whisper on CPU."
            )
            infoBullet(
                "3. Speaker Diarization — Sherpa-ONNX (pyannote + English WeSpeaker VoxCeleb):",
                "Diarization (literally 'diary-zation'—logging who spoke when) splits audio into speech segments, computes an English VoxCeleb voiceprint per segment using WeSpeaker ResNet34-LM, clusters them into N speakers, and applies sentence-boundary majority smoothing so sentences are never torn mid-clause across two speakers (~32 MB total, zero HF token)."
            )
            infoBullet(
                "4. Optional Semantic Speaker Repair — Local MLX LLM (Qwen2.5-7B-Instruct 4-bit):",
                "When enabled, runs a 4-bit quantized 7B language model natively on your Apple Silicon GPU via mlx-lm. It reads numbered sentences in overlapping chunks and outputs only corrected speaker IDs (JSON), fixing rapid Q&A / greeting swaps while preserving every word and timestamp verbatim."
            )
            infoBullet(
                "5. Acoustic & Prosody Analysis — openSMILE + Optional SpeechBrain SER:",
                "Plain text loses how something was said. openSMILE extracts eGeMAPS vocal biomarkers (pitch, pitch jitter, spectral flux) alongside timestamp-based speaking pace (WPM) and mid-clause hesitation blocks. Optionally, SpeechBrain's wav2vec2 Speech Emotion Recognition (SER) model scores vocal tension in 8-second windows."
            )
        }
    }

    private var privacySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Privacy & Offline Execution")
                .font(.headline)
            Text("After the one-time download of public open-weight models, all audio decoding, transcription, diarization, and prosody analysis run 100% locally on your Mac. Outbound network sockets and Hugging Face telemetry are actively blocked during transcription.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func infoBullet(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
