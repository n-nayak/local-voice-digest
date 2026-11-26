import AppKit
import Foundation
import SwiftUI

enum WhisperModelSize: String, CaseIterable, Identifiable {
    case base = "base"
    case small = "small"
    case medium = "medium"
    case turbo = "turbo"
    case largeV3 = "large-v3"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .base: return "Base (Fastest, ~150 MB)"
        case .small: return "Small (Balanced, ~500 MB)"
        case .medium: return "Medium (Accurate, ~1.5 GB)"
        case .turbo: return "Turbo (Recommended, ~1.6 GB)"
        case .largeV3: return "Large v3 (Highest Accuracy, ~3 GB)"
        }
    }
}

enum LanguageOption: String, CaseIterable, Identifiable {
    case en = "en"
    case auto = "auto"
    case es = "es"
    case fr = "fr"
    case de = "de"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .en: return "English (en)"
        case .auto: return "Auto-detect"
        case .es: return "Spanish (es)"
        case .fr: return "French (fr)"
        case .de: return "German (de)"
        }
    }
}

struct ProgressPayload: Decodable {
    let percent: Double
    let stage: String
    let detail: String?
    let output: String?
}

struct BackendCommand {
    let executableURL: URL
    let baseArgs: [String]
    let workingDirectory: URL
    let ffmpegPath: String?
    let modelDir: URL
}

@MainActor
final class TranscriptionViewModel: ObservableObject {
    @Published var inputFileURL: URL?
    @Published var outputFolderURL: URL
    @Published var modelSize: WhisperModelSize = .turbo
    @Published var speakers: Int = 2
    @Published var language: LanguageOption = .en
    @Published var normalizeLoudness: Bool = true
    @Published var loudnessTargetLUFS: Double = -16.0
    @Published var enableLLMRepair: Bool = false
    @Published var publishSummary: Bool = false
    @Published var summaryFocus: String = ""
    @Published var enableProsody: Bool = true
    @Published var fastDspOnlyProsody: Bool = true

    @Published var isRunning: Bool = false
    @Published var progress: Double = 0.0
    @Published var stageLabel: String = "Ready"
    @Published var detailLabel: String = "Drop an audio or video file above to begin"
    @Published var elapsedSeconds: Int = 0
    @Published var completedOutputURL: URL?
    @Published var errorMessage: String?

    private var activeProcess: Process?
    private var timer: Timer?

    init() {
        let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
        self.outputFolderURL = desktop
    }

    var inputFileSummary: String {
        guard let url = inputFileURL else { return "No file selected" }
        let sizeStr: String
        if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
           let bytes = attrs[.size] as? Int64 {
            sizeStr = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
        } else {
            sizeStr = ""
        }
        return sizeStr.isEmpty ? url.lastPathComponent : "\(url.lastPathComponent) (\(sizeStr))"
    }

    var formattedElapsed: String {
        let mins = elapsedSeconds / 60
        let secs = elapsedSeconds % 60
        return mins > 0 ? String(format: "%dm %02ds", mins, secs) : "\(secs)s"
    }

    func chooseInputFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .movie, .mpeg4Audio, .mpeg4Movie]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select an audio or video recording to transcribe"
        if panel.runModal() == .OK, let url = panel.url {
            inputFileURL = url
            errorMessage = nil
            completedOutputURL = nil
        }
    }

    func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = outputFolderURL
        panel.message = "Select folder where the transcript .txt will be saved"
        if panel.runModal() == .OK, let url = panel.url {
            outputFolderURL = url
        }
    }

    func start() {
        guard let inputURL = inputFileURL else {
            errorMessage = "Please drop or select an .m4a audio file first."
            return
        }
        guard !isRunning else { return }

        guard let backend = resolveBackend() else {
            errorMessage = "Could not locate bundled transcription engine or local .venv."
            return
        }

        isRunning = true
        errorMessage = nil
        completedOutputURL = nil
        progress = 0.0
        elapsedSeconds = 0
        startTimer()

        let stem = inputURL.deletingPathExtension().lastPathComponent
        let outputFileURL = outputFolderURL.appendingPathComponent("\(stem)_transcript.txt")

        Task.detached { [weak self] in
            guard let self else { return }
            do {
                // Step 0: Auto-download missing public models (zero HF token required)
                let wantLLMRepair = await self.enableLLMRepair
                let wantSummary = await self.publishSummary
                let wantLLM = wantLLMRepair || wantSummary
                let needsDownload = self.needsModelDownload(
                    modelDir: backend.modelDir,
                    size: await self.modelSize,
                    includeLLM: wantLLM
                )
                if needsDownload {
                    await MainActor.run {
                        self.stageLabel = "First-time setup: downloading AI models"
                        self.detailLabel = wantLLM
                            ? "Downloading Whisper (\(self.modelSize.rawValue)), English speaker & Qwen 7B models"
                            : "Downloading Whisper (\(self.modelSize.rawValue)) & English speaker models"
                        self.progress = 3.0
                    }
                    var dlArgs = backend.baseArgs + [
                        "--download-models",
                        "--asr-backend", "mlx",
                        "--model-size", await self.modelSize.rawValue,
                        "--diarizer", "sherpa-onnx",
                        "--model-dir", backend.modelDir.path
                    ]
                    if wantLLM {
                        dlArgs.append("--llm-repair")
                    }
                    try await self.runSubprocess(backend: backend, args: dlArgs)
                }

                // Step 1: Run offline transcription
                var args = backend.baseArgs + [
                    inputURL.path,
                    "--speakers", "\(await self.speakers)",
                    "--asr-backend", "mlx",
                    "--model-size", await self.modelSize.rawValue,
                    "--diarizer", "sherpa-onnx",
                    "--model-dir", backend.modelDir.path,
                    "--output", outputFileURL.path,
                    "--json-progress"
                ]

                let lang = await self.language
                if lang != .auto {
                    args += ["--language", lang.rawValue]
                }
                if await self.normalizeLoudness {
                    let targetInt = Int(await self.loudnessTargetLUFS.rounded())
                    args += ["--loudness-target", "\(targetInt)"]
                } else {
                    args.append("--no-loudnorm")
                }
                if wantLLMRepair {
                    args.append("--llm-repair")
                }
                if wantSummary {
                    args.append("--summary")
                    let focus = String((await self.summaryFocus).trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))
                    if !focus.isEmpty {
                        args += ["--summary-focus", focus]
                    }
                }
                if await self.enableProsody {
                    args.append("--prosody")
                    if await self.fastDspOnlyProsody {
                        args.append("--no-ser")
                    }
                }

                try await self.runSubprocess(backend: backend, args: args)

                await MainActor.run {
                    self.progress = 100.0
                    self.stageLabel = "Complete"
                    self.detailLabel = "Saved to \(outputFileURL.lastPathComponent)"
                    self.completedOutputURL = outputFileURL
                    self.isRunning = false
                    self.stopTimer()
                }
            } catch {
                await MainActor.run {
                    if self.isRunning {
                        self.errorMessage = error.localizedDescription
                        self.stageLabel = "Failed"
                        self.detailLabel = error.localizedDescription
                    }
                    self.isRunning = false
                    self.stopTimer()
                }
            }
        }
    }

    func cancel() {
        activeProcess?.terminate()
        activeProcess = nil
        isRunning = false
        stopTimer()
        stageLabel = "Cancelled"
        detailLabel = "Transcription stopped"
    }

    func openCompletedTranscript() {
        guard let url = completedOutputURL else { return }
        NSWorkspace.shared.open(url)
    }

    func revealInFinder() {
        guard let url = completedOutputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    // MARK: - Helpers

    private nonisolated func needsModelDownload(modelDir: URL, size: WhisperModelSize, includeLLM: Bool) -> Bool {
        let fm = FileManager.default
        let mlxDir = modelDir.appendingPathComponent("whisper-mlx/\(size.rawValue)")
        let segFile = modelDir.appendingPathComponent("sherpa-onnx/segmentation.onnx")
        let embFile = modelDir.appendingPathComponent("sherpa-onnx/embedding_en_voxceleb.onnx")
        let llmDir = modelDir.appendingPathComponent("llm/qwen2.5-7b-4bit")

        let hasMLX = (try? fm.contentsOfDirectory(atPath: mlxDir.path).isEmpty == false) ?? false
        let hasSherpa = fm.fileExists(atPath: segFile.path) && fm.fileExists(atPath: embFile.path)
        let hasLLM = !includeLLM || ((try? fm.contentsOfDirectory(atPath: llmDir.path).isEmpty == false) ?? false)
        return !(hasMLX && hasSherpa && hasLLM)
    }

    private func resolveBackend() -> BackendCommand? {
        let fm = FileManager.default
        let appSupportModels = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("VoiceDigest/models")

        // 1. Self-contained packaged .app bundle
        if let resURL = Bundle.main.resourceURL {
            let bundledCLI = resURL.appendingPathComponent("backend/transcribe-cli")
            if fm.isExecutableFile(atPath: bundledCLI.path) {
                let bundledFFmpeg = resURL.appendingPathComponent("bin/ffmpeg")
                return BackendCommand(
                    executableURL: bundledCLI,
                    baseArgs: [],
                    workingDirectory: resURL,
                    ffmpegPath: fm.isExecutableFile(atPath: bundledFFmpeg.path) ? bundledFFmpeg.path : nil,
                    modelDir: appSupportModels
                )
            }
        }

        // 2. Development mode: walk upward from executable to locate repo root (.venv + transcribe.py)
        var cursor = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        for _ in 0..<8 {
            let script = cursor.appendingPathComponent("transcribe.py")
            let venvPy = cursor.appendingPathComponent(".venv/bin/python")
            if fm.fileExists(atPath: script.path) && fm.isExecutableFile(atPath: venvPy.path) {
                let localModels = cursor.appendingPathComponent("models")
                return BackendCommand(
                    executableURL: venvPy,
                    baseArgs: [script.path],
                    workingDirectory: cursor,
                    ffmpegPath: nil,
                    modelDir: fm.fileExists(atPath: localModels.path) ? localModels : appSupportModels
                )
            }
            cursor.deleteLastPathComponent()
        }
        return nil
    }

    private nonisolated func runSubprocess(backend: BackendCommand, args: [String]) async throws {
        let proc = Process()
        proc.executableURL = backend.executableURL
        proc.arguments = args
        proc.currentDirectoryURL = backend.workingDirectory

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        if let ffmpegPath = backend.ffmpegPath {
            env["LOCAL_TRANSCRIBE_FFMPEG"] = ffmpegPath
        }
        // Ensure Homebrew PATH is visible in dev mode
        let existingPath = env["PATH"] ?? "/usr/bin:/bin"
        if !existingPath.contains("/opt/homebrew/bin") {
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + existingPath
        }
        proc.environment = env

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe

        await MainActor.run {
            self.activeProcess = proc
        }

        try proc.run()

        let handle = stdoutPipe.fileHandleForReading
        var stderrLines: [String] = []

        // Read stderr concurrently to capture any error details
        let stderrTask = Task.detached {
            let data = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        }

        var lineBuffer = Data()
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            lineBuffer.append(chunk)
            while let nlRange = lineBuffer.firstRange(of: Data([0x0A])) {
                let lineData = lineBuffer.subdata(in: 0..<nlRange.lowerBound)
                lineBuffer.removeSubrange(0..<nlRange.upperBound)
                if let line = String(data: lineData, encoding: .utf8) {
                    await handleStdoutLine(line)
                }
            }
        }

        proc.waitUntilExit()
        let stderrText = await stderrTask.value
        if !stderrText.isEmpty {
            stderrLines = stderrText.split(separator: "\n").map(String.init)
        }

        await MainActor.run {
            self.activeProcess = nil
        }

        if proc.terminationStatus != 0 {
            let reason = stderrLines.last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty })
                ?? "Process exited with code \(proc.terminationStatus)"
            throw NSError(domain: "LocalTranscribe", code: Int(proc.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: reason
            ])
        }
    }

    private func handleStdoutLine(_ line: String) {
        let prefix = "@@PROGRESS@@ "
        guard line.hasPrefix(prefix) else { return }
        let jsonStr = String(line.dropFirst(prefix.count))
        guard let data = jsonStr.data(using: .utf8),
              let payload = try? JSONDecoder().decode(ProgressPayload.self, from: data) else {
            return
        }
        self.progress = payload.percent
        self.stageLabel = payload.stage
        self.detailLabel = payload.detail ?? ""
    }

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.elapsedSeconds += 1
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}
