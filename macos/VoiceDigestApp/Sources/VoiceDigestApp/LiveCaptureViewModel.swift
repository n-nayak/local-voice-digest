import AppKit
import Foundation
import SwiftUI

enum CaptureProcessingMode: String, CaseIterable, Identifiable {
    case processOnStop = "Process on Stop"
    case liveRolling60s = "Live 60s Rolling"

    var id: String { rawValue }
}

struct BackgroundJobStatus: Identifiable {
    let id: String
    let title: String
    var percent: Double
    var stage: String
    var outputURL: URL?
    var error: String?
    var isFinished: Bool = false
}

@MainActor
final class LiveCaptureViewModel: ObservableObject {
    @Published var isRecording: Bool = false
    @Published var elapsedSeconds: Int = 0
    @Published var systemAudioWarning: String?
    @Published var errorMessage: String?

    @Published var mode: CaptureProcessingMode = .processOnStop
    @Published var modelSize: WhisperModelSize = .turbo
    @Published var liveChunkModelSize: WhisperModelSize = .small
    @Published var remoteSpeakers: Int = 1
    @Published var enableProsody: Bool = true
    @Published var publishSummary: Bool = false
    @Published var summaryFocus: String = ""
    @Published var outputFolderURL: URL

    @Published var jobs: [BackgroundJobStatus] = []
    @Published var tempBytes: Int64 = 0
    @Published var systemAudioVerified: Bool? = nil
    @Published var isVerifyingSystemAudio: Bool = false

    private let recorder = DualAudioRecorder()
    private var activeSession: CaptureSession?
    private var liveOutputURL: URL?
    private var lastFlushedSec: Double = 0.0
    private var clockTimer: Timer?
    private var liveFlushTimer: Timer?
    private var activeSessionIDs: Set<String> = []

    init() {
        let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
        self.outputFolderURL = desktop
        // Sweep any orphaned temp sessions from previous crashes on launch
        SessionStorage.cleanOrphanedSessions()
        refreshTempUsage()
    }

    var formattedElapsed: String {
        let hrs = elapsedSeconds / 3600
        let mins = (elapsedSeconds % 3600) / 60
        let secs = elapsedSeconds % 60
        if hrs > 0 {
            return String(format: "%d:%02d:%02d", hrs, mins, secs)
        }
        return String(format: "%02d:%02d", mins, secs)
    }

    var formattedTempUsage: String {
        ByteCountFormatter.string(fromByteCount: tempBytes, countStyle: .file)
    }

    func refreshTempUsage() {
        tempBytes = SessionStorage.currentTempBytes()
    }

    func verifySystemAudio() {
        guard !isVerifyingSystemAudio else { return }
        isVerifyingSystemAudio = true
        Task.detached(priority: .userInitiated) {
            let result = DualAudioRecorder.probeSystemAudioTap()
            await MainActor.run { [weak self] in
                self?.isVerifyingSystemAudio = false
                self?.systemAudioVerified = result.granted
                if !result.granted {
                    self?.systemAudioWarning = result.message
                } else {
                    self?.systemAudioWarning = nil
                }
            }
        }
    }

    func clearTempFiles() {
        SessionStorage.clearAllTempSessions(excluding: activeSessionIDs)
        refreshTempUsage()
    }

    func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = outputFolderURL
        panel.message = "Select folder where live call transcripts will be saved"
        if panel.runModal() == .OK, let url = panel.url {
            outputFolderURL = url
        }
    }

    func startCallCapture() {
        guard !isRecording else { return }
        errorMessage = nil
        systemAudioWarning = nil

        Task {
            let micGranted = await DualAudioRecorder.ensureMicrophonePermission()
            guard micGranted else {
                self.errorMessage = "Microphone permission is required. Enable VoiceDigest in System Settings → Privacy & Security → Microphone."
                return
            }
            self.beginRecordingAfterPermission()
        }
    }

    private func beginRecordingAfterPermission() {
        guard !isRecording else { return }
        do {
            let session = try recorder.startRecording()
            activeSession = session
            activeSessionIDs.insert(session.id)
            isRecording = true
            elapsedSeconds = 0
            lastFlushedSec = 0.0
            liveOutputURL = outputFolderURL.appendingPathComponent(session.defaultTranscriptFilename)

            if !recorder.systemAudioActive {
                systemAudioWarning = "Mic only (grant System Audio Recording in System Settings for remote voice)"
            }

            clockTimer?.invalidate()
            clockTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    self.elapsedSeconds += 1
                    if self.elapsedSeconds % 5 == 0 {
                        self.refreshTempUsage()
                    }
                }
            }

            if mode == .liveRolling60s {
                liveFlushTimer?.invalidate()
                liveFlushTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
                    Task { @MainActor in
                        self?.flushLiveChunk()
                    }
                }
            }
            refreshTempUsage()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stopCallCapture() {
        guard isRecording else { return }
        clockTimer?.invalidate()
        clockTimer = nil
        liveFlushTimer?.invalidate()
        liveFlushTimer = nil
        isRecording = false

        guard let finishedSession = recorder.stopRecording() else { return }
        let outURL = liveOutputURL ?? outputFolderURL.appendingPathComponent(finishedSession.defaultTranscriptFilename)
        activeSession = nil
        liveOutputURL = nil

        let selectedModel = (mode == .liveRolling60s) ? liveChunkModelSize : modelSize
        let remoteCount = remoteSpeakers
        let wantProsody = enableProsody
        let wantSummary = publishSummary
        let focus = String(summaryFocus.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))

        // Enqueue post-call processing in background so user can immediately start Call #2!
        let job = BackgroundJobStatus(
            id: finishedSession.id,
            title: outURL.lastPathComponent,
            percent: 2.0,
            stage: "Queued for transcription"
        )
        jobs.insert(job, at: 0)
        if jobs.count > 5 { jobs.removeLast() }

        Task.detached { [weak self] in
            guard let self else { return }
            await self.processFinishedSession(
                session: finishedSession,
                outputURL: outURL,
                modelSize: selectedModel,
                remoteSpeakers: remoteCount,
                prosody: wantProsody,
                summary: wantSummary,
                summaryFocus: focus
            )
        }
    }

    func cancelCallCapture() {
        clockTimer?.invalidate()
        clockTimer = nil
        liveFlushTimer?.invalidate()
        liveFlushTimer = nil
        isRecording = false
        if let session = recorder.stopRecording() {
            activeSessionIDs.remove(session.id)
            SessionStorage.deleteSession(session)
        }
        activeSession = nil
        liveOutputURL = nil
        refreshTempUsage()
    }

    func openOutput(_ url: URL) {
        NSWorkspace.shared.open(url)
    }

    func revealOutput(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    // MARK: - Live Rolling 60s Chunk Flush

    private func flushLiveChunk() {
        guard isRecording, let session = activeSession, let outURL = liveOutputURL else { return }
        let endSec = Double(elapsedSeconds)
        guard endSec - lastFlushedSec >= 10.0 else { return }
        let startSec = lastFlushedSec
        lastFlushedSec = endSec

        guard let chunk = recorder.snapshotChunk(session: session, startSec: startSec, endSec: endSec) else {
            return
        }
        let chunkModel = liveChunkModelSize
        Task.detached { [weak self] in
            guard let self, let backend = BackendCommand.resolve() else { return }
            defer {
                try? FileManager.default.removeItem(at: chunk.micChunk)
                try? FileManager.default.removeItem(at: chunk.sysChunk)
            }
            let args = backend.baseArgs + [
                "--mic-file", chunk.micChunk.path,
                "--system-file", chunk.sysChunk.path,
                "--remote-speakers", "1",
                "--asr-backend", "mlx",
                "--model-size", chunkModel.rawValue,
                "--model-dir", backend.modelDir.path,
                "--language", "en",
                "--no-loudnorm",
                "--append",
                "--output", outURL.path
            ]
            _ = try? await self.runCLI(backend: backend, args: args, jobID: nil)
        }
    }

    // MARK: - Post-Call Background Processing

    private nonisolated func processFinishedSession(
        session: CaptureSession,
        outputURL: URL,
        modelSize: WhisperModelSize,
        remoteSpeakers: Int,
        prosody: Bool,
        summary: Bool,
        summaryFocus: String
    ) async {
        defer {
            // Auto-delete temporary session WAVs immediately after processing completes
            SessionStorage.deleteSession(session)
            Task { @MainActor in
                self.activeSessionIDs.remove(session.id)
                self.refreshTempUsage()
            }
        }

        guard let backend = BackendCommand.resolve() else {
            await updateJob(session.id, percent: 0, stage: "Failed", error: "Backend not found", finished: true)
            return
        }

        do {
            // Auto-download missing public models (no token required)
            let needsDL = needsModelDownload(
                modelDir: backend.modelDir,
                size: modelSize,
                needSherpa: remoteSpeakers >= 2,
                needLLM: summary
            )
            if needsDL {
                await updateJob(session.id, percent: 4, stage: "Downloading public AI models…")
                var dlArgs = backend.baseArgs + [
                    "--download-models",
                    "--asr-backend", "mlx",
                    "--model-size", modelSize.rawValue,
                    "--diarizer", "sherpa-onnx",
                    "--model-dir", backend.modelDir.path
                ]
                if summary { dlArgs.append("--llm-repair") }
                try await runCLI(backend: backend, args: dlArgs, jobID: nil)
            }

            var args = backend.baseArgs + [
                "--mic-file", session.micWAVURL.path,
                "--system-file", session.systemWAVURL.path,
                "--remote-speakers", "\(remoteSpeakers)",
                "--asr-backend", "mlx",
                "--model-size", modelSize.rawValue,
                "--diarizer", "sherpa-onnx",
                "--model-dir", backend.modelDir.path,
                "--language", "en",
                "--output", outputURL.path,
                "--json-progress"
            ]
            if prosody {
                args += ["--prosody", "--no-ser"]
            }
            if summary {
                args.append("--summary")
                if !summaryFocus.isEmpty {
                    args += ["--summary-focus", summaryFocus]
                }
            }

            try await runCLI(backend: backend, args: args, jobID: session.id)
            await updateJob(
                session.id,
                percent: 100,
                stage: "Saved",
                outputURL: outputURL,
                finished: true
            )
        } catch {
            await updateJob(
                session.id,
                percent: 0,
                stage: "Failed",
                error: error.localizedDescription,
                finished: true
            )
        }
    }

    private func updateJob(
        _ id: String,
        percent: Double,
        stage: String,
        outputURL: URL? = nil,
        error: String? = nil,
        finished: Bool = false
    ) {
        guard let idx = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[idx].percent = percent
        jobs[idx].stage = stage
        if let outputURL { jobs[idx].outputURL = outputURL }
        if let error { jobs[idx].error = error }
        jobs[idx].isFinished = finished
    }

    private nonisolated func needsModelDownload(
        modelDir: URL,
        size: WhisperModelSize,
        needSherpa: Bool,
        needLLM: Bool
    ) -> Bool {
        let fm = FileManager.default
        let mlxDir = modelDir.appendingPathComponent("whisper-mlx/\(size.rawValue)")
        let hasMLX = (try? fm.contentsOfDirectory(atPath: mlxDir.path).isEmpty == false) ?? false
        let hasSherpa = !needSherpa || (
            fm.fileExists(atPath: modelDir.appendingPathComponent("sherpa-onnx/segmentation.onnx").path) &&
            fm.fileExists(atPath: modelDir.appendingPathComponent("sherpa-onnx/embedding_en_voxceleb.onnx").path)
        )
        let llmDir = modelDir.appendingPathComponent("llm/qwen3-4b-instruct-4bit")
        let hasLLM = !needLLM || ((try? fm.contentsOfDirectory(atPath: llmDir.path).isEmpty == false) ?? false)
        return !(hasMLX && hasSherpa && hasLLM)
    }

    private nonisolated func runCLI(backend: BackendCommand, args: [String], jobID: String?) async throws {
        let proc = Process()
        proc.executableURL = backend.executableURL
        proc.arguments = args
        proc.currentDirectoryURL = backend.workingDirectory

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        if let ffmpegPath = backend.ffmpegPath {
            env["LOCAL_TRANSCRIBE_FFMPEG"] = ffmpegPath
        }
        let existingPath = env["PATH"] ?? "/usr/bin:/bin"
        if !existingPath.contains("/opt/homebrew/bin") {
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + existingPath
        }
        proc.environment = env

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe

        try proc.run()

        let stderrTask = Task.detached {
            let data = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        }

        let handle = stdoutPipe.fileHandleForReading
        var buf = Data()
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            buf.append(chunk)
            while let nl = buf.firstRange(of: Data([0x0A])) {
                let lineData = buf.subdata(in: 0..<nl.lowerBound)
                buf.removeSubrange(0..<nl.upperBound)
                if let jobID,
                   let line = String(data: lineData, encoding: .utf8),
                   line.hasPrefix("@@PROGRESS@@ ") {
                    let jsonStr = String(line.dropFirst("@@PROGRESS@@ ".count))
                    if let data = jsonStr.data(using: .utf8),
                       let payload = try? JSONDecoder().decode(ProgressPayload.self, from: data) {
                        await updateJob(jobID, percent: payload.percent, stage: payload.stage)
                    }
                }
            }
        }

        proc.waitUntilExit()
        let stderrText = await stderrTask.value
        if proc.terminationStatus != 0 {
            let lastLine = stderrText.split(separator: "\n").map(String.init)
                .last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty })
                ?? "Exited with code \(proc.terminationStatus)"
            throw NSError(domain: "VoiceDigest", code: Int(proc.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: lastLine
            ])
        }
    }
}
