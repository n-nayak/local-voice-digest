import AppKit
import AVFoundation
import Foundation
import SwiftUI

enum AppLanguageOption: String, CaseIterable, Identifiable {
    case english = "en"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .english: return "English"
        }
    }
}

struct LocalModelStatus: Identifiable {
    let id: String
    let name: String
    let sizeLabel: String
    let purpose: String
    let isDownloaded: Bool
}

@MainActor
final class ModelInventoryViewModel: ObservableObject {
    @Published var models: [LocalModelStatus] = []
    @Published var isDownloading: Bool = false
    @Published var statusMessage: String = ""
    @Published var errorMessage: String?

    var allDownloaded: Bool {
        !models.isEmpty && models.allSatisfy(\.isDownloaded)
    }

    func refresh() {
        let modelDir = BackendCommand.resolve()?.modelDir ?? defaultModelDir()
        let fm = FileManager.default

        func dirHasFiles(_ rel: String) -> Bool {
            let url = modelDir.appendingPathComponent(rel)
            return ((try? fm.contentsOfDirectory(atPath: url.path).isEmpty == false) ?? false)
        }

        let hasSherpa = fm.fileExists(atPath: modelDir.appendingPathComponent("sherpa-onnx/segmentation.onnx").path)
            && fm.fileExists(atPath: modelDir.appendingPathComponent("sherpa-onnx/embedding_en_voxceleb.onnx").path)

        models = [
            LocalModelStatus(
                id: "whisper-turbo",
                name: "Whisper Turbo (MLX)",
                sizeLabel: "~1.6 GB",
                purpose: "File & Live Call transcription",
                isDownloaded: dirHasFiles("whisper-mlx/turbo")
            ),
            LocalModelStatus(
                id: "sherpa-onnx",
                name: "Sherpa-ONNX Diarizer",
                sizeLabel: "~32 MB",
                purpose: "Speaker separation (zero token)",
                isDownloaded: hasSherpa
            ),
            LocalModelStatus(
                id: "qwen3-4b",
                name: "Qwen3 4B Instruct (4-bit)",
                sizeLabel: "~2.3 GB",
                purpose: "Speaker repair & AI summaries",
                isDownloaded: dirHasFiles("llm/qwen3-4b-instruct-4bit")
            ),
            LocalModelStatus(
                id: "parakeet-v3",
                name: "Parakeet TDT 0.6B v3 (MLX)",
                sizeLabel: "~1.2 GB",
                purpose: "System-wide dictation (<500 ms)",
                isDownloaded: dirHasFiles("dictation/parakeet-tdt-0.6b-v3")
            ),
            LocalModelStatus(
                id: "qwen3-1.7b",
                name: "Qwen3 1.7B (4-bit)",
                sizeLabel: "~1.0 GB",
                purpose: "Dictation AI cleanup (fillers/edits)",
                isDownloaded: dirHasFiles("llm/qwen3-1.7b-4bit")
            ),
        ]
    }

    func revealModelsFolder() {
        let dir = BackendCommand.resolve()?.modelDir ?? defaultModelDir()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(dir)
    }

    func downloadAllModels() {
        runDownload(
            label: "Downloading all public models (~6.1 GB total)…",
            extraArgs: [
                "--download-models",
                "--asr-backend", "mlx",
                "--model-size", "turbo",
                "--diarizer", "sherpa-onnx",
                "--llm-repair",
                "--include-dictation",
            ]
        )
    }

    func downloadDictationModels() {
        runDownload(
            label: "Downloading Parakeet v3 + Qwen3 1.7B (~2.2 GB)…",
            extraArgs: [
                "--download-models",
                "--dictation-models",
            ]
        )
    }

    func downloadCoreAndSummaryModels() {
        runDownload(
            label: "Downloading Whisper Turbo, Sherpa-ONNX & Qwen3 4B (~3.9 GB)…",
            extraArgs: [
                "--download-models",
                "--asr-backend", "mlx",
                "--model-size", "turbo",
                "--diarizer", "sherpa-onnx",
                "--llm-repair",
            ]
        )
    }

    private func runDownload(label: String, extraArgs: [String]) {
        guard !isDownloading else { return }
        guard let backend = BackendCommand.resolve() else {
            errorMessage = "Backend not found. Run from VoiceDigest.app or project .venv."
            return
        }

        isDownloading = true
        errorMessage = nil
        statusMessage = label

        Task {
            defer {
                self.isDownloading = false
                self.refresh()
            }
            do {
                try await Self.executeDownload(
                    backend: backend,
                    args: backend.baseArgs + extraArgs + ["--model-dir", backend.modelDir.path]
                )
                self.statusMessage = "All requested models are downloaded and ready for offline use."
            } catch {
                self.errorMessage = error.localizedDescription
                self.statusMessage = ""
            }
        }
    }

    private nonisolated static func executeDownload(backend: BackendCommand, args: [String]) async throws {
        let proc = Process()
        proc.executableURL = backend.executableURL
        proc.arguments = args
        proc.currentDirectoryURL = backend.workingDirectory

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        let stderrPipe = Pipe()
        proc.standardOutput = Pipe()
        proc.standardError = stderrPipe

        try proc.run()
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            proc.terminationHandler = { _ in cont.resume() }
        }

        if proc.terminationStatus != 0 {
            let data = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let errText = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "Download failed."
            throw NSError(domain: "VoiceDigest", code: Int(proc.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: errText,
            ])
        }
    }

    private func defaultModelDir() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("VoiceDigest/models")
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var dictationManager: DictationManager
    @StateObject private var inventory = ModelInventoryViewModel()
    @AppStorage("hideDockIcon") private var hideDockIcon: Bool = false
    @AppStorage("appLanguage") private var appLanguageRaw: String = AppLanguageOption.english.rawValue
    @State private var isVerifyingSystemAudio: Bool = false
    @State private var systemAudioVerified: Bool? = nil
    @State private var systemAudioStatusText: String = "Click Verify to pre-authorize and test Core Audio system tap."

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "gearshape.fill")
                    .font(.title2)
                    .foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Settings")
                        .font(.title3.bold())
                    Text("Offline AI models, dictation hotkey & app preferences")
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
                VStack(alignment: .leading, spacing: 18) {
                    modelsSection
                    liveCallPermissionsSection
                    dictationHotkeySection
                    generalSection
                }
                .padding(20)
            }
        }
        .frame(width: 600, height: 660)
        .onAppear {
            inventory.refresh()
            dictationManager.refreshPermissions()
        }
    }

    private var modelsSection: some View {
        GroupBox("Offline AI Models (Public, Zero Login Required)") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Models download automatically on first use, or you can pre-download them here before going offline.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                VStack(spacing: 8) {
                    ForEach(inventory.models) { item in
                        HStack(spacing: 10) {
                            Image(systemName: item.isDownloaded ? "checkmark.circle.fill" : "arrow.down.circle")
                                .foregroundStyle(item.isDownloaded ? Color.green : Color.secondary)
                            VStack(alignment: .leading, spacing: 1) {
                                HStack(spacing: 6) {
                                    Text(item.name)
                                        .font(.subheadline.weight(.medium))
                                    Text(item.sizeLabel)
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                }
                                Text(item.purpose)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(item.isDownloaded ? "Downloaded" : "Not Downloaded")
                                .font(.caption.weight(.medium))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(
                                    (item.isDownloaded ? Color.green : Color.secondary).opacity(0.14),
                                    in: Capsule()
                                )
                                .foregroundStyle(item.isDownloaded ? Color.green : Color.secondary)
                        }
                    }
                }
                .padding(.vertical, 4)

                if inventory.isDownloading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text(inventory.statusMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else if !inventory.statusMessage.isEmpty {
                    Text(inventory.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.green)
                }

                if let err = inventory.errorMessage {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(3)
                }

                Divider()

                HStack(spacing: 8) {
                    Button(inventory.allDownloaded ? "All Models Downloaded" : "Download All Models") {
                        inventory.downloadAllModels()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(inventory.isDownloading || inventory.allDownloaded)

                    Button("Core + Summary") {
                        inventory.downloadCoreAndSummaryModels()
                    }
                    .buttonStyle(.bordered)
                    .disabled(inventory.isDownloading)

                    Button("Dictation Only") {
                        inventory.downloadDictationModels()
                    }
                    .buttonStyle(.bordered)
                    .disabled(inventory.isDownloading)

                    Spacer()

                    Button("Show Folder") {
                        inventory.revealModelsFolder()
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(8)
        }
    }

    private var liveCallPermissionsSection: some View {
        GroupBox("Live Call Audio Permissions (Pre-Authorize & Verify)") {
            VStack(alignment: .leading, spacing: 11) {
                Text("Pre-authorize Microphone ('Me') and System Audio ('Other') so live call capture starts cleanly without interruptions.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                // Microphone permission row
                HStack(spacing: 10) {
                    Image(systemName: dictationManager.hasMicrophonePermission ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                        .foregroundStyle(dictationManager.hasMicrophonePermission ? Color.green : Color.orange)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Microphone (Me)")
                            .font(.subheadline.weight(.medium))
                        Text(dictationManager.hasMicrophonePermission ? "Authorized and ready." : "Required for recording your voice in calls and dictation.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if !dictationManager.hasMicrophonePermission {
                        if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
                            Button("Allow Microphone") {
                                dictationManager.requestMicrophonePermission()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        } else {
                            Button("Open Mic Settings") {
                                DualAudioRecorder.openMicrophoneSettings()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    } else {
                        Text("Verified")
                            .font(.caption.weight(.medium))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.14), in: Capsule())
                            .foregroundStyle(.green)
                    }
                }

                Divider()

                // System audio tap row
                HStack(spacing: 10) {
                    Image(systemName: systemAudioIconName)
                        .foregroundStyle(systemAudioIconColor)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("System Audio Capture (Other)")
                            .font(.subheadline.weight(.medium))
                        Text(systemAudioStatusText)
                            .font(.caption)
                            .foregroundStyle(systemAudioVerified == false ? Color.orange : Color.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    if isVerifyingSystemAudio {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        if systemAudioVerified == false {
                            Button("Open Settings") {
                                DualAudioRecorder.openSystemAudioSettings()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        Button(systemAudioVerified == true ? "Re-verify" : "Verify System Audio") {
                            verifySystemAudioCapture()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }
            .padding(8)
        }
    }

    private var systemAudioIconName: String {
        switch systemAudioVerified {
        case .some(true): return "checkmark.circle.fill"
        case .some(false): return "exclamationmark.circle.fill"
        case .none: return "speaker.wave.2.circle"
        }
    }

    private var systemAudioIconColor: Color {
        switch systemAudioVerified {
        case .some(true): return .green
        case .some(false): return .orange
        case .none: return .secondary
        }
    }

    private func verifySystemAudioCapture() {
        isVerifyingSystemAudio = true
        systemAudioStatusText = "Testing Core Audio Process Tap…"
        Task.detached(priority: .userInitiated) {
            let result = DualAudioRecorder.probeSystemAudioTap()
            await MainActor.run {
                self.isVerifyingSystemAudio = false
                self.systemAudioVerified = result.granted
                self.systemAudioStatusText = result.message
            }
        }
    }

    private var dictationHotkeySection: some View {
        GroupBox("System-Wide Dictation Hotkey") {
            VStack(alignment: .leading, spacing: 11) {
                HStack {
                    Text("Trigger Hotkey")
                        .frame(width: 120, alignment: .leading)
                    Picker("", selection: $dictationManager.hotkeyOption) {
                        ForEach(DictationHotkeyOption.allCases) { opt in
                            Text(opt.label).tag(opt)
                        }
                    }
                    .labelsHidden()
                }

                HStack {
                    Text("Trigger Mode")
                        .frame(width: 120, alignment: .leading)
                    Picker("", selection: $dictationManager.triggerMode) {
                        ForEach(DictationTriggerMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }

                Toggle("Show floating recording pill with live preview", isOn: $dictationManager.showFloatingPill)

                Toggle("Auto-paste chunks on pause or end of sentence (. ? !)", isOn: $dictationManager.streamPasteChunks)

                HStack(spacing: 12) {
                    HStack(spacing: 6) {
                        Image(systemName: dictationManager.hasAccessibilityPermission ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                            .foregroundStyle(dictationManager.hasAccessibilityPermission ? Color.green : Color.orange)
                        Text("Accessibility Permission")
                            .font(.caption.weight(.medium))
                    }
                    if !dictationManager.hasAccessibilityPermission {
                        Button("Open Accessibility Settings") {
                            dictationManager.openAccessibilitySettings()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    Spacer()
                }
            }
            .padding(8)
        }
    }

    private var generalSection: some View {
        GroupBox("General") {
            VStack(alignment: .leading, spacing: 11) {
                HStack {
                    Text("App Language")
                        .frame(width: 120, alignment: .leading)
                    Picker("", selection: $appLanguageRaw) {
                        ForEach(AppLanguageOption.allCases) { lang in
                            Text(lang.label).tag(lang.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 180)
                    Spacer()
                }
                Text("Additional interface languages (including German) are planned for a future update.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Divider()

                Toggle("Menu Bar Only (Hide Dock Icon)", isOn: $hideDockIcon)
                    .onChange(of: hideDockIcon) { _, hidden in
                        NSApplication.shared.setActivationPolicy(hidden ? .accessory : .regular)
                    }
            }
            .padding(8)
        }
    }
}
