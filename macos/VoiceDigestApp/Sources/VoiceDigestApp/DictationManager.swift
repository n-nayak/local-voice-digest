import AppKit
import ApplicationServices
import AVFoundation
import Carbon.HIToolbox
import Foundation
import SwiftUI

enum DictationTriggerMode: String, CaseIterable, Identifiable {
    case pressAndHold = "pressAndHold"
    case tapToToggle = "tapToToggle"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .pressAndHold: return "Press & Hold"
        case .tapToToggle: return "Tap to Toggle"
        }
    }
}

enum DictationHotkeyOption: String, CaseIterable, Identifiable {
    case rightOption = "rightOption"
    case leftControlOption = "leftControlOption"
    case rightCommandOption = "rightCommandOption"
    case rightCommand = "rightCommand"
    case fnGlobe = "fnGlobe"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .rightOption: return "Right Option (⌥) — Recommended"
        case .leftControlOption: return "Left Control + Option (⌃⌥) — Universal"
        case .rightCommandOption: return "Right Command + Option (⌥⌘)"
        case .rightCommand: return "Right Command (⌘)"
        case .fnGlobe: return "Fn / Globe (MacBook / Apple Keyboards)"
        }
    }

    var shortBadge: String {
        switch self {
        case .rightOption: return "Right ⌥"
        case .leftControlOption: return "⌃⌥"
        case .rightCommandOption: return "Right ⌥⌘"
        case .rightCommand: return "Right ⌘"
        case .fnGlobe: return "Fn"
        }
    }
}

enum DictationDaemonStatus: Equatable {
    case off
    case waitingForPermission
    case downloading(String)
    case warmingUp
    case ready
    case listening
    case transcribing
    case error(String)

    var label: String {
        switch self {
        case .off: return "Off"
        case .waitingForPermission: return "Waiting for permission…"
        case .downloading(let detail): return detail
        case .warmingUp: return "Loading warm models…"
        case .ready: return "Ready (<500 ms)"
        case .listening: return "Listening…"
        case .transcribing: return "Transcribing…"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    var isReadyOrActive: Bool {
        switch self {
        case .ready, .listening, .transcribing: return true
        default: return false
        }
    }
}

private struct DictationRPCResponse: Decodable {
    let id: Int?
    let event: String?
    let ok: Bool?
    let text: String?
    let raw_text: String?
    let asr_ms: Int?
    let llm_ms: Int?
    let total_ms: Int?
    let used_llm: Bool?
    let llm_loaded: Bool?
    let error: String?
}

@MainActor
final class DictationManager: ObservableObject {
    @Published var isEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnabled, forKey: "dictation_enabled")
            if isEnabled {
                startDictationService(promptForPermissions: true)
            } else {
                stopDictationService()
            }
        }
    }

    @Published var triggerMode: DictationTriggerMode {
        didSet {
            UserDefaults.standard.set(triggerMode.rawValue, forKey: "dictation_trigger_mode")
        }
    }

    @Published var hotkeyOption: DictationHotkeyOption {
        didSet {
            UserDefaults.standard.set(hotkeyOption.rawValue, forKey: "dictation_hotkey")
        }
    }

    @Published var enableAICleanup: Bool {
        didSet {
            UserDefaults.standard.set(enableAICleanup, forKey: "dictation_ai_cleanup")
        }
    }

    @Published var showFloatingPill: Bool {
        didSet {
            UserDefaults.standard.set(showFloatingPill, forKey: "dictation_show_pill")
            if !showFloatingPill {
                hud.hide()
            }
        }
    }

    @Published var streamPasteChunks: Bool {
        didSet {
            UserDefaults.standard.set(streamPasteChunks, forKey: "dictation_stream_chunks")
        }
    }

    @Published var customVocabulary: String {
        didSet {
            UserDefaults.standard.set(customVocabulary, forKey: "dictation_custom_vocab")
        }
    }

    @Published var status: DictationDaemonStatus = .off
    @Published var hasAccessibilityPermission: Bool = false
    @Published var hasMicrophonePermission: Bool = false
    @Published var lastTranscribedText: String = ""
    @Published var lastLatencySummary: String = ""
    @Published var errorMessage: String?

    private let hud = DictationHUDController()
    private var daemonProcess: Process?
    private var daemonStdin: FileHandle?
    private var stdoutBuffer = Data()
    private var pendingContinuation: CheckedContinuation<DictationRPCResponse, Error>?
    private var requestCounter = 0

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var isHotkeyPressed = false
    private var activeObserver: NSObjectProtocol?
    private var permissionPollTimer: Timer?

    private var recorder: AVAudioRecorder?
    private var currentClipURL: URL?
    private var currentAppName: String = ""
    private var currentWindowTitle: String = ""
    private var isTranscribing = false
    private var isRPCInFlight = false

    // Live preview & optional pause/sentence-boundary chunk streaming state
    private var liveMonitorTimer: Timer?
    private var segmentStartSec: Double = 0.0
    private var hasSpokenInSegment: Bool = false
    private var silenceDuration: Double = 0.0
    private var ticksSincePreview: Int = 0
    private var latestPreviewText: String = ""
    private var chunksPastedCount: Int = 0

    init() {
        let defaults = UserDefaults.standard
        let savedTrigger = defaults.string(forKey: "dictation_trigger_mode")
            .flatMap(DictationTriggerMode.init(rawValue:)) ?? .pressAndHold
        let savedHotkey = defaults.string(forKey: "dictation_hotkey")
            .flatMap(DictationHotkeyOption.init(rawValue:)) ?? .rightOption
        let savedCleanup = defaults.object(forKey: "dictation_ai_cleanup") as? Bool ?? true
        let savedPill = defaults.object(forKey: "dictation_show_pill") as? Bool ?? true
        let savedStreamChunks = defaults.bool(forKey: "dictation_stream_chunks")
        let savedVocab = defaults.string(forKey: "dictation_custom_vocab") ?? ""
        let savedEnabled = defaults.bool(forKey: "dictation_enabled")

        self.triggerMode = savedTrigger
        self.hotkeyOption = savedHotkey
        self.enableAICleanup = savedCleanup
        self.showFloatingPill = savedPill
        self.streamPasteChunks = savedStreamChunks
        self.customVocabulary = savedVocab
        self.isEnabled = false

        refreshPermissions()

        activeObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshPermissions()
            }
        }

        // Restore enabled state on launch ONLY if permissions are already granted
        // (avoids flooding the user with permission dialogs on first launch, and
        // explicitly boots the daemon since `didSet` does not fire inside `init()`).
        if savedEnabled && hasMicrophonePermission && hasAccessibilityPermission {
            self.isEnabled = true
            Task { @MainActor [weak self] in
                self?.startDictationService(promptForPermissions: false)
            }
        } else if savedEnabled {
            self.isEnabled = false
            defaults.set(false, forKey: "dictation_enabled")
        }
    }

    deinit {
        if let observer = activeObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - Permissions

    func refreshPermissions() {
        let axTrusted = AXIsProcessTrusted()
        let micGranted = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        hasAccessibilityPermission = axTrusted
        hasMicrophonePermission = micGranted

        if isEnabled && axTrusted && micGranted && eventTap == nil && daemonProcess?.isRunning == true {
            errorMessage = nil
            if installGlobalEventTap() {
                stopPermissionPollTimer()
                status = .ready
            }
        }
    }

    private func startPermissionPollTimer() {
        stopPermissionPollTimer()
        permissionPollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isEnabled else {
                    self?.stopPermissionPollTimer()
                    return
                }
                self.refreshPermissions()
            }
        }
    }

    private func stopPermissionPollTimer() {
        permissionPollTimer?.invalidate()
        permissionPollTimer = nil
    }

    func requestAccessibilityPermission() {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let options: CFDictionary = [key: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
        refreshPermissions()
    }

    func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            let opened = NSWorkspace.shared.open(url)
            if !opened {
                NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Applications/System Settings.app"))
            }
        }
    }

    func requestMicrophonePermission() {
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            Task { @MainActor [weak self] in
                self?.hasMicrophonePermission = granted
                self?.refreshPermissions()
            }
        }
    }

    // MARK: - Lifecycle

    func shutdown() {
        stopPermissionPollTimer()
        stopRecordingIfNeeded(discard: true)
        removeGlobalEventTap()
        terminateDaemon()
        hud.hide()
        status = .off
    }

    private func startDictationService(promptForPermissions: Bool) {
        errorMessage = nil
        refreshPermissions()

        if promptForPermissions {
            if !hasMicrophonePermission {
                requestMicrophonePermission()
            }
            if !hasAccessibilityPermission {
                openAccessibilitySettings()
            }
        }

        Task {
            await bootDaemonAndTap()
        }
    }

    private func stopDictationService() {
        shutdown()
    }

    private func bootDaemonAndTap() async {
        guard let backend = BackendCommand.resolve() else {
            let msg = "Backend not found. Run from VoiceDigest.app or project .venv."
            errorMessage = msg
            status = .error(msg)
            return
        }

        do {
            if needsDictationModelDownload(modelDir: backend.modelDir) {
                status = .downloading("Downloading Parakeet v3 + Qwen3-1.7B (~2.2 GB)…")
                try await downloadDictationModels(backend: backend)
            }

            guard isEnabled else {
                status = .off
                return
            }

            status = .warmingUp
            try await launchWarmDaemon(backend: backend)

            guard isEnabled else {
                shutdown()
                return
            }

            refreshPermissions()
            if hasAccessibilityPermission && hasMicrophonePermission && installGlobalEventTap() {
                stopPermissionPollTimer()
                errorMessage = nil
                status = .ready
            } else {
                // Keep daemon warm and poll every 1s until user clicks Allow in macOS prompt/Settings
                status = .waitingForPermission
                startPermissionPollTimer()
            }
        } catch {
            let msg = error.localizedDescription
            errorMessage = msg
            status = .error(msg)
            terminateDaemon()
        }
    }

    private nonisolated func needsDictationModelDownload(modelDir: URL) -> Bool {
        let fm = FileManager.default
        let asrDir = modelDir.appendingPathComponent("dictation/parakeet-tdt-0.6b-v3")
        let llmDir = modelDir.appendingPathComponent("llm/qwen3-1.7b-4bit")
        let hasASR = (try? fm.contentsOfDirectory(atPath: asrDir.path).isEmpty == false) ?? false
        let hasLLM = (try? fm.contentsOfDirectory(atPath: llmDir.path).isEmpty == false) ?? false
        return !(hasASR && hasLLM)
    }

    private nonisolated func downloadDictationModels(backend: BackendCommand) async throws {
        let proc = Process()
        proc.executableURL = backend.executableURL
        proc.arguments = backend.baseArgs + [
            "--download-models",
            "--dictation-models",
            "--model-dir", backend.modelDir.path,
        ]
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
            let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let errText = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "Download failed"
            throw NSError(domain: "VoiceDigest", code: Int(proc.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: errText,
            ])
        }
    }

    private func launchWarmDaemon(backend: BackendCommand) async throws {
        terminateDaemon()

        let proc = Process()
        proc.executableURL = backend.executableURL
        proc.arguments = backend.baseArgs + [
            "--dictation-daemon",
            "--model-dir", backend.modelDir.path,
        ]
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

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardInput = stdinPipe
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe

        stdoutBuffer.removeAll()

        let readyResponse: DictationRPCResponse = try await withCheckedThrowingContinuation { cont in
            self.pendingContinuation = cont

            stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty else { return }
                Task { @MainActor [weak self] in
                    self?.handleDaemonStdout(data)
                }
            }

            proc.terminationHandler = { [weak self] terminated in
                stdoutPipe.fileHandleForReading.readabilityHandler = nil
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if let pending = self.pendingContinuation {
                        self.pendingContinuation = nil
                        let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                        let errMsg = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
                        pending.resume(throwing: NSError(
                            domain: "VoiceDigest",
                            code: Int(terminated.terminationStatus),
                            userInfo: [NSLocalizedDescriptionKey: errMsg?.isEmpty == false ? errMsg! : "Dictation daemon exited unexpectedly."]
                        ))
                    }
                    if self.isEnabled && self.daemonProcess === terminated {
                        self.status = .error("Daemon stopped")
                    }
                }
            }

            do {
                try proc.run()
                self.daemonProcess = proc
                self.daemonStdin = stdinPipe.fileHandleForWriting
            } catch {
                self.pendingContinuation = nil
                cont.resume(throwing: error)
            }
        }

        if readyResponse.ok == false {
            throw NSError(domain: "VoiceDigest", code: 1, userInfo: [
                NSLocalizedDescriptionKey: readyResponse.error ?? "Failed to warm up dictation models.",
            ])
        }
    }

    private func handleDaemonStdout(_ data: Data) {
        stdoutBuffer.append(data)
        while let newlineRange = stdoutBuffer.range(of: Data([0x0A])) {
            let lineData = stdoutBuffer.subdata(in: stdoutBuffer.startIndex..<newlineRange.lowerBound)
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newlineRange.lowerBound)
            guard !lineData.isEmpty else { continue }
            if let decoded = try? JSONDecoder().decode(DictationRPCResponse.self, from: lineData) {
                if let pending = pendingContinuation {
                    pendingContinuation = nil
                    pending.resume(returning: decoded)
                }
            }
        }
    }

    private func sendDaemonRequest(_ dict: [String: Any]) async throws -> DictationRPCResponse {
        guard let stdinHandle = daemonStdin, let proc = daemonProcess, proc.isRunning else {
            throw NSError(domain: "VoiceDigest", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Dictation daemon is not running.",
            ])
        }
        requestCounter += 1
        var payload = dict
        payload["id"] = requestCounter
        let jsonData = try JSONSerialization.data(withJSONObject: payload)
        var lineData = jsonData
        lineData.append(0x0A)

        return try await withCheckedThrowingContinuation { cont in
            self.pendingContinuation = cont
            do {
                try stdinHandle.write(contentsOf: lineData)
            } catch {
                self.pendingContinuation = nil
                cont.resume(throwing: error)
            }
        }
    }

    private func terminateDaemon() {
        if let stdinHandle = daemonStdin {
            if let quitData = "{\"cmd\":\"quit\"}\n".data(using: .utf8) {
                try? stdinHandle.write(contentsOf: quitData)
            }
            try? stdinHandle.close()
        }
        daemonStdin = nil
        if let proc = daemonProcess, proc.isRunning {
            proc.terminate()
        }
        daemonProcess = nil
        pendingContinuation = nil
    }

    // MARK: - Global Hotkey (CGEventTap)

    @discardableResult
    private func installGlobalEventTap() -> Bool {
        removeGlobalEventTap()

        let mask: CGEventMask =
            (1 << CGEventType.flagsChanged.rawValue) |
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.keyUp.rawValue)

        let userInfo = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: mask,
            callback: { _, type, event, refcon in
                guard let refcon else { return Unmanaged.passUnretained(event) }
                let manager = Unmanaged<DictationManager>.fromOpaque(refcon).takeUnretainedValue()
                let keyCode = Int(event.getIntegerValueField(.keyboardEventKeycode))
                let flags = event.flags
                DispatchQueue.main.async {
                    manager.handleCGEvent(type: type, keyCode: keyCode, flags: flags)
                }
                return Unmanaged.passUnretained(event)
            },
            userInfo: userInfo
        ) else {
            hasAccessibilityPermission = AXIsProcessTrusted()
            errorMessage = "Enable Accessibility permission in System Settings to activate the global dictation hotkey."
            return false
        }

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        eventTap = tap
        runLoopSource = source
        return true
    }

    private func removeGlobalEventTap() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        isHotkeyPressed = false
    }

    private func handleCGEvent(type: CGEventType, keyCode: Int, flags: CGEventFlags) {
        guard isEnabled, status.isReadyOrActive else { return }

        // Allow Esc to cancel an active recording without pasting
        if type == .keyDown, keyCode == 53, recorder?.isRecording == true {
            isHotkeyPressed = false
            stopRecordingIfNeeded(discard: true)
            status = .ready
            hud.hide()
            return
        }

        guard type == .flagsChanged else { return }

        switch hotkeyOption {
        case .rightOption:
            guard keyCode == 61 else { return }
            handleHotkeyTransition(isDown: flags.contains(.maskAlternate))

        case .rightCommand:
            guard keyCode == 54 else { return }
            handleHotkeyTransition(isDown: flags.contains(.maskCommand))

        case .fnGlobe:
            guard keyCode == 63 else { return }
            handleHotkeyTransition(isDown: flags.contains(.maskSecondaryFn))

        case .leftControlOption:
            // Both Left Control (59) and Left Option (58) held together
            let bothHeld = flags.contains(.maskControl) && flags.contains(.maskAlternate)
            if bothHeld && !isHotkeyPressed {
                handleHotkeyTransition(isDown: true)
            } else if !bothHeld && isHotkeyPressed {
                handleHotkeyTransition(isDown: false)
            }

        case .rightCommandOption:
            // Command + Option held together (triggered via Right Cmd 54 or Right Opt 61)
            let bothHeld = flags.contains(.maskCommand) && flags.contains(.maskAlternate)
            if bothHeld && !isHotkeyPressed && (keyCode == 54 || keyCode == 61) {
                handleHotkeyTransition(isDown: true)
            } else if !bothHeld && isHotkeyPressed {
                handleHotkeyTransition(isDown: false)
            }
        }
    }

    private func handleHotkeyTransition(isDown: Bool) {
        switch triggerMode {
        case .pressAndHold:
            if isDown && !isHotkeyPressed {
                isHotkeyPressed = true
                beginListening()
            } else if !isDown && isHotkeyPressed {
                isHotkeyPressed = false
                finishListeningAndTranscribe()
            }
        case .tapToToggle:
            if isDown && !isHotkeyPressed {
                isHotkeyPressed = true
                if recorder?.isRecording == true {
                    finishListeningAndTranscribe()
                } else {
                    beginListening()
                }
            } else if !isDown {
                isHotkeyPressed = false
            }
        }
    }

    // MARK: - Recording & Active App Context

    private func beginListening() {
        guard !isTranscribing, recorder == nil else { return }

        let (appName, winTitle) = captureActiveWindowContext()
        currentAppName = appName
        currentWindowTitle = winTitle

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicedigest_dictation_\(UUID().uuidString.prefix(8)).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]

        do {
            let rec = try AVAudioRecorder(url: tmpURL, settings: settings)
            rec.isMeteringEnabled = true
            rec.prepareToRecord()
            guard rec.record() else {
                errorMessage = "Could not start microphone recording."
                return
            }
            recorder = rec
            currentClipURL = tmpURL
            segmentStartSec = 0.0
            hasSpokenInSegment = false
            silenceDuration = 0.0
            ticksSincePreview = 0
            latestPreviewText = ""
            chunksPastedCount = 0
            status = .listening
            if showFloatingPill {
                hud.update(state: .listening(preview: ""))
            }
            startLiveMonitorTimer()
        } catch {
            errorMessage = "Mic error: \(error.localizedDescription)"
            if showFloatingPill {
                hud.update(state: .error("Mic Error"))
            }
        }
    }

    private func startLiveMonitorTimer() {
        stopLiveMonitorTimer()
        liveMonitorTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.onLiveMonitorTick()
            }
        }
    }

    private func stopLiveMonitorTimer() {
        liveMonitorTimer?.invalidate()
        liveMonitorTimer = nil
    }

    private func onLiveMonitorTick() {
        guard let rec = recorder, rec.isRecording, let clipURL = currentClipURL else { return }
        rec.updateMeters()
        let power = rec.averagePower(forChannel: 0)
        let isSpeech = power > -42.0

        if isSpeech {
            hasSpokenInSegment = true
            silenceDuration = 0.0
        } else if hasSpokenInSegment {
            silenceDuration += 0.2
        }

        ticksSincePreview += 1
        let nowSec = rec.currentTime

        // Check optional streaming chunk paste trigger:
        // 1) Long silence pause (>= 1.2s) OR
        // 2) Semantic end-of-sentence punctuation (., ?, !) + brief natural beat (>= 0.45s)
        if streamPasteChunks && hasSpokenInSegment && !isRPCInFlight {
            let trimmedPreview = latestPreviewText.trimmingCharacters(in: .whitespacesAndNewlines)
            let endsSentence = trimmedPreview.hasSuffix(".") || trimmedPreview.hasSuffix("?") || trimmedPreview.hasSuffix("!")
            let hitLongPause = silenceDuration >= 1.2
            let hitSentenceBoundary = endsSentence && silenceDuration >= 0.45 && trimmedPreview.count >= 6

            if (hitLongPause || hitSentenceBoundary) && (nowSec - segmentStartSec) >= 0.6 {
                let startSec = segmentStartSec
                let endSec = nowSec
                segmentStartSec = endSec
                hasSpokenInSegment = false
                silenceDuration = 0.0
                latestPreviewText = ""
                ticksSincePreview = 0
                flushStreamingChunk(clipURL: clipURL, startSec: startSec, endSec: endSec)
                return
            }
        }

        // Live preview in the floating HUD bubble every ~0.8s (4 ticks) using raw Parakeet
        if showFloatingPill && hasSpokenInSegment && !isRPCInFlight && ticksSincePreview >= 4 && (nowSec - segmentStartSec) >= 0.45 {
            ticksSincePreview = 0
            runLivePreview(clipURL: clipURL, startSec: segmentStartSec, endSec: nowSec)
        }
    }

    private func runLivePreview(clipURL: URL, startSec: Double, endSec: Double) {
        let sliceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicedigest_prev_\(UUID().uuidString.prefix(8)).wav")
        guard Self.sliceLivePCM16WAV(src: clipURL, dst: sliceURL, startSec: startSec, endSec: endSec) else {
            return
        }
        isRPCInFlight = true
        Task {
            defer {
                try? FileManager.default.removeItem(at: sliceURL)
                self.isRPCInFlight = false
            }
            do {
                let resp = try await sendDaemonRequest([
                    "cmd": "transcribe",
                    "wav": sliceURL.path,
                    "cleanup": false,
                    "app_name": "",
                    "window_title": "",
                    "vocab": "",
                ])
                if resp.ok != false,
                   let text = resp.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !text.isEmpty,
                   self.recorder?.isRecording == true {
                    self.latestPreviewText = text
                    if self.showFloatingPill {
                        self.hud.update(state: .listening(preview: text))
                    }
                }
            } catch {
                // Ignore transient preview errors; final transcription will handle full audio
            }
        }
    }

    private func flushStreamingChunk(clipURL: URL, startSec: Double, endSec: Double) {
        let sliceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicedigest_chunk_\(UUID().uuidString.prefix(8)).wav")
        guard Self.sliceLivePCM16WAV(src: clipURL, dst: sliceURL, startSec: startSec, endSec: endSec) else {
            return
        }

        isRPCInFlight = true
        let cleanup = enableAICleanup
        let appName = currentAppName
        let winTitle = currentWindowTitle
        let vocab = customVocabulary

        Task {
            defer {
                try? FileManager.default.removeItem(at: sliceURL)
                self.isRPCInFlight = false
            }
            do {
                let resp = try await sendDaemonRequest([
                    "cmd": "transcribe",
                    "wav": sliceURL.path,
                    "cleanup": cleanup,
                    "app_name": appName,
                    "window_title": winTitle,
                    "vocab": vocab,
                ])
                if resp.ok != false,
                   let text = resp.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !text.isEmpty {
                    let asrMs = resp.asr_ms ?? 0
                    let llmMs = resp.llm_ms ?? 0
                    let totalMs = resp.total_ms ?? (asrMs + llmMs)
                    self.chunksPastedCount += 1
                    self.lastTranscribedText = text
                    self.lastLatencySummary = resp.used_llm == true
                        ? "Chunk · ASR \(asrMs) ms + AI \(llmMs) ms = \(totalMs) ms"
                        : "Chunk · ASR \(asrMs) ms (raw Parakeet)"
                    self.pasteTextAtCursor(text + " ")
                    if self.showFloatingPill && self.recorder?.isRecording == true {
                        self.hud.update(state: .listening(preview: ""))
                    }
                }
            } catch {
                // Non-fatal; continue listening
            }
        }
    }

    private func stopRecordingIfNeeded(discard: Bool) {
        stopLiveMonitorTimer()
        if let rec = recorder {
            rec.stop()
            recorder = nil
        }
        if discard, let url = currentClipURL {
            try? FileManager.default.removeItem(at: url)
            currentClipURL = nil
        }
    }

    private func finishListeningAndTranscribe() {
        stopLiveMonitorTimer()
        guard let rec = recorder, let clipURL = currentClipURL else { return }
        let finalDuration = rec.currentTime
        rec.stop()
        recorder = nil
        currentClipURL = nil

        let trailingStart = segmentStartSec
        let hadPriorChunks = chunksPastedCount > 0

        isTranscribing = true
        status = .transcribing
        if showFloatingPill {
            hud.update(state: .transcribing(polishing: enableAICleanup))
        }

        let cleanup = enableAICleanup
        let appName = currentAppName
        let winTitle = currentWindowTitle
        let vocab = customVocabulary

        Task {
            // Wait briefly if a live preview / chunk RPC is finishing so stdin/stdout stays serialized
            while self.isRPCInFlight {
                try? await Task.sleep(nanoseconds: 25_000_000)
            }

            // If streaming chunks already pasted earlier audio, transcribe only the trailing segment
            var targetURL = clipURL
            var tempTailURL: URL? = nil
            if hadPriorChunks {
                let tailURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("voicedigest_tail_\(UUID().uuidString.prefix(8)).wav")
                if Self.sliceLivePCM16WAV(src: clipURL, dst: tailURL, startSec: trailingStart, endSec: finalDuration + 0.5) {
                    targetURL = tailURL
                    tempTailURL = tailURL
                } else {
                    // Nothing left to paste after the last flushed chunk
                    try? FileManager.default.removeItem(at: clipURL)
                    self.isTranscribing = false
                    if self.isEnabled && self.daemonProcess?.isRunning == true {
                        self.status = .ready
                    }
                    if self.showFloatingPill {
                        self.hud.hide()
                    }
                    return
                }
            }

            self.isRPCInFlight = true
            defer {
                self.isRPCInFlight = false
                try? FileManager.default.removeItem(at: clipURL)
                if let t = tempTailURL {
                    try? FileManager.default.removeItem(at: t)
                }
                self.isTranscribing = false
                if self.isEnabled && self.daemonProcess?.isRunning == true {
                    self.status = .ready
                }
            }

            do {
                let resp = try await sendDaemonRequest([
                    "cmd": "transcribe",
                    "wav": targetURL.path,
                    "cleanup": cleanup,
                    "app_name": appName,
                    "window_title": winTitle,
                    "vocab": vocab,
                ])

                if resp.ok == false {
                    let msg = resp.error ?? "Transcription failed"
                    self.errorMessage = msg
                    if self.showFloatingPill {
                        self.hud.update(state: .error("Failed"))
                    }
                    return
                }

                let text = (resp.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else {
                    if self.showFloatingPill {
                        self.hud.hide()
                    }
                    return
                }

                let asrMs = resp.asr_ms ?? 0
                let llmMs = resp.llm_ms ?? 0
                let totalMs = resp.total_ms ?? (asrMs + llmMs)

                self.lastTranscribedText = text
                if resp.used_llm == true {
                    self.lastLatencySummary = "ASR \(asrMs) ms + AI \(llmMs) ms = \(totalMs) ms"
                } else {
                    self.lastLatencySummary = "ASR \(asrMs) ms (raw Parakeet)"
                }

                pasteTextAtCursor(text)

                if self.showFloatingPill {
                    self.hud.update(state: .pasted(totalMs: totalMs))
                }
            } catch {
                self.errorMessage = error.localizedDescription
                if self.showFloatingPill {
                    self.hud.update(state: .error("Error"))
                }
            }
        }
    }

    /// Safely slice [startSec, endSec] directly from a growing 16 kHz mono 16-bit PCM WAV file.
    private nonisolated static func sliceLivePCM16WAV(
        src: URL,
        dst: URL,
        startSec: Double,
        endSec: Double
    ) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: src) else { return false }
        defer { try? handle.close() }

        let fileSize = Int((try? FileManager.default.attributesOfItem(atPath: src.path)[.size] as? NSNumber)?.int64Value ?? 0)
        guard fileSize > 44 else { return false }

        // Locate the "data" chunk header in the first 4 KB (AVAudioRecorder may include FLLR padding)
        let headerScanLen = min(4096, fileSize)
        guard let headerData = try? handle.read(upToCount: headerScanLen),
              let dataMarkerRange = headerData.range(of: Data("data".utf8)) else {
            return false
        }
        let pcmOffset = dataMarkerRange.upperBound + 4
        let totalPCMBytes = fileSize - pcmOffset
        guard totalPCMBytes > 3200 else { return false } // At least ~0.1s of 16 kHz mono PCM16

        let bytesPerSec = 32_000.0
        let rawStart = max(0, Int(startSec * bytesPerSec)) & ~1
        let rawEnd = min(totalPCMBytes, max(rawStart, Int(endSec * bytesPerSec)) & ~1)
        let sliceLen = rawEnd - rawStart
        guard sliceLen >= 3200 else { return false }

        do {
            try handle.seek(toOffset: UInt64(pcmOffset + rawStart))
            guard let pcmSlice = try handle.read(upToCount: sliceLen), !pcmSlice.isEmpty else {
                return false
            }
            var outData = PCM16MonoWAVWriter.makeWAVHeader(dataBytes: UInt32(pcmSlice.count), sampleRate: 16_000)
            outData.append(pcmSlice)
            try outData.write(to: dst, options: .atomic)
            return true
        } catch {
            return false
        }
    }

    private func captureActiveWindowContext() -> (String, String) {
        guard let app = NSWorkspace.shared.frontmostApplication else {
            return ("", "")
        }
        let appName = app.localizedName ?? ""
        guard AXIsProcessTrusted() else {
            return (appName, "")
        }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        var focusedWindow: CFTypeRef?
        let winErr = AXUIElementCopyAttributeValue(axApp, kAXFocusedWindowAttribute as CFString, &focusedWindow)
        guard winErr == .success, let winObj = focusedWindow else {
            return (appName, "")
        }
        let axWin = winObj as! AXUIElement
        var titleVal: CFTypeRef?
        let titleErr = AXUIElementCopyAttributeValue(axWin, kAXTitleAttribute as CFString, &titleVal)
        let title = (titleErr == .success ? (titleVal as? String) : nil) ?? ""
        return (appName, title)
    }

    // MARK: - Clipboard-Safe Auto-Paste

    private func pasteTextAtCursor(_ text: String) {
        let pasteboard = NSPasteboard.general
        let savedString = pasteboard.string(forType: .string)
        let savedChangeCount: Int

        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        savedChangeCount = pasteboard.changeCount

        // Synthesize Cmd+V into the currently focused app
        let source = CGEventSource(stateID: .combinedSessionState)
        let vKeyCode = CGKeyCode(kVK_ANSI_V)
        if let keyDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true),
           let keyUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false) {
            keyDown.flags = .maskCommand
            keyUp.flags = .maskCommand
            keyDown.post(tap: .cghidEventTap)
            keyUp.post(tap: .cghidEventTap)
        }

        // Restore the user's previous clipboard after the target app finishes reading Cmd+V
        if let previous = savedString {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                if pasteboard.changeCount == savedChangeCount {
                    pasteboard.clearContents()
                    pasteboard.setString(previous, forType: .string)
                }
            }
        }
    }
}
