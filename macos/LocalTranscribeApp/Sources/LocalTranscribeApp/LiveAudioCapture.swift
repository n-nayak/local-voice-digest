import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation

/// Unique per-call temporary session folder under ~/Library/Caches/LocalTranscribe/sessions/.
struct CaptureSession: Identifiable, Hashable {
    let id: String
    let startedAt: Date
    let directoryURL: URL
    let micWAVURL: URL
    let systemWAVURL: URL

    var defaultTranscriptFilename: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH-mm"
        return "\(fmt.string(from: startedAt))_transcript.txt"
    }
}

enum SessionStorage {
    static var sessionsRootURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return caches.appendingPathComponent("VoiceDigest/sessions", isDirectory: true)
    }

    static func createSession() throws -> CaptureSession {
        let now = Date()
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let shortID = String(UUID().uuidString.prefix(8)).lowercased()
        let folderName = "\(fmt.string(from: now))_\(shortID)"
        let dir = sessionsRootURL.appendingPathComponent(folderName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return CaptureSession(
            id: folderName,
            startedAt: now,
            directoryURL: dir,
            micWAVURL: dir.appendingPathComponent("mic.wav"),
            systemWAVURL: dir.appendingPathComponent("system.wav")
        )
    }

    static func deleteSession(_ session: CaptureSession) {
        try? FileManager.default.removeItem(at: session.directoryURL)
    }

    /// Remove abandoned session folders left behind by a previous crash or power loss.
    static func cleanOrphanedSessions(excluding activeIDs: Set<String> = []) {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: sessionsRootURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return }
        for item in items where !activeIDs.contains(item.lastPathComponent) {
            try? fm.removeItem(at: item)
        }
    }

    static func currentTempBytes() -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: sessionsRootURL,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    static func clearAllTempSessions(excluding activeIDs: Set<String> = []) {
        cleanOrphanedSessions(excluding: activeIDs)
    }
}

/// Thread-safe 16 kHz, 16-bit, mono PCM WAV writer with automatic resampling.
final class PCM16MonoWAVWriter {
    private let lock = NSLock()
    private let targetFormat: AVAudioFormat
    private var audioFile: AVAudioFile?
    private var converter: AVAudioConverter?
    private var lastInputFormat: AVAudioFormat?
    private(set) var framesWritten: AVAudioFramePosition = 0

    init(outputURL: URL) throws {
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16_000,
            channels: 1,
            interleaved: true
        ) else {
            throw NSError(domain: "LocalTranscribe", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create 16 kHz mono PCM format."
            ])
        }
        self.targetFormat = fmt
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        self.audioFile = try AVAudioFile(
            forWriting: outputURL,
            settings: settings,
            commonFormat: .pcmFormatInt16,
            interleaved: true
        )
    }

    func append(buffer: AVAudioPCMBuffer) {
        guard buffer.frameLength > 0 else { return }
        lock.lock()
        defer { lock.unlock() }
        guard let file = audioFile else { return }

        if lastInputFormat == nil || lastInputFormat != buffer.format {
            lastInputFormat = buffer.format
            converter = AVAudioConverter(from: buffer.format, to: targetFormat)
        }
        guard let converter else { return }

        let ratio = targetFormat.sampleRate / max(1.0, buffer.format.sampleRate)
        let outCapacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio)) + 32
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else {
            return
        }

        var consumed = false
        var convError: NSError?
        converter.convert(to: outBuffer, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        if convError == nil, outBuffer.frameLength > 0 {
            try? file.write(from: outBuffer)
            framesWritten += AVAudioFramePosition(outBuffer.frameLength)
        }
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        audioFile = nil
        converter = nil
    }
}

/// Captures both Microphone ('Me') and System Audio ('Other') natively on macOS 14.2+
/// using AVAudioEngine + Core Audio Process Tap (no Screen Recording permission, no virtual driver).
final class DualAudioRecorder {
    private var session: CaptureSession?
    private var micEngine: AVAudioEngine?
    private var micWriter: PCM16MonoWAVWriter?
    private var sysWriter: PCM16MonoWAVWriter?

    // Core Audio Process Tap handles
    private var tapID: AudioObjectID = kAudioObjectUnknown
    private var aggregateDeviceID: AudioObjectID = kAudioObjectUnknown
    private var ioProcID: AudioDeviceIOProcID?

    private(set) var isRecording: Bool = false
    private(set) var systemAudioActive: Bool = false

    func startRecording() throws -> CaptureSession {
        guard !isRecording else {
            throw NSError(domain: "LocalTranscribe", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "A recording session is already in progress."
            ])
        }

        let newSession = try SessionStorage.createSession()
        do {
            let mWriter = try PCM16MonoWAVWriter(outputURL: newSession.micWAVURL)
            let sWriter = try PCM16MonoWAVWriter(outputURL: newSession.systemWAVURL)
            self.micWriter = mWriter
            self.sysWriter = sWriter
            self.session = newSession

            try startMicrophoneCapture(writer: mWriter)
            if #available(macOS 14.2, *) {
                do {
                    try startSystemAudioTap(writer: sWriter)
                    self.systemAudioActive = true
                } catch {
                    // Continue with mic-only if System Audio permission was denied
                    self.systemAudioActive = false
                }
            } else {
                self.systemAudioActive = false
            }

            self.isRecording = true
            return newSession
        } catch {
            stopAndCleanupOnError()
            SessionStorage.deleteSession(newSession)
            throw error
        }
    }

    func stopRecording() -> CaptureSession? {
        guard isRecording else { return nil }
        isRecording = false

        stopMicrophoneCapture()
        stopSystemAudioTap()

        micWriter?.close()
        sysWriter?.close()
        micWriter = nil
        sysWriter = nil

        let finished = session
        session = nil
        return finished
    }

    /// Copy a time slice [startSec, endSec] of the currently recording WAVs into temporary chunk files
    /// for optional minute-by-minute live rolling transcription.
    func snapshotChunk(
        session: CaptureSession,
        startSec: Double,
        endSec: Double
    ) -> (micChunk: URL, sysChunk: URL)? {
        let chunkDir = session.directoryURL.appendingPathComponent("chunks", isDirectory: true)
        try? FileManager.default.createDirectory(at: chunkDir, withIntermediateDirectories: true)
        let tag = "\(Int(startSec))_\(Int(endSec))"
        let micChunk = chunkDir.appendingPathComponent("mic_\(tag).wav")
        let sysChunk = chunkDir.appendingPathComponent("sys_\(tag).wav")

        let okMic = slice16kMonoWAV(src: session.micWAVURL, dst: micChunk, startSec: startSec, endSec: endSec)
        let okSys = slice16kMonoWAV(src: session.systemWAVURL, dst: sysChunk, startSec: startSec, endSec: endSec)
        if okMic || okSys {
            return (micChunk, sysChunk)
        }
        return nil
    }

    // MARK: - Microphone Capture (AVAudioEngine)

    private func startMicrophoneCapture(writer: PCM16MonoWAVWriter) throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0 && inputFormat.channelCount > 0 else {
            throw NSError(domain: "LocalTranscribe", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "No active microphone input device found."
            ])
        }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            writer.append(buffer: buffer)
        }
        engine.prepare()
        try engine.start()
        self.micEngine = engine
    }

    private func stopMicrophoneCapture() {
        micEngine?.inputNode.removeTap(onBus: 0)
        micEngine?.stop()
        micEngine = nil
    }

    // MARK: - System Audio Capture (macOS 14.2+ Core Audio Process Tap)

    @available(macOS 14.2, *)
    private func startSystemAudioTap(writer: PCM16MonoWAVWriter) throws {
        // 1. Create a global system-audio process tap (stereo, non-muted)
        let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        tapDesc.name = "LocalTranscribeSystemTap"
        tapDesc.isPrivate = true
        tapDesc.muteBehavior = .unmuted

        var createdTapID: AudioObjectID = kAudioObjectUnknown
        let tapStatus = AudioHardwareCreateProcessTap(tapDesc, &createdTapID)
        guard tapStatus == noErr, createdTapID != kAudioObjectUnknown else {
            throw NSError(domain: "LocalTranscribe", code: Int(tapStatus), userInfo: [
                NSLocalizedDescriptionKey: "Failed to create Core Audio system tap (OSStatus \(tapStatus)). Grant System Audio Recording permission in System Settings."
            ])
        }
        self.tapID = createdTapID

        // 2. Read the tap's unique UUID string
        var tapUUIDRef: Unmanaged<CFString>?
        var uuidSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var uuidAddr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let uuidStatus = AudioObjectGetPropertyData(createdTapID, &uuidAddr, 0, nil, &uuidSize, &tapUUIDRef)
        guard uuidStatus == noErr, let tapUID = tapUUIDRef?.takeRetainedValue() as String? else {
            stopSystemAudioTap()
            throw NSError(domain: "LocalTranscribe", code: Int(uuidStatus), userInfo: [
                NSLocalizedDescriptionKey: "Could not read Core Audio tap UID."
            ])
        }

        // 3. Wrap the tap in a private Aggregate Device so we can attach an IOProc
        let aggUID = "com.localtranscribe.tap.aggregate.\(UUID().uuidString)"
        let aggDesc: [String: Any] = [
            kAudioAggregateDeviceNameKey as String: "LocalTranscribeTapDevice",
            kAudioAggregateDeviceUIDKey as String: aggUID,
            kAudioAggregateDeviceIsPrivateKey as String: true,
            kAudioAggregateDeviceIsStackedKey as String: false,
            kAudioAggregateDeviceTapAutoStartKey as String: true,
            kAudioAggregateDeviceTapListKey as String: [
                [
                    kAudioSubTapUIDKey as String: tapUID,
                    kAudioSubTapDriftCompensationKey as String: true
                ]
            ]
        ]

        var createdAggID: AudioObjectID = kAudioObjectUnknown
        let aggStatus = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &createdAggID)
        guard aggStatus == noErr, createdAggID != kAudioObjectUnknown else {
            stopSystemAudioTap()
            throw NSError(domain: "LocalTranscribe", code: Int(aggStatus), userInfo: [
                NSLocalizedDescriptionKey: "Failed to create private aggregate device (OSStatus \(aggStatus))."
            ])
        }
        self.aggregateDeviceID = createdAggID

        // 4. Query the tap's stream ASBD format
        var streamDesc = AudioStreamBasicDescription()
        var descSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var fmtAddr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let fmtStatus = AudioObjectGetPropertyData(createdTapID, &fmtAddr, 0, nil, &descSize, &streamDesc)
        guard fmtStatus == noErr,
              let inputFormat = AVAudioFormat(streamDescription: &streamDesc) else {
            stopSystemAudioTap()
            throw NSError(domain: "LocalTranscribe", code: Int(fmtStatus), userInfo: [
                NSLocalizedDescriptionKey: "Failed to read system audio tap stream format."
            ])
        }

        // 5. Register IOProc block to stream incoming system audio buffers into system.wav
        var createdIOProcID: AudioDeviceIOProcID?
        let queue = DispatchQueue(label: "com.localtranscribe.systemtap.io", qos: .utility)
        let ioStatus = AudioDeviceCreateIOProcIDWithBlock(&createdIOProcID, createdAggID, queue) { _, inInputData, _, _, _ in
            let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
            guard let first = abl.first, first.mDataByteSize > 0 else { return }
            let bytesPerFrame = max(1, Int(inputFormat.streamDescription.pointee.mBytesPerFrame))
            let frameCount = AVAudioFrameCount(Int(first.mDataByteSize) / bytesPerFrame)
            guard frameCount > 0,
                  let pcmBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
                return
            }
            pcmBuffer.frameLength = frameCount
            let dstABL = UnsafeMutableAudioBufferListPointer(pcmBuffer.mutableAudioBufferList)
            let count = min(abl.count, dstABL.count)
            for i in 0..<count {
                if let srcData = abl[i].mData, let dstData = dstABL[i].mData {
                    let copyBytes = min(Int(abl[i].mDataByteSize), Int(dstABL[i].mDataByteSize))
                    memcpy(dstData, srcData, copyBytes)
                }
            }
            writer.append(buffer: pcmBuffer)
        }
        guard ioStatus == noErr, let procID = createdIOProcID else {
            stopSystemAudioTap()
            throw NSError(domain: "LocalTranscribe", code: Int(ioStatus), userInfo: [
                NSLocalizedDescriptionKey: "Failed to register system audio IOProc (OSStatus \(ioStatus))."
            ])
        }
        self.ioProcID = procID

        let startStatus = AudioDeviceStart(createdAggID, procID)
        guard startStatus == noErr else {
            stopSystemAudioTap()
            throw NSError(domain: "LocalTranscribe", code: Int(startStatus), userInfo: [
                NSLocalizedDescriptionKey: "Failed to start system audio capture (OSStatus \(startStatus))."
            ])
        }
    }

    private func stopSystemAudioTap() {
        if let procID = ioProcID, aggregateDeviceID != kAudioObjectUnknown {
            AudioDeviceStop(aggregateDeviceID, procID)
            AudioDeviceDestroyIOProcID(aggregateDeviceID, procID)
            ioProcID = nil
        }
        if aggregateDeviceID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            aggregateDeviceID = kAudioObjectUnknown
        }
        if tapID != kAudioObjectUnknown {
            if #available(macOS 14.2, *) {
                AudioHardwareDestroyProcessTap(tapID)
            }
            tapID = kAudioObjectUnknown
        }
        systemAudioActive = false
    }

    private func stopAndCleanupOnError() {
        stopMicrophoneCapture()
        stopSystemAudioTap()
        micWriter?.close()
        sysWriter?.close()
        micWriter = nil
        sysWriter = nil
        session = nil
        isRecording = false
    }

    /// Extract a [startSec, endSec] slice from a growing 16 kHz mono 16-bit WAV file.
    private func slice16kMonoWAV(src: URL, dst: URL, startSec: Double, endSec: Double) -> Bool {
        guard let inFile = try? AVAudioFile(forReading: src) else { return false }
        let sr = inFile.processingFormat.sampleRate
        let totalFrames = inFile.length
        let startFrame = max(0, min(totalFrames, AVAudioFramePosition(startSec * sr)))
        let endFrame = max(startFrame, min(totalFrames, AVAudioFramePosition(endSec * sr)))
        let frameCount = AVAudioFrameCount(endFrame - startFrame)
        guard frameCount > 1600 else { return false }

        inFile.framePosition = startFrame
        guard let buf = AVAudioPCMBuffer(pcmFormat: inFile.processingFormat, frameCapacity: frameCount) else {
            return false
        }
        do {
            try inFile.read(into: buf, frameCount: frameCount)
            let writer = try PCM16MonoWAVWriter(outputURL: dst)
            writer.append(buffer: buf)
            writer.close()
            return true
        } catch {
            return false
        }
    }
}
