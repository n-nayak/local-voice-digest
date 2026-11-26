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

/// Thread-safe, zero-dependency 16 kHz 16-bit mono PCM WAV writer.
/// Handles any input sample rate, channel count, and interleaved/non-interleaved Float32 or Int16 buffer
/// directly without relying on fragile AVAudioConverter / AVAudioFile format conversions.
final class PCM16MonoWAVWriter {
    private let lock = NSLock()
    private let targetRate: Double = 16_000.0
    private var fileHandle: FileHandle?
    private(set) var framesWritten: Int64 = 0
    private var dataBytesWritten: UInt32 = 0

    init(outputURL: URL) throws {
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: outputURL)
        // Write placeholder 44-byte RIFF WAV header (finalized with exact sizes on close)
        try handle.write(contentsOf: Self.makeWAVHeader(dataBytes: 0, sampleRate: 16_000))
        self.fileHandle = handle
    }

    func append(buffer: AVAudioPCMBuffer) {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }

        let fmt = buffer.format
        let inRate = fmt.sampleRate > 0 ? fmt.sampleRate : targetRate
        let channels = max(1, Int(fmt.channelCount))

        // 1. Extract mono Float32 samples from any Float32 or Int16 PCM buffer
        var mono = [Float](repeating: 0.0, count: frameCount)
        if let floatData = buffer.floatChannelData {
            if fmt.isInterleaved {
                let ptr = floatData[0]
                if channels == 1 {
                    for i in 0..<frameCount { mono[i] = ptr[i] }
                } else {
                    let invCh = 1.0 / Float(channels)
                    for i in 0..<frameCount {
                        var sum: Float = 0.0
                        let base = i * channels
                        for c in 0..<channels { sum += ptr[base + c] }
                        mono[i] = sum * invCh
                    }
                }
            } else {
                if channels == 1 {
                    let ptr = floatData[0]
                    for i in 0..<frameCount { mono[i] = ptr[i] }
                } else {
                    let invCh = 1.0 / Float(channels)
                    for i in 0..<frameCount {
                        var sum: Float = 0.0
                        for c in 0..<channels { sum += floatData[c][i] }
                        mono[i] = sum * invCh
                    }
                }
            }
        } else if let int16Data = buffer.int16ChannelData {
            let scale: Float = 1.0 / 32768.0
            if fmt.isInterleaved {
                let ptr = int16Data[0]
                let invCh = scale / Float(channels)
                for i in 0..<frameCount {
                    var sum: Float = 0.0
                    let base = i * channels
                    for c in 0..<channels { sum += Float(ptr[base + c]) }
                    mono[i] = sum * invCh
                }
            } else {
                let invCh = scale / Float(channels)
                for i in 0..<frameCount {
                    var sum: Float = 0.0
                    for c in 0..<channels { sum += Float(int16Data[c][i]) }
                    mono[i] = sum * invCh
                }
            }
        } else {
            return
        }

        // 2. Linear-resample to 16 kHz mono Int16
        let ratio = inRate / targetRate
        let outCount = max(1, Int(floor(Double(frameCount) / ratio)))
        var pcm16 = [Int16](repeating: 0, count: outCount)
        for o in 0..<outCount {
            let srcPos = Double(o) * ratio
            let idx = Int(srcPos)
            let frac = Float(srcPos - Double(idx))
            let s0 = mono[min(idx, frameCount - 1)]
            let s1 = mono[min(idx + 1, frameCount - 1)]
            let sample = max(-1.0, min(1.0, s0 + (s1 - s0) * frac))
            pcm16[o] = Int16(sample * 32767.0).littleEndian
        }

        let byteCount = outCount * MemoryLayout<Int16>.size
        let data = pcm16.withUnsafeBytes { Data($0) }

        lock.lock()
        defer { lock.unlock() }
        guard let handle = fileHandle else { return }
        do {
            try handle.write(contentsOf: data)
            framesWritten += Int64(outCount)
            dataBytesWritten &+= UInt32(byteCount)
        } catch {
            // Ignore write errors after close
        }
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        guard let handle = fileHandle else { return }
        do {
            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: Self.makeWAVHeader(dataBytes: dataBytesWritten, sampleRate: 16_000))
            try handle.close()
        } catch {
            try? handle.close()
        }
        fileHandle = nil
    }

    private static func makeWAVHeader(dataBytes: UInt32, sampleRate: UInt32) -> Data {
        var header = Data(capacity: 44)
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate: UInt32 = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign: UInt16 = channels * (bitsPerSample / 8)
        let riffSize: UInt32 = 36 &+ dataBytes

        func appendU32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { header.append(contentsOf: $0) }
        }
        func appendU16(_ v: UInt16) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { header.append(contentsOf: $0) }
        }

        header.append(contentsOf: "RIFF".utf8)
        appendU32(riffSize)
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        appendU32(16) // PCM fmt chunk size
        appendU16(1)  // PCM format
        appendU16(channels)
        appendU32(sampleRate)
        appendU32(byteRate)
        appendU16(blockAlign)
        appendU16(bitsPerSample)
        header.append(contentsOf: "data".utf8)
        appendU32(dataBytes)
        return header
    }
}

/// Captures both Microphone ('Me') and System Audio ('Other') natively on macOS 14.2+
/// using AVAudioRecorder + Core Audio Process Tap (no Screen Recording permission, no virtual driver).
final class DualAudioRecorder {
    private var session: CaptureSession?
    private var micRecorder: AVAudioRecorder?
    private var sysWriter: PCM16MonoWAVWriter?

    // Core Audio Process Tap handles
    private var tapID: AudioObjectID = kAudioObjectUnknown
    private var aggregateDeviceID: AudioObjectID = kAudioObjectUnknown
    private var ioProcID: AudioDeviceIOProcID?

    private(set) var isRecording: Bool = false
    private(set) var systemAudioActive: Bool = false

    /// Ensure macOS microphone permission is explicitly requested and granted before capturing.
    static func ensureMicrophonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    cont.resume(returning: granted)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    func startRecording() throws -> CaptureSession {
        guard !isRecording else {
            throw NSError(domain: "LocalTranscribe", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "A recording session is already in progress."
            ])
        }

        let newSession = try SessionStorage.createSession()
        do {
            let sWriter = try PCM16MonoWAVWriter(outputURL: newSession.systemWAVURL)
            self.sysWriter = sWriter
            self.session = newSession

            try startMicrophoneCapture(outputURL: newSession.micWAVURL)
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

        sysWriter?.close()
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

    // MARK: - Microphone Capture (AVAudioRecorder — native 16 kHz mono PCM WAV)

    private func startMicrophoneCapture(outputURL: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]
        let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
        recorder.isMeteringEnabled = true
        guard recorder.prepareToRecord(), recorder.record() else {
            throw NSError(domain: "VoiceDigest", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Could not start microphone recording. Check System Settings → Privacy & Security → Microphone."
            ])
        }
        self.micRecorder = recorder
    }

    private func stopMicrophoneCapture() {
        micRecorder?.stop()
        micRecorder = nil
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
        sysWriter?.close()
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
