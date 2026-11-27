import AppKit
import SwiftUI

/// Marks the hosting NSWindow as excluded from screen-sharing captures.
struct ScreenShareHiddenWindowAccessor: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            view.window?.sharingType = .none
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            nsView.window?.sharingType = .none
        }
    }
}

/// Fixed-size menu bar companion popover: never shrinks dynamically (avoids macOS MenuBarExtra resize artifacts).
struct MenuBarView: View {
    @ObservedObject var liveVM: LiveCaptureViewModel
    @ObservedObject var dictationManager: DictationManager
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("VoiceDigest")
                    .font(.headline)
                Spacer()
                if liveVM.isRecording {
                    Text(liveVM.formattedElapsed)
                        .font(.caption.monospacedDigit().weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 8) {
                if liveVM.isRecording {
                    Button {
                        liveVM.stopCallCapture()
                    } label: {
                        Label("Stop & Transcribe", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Cancel", role: .destructive) {
                        liveVM.cancelCallCapture()
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button {
                        liveVM.startCallCapture()
                    } label: {
                        Label(
                            dictationManager.isEnabled ? "Turn Off Dictation to Record" : "Start Call Capture",
                            systemImage: "record.circle"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(dictationManager.isEnabled)
                }
            }

            if let warn = liveVM.systemAudioWarning {
                Text(warn)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
            } else if let err = liveVM.errorMessage {
                Text(err)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }

            Divider()

            HStack(spacing: 8) {
                Circle()
                    .fill(dictationManager.status.isReadyOrActive ? Color.green : Color.secondary.opacity(0.4))
                    .frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Dictation (\(dictationManager.hotkeyOption.shortBadge))")
                        .font(.caption.weight(.semibold))
                    Text(dictationManager.status.label)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Toggle("", isOn: $dictationManager.isEnabled)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .labelsHidden()
                    .disabled(liveVM.isRecording)
            }

            Divider()

            Text("Recent Captures")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            ScrollView {
                if liveVM.jobs.isEmpty {
                    Text("No captures yet this session.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(liveVM.jobs.prefix(5)) { job in
                            VStack(alignment: .leading, spacing: 3) {
                                HStack {
                                    Text(job.title)
                                        .font(.caption)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    Spacer()
                                    if let url = job.outputURL, job.isFinished {
                                        Button("Open") { liveVM.openOutput(url) }
                                            .buttonStyle(.borderless)
                                            .font(.caption)
                                    } else if let err = job.error {
                                        Text(err).font(.caption2).foregroundStyle(.red).lineLimit(1)
                                    } else {
                                        Text("\(Int(job.percent))%").font(.caption2.monospacedDigit())
                                    }
                                }
                                if !job.isFinished {
                                    ProgressView(value: job.percent, total: 100.0)
                                        .progressViewStyle(.linear)
                                    Text(job.stage)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                } else if job.error == nil {
                                    Text("Saved")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .frame(maxHeight: .infinity)

            Divider()

            HStack {
                Button {
                    NSApplication.shared.setActivationPolicy(.regular)
                    openWindow(id: "main")
                    NSApplication.shared.activate(ignoringOtherApps: true)
                } label: {
                    Text("Open VoiceDigest")
                        .font(.subheadline.weight(.medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.28), in: RoundedRectangle(cornerRadius: 7))
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .strokeBorder(Color.blue.opacity(0.45), lineWidth: 1)
                        )
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)

                Spacer()

                Button {
                    NSApplication.shared.terminate(nil)
                } label: {
                    Text("Quit")
                        .font(.subheadline.weight(.medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.red.opacity(0.28), in: RoundedRectangle(cornerRadius: 7))
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .strokeBorder(Color.red.opacity(0.45), lineWidth: 1)
                        )
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .frame(width: 290, height: 300)
        .background(ScreenShareHiddenWindowAccessor())
    }
}
