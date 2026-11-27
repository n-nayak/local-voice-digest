import AppKit
import SwiftUI

enum DictationHUDState: Equatable {
    case hidden
    case listening(preview: String)
    case transcribing(polishing: Bool)
    case pasted(totalMs: Int)
    case error(String)
}

@MainActor
final class DictationHUDController {
    private var panel: NSPanel?
    private var hostingView: NSHostingView<DictationHUDContentView>?
    private var autoHideWorkItem: DispatchWorkItem?

    func update(state: DictationHUDState) {
        autoHideWorkItem?.cancel()
        autoHideWorkItem = nil

        if state == .hidden {
            panel?.orderOut(nil)
            return
        }

        let panel = ensurePanel()
        let content = DictationHUDContentView(state: state)
        if let hosting = hostingView {
            hosting.rootView = content
        } else {
            let hosting = NSHostingView(rootView: content)
            panel.contentView = hosting
            hostingView = hosting
        }

        let size = panelSize(for: state)
        positionBottomCenter(panel, size: size)
        panel.orderFrontRegardless()

        switch state {
        case .pasted:
            let item = DispatchWorkItem { [weak self] in
                self?.panel?.orderOut(nil)
            }
            autoHideWorkItem = item
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.4, execute: item)
        case .error:
            let item = DispatchWorkItem { [weak self] in
                self?.panel?.orderOut(nil)
            }
            autoHideWorkItem = item
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.4, execute: item)
        default:
            break
        }
    }

    func hide() {
        update(state: .hidden)
    }

    private func panelSize(for state: DictationHUDState) -> NSSize {
        switch state {
        case .listening(let preview) where !preview.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
            return NSSize(width: 420, height: 56)
        default:
            return NSSize(width: 220, height: 42)
        }
    }

    private func ensurePanel() -> NSPanel {
        if let existing = panel {
            return existing
        }
        let p = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 220, height: 42),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.level = .floating
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        p.ignoresMouseEvents = true
        p.sharingType = .none
        p.isReleasedWhenClosed = false
        panel = p
        return p
    }

    private func positionBottomCenter(_ panel: NSPanel, size: NSSize) {
        let mouseLocation = NSEvent.mouseLocation
        let screen = NSScreen.screens.first(where: { NSMouseInRect(mouseLocation, $0.frame, false) })
            ?? NSScreen.main
            ?? NSScreen.screens.first
        guard let visible = screen?.visibleFrame else { return }
        let x = visible.midX - (size.width / 2)
        let y = visible.minY + 32
        panel.setFrame(NSRect(x: x, y: y, width: size.width, height: size.height), display: true)
    }
}

private struct DictationHUDContentView: View {
    let state: DictationHUDState
    @State private var pulse = false

    private var hasLivePreview: Bool {
        if case .listening(let preview) = state {
            return !preview.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return false
    }

    var body: some View {
        HStack(spacing: 10) {
            iconView
            if case .listening(let preview) = state,
               !preview.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(tailPreview(preview))
                    .font(.system(size: 12.5, weight: .medium, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .truncationMode(.head)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text(labelText)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .frame(
            minWidth: hasLivePreview ? 360 : 168,
            maxWidth: hasLivePreview ? 420 : 220,
            minHeight: hasLivePreview ? 52 : 40,
            maxHeight: hasLivePreview ? 56 : 40
        )
        .background(
            RoundedRectangle(cornerRadius: hasLivePreview ? 16 : 20, style: .continuous)
                .fill(Color.black.opacity(0.86))
                .overlay(
                    RoundedRectangle(cornerRadius: hasLivePreview ? 16 : 20, style: .continuous)
                        .strokeBorder(borderColor.opacity(0.65), lineWidth: 1)
                )
        )
        .onAppear {
            withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
    }

    private func tailPreview(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= 95 {
            return trimmed
        }
        return "…" + String(trimmed.suffix(92))
    }

    @ViewBuilder
    private var iconView: some View {
        switch state {
        case .listening:
            Circle()
                .fill(Color.red)
                .frame(width: 10, height: 10)
                .scaleEffect(pulse ? 1.25 : 0.85)
        case .transcribing:
            ProgressView()
                .controlSize(.small)
                .colorScheme(.dark)
        case .pasted:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(Color.green)
                .font(.system(size: 14, weight: .bold))
        case .error:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.orange)
                .font(.system(size: 14, weight: .bold))
        case .hidden:
            EmptyView()
        }
    }

    private var labelText: String {
        switch state {
        case .listening:
            return "Listening…"
        case .transcribing(let polishing):
            return polishing ? "Polishing…" : "Transcribing…"
        case .pasted(let ms):
            return "Pasted · \(ms) ms"
        case .error(let msg):
            return msg
        case .hidden:
            return ""
        }
    }

    private var borderColor: Color {
        switch state {
        case .listening: return .red
        case .transcribing: return .cyan
        case .pasted: return .green
        case .error: return .orange
        case .hidden: return .clear
        }
    }
}
