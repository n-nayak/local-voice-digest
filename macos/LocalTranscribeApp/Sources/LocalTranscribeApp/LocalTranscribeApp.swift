import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var openMainWindowAction: (() -> Void)?

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            openMainWindowAction?()
        }
        return true
    }
}

@main
struct LocalTranscribeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var liveVM = LiveCaptureViewModel()
    @Environment(\.openWindow) private var openWindow

    init() {
        let hidden = UserDefaults.standard.bool(forKey: "hideDockIcon")
        NSApplication.shared.setActivationPolicy(hidden ? .accessory : .regular)
        Self.applyDockIconIfNeeded()
        if !hidden {
            NSApplication.shared.activate(ignoringOtherApps: true)
        }
    }

    var body: some Scene {
        WindowGroup("VoiceDigest", id: "main") {
            ContentView(liveVM: liveVM)
                .frame(minWidth: 640, idealWidth: 680, maxWidth: 880, minHeight: 780, idealHeight: 820)
                .onAppear {
                    appDelegate.openMainWindowAction = {
                        openWindow(id: "main")
                        NSApplication.shared.activate(ignoringOtherApps: true)
                    }
                }
        }
        .defaultSize(width: 680, height: 820)

        // Menu bar popover for live call capture
        MenuBarExtra("VoiceDigest", systemImage: "waveform") {
            MenuBarView(liveVM: liveVM)
        }
        .menuBarExtraStyle(.window)
    }

    private static func applyDockIconIfNeeded() {
        let fm = FileManager.default
        if let resURL = Bundle.main.resourceURL {
            for name in ["AppIcon.icns", "AppIcon.png"] {
                let candidate = resURL.appendingPathComponent(name)
                if fm.fileExists(atPath: candidate.path), let img = NSImage(contentsOf: candidate) {
                    NSApplication.shared.applicationIconImage = img
                    return
                }
            }
        }
        var cursor = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = cursor.appendingPathComponent("macos/LocalTranscribeApp/Assets/AppIcon.png")
            if fm.fileExists(atPath: candidate.path), let img = NSImage(contentsOf: candidate) {
                NSApplication.shared.applicationIconImage = img
                return
            }
            cursor.deleteLastPathComponent()
        }
    }
}
