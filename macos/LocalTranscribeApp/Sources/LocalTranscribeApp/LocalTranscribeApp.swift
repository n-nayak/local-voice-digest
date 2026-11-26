import AppKit
import SwiftUI

@main
struct LocalTranscribeApp: App {
    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        Self.applyDockIconIfNeeded()
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 640, idealWidth: 680, maxWidth: 880, minHeight: 780, idealHeight: 820)
        }
        .defaultSize(width: 680, height: 820)
    }

    private static func applyDockIconIfNeeded() {
        // When launched via `swift run` (outside a packaged .app bundle), load Assets/AppIcon.png for the Dock icon.
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
