// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VoiceDigest",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "VoiceDigest", targets: ["VoiceDigest"])
    ],
    targets: [
        .executableTarget(
            name: "VoiceDigest",
            path: "Sources/LocalTranscribeApp"
        )
    ]
)
