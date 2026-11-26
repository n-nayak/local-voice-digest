// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LocalTranscribeApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "LocalTranscribeApp", targets: ["LocalTranscribeApp"])
    ],
    targets: [
        .executableTarget(
            name: "LocalTranscribeApp",
            path: "Sources/LocalTranscribeApp"
        )
    ]
)
