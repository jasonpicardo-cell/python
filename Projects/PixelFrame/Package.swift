// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PixelFrame",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "PixelFrame",
            path: "Sources/PixelFrame",
            swiftSettings: [.unsafeFlags(["-Osize", "-wmo"])],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-dead_strip"])]
        )
    ]
)
