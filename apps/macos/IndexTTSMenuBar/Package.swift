// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "IndexTTSMenuBar",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "IndexTTSMenuBar", targets: ["IndexTTSMenuBar"]),
    ],
    targets: [
        .target(name: "IndexTTSMenuBarCore"),
        .executableTarget(
            name: "IndexTTSMenuBar",
            dependencies: ["IndexTTSMenuBarCore"]
        ),
        .testTarget(
            name: "IndexTTSMenuBarCoreTests",
            dependencies: ["IndexTTSMenuBarCore"]
        ),
    ]
)
