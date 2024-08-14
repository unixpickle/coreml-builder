// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CoreMLBuilder",
    platforms: [.macOS(.v13)],
    products: [
        .library(
            name: "CoreMLProto",
            targets: ["CoreMLProto"]),
        .library(
            name: "CoreMLBuilder",
            targets: ["CoreMLBuilder"]),
        .executable(
            name: "MatrixBench",
            targets: ["MatrixBench"]),
        .executable(
            name: "BlockANEMatmul",
            targets: ["BlockANEMatmul"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-protobuf", "1.9.0" ..< "2.0.0"),
    ],
    targets: [
        .target(
            name: "CoreMLProto",
            dependencies: [
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
            ]),
        .target(
            name: "CoreMLBuilder",
            dependencies: ["CoreMLProto"]),
        .testTarget(
            name: "CoreMLBuilderTests",
            dependencies: ["CoreMLBuilder"]),
        .executableTarget(
            name: "MatrixBench",
            dependencies: ["CoreMLBuilder"]),
        .executableTarget(
            name: "BlockANEMatmul",
            dependencies: [
                "CoreMLBuilder",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]),
    ]
)
