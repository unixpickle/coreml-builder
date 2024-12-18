// DO NOT EDIT.
// swift-format-ignore-file
// swiftlint:disable all
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: VisionFeaturePrint.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2018, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///
/// A model which takes an input image and outputs array(s) of features
/// according to the specified feature types
public struct CoreMLModels_VisionFeaturePrint: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Vision feature print type
  public var visionFeaturePrintType: CoreMLModels_VisionFeaturePrint.OneOf_VisionFeaturePrintType? = nil

  public var scene: CoreMLModels_VisionFeaturePrint.Scene {
    get {
      if case .scene(let v)? = visionFeaturePrintType {return v}
      return CoreMLModels_VisionFeaturePrint.Scene()
    }
    set {visionFeaturePrintType = .scene(newValue)}
  }

  public var objects: CoreMLModels_VisionFeaturePrint.Objects {
    get {
      if case .objects(let v)? = visionFeaturePrintType {return v}
      return CoreMLModels_VisionFeaturePrint.Objects()
    }
    set {visionFeaturePrintType = .objects(newValue)}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  /// Vision feature print type
  public enum OneOf_VisionFeaturePrintType: Equatable, Sendable {
    case scene(CoreMLModels_VisionFeaturePrint.Scene)
    case objects(CoreMLModels_VisionFeaturePrint.Objects)

  }

  /// Scene extracts features useful for identifying contents of natural images
  /// in both indoor and outdoor environments
  public struct Scene: Sendable {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var version: CoreMLModels_VisionFeaturePrint.Scene.SceneVersion = .invalid

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public enum SceneVersion: SwiftProtobuf.Enum, Swift.CaseIterable {
      public typealias RawValue = Int
      case invalid // = 0

      /// VERSION_1 is available on iOS,tvOS 12.0+, macOS 10.14+
      /// It uses a 299x299 input image and yields a 2048 float feature vector
      case sceneVersion1 // = 1

      /// VERSION_2 is available on iOS,tvOS 17.0+, macOS 14.0+
      /// It uses a 360x360 input image and yields a 768 float feature vector
      case sceneVersion2 // = 2
      case UNRECOGNIZED(Int)

      public init() {
        self = .invalid
      }

      public init?(rawValue: Int) {
        switch rawValue {
        case 0: self = .invalid
        case 1: self = .sceneVersion1
        case 2: self = .sceneVersion2
        default: self = .UNRECOGNIZED(rawValue)
        }
      }

      public var rawValue: Int {
        switch self {
        case .invalid: return 0
        case .sceneVersion1: return 1
        case .sceneVersion2: return 2
        case .UNRECOGNIZED(let i): return i
        }
      }

      // The compiler won't synthesize support with the UNRECOGNIZED case.
      public static let allCases: [CoreMLModels_VisionFeaturePrint.Scene.SceneVersion] = [
        .invalid,
        .sceneVersion1,
        .sceneVersion2,
      ]

    }

    public init() {}
  }

  /// Objects extracts features useful for identifying and localizing
  /// objects in natural images
  public struct Objects: Sendable {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var version: CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion = .invalid

    ///
    /// Stores the names of the output features according to the
    /// order of them being computed from the neural network, i.e.,
    /// the first element in the output is the earliest being
    /// computed, while the last is the latest being computed. In
    /// general, the order reflects the resolution of the feature.
    /// The earlier it is computed, the higher the feature resolution.
    public var output: [String] = []

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public enum ObjectsVersion: SwiftProtobuf.Enum, Swift.CaseIterable {
      public typealias RawValue = Int
      case invalid // = 0

      /// VERSION_1 is available on iOS,tvOS 14.0+, macOS 11.0+
      /// It uses a 299x299 input image and yields two multiarray
      /// features: one at high resolution of shape (288, 35, 35)
      /// the other at low resolution of shape (768, 17, 17)
      case objectsVersion1 // = 1
      case UNRECOGNIZED(Int)

      public init() {
        self = .invalid
      }

      public init?(rawValue: Int) {
        switch rawValue {
        case 0: self = .invalid
        case 1: self = .objectsVersion1
        default: self = .UNRECOGNIZED(rawValue)
        }
      }

      public var rawValue: Int {
        switch self {
        case .invalid: return 0
        case .objectsVersion1: return 1
        case .UNRECOGNIZED(let i): return i
        }
      }

      // The compiler won't synthesize support with the UNRECOGNIZED case.
      public static let allCases: [CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion] = [
        .invalid,
        .objectsVersion1,
      ]

    }

    public init() {}
  }

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification.CoreMLModels"

extension CoreMLModels_VisionFeaturePrint: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".VisionFeaturePrint"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    20: .same(proto: "scene"),
    21: .same(proto: "objects"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 20: try {
        var v: CoreMLModels_VisionFeaturePrint.Scene?
        var hadOneofValue = false
        if let current = self.visionFeaturePrintType {
          hadOneofValue = true
          if case .scene(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.visionFeaturePrintType = .scene(v)
        }
      }()
      case 21: try {
        var v: CoreMLModels_VisionFeaturePrint.Objects?
        var hadOneofValue = false
        if let current = self.visionFeaturePrintType {
          hadOneofValue = true
          if case .objects(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.visionFeaturePrintType = .objects(v)
        }
      }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    switch self.visionFeaturePrintType {
    case .scene?: try {
      guard case .scene(let v)? = self.visionFeaturePrintType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 20)
    }()
    case .objects?: try {
      guard case .objects(let v)? = self.visionFeaturePrintType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 21)
    }()
    case nil: break
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: CoreMLModels_VisionFeaturePrint, rhs: CoreMLModels_VisionFeaturePrint) -> Bool {
    if lhs.visionFeaturePrintType != rhs.visionFeaturePrintType {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreMLModels_VisionFeaturePrint.Scene: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = CoreMLModels_VisionFeaturePrint.protoMessageName + ".Scene"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "version"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularEnumField(value: &self.version) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.version != .invalid {
      try visitor.visitSingularEnumField(value: self.version, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: CoreMLModels_VisionFeaturePrint.Scene, rhs: CoreMLModels_VisionFeaturePrint.Scene) -> Bool {
    if lhs.version != rhs.version {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreMLModels_VisionFeaturePrint.Scene.SceneVersion: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "SCENE_VERSION_INVALID"),
    1: .same(proto: "SCENE_VERSION_1"),
    2: .same(proto: "SCENE_VERSION_2"),
  ]
}

extension CoreMLModels_VisionFeaturePrint.Objects: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = CoreMLModels_VisionFeaturePrint.protoMessageName + ".Objects"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "version"),
    100: .same(proto: "output"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularEnumField(value: &self.version) }()
      case 100: try { try decoder.decodeRepeatedStringField(value: &self.output) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.version != .invalid {
      try visitor.visitSingularEnumField(value: self.version, fieldNumber: 1)
    }
    if !self.output.isEmpty {
      try visitor.visitRepeatedStringField(value: self.output, fieldNumber: 100)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: CoreMLModels_VisionFeaturePrint.Objects, rhs: CoreMLModels_VisionFeaturePrint.Objects) -> Bool {
    if lhs.version != rhs.version {return false}
    if lhs.output != rhs.output {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "OBJECTS_VERSION_INVALID"),
    1: .same(proto: "OBJECTS_VERSION_1"),
  ]
}
