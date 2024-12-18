// DO NOT EDIT.
// swift-format-ignore-file
// swiftlint:disable all
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: Normalizer.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
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
/// A normalization preprocessor.
public struct Normalizer: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var normType: Normalizer.NormType = .lmax

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  ///
  /// There are three normalization modes,
  /// which have the corresponding formulas:
  ///
  /// Max
  ///     .. math::
  ///         max(x_i)
  ///
  /// L1
  ///     .. math::
  ///         z = ||x||_1 = \sum_{i=1}^{n} |x_i|
  ///
  /// L2
  ///     .. math::
  ///         z = ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
  public enum NormType: SwiftProtobuf.Enum, Swift.CaseIterable {
    public typealias RawValue = Int
    case lmax // = 0
    case l1 // = 1
    case l2 // = 2
    case UNRECOGNIZED(Int)

    public init() {
      self = .lmax
    }

    public init?(rawValue: Int) {
      switch rawValue {
      case 0: self = .lmax
      case 1: self = .l1
      case 2: self = .l2
      default: self = .UNRECOGNIZED(rawValue)
      }
    }

    public var rawValue: Int {
      switch self {
      case .lmax: return 0
      case .l1: return 1
      case .l2: return 2
      case .UNRECOGNIZED(let i): return i
      }
    }

    // The compiler won't synthesize support with the UNRECOGNIZED case.
    public static let allCases: [Normalizer.NormType] = [
      .lmax,
      .l1,
      .l2,
    ]

  }

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension Normalizer: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Normalizer"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "normType"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularEnumField(value: &self.normType) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.normType != .lmax {
      try visitor.visitSingularEnumField(value: self.normType, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Normalizer, rhs: Normalizer) -> Bool {
    if lhs.normType != rhs.normType {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Normalizer.NormType: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "LMax"),
    1: .same(proto: "L1"),
    2: .same(proto: "L2"),
  ]
}
