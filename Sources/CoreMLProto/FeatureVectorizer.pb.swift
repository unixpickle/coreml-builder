// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: FeatureVectorizer.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
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
/// A FeatureVectorizer puts one or more features into a single array.
///
/// The ordering of features in the output array is determined by
/// ``inputList``.
///
/// ``inputDimensions`` is a zero based index.
public struct FeatureVectorizer {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var inputList: [FeatureVectorizer.InputColumn] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public struct InputColumn {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var inputColumn: String = String()

    public var inputDimensions: UInt64 = 0

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}
  }

  public init() {}
}

#if swift(>=5.5) && canImport(_Concurrency)
extension FeatureVectorizer: @unchecked Sendable {}
extension FeatureVectorizer.InputColumn: @unchecked Sendable {}
#endif  // swift(>=5.5) && canImport(_Concurrency)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension FeatureVectorizer: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".FeatureVectorizer"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "inputList"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedMessageField(value: &self.inputList) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.inputList.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.inputList, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: FeatureVectorizer, rhs: FeatureVectorizer) -> Bool {
    if lhs.inputList != rhs.inputList {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension FeatureVectorizer.InputColumn: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = FeatureVectorizer.protoMessageName + ".InputColumn"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "inputColumn"),
    2: .same(proto: "inputDimensions"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularStringField(value: &self.inputColumn) }()
      case 2: try { try decoder.decodeSingularUInt64Field(value: &self.inputDimensions) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.inputColumn.isEmpty {
      try visitor.visitSingularStringField(value: self.inputColumn, fieldNumber: 1)
    }
    if self.inputDimensions != 0 {
      try visitor.visitSingularUInt64Field(value: self.inputDimensions, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: FeatureVectorizer.InputColumn, rhs: FeatureVectorizer.InputColumn) -> Bool {
    if lhs.inputColumn != rhs.inputColumn {return false}
    if lhs.inputDimensions != rhs.inputDimensions {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
