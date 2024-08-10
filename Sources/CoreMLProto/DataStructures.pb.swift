// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: DataStructures.proto
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
/// A mapping from a string
/// to a 64-bit integer.
public struct StringToInt64Map {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var map: Dictionary<String,Int64> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A mapping from a 64-bit integer
/// to a string.
public struct Int64ToStringMap {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var map: Dictionary<Int64,String> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A mapping from a string
/// to a double-precision floating point number.
public struct StringToDoubleMap {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var map: Dictionary<String,Double> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A mapping from a 64-bit integer
/// to a double-precision floating point number.
public struct Int64ToDoubleMap {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var map: Dictionary<Int64,Double> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A vector of strings.
public struct StringVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var vector: [String] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A vector of 64-bit integers.
public struct Int64Vector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var vector: [Int64] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A vector of floating point numbers.
public struct FloatVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var vector: [Float] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A vector of double-precision floating point numbers.
public struct DoubleVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var vector: [Double] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A range of int64 values
public struct Int64Range {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var minValue: Int64 = 0

  public var maxValue: Int64 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A set of int64 values
public struct Int64Set {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var values: [Int64] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A range of double values
public struct DoubleRange {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var minValue: Double = 0

  public var maxValue: Double = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///*
/// Precision/Recall curve.
///
/// The syntax comprises two tables, one to look up the confidence value threshold
/// for a given precision, and the other for a given recall.
///
/// Example:
/// ----------------------+----+----+----+----+----+----+----+----+----
/// precisionValues       | .1 | .2 | .3 | .4 | .5 | .6 | .7 |
/// precisionConfidence   | .0 | .0 | .0 | .0 | .1 | .3 | .4 |
/// ----------------------+----+----+----+----+----+----+----+----+----
///
/// ----------------------+----+----+----+----+----+----+----+----+----
/// recallValues          | .1 | .2 | .3 | .4 | .5 | .6 | .7 | .8 | .9
/// recallConfidence      | .7 | .6 | .5 | .4 | .3 | .3 | .2 | .1 | .0
/// ----------------------+----+----+----+----+----+----+----+----+----
///
/// The application expects that, when it filters out samples with
/// confidence threshold = 0.1, it gets precision = 0.5. Likewise,
/// with threshold = 0.2 it gets recall = 0.7.
///
/// The table must have only valid values; do not use `NaN`, `+/- INF`,
/// or negative values. The application is responsible for inter/extrapolating
/// approprate confidence threshold based on the application's specific need.
public struct PrecisionRecallCurve {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var precisionValues: FloatVector {
    get {return _precisionValues ?? FloatVector()}
    set {_precisionValues = newValue}
  }
  /// Returns true if `precisionValues` has been explicitly set.
  public var hasPrecisionValues: Bool {return self._precisionValues != nil}
  /// Clears the value of `precisionValues`. Subsequent reads from it will return its default value.
  public mutating func clearPrecisionValues() {self._precisionValues = nil}

  public var precisionConfidenceThresholds: FloatVector {
    get {return _precisionConfidenceThresholds ?? FloatVector()}
    set {_precisionConfidenceThresholds = newValue}
  }
  /// Returns true if `precisionConfidenceThresholds` has been explicitly set.
  public var hasPrecisionConfidenceThresholds: Bool {return self._precisionConfidenceThresholds != nil}
  /// Clears the value of `precisionConfidenceThresholds`. Subsequent reads from it will return its default value.
  public mutating func clearPrecisionConfidenceThresholds() {self._precisionConfidenceThresholds = nil}

  public var recallValues: FloatVector {
    get {return _recallValues ?? FloatVector()}
    set {_recallValues = newValue}
  }
  /// Returns true if `recallValues` has been explicitly set.
  public var hasRecallValues: Bool {return self._recallValues != nil}
  /// Clears the value of `recallValues`. Subsequent reads from it will return its default value.
  public mutating func clearRecallValues() {self._recallValues = nil}

  public var recallConfidenceThresholds: FloatVector {
    get {return _recallConfidenceThresholds ?? FloatVector()}
    set {_recallConfidenceThresholds = newValue}
  }
  /// Returns true if `recallConfidenceThresholds` has been explicitly set.
  public var hasRecallConfidenceThresholds: Bool {return self._recallConfidenceThresholds != nil}
  /// Clears the value of `recallConfidenceThresholds`. Subsequent reads from it will return its default value.
  public mutating func clearRecallConfidenceThresholds() {self._recallConfidenceThresholds = nil}

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _precisionValues: FloatVector? = nil
  fileprivate var _precisionConfidenceThresholds: FloatVector? = nil
  fileprivate var _recallValues: FloatVector? = nil
  fileprivate var _recallConfidenceThresholds: FloatVector? = nil
}

#if swift(>=5.5) && canImport(_Concurrency)
extension StringToInt64Map: @unchecked Sendable {}
extension Int64ToStringMap: @unchecked Sendable {}
extension StringToDoubleMap: @unchecked Sendable {}
extension Int64ToDoubleMap: @unchecked Sendable {}
extension StringVector: @unchecked Sendable {}
extension Int64Vector: @unchecked Sendable {}
extension FloatVector: @unchecked Sendable {}
extension DoubleVector: @unchecked Sendable {}
extension Int64Range: @unchecked Sendable {}
extension Int64Set: @unchecked Sendable {}
extension DoubleRange: @unchecked Sendable {}
extension PrecisionRecallCurve: @unchecked Sendable {}
#endif  // swift(>=5.5) && canImport(_Concurrency)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension StringToInt64Map: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".StringToInt64Map"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufInt64>.self, value: &self.map) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufInt64>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: StringToInt64Map, rhs: StringToInt64Map) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Int64ToStringMap: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Int64ToStringMap"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufString>.self, value: &self.map) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufString>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Int64ToStringMap, rhs: Int64ToStringMap) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension StringToDoubleMap: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".StringToDoubleMap"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufDouble>.self, value: &self.map) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufDouble>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: StringToDoubleMap, rhs: StringToDoubleMap) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Int64ToDoubleMap: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Int64ToDoubleMap"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufDouble>.self, value: &self.map) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufDouble>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Int64ToDoubleMap, rhs: Int64ToDoubleMap) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension StringVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".StringVector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedStringField(value: &self.vector) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitRepeatedStringField(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: StringVector, rhs: StringVector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Int64Vector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Int64Vector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedInt64Field(value: &self.vector) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitPackedInt64Field(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Int64Vector, rhs: Int64Vector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension FloatVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".FloatVector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedFloatField(value: &self.vector) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitPackedFloatField(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: FloatVector, rhs: FloatVector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension DoubleVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".DoubleVector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedDoubleField(value: &self.vector) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitPackedDoubleField(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: DoubleVector, rhs: DoubleVector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Int64Range: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Int64Range"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "minValue"),
    2: .same(proto: "maxValue"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt64Field(value: &self.minValue) }()
      case 2: try { try decoder.decodeSingularInt64Field(value: &self.maxValue) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.minValue != 0 {
      try visitor.visitSingularInt64Field(value: self.minValue, fieldNumber: 1)
    }
    if self.maxValue != 0 {
      try visitor.visitSingularInt64Field(value: self.maxValue, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Int64Range, rhs: Int64Range) -> Bool {
    if lhs.minValue != rhs.minValue {return false}
    if lhs.maxValue != rhs.maxValue {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Int64Set: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Int64Set"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "values"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedInt64Field(value: &self.values) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.values.isEmpty {
      try visitor.visitPackedInt64Field(value: self.values, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Int64Set, rhs: Int64Set) -> Bool {
    if lhs.values != rhs.values {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension DoubleRange: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".DoubleRange"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "minValue"),
    2: .same(proto: "maxValue"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularDoubleField(value: &self.minValue) }()
      case 2: try { try decoder.decodeSingularDoubleField(value: &self.maxValue) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.minValue != 0 {
      try visitor.visitSingularDoubleField(value: self.minValue, fieldNumber: 1)
    }
    if self.maxValue != 0 {
      try visitor.visitSingularDoubleField(value: self.maxValue, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: DoubleRange, rhs: DoubleRange) -> Bool {
    if lhs.minValue != rhs.minValue {return false}
    if lhs.maxValue != rhs.maxValue {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension PrecisionRecallCurve: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".PrecisionRecallCurve"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "precisionValues"),
    2: .same(proto: "precisionConfidenceThresholds"),
    3: .same(proto: "recallValues"),
    4: .same(proto: "recallConfidenceThresholds"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._precisionValues) }()
      case 2: try { try decoder.decodeSingularMessageField(value: &self._precisionConfidenceThresholds) }()
      case 3: try { try decoder.decodeSingularMessageField(value: &self._recallValues) }()
      case 4: try { try decoder.decodeSingularMessageField(value: &self._recallConfidenceThresholds) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    try { if let v = self._precisionValues {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    try { if let v = self._precisionConfidenceThresholds {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    } }()
    try { if let v = self._recallValues {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    } }()
    try { if let v = self._recallConfidenceThresholds {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: PrecisionRecallCurve, rhs: PrecisionRecallCurve) -> Bool {
    if lhs._precisionValues != rhs._precisionValues {return false}
    if lhs._precisionConfidenceThresholds != rhs._precisionConfidenceThresholds {return false}
    if lhs._recallValues != rhs._recallValues {return false}
    if lhs._recallConfidenceThresholds != rhs._recallConfidenceThresholds {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
