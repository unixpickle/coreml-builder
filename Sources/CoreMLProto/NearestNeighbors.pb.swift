// DO NOT EDIT.
// swift-format-ignore-file
// swiftlint:disable all
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: NearestNeighbors.proto
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
/// A k-Nearest-Neighbor classifier
public struct KNearestNeighborsClassifier: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///
  /// The "core" nearest neighbor model attributes.
  public var nearestNeighborsIndex: NearestNeighborsIndex {
    get {return _nearestNeighborsIndex ?? NearestNeighborsIndex()}
    set {_nearestNeighborsIndex = newValue}
  }
  /// Returns true if `nearestNeighborsIndex` has been explicitly set.
  public var hasNearestNeighborsIndex: Bool {return self._nearestNeighborsIndex != nil}
  /// Clears the value of `nearestNeighborsIndex`. Subsequent reads from it will return its default value.
  public mutating func clearNearestNeighborsIndex() {self._nearestNeighborsIndex = nil}

  ///
  /// Number of neighbors to use for classification.
  public var numberOfNeighbors: Int64Parameter {
    get {return _numberOfNeighbors ?? Int64Parameter()}
    set {_numberOfNeighbors = newValue}
  }
  /// Returns true if `numberOfNeighbors` has been explicitly set.
  public var hasNumberOfNeighbors: Bool {return self._numberOfNeighbors != nil}
  /// Clears the value of `numberOfNeighbors`. Subsequent reads from it will return its default value.
  public mutating func clearNumberOfNeighbors() {self._numberOfNeighbors = nil}

  ///
  /// Type of labels supported by the model. Currently supports String or Int64
  /// labels.
  public var classLabels: KNearestNeighborsClassifier.OneOf_ClassLabels? = nil

  public var stringClassLabels: StringVector {
    get {
      if case .stringClassLabels(let v)? = classLabels {return v}
      return StringVector()
    }
    set {classLabels = .stringClassLabels(newValue)}
  }

  public var int64ClassLabels: Int64Vector {
    get {
      if case .int64ClassLabels(let v)? = classLabels {return v}
      return Int64Vector()
    }
    set {classLabels = .int64ClassLabels(newValue)}
  }

  ///
  /// Default value of class label (useful when prediction is called on an empty kNN classifier)
  public var defaultClassLabel: KNearestNeighborsClassifier.OneOf_DefaultClassLabel? = nil

  public var defaultStringLabel: String {
    get {
      if case .defaultStringLabel(let v)? = defaultClassLabel {return v}
      return String()
    }
    set {defaultClassLabel = .defaultStringLabel(newValue)}
  }

  public var defaultInt64Label: Int64 {
    get {
      if case .defaultInt64Label(let v)? = defaultClassLabel {return v}
      return 0
    }
    set {defaultClassLabel = .defaultInt64Label(newValue)}
  }

  ///
  /// Weighting scheme to be used when computing the majority label of a 
  /// new data point.
  public var weightingScheme: KNearestNeighborsClassifier.OneOf_WeightingScheme? = nil

  public var uniformWeighting: UniformWeighting {
    get {
      if case .uniformWeighting(let v)? = weightingScheme {return v}
      return UniformWeighting()
    }
    set {weightingScheme = .uniformWeighting(newValue)}
  }

  public var inverseDistanceWeighting: InverseDistanceWeighting {
    get {
      if case .inverseDistanceWeighting(let v)? = weightingScheme {return v}
      return InverseDistanceWeighting()
    }
    set {weightingScheme = .inverseDistanceWeighting(newValue)}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  ///
  /// Type of labels supported by the model. Currently supports String or Int64
  /// labels.
  public enum OneOf_ClassLabels: Equatable, Sendable {
    case stringClassLabels(StringVector)
    case int64ClassLabels(Int64Vector)

  }

  ///
  /// Default value of class label (useful when prediction is called on an empty kNN classifier)
  public enum OneOf_DefaultClassLabel: Equatable, Sendable {
    case defaultStringLabel(String)
    case defaultInt64Label(Int64)

  }

  ///
  /// Weighting scheme to be used when computing the majority label of a 
  /// new data point.
  public enum OneOf_WeightingScheme: Equatable, Sendable {
    case uniformWeighting(UniformWeighting)
    case inverseDistanceWeighting(InverseDistanceWeighting)

  }

  public init() {}

  fileprivate var _nearestNeighborsIndex: NearestNeighborsIndex? = nil
  fileprivate var _numberOfNeighbors: Int64Parameter? = nil
}

///
/// The "core" attributes of a Nearest Neighbors model.
public struct NearestNeighborsIndex: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// 
  /// Number of dimensions of the input data.
  public var numberOfDimensions: Int32 = 0

  ///
  /// Vector of floating point data that makes up the model. Each data point must have 'numberOfDimensions'
  /// dimensions.
  public var floatSamples: [FloatVector] = []

  /// 
  /// Backing data structure for the Nearest Neighbors Index. Currently supports 
  /// a linear index or a kd-tree index.
  public var indexType: NearestNeighborsIndex.OneOf_IndexType? = nil

  public var linearIndex: LinearIndex {
    get {
      if case .linearIndex(let v)? = indexType {return v}
      return LinearIndex()
    }
    set {indexType = .linearIndex(newValue)}
  }

  public var singleKdTreeIndex: SingleKdTreeIndex {
    get {
      if case .singleKdTreeIndex(let v)? = indexType {return v}
      return SingleKdTreeIndex()
    }
    set {indexType = .singleKdTreeIndex(newValue)}
  }

  /// 
  /// Distance function to be used to find neighbors. Currently only Squared Euclidean
  /// Distance is supported.
  public var distanceFunction: NearestNeighborsIndex.OneOf_DistanceFunction? = nil

  public var squaredEuclideanDistance: SquaredEuclideanDistance {
    get {
      if case .squaredEuclideanDistance(let v)? = distanceFunction {return v}
      return SquaredEuclideanDistance()
    }
    set {distanceFunction = .squaredEuclideanDistance(newValue)}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  /// 
  /// Backing data structure for the Nearest Neighbors Index. Currently supports 
  /// a linear index or a kd-tree index.
  public enum OneOf_IndexType: Equatable, Sendable {
    case linearIndex(LinearIndex)
    case singleKdTreeIndex(SingleKdTreeIndex)

  }

  /// 
  /// Distance function to be used to find neighbors. Currently only Squared Euclidean
  /// Distance is supported.
  public enum OneOf_DistanceFunction: Equatable, Sendable {
    case squaredEuclideanDistance(SquaredEuclideanDistance)

  }

  public init() {}
}

///
/// Specifies a uniform weighting scheme (i.e. each neighbor receives equal
/// voting power).
public struct UniformWeighting: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// Specifies a inverse-distance weighting scheme (i.e. closest neighbors receives higher
/// voting power). A nearest neighbor with highest sum of (1 / distance) is picked.
public struct InverseDistanceWeighting: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// Specifies a flat index of data points to be searched by brute force.
public struct LinearIndex: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// Specifies a kd-tree backend for the nearest neighbors model.
public struct SingleKdTreeIndex: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///
  /// Number of data points contained within a leaf node of the kd-tree.
  public var leafSize: Int32 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// Specifies the Squared Euclidean Distance function.
public struct SquaredEuclideanDistance: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension KNearestNeighborsClassifier: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".KNearestNeighborsClassifier"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "nearestNeighborsIndex"),
    3: .same(proto: "numberOfNeighbors"),
    100: .same(proto: "stringClassLabels"),
    101: .same(proto: "int64ClassLabels"),
    110: .same(proto: "defaultStringLabel"),
    111: .same(proto: "defaultInt64Label"),
    200: .same(proto: "uniformWeighting"),
    210: .same(proto: "inverseDistanceWeighting"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._nearestNeighborsIndex) }()
      case 3: try { try decoder.decodeSingularMessageField(value: &self._numberOfNeighbors) }()
      case 100: try {
        var v: StringVector?
        var hadOneofValue = false
        if let current = self.classLabels {
          hadOneofValue = true
          if case .stringClassLabels(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.classLabels = .stringClassLabels(v)
        }
      }()
      case 101: try {
        var v: Int64Vector?
        var hadOneofValue = false
        if let current = self.classLabels {
          hadOneofValue = true
          if case .int64ClassLabels(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.classLabels = .int64ClassLabels(v)
        }
      }()
      case 110: try {
        var v: String?
        try decoder.decodeSingularStringField(value: &v)
        if let v = v {
          if self.defaultClassLabel != nil {try decoder.handleConflictingOneOf()}
          self.defaultClassLabel = .defaultStringLabel(v)
        }
      }()
      case 111: try {
        var v: Int64?
        try decoder.decodeSingularInt64Field(value: &v)
        if let v = v {
          if self.defaultClassLabel != nil {try decoder.handleConflictingOneOf()}
          self.defaultClassLabel = .defaultInt64Label(v)
        }
      }()
      case 200: try {
        var v: UniformWeighting?
        var hadOneofValue = false
        if let current = self.weightingScheme {
          hadOneofValue = true
          if case .uniformWeighting(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.weightingScheme = .uniformWeighting(v)
        }
      }()
      case 210: try {
        var v: InverseDistanceWeighting?
        var hadOneofValue = false
        if let current = self.weightingScheme {
          hadOneofValue = true
          if case .inverseDistanceWeighting(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.weightingScheme = .inverseDistanceWeighting(v)
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
    try { if let v = self._nearestNeighborsIndex {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    try { if let v = self._numberOfNeighbors {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    } }()
    switch self.classLabels {
    case .stringClassLabels?: try {
      guard case .stringClassLabels(let v)? = self.classLabels else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 100)
    }()
    case .int64ClassLabels?: try {
      guard case .int64ClassLabels(let v)? = self.classLabels else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 101)
    }()
    case nil: break
    }
    switch self.defaultClassLabel {
    case .defaultStringLabel?: try {
      guard case .defaultStringLabel(let v)? = self.defaultClassLabel else { preconditionFailure() }
      try visitor.visitSingularStringField(value: v, fieldNumber: 110)
    }()
    case .defaultInt64Label?: try {
      guard case .defaultInt64Label(let v)? = self.defaultClassLabel else { preconditionFailure() }
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 111)
    }()
    case nil: break
    }
    switch self.weightingScheme {
    case .uniformWeighting?: try {
      guard case .uniformWeighting(let v)? = self.weightingScheme else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 200)
    }()
    case .inverseDistanceWeighting?: try {
      guard case .inverseDistanceWeighting(let v)? = self.weightingScheme else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 210)
    }()
    case nil: break
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: KNearestNeighborsClassifier, rhs: KNearestNeighborsClassifier) -> Bool {
    if lhs._nearestNeighborsIndex != rhs._nearestNeighborsIndex {return false}
    if lhs._numberOfNeighbors != rhs._numberOfNeighbors {return false}
    if lhs.classLabels != rhs.classLabels {return false}
    if lhs.defaultClassLabel != rhs.defaultClassLabel {return false}
    if lhs.weightingScheme != rhs.weightingScheme {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension NearestNeighborsIndex: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".NearestNeighborsIndex"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "numberOfDimensions"),
    2: .same(proto: "floatSamples"),
    100: .same(proto: "linearIndex"),
    110: .same(proto: "singleKdTreeIndex"),
    200: .same(proto: "squaredEuclideanDistance"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.numberOfDimensions) }()
      case 2: try { try decoder.decodeRepeatedMessageField(value: &self.floatSamples) }()
      case 100: try {
        var v: LinearIndex?
        var hadOneofValue = false
        if let current = self.indexType {
          hadOneofValue = true
          if case .linearIndex(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.indexType = .linearIndex(v)
        }
      }()
      case 110: try {
        var v: SingleKdTreeIndex?
        var hadOneofValue = false
        if let current = self.indexType {
          hadOneofValue = true
          if case .singleKdTreeIndex(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.indexType = .singleKdTreeIndex(v)
        }
      }()
      case 200: try {
        var v: SquaredEuclideanDistance?
        var hadOneofValue = false
        if let current = self.distanceFunction {
          hadOneofValue = true
          if case .squaredEuclideanDistance(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.distanceFunction = .squaredEuclideanDistance(v)
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
    if self.numberOfDimensions != 0 {
      try visitor.visitSingularInt32Field(value: self.numberOfDimensions, fieldNumber: 1)
    }
    if !self.floatSamples.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.floatSamples, fieldNumber: 2)
    }
    switch self.indexType {
    case .linearIndex?: try {
      guard case .linearIndex(let v)? = self.indexType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 100)
    }()
    case .singleKdTreeIndex?: try {
      guard case .singleKdTreeIndex(let v)? = self.indexType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 110)
    }()
    case nil: break
    }
    try { if case .squaredEuclideanDistance(let v)? = self.distanceFunction {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 200)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: NearestNeighborsIndex, rhs: NearestNeighborsIndex) -> Bool {
    if lhs.numberOfDimensions != rhs.numberOfDimensions {return false}
    if lhs.floatSamples != rhs.floatSamples {return false}
    if lhs.indexType != rhs.indexType {return false}
    if lhs.distanceFunction != rhs.distanceFunction {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension UniformWeighting: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".UniformWeighting"
  public static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    // Load everything into unknown fields
    while try decoder.nextFieldNumber() != nil {}
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: UniformWeighting, rhs: UniformWeighting) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension InverseDistanceWeighting: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".InverseDistanceWeighting"
  public static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    // Load everything into unknown fields
    while try decoder.nextFieldNumber() != nil {}
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: InverseDistanceWeighting, rhs: InverseDistanceWeighting) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension LinearIndex: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".LinearIndex"
  public static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    // Load everything into unknown fields
    while try decoder.nextFieldNumber() != nil {}
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: LinearIndex, rhs: LinearIndex) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SingleKdTreeIndex: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SingleKdTreeIndex"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "leafSize"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.leafSize) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.leafSize != 0 {
      try visitor.visitSingularInt32Field(value: self.leafSize, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SingleKdTreeIndex, rhs: SingleKdTreeIndex) -> Bool {
    if lhs.leafSize != rhs.leafSize {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SquaredEuclideanDistance: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SquaredEuclideanDistance"
  public static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    // Load everything into unknown fields
    while try decoder.nextFieldNumber() != nil {}
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SquaredEuclideanDistance, rhs: SquaredEuclideanDistance) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
