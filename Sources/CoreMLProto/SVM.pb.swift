// DO NOT EDIT.
// swift-format-ignore-file
// swiftlint:disable all
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: SVM.proto
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
/// A linear kernel.
///
/// This function has the following formula:
///
/// .. math::
///     K(\boldsymbol{x}, \boldsymbol{x'}) = \boldsymbol{x}^T \boldsymbol{x'}
public struct LinearKernel: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A Gaussian radial basis function (RBF) kernel.
///
/// This function has the following formula:
///
/// .. math::
///     K(\boldsymbol{x}, \boldsymbol{x'}) = \
///          \exp(-\gamma || \boldsymbol{x} - \boldsymbol{x'} ||^2 )
public struct RBFKernel: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var gamma: Double = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A polynomial kernel.
///
/// This function has the following formula:
///
/// .. math::
///     K(\boldsymbol{x}, \boldsymbol{x'}) = \
///           (\gamma \boldsymbol{x}^T \boldsymbol{x'} + c)^{degree}
public struct PolyKernel: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var degree: Int32 = 0

  public var c: Double = 0

  public var gamma: Double = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A sigmoid kernel.
///
/// This function has the following formula:
///
/// .. math::
///     K(\boldsymbol{x}, \boldsymbol{x'}) = \
///           \tanh(\gamma \boldsymbol{x}^T \boldsymbol{x'} + c)
public struct SigmoidKernel: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var gamma: Double = 0

  public var c: Double = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A kernel.
public struct Kernel: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var kernel: Kernel.OneOf_Kernel? = nil

  public var linearKernel: LinearKernel {
    get {
      if case .linearKernel(let v)? = kernel {return v}
      return LinearKernel()
    }
    set {kernel = .linearKernel(newValue)}
  }

  public var rbfKernel: RBFKernel {
    get {
      if case .rbfKernel(let v)? = kernel {return v}
      return RBFKernel()
    }
    set {kernel = .rbfKernel(newValue)}
  }

  public var polyKernel: PolyKernel {
    get {
      if case .polyKernel(let v)? = kernel {return v}
      return PolyKernel()
    }
    set {kernel = .polyKernel(newValue)}
  }

  public var sigmoidKernel: SigmoidKernel {
    get {
      if case .sigmoidKernel(let v)? = kernel {return v}
      return SigmoidKernel()
    }
    set {kernel = .sigmoidKernel(newValue)}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public enum OneOf_Kernel: Equatable, Sendable {
    case linearKernel(LinearKernel)
    case rbfKernel(RBFKernel)
    case polyKernel(PolyKernel)
    case sigmoidKernel(SigmoidKernel)

  }

  public init() {}
}

///
/// A sparse node.
public struct SparseNode: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// 1-based indexes, like libsvm
  public var index: Int32 = 0

  public var value: Double = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A sparse vector.
public struct SparseVector: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var nodes: [SparseNode] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// One or more sparse support vectors.
public struct SparseSupportVectors: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var vectors: [SparseVector] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A dense vector.
public struct DenseVector: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var values: [Double] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// One or more dense support vectors.
public struct DenseSupportVectors: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var vectors: [DenseVector] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// One or more coefficients.
public struct Coefficients: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var alpha: [Double] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

///
/// A support vector regressor.
public struct SupportVectorRegressor: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var kernel: Kernel {
    get {return _kernel ?? Kernel()}
    set {_kernel = newValue}
  }
  /// Returns true if `kernel` has been explicitly set.
  public var hasKernel: Bool {return self._kernel != nil}
  /// Clears the value of `kernel`. Subsequent reads from it will return its default value.
  public mutating func clearKernel() {self._kernel = nil}

  /// Support vectors, either sparse or dense format
  public var supportVectors: SupportVectorRegressor.OneOf_SupportVectors? = nil

  public var sparseSupportVectors: SparseSupportVectors {
    get {
      if case .sparseSupportVectors(let v)? = supportVectors {return v}
      return SparseSupportVectors()
    }
    set {supportVectors = .sparseSupportVectors(newValue)}
  }

  public var denseSupportVectors: DenseSupportVectors {
    get {
      if case .denseSupportVectors(let v)? = supportVectors {return v}
      return DenseSupportVectors()
    }
    set {supportVectors = .denseSupportVectors(newValue)}
  }

  /// Coefficients, one for each support vector
  public var coefficients: Coefficients {
    get {return _coefficients ?? Coefficients()}
    set {_coefficients = newValue}
  }
  /// Returns true if `coefficients` has been explicitly set.
  public var hasCoefficients: Bool {return self._coefficients != nil}
  /// Clears the value of `coefficients`. Subsequent reads from it will return its default value.
  public mutating func clearCoefficients() {self._coefficients = nil}

  public var rho: Double = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  /// Support vectors, either sparse or dense format
  public enum OneOf_SupportVectors: Equatable, Sendable {
    case sparseSupportVectors(SparseSupportVectors)
    case denseSupportVectors(DenseSupportVectors)

  }

  public init() {}

  fileprivate var _kernel: Kernel? = nil
  fileprivate var _coefficients: Coefficients? = nil
}

///
/// A support vector classifier
public struct SupportVectorClassifier: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var kernel: Kernel {
    get {return _kernel ?? Kernel()}
    set {_kernel = newValue}
  }
  /// Returns true if `kernel` has been explicitly set.
  public var hasKernel: Bool {return self._kernel != nil}
  /// Clears the value of `kernel`. Subsequent reads from it will return its default value.
  public mutating func clearKernel() {self._kernel = nil}

  ///
  /// The number of support vectors for each class.
  public var numberOfSupportVectorsPerClass: [Int32] = []

  ///
  /// The support vectors, in either sparse or dense format.
  public var supportVectors: SupportVectorClassifier.OneOf_SupportVectors? = nil

  public var sparseSupportVectors: SparseSupportVectors {
    get {
      if case .sparseSupportVectors(let v)? = supportVectors {return v}
      return SparseSupportVectors()
    }
    set {supportVectors = .sparseSupportVectors(newValue)}
  }

  public var denseSupportVectors: DenseSupportVectors {
    get {
      if case .denseSupportVectors(let v)? = supportVectors {return v}
      return DenseSupportVectors()
    }
    set {supportVectors = .denseSupportVectors(newValue)}
  }

  ///
  /// The coefficients, essentially a two dimensional array of
  /// size: (numberOfClasses-1) by (total number of support vectors)
  public var coefficients: [Coefficients] = []

  ///
  /// Constants for decision function,
  /// with K*(K-1) / 2 elements,
  /// where K is the number of classes.
  public var rho: [Double] = []

  ///
  /// Pairwise probability information for A vs B classifier.
  /// Total of K*(K-1)/2 elements where K is the number of classes.
  /// These fields are optional,
  /// and only required if you want probabilities or multi class predictions.
  public var probA: [Double] = []

  public var probB: [Double] = []

  ///
  /// Class label mapping.
  public var classLabels: SupportVectorClassifier.OneOf_ClassLabels? = nil

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

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  ///
  /// The support vectors, in either sparse or dense format.
  public enum OneOf_SupportVectors: Equatable, Sendable {
    case sparseSupportVectors(SparseSupportVectors)
    case denseSupportVectors(DenseSupportVectors)

  }

  ///
  /// Class label mapping.
  public enum OneOf_ClassLabels: Equatable, Sendable {
    case stringClassLabels(StringVector)
    case int64ClassLabels(Int64Vector)

  }

  public init() {}

  fileprivate var _kernel: Kernel? = nil
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension LinearKernel: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".LinearKernel"
  public static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    // Load everything into unknown fields
    while try decoder.nextFieldNumber() != nil {}
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: LinearKernel, rhs: LinearKernel) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension RBFKernel: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".RBFKernel"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "gamma"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularDoubleField(value: &self.gamma) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.gamma.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.gamma, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: RBFKernel, rhs: RBFKernel) -> Bool {
    if lhs.gamma != rhs.gamma {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension PolyKernel: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".PolyKernel"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "degree"),
    2: .same(proto: "c"),
    3: .same(proto: "gamma"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.degree) }()
      case 2: try { try decoder.decodeSingularDoubleField(value: &self.c) }()
      case 3: try { try decoder.decodeSingularDoubleField(value: &self.gamma) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.degree != 0 {
      try visitor.visitSingularInt32Field(value: self.degree, fieldNumber: 1)
    }
    if self.c.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.c, fieldNumber: 2)
    }
    if self.gamma.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.gamma, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: PolyKernel, rhs: PolyKernel) -> Bool {
    if lhs.degree != rhs.degree {return false}
    if lhs.c != rhs.c {return false}
    if lhs.gamma != rhs.gamma {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SigmoidKernel: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SigmoidKernel"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "gamma"),
    2: .same(proto: "c"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularDoubleField(value: &self.gamma) }()
      case 2: try { try decoder.decodeSingularDoubleField(value: &self.c) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.gamma.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.gamma, fieldNumber: 1)
    }
    if self.c.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.c, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SigmoidKernel, rhs: SigmoidKernel) -> Bool {
    if lhs.gamma != rhs.gamma {return false}
    if lhs.c != rhs.c {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Kernel: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Kernel"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "linearKernel"),
    2: .same(proto: "rbfKernel"),
    3: .same(proto: "polyKernel"),
    4: .same(proto: "sigmoidKernel"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try {
        var v: LinearKernel?
        var hadOneofValue = false
        if let current = self.kernel {
          hadOneofValue = true
          if case .linearKernel(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.kernel = .linearKernel(v)
        }
      }()
      case 2: try {
        var v: RBFKernel?
        var hadOneofValue = false
        if let current = self.kernel {
          hadOneofValue = true
          if case .rbfKernel(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.kernel = .rbfKernel(v)
        }
      }()
      case 3: try {
        var v: PolyKernel?
        var hadOneofValue = false
        if let current = self.kernel {
          hadOneofValue = true
          if case .polyKernel(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.kernel = .polyKernel(v)
        }
      }()
      case 4: try {
        var v: SigmoidKernel?
        var hadOneofValue = false
        if let current = self.kernel {
          hadOneofValue = true
          if case .sigmoidKernel(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.kernel = .sigmoidKernel(v)
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
    switch self.kernel {
    case .linearKernel?: try {
      guard case .linearKernel(let v)? = self.kernel else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    }()
    case .rbfKernel?: try {
      guard case .rbfKernel(let v)? = self.kernel else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    }()
    case .polyKernel?: try {
      guard case .polyKernel(let v)? = self.kernel else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    }()
    case .sigmoidKernel?: try {
      guard case .sigmoidKernel(let v)? = self.kernel else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
    }()
    case nil: break
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Kernel, rhs: Kernel) -> Bool {
    if lhs.kernel != rhs.kernel {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SparseNode: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SparseNode"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "index"),
    2: .same(proto: "value"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.index) }()
      case 2: try { try decoder.decodeSingularDoubleField(value: &self.value) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.index != 0 {
      try visitor.visitSingularInt32Field(value: self.index, fieldNumber: 1)
    }
    if self.value.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.value, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SparseNode, rhs: SparseNode) -> Bool {
    if lhs.index != rhs.index {return false}
    if lhs.value != rhs.value {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SparseVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SparseVector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "nodes"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedMessageField(value: &self.nodes) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.nodes.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodes, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SparseVector, rhs: SparseVector) -> Bool {
    if lhs.nodes != rhs.nodes {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SparseSupportVectors: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SparseSupportVectors"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vectors"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedMessageField(value: &self.vectors) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vectors.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.vectors, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SparseSupportVectors, rhs: SparseSupportVectors) -> Bool {
    if lhs.vectors != rhs.vectors {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension DenseVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".DenseVector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "values"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedDoubleField(value: &self.values) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.values.isEmpty {
      try visitor.visitPackedDoubleField(value: self.values, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: DenseVector, rhs: DenseVector) -> Bool {
    if lhs.values != rhs.values {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension DenseSupportVectors: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".DenseSupportVectors"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vectors"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedMessageField(value: &self.vectors) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vectors.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.vectors, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: DenseSupportVectors, rhs: DenseSupportVectors) -> Bool {
    if lhs.vectors != rhs.vectors {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Coefficients: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Coefficients"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "alpha"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedDoubleField(value: &self.alpha) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.alpha.isEmpty {
      try visitor.visitPackedDoubleField(value: self.alpha, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: Coefficients, rhs: Coefficients) -> Bool {
    if lhs.alpha != rhs.alpha {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SupportVectorRegressor: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SupportVectorRegressor"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "kernel"),
    2: .same(proto: "sparseSupportVectors"),
    3: .same(proto: "denseSupportVectors"),
    4: .same(proto: "coefficients"),
    5: .same(proto: "rho"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._kernel) }()
      case 2: try {
        var v: SparseSupportVectors?
        var hadOneofValue = false
        if let current = self.supportVectors {
          hadOneofValue = true
          if case .sparseSupportVectors(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.supportVectors = .sparseSupportVectors(v)
        }
      }()
      case 3: try {
        var v: DenseSupportVectors?
        var hadOneofValue = false
        if let current = self.supportVectors {
          hadOneofValue = true
          if case .denseSupportVectors(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.supportVectors = .denseSupportVectors(v)
        }
      }()
      case 4: try { try decoder.decodeSingularMessageField(value: &self._coefficients) }()
      case 5: try { try decoder.decodeSingularDoubleField(value: &self.rho) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    try { if let v = self._kernel {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    switch self.supportVectors {
    case .sparseSupportVectors?: try {
      guard case .sparseSupportVectors(let v)? = self.supportVectors else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    }()
    case .denseSupportVectors?: try {
      guard case .denseSupportVectors(let v)? = self.supportVectors else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    }()
    case nil: break
    }
    try { if let v = self._coefficients {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
    } }()
    if self.rho.bitPattern != 0 {
      try visitor.visitSingularDoubleField(value: self.rho, fieldNumber: 5)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SupportVectorRegressor, rhs: SupportVectorRegressor) -> Bool {
    if lhs._kernel != rhs._kernel {return false}
    if lhs.supportVectors != rhs.supportVectors {return false}
    if lhs._coefficients != rhs._coefficients {return false}
    if lhs.rho != rhs.rho {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension SupportVectorClassifier: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SupportVectorClassifier"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "kernel"),
    2: .same(proto: "numberOfSupportVectorsPerClass"),
    3: .same(proto: "sparseSupportVectors"),
    4: .same(proto: "denseSupportVectors"),
    5: .same(proto: "coefficients"),
    6: .same(proto: "rho"),
    7: .same(proto: "probA"),
    8: .same(proto: "probB"),
    100: .same(proto: "stringClassLabels"),
    101: .same(proto: "int64ClassLabels"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._kernel) }()
      case 2: try { try decoder.decodeRepeatedInt32Field(value: &self.numberOfSupportVectorsPerClass) }()
      case 3: try {
        var v: SparseSupportVectors?
        var hadOneofValue = false
        if let current = self.supportVectors {
          hadOneofValue = true
          if case .sparseSupportVectors(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.supportVectors = .sparseSupportVectors(v)
        }
      }()
      case 4: try {
        var v: DenseSupportVectors?
        var hadOneofValue = false
        if let current = self.supportVectors {
          hadOneofValue = true
          if case .denseSupportVectors(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.supportVectors = .denseSupportVectors(v)
        }
      }()
      case 5: try { try decoder.decodeRepeatedMessageField(value: &self.coefficients) }()
      case 6: try { try decoder.decodeRepeatedDoubleField(value: &self.rho) }()
      case 7: try { try decoder.decodeRepeatedDoubleField(value: &self.probA) }()
      case 8: try { try decoder.decodeRepeatedDoubleField(value: &self.probB) }()
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
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    try { if let v = self._kernel {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    if !self.numberOfSupportVectorsPerClass.isEmpty {
      try visitor.visitPackedInt32Field(value: self.numberOfSupportVectorsPerClass, fieldNumber: 2)
    }
    switch self.supportVectors {
    case .sparseSupportVectors?: try {
      guard case .sparseSupportVectors(let v)? = self.supportVectors else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    }()
    case .denseSupportVectors?: try {
      guard case .denseSupportVectors(let v)? = self.supportVectors else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
    }()
    case nil: break
    }
    if !self.coefficients.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.coefficients, fieldNumber: 5)
    }
    if !self.rho.isEmpty {
      try visitor.visitPackedDoubleField(value: self.rho, fieldNumber: 6)
    }
    if !self.probA.isEmpty {
      try visitor.visitPackedDoubleField(value: self.probA, fieldNumber: 7)
    }
    if !self.probB.isEmpty {
      try visitor.visitPackedDoubleField(value: self.probB, fieldNumber: 8)
    }
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
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: SupportVectorClassifier, rhs: SupportVectorClassifier) -> Bool {
    if lhs._kernel != rhs._kernel {return false}
    if lhs.numberOfSupportVectorsPerClass != rhs.numberOfSupportVectorsPerClass {return false}
    if lhs.supportVectors != rhs.supportVectors {return false}
    if lhs.coefficients != rhs.coefficients {return false}
    if lhs.rho != rhs.rho {return false}
    if lhs.probA != rhs.probA {return false}
    if lhs.probB != rhs.probB {return false}
    if lhs.classLabels != rhs.classLabels {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
