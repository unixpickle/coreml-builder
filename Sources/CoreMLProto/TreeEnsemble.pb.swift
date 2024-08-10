// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: TreeEnsemble.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

//
// Each tree is a collection of nodes,
// each of which is identified by a unique identifier.
//
// Each node is either a branch or a leaf node.
// A branch node evaluates a value according to a behavior;
// if true, the node identified by ``true_child_node_id`` is evaluated next,
// if false, the node identified by ``false_child_node_id`` is evaluated next.
// A leaf node adds the evaluation value to the base prediction value
// to get the final prediction.
//
// A tree must have exactly one root node,
// which has no parent node.
// A tree must not terminate on a branch node.
// All leaf nodes must be accessible
// by evaluating one or more branch nodes in sequence,
// starting from the root node.

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
/// A tree ensemble post-evaluation transform.
public enum TreeEnsemblePostEvaluationTransform: SwiftProtobuf.Enum {
  public typealias RawValue = Int
  case noTransform // = 0
  case classificationSoftMax // = 1
  case regressionLogistic // = 2
  case classificationSoftMaxWithZeroClassReference // = 3
  case UNRECOGNIZED(Int)

  public init() {
    self = .noTransform
  }

  public init?(rawValue: Int) {
    switch rawValue {
    case 0: self = .noTransform
    case 1: self = .classificationSoftMax
    case 2: self = .regressionLogistic
    case 3: self = .classificationSoftMaxWithZeroClassReference
    default: self = .UNRECOGNIZED(rawValue)
    }
  }

  public var rawValue: Int {
    switch self {
    case .noTransform: return 0
    case .classificationSoftMax: return 1
    case .regressionLogistic: return 2
    case .classificationSoftMaxWithZeroClassReference: return 3
    case .UNRECOGNIZED(let i): return i
    }
  }

}

#if swift(>=4.2)

extension TreeEnsemblePostEvaluationTransform: CaseIterable {
  // The compiler won't synthesize support with the UNRECOGNIZED case.
  public static var allCases: [TreeEnsemblePostEvaluationTransform] = [
    .noTransform,
    .classificationSoftMax,
    .regressionLogistic,
    .classificationSoftMaxWithZeroClassReference,
  ]
}

#endif  // swift(>=4.2)

///
/// Tree ensemble parameters.
public struct TreeEnsembleParameters {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var nodes: [TreeEnsembleParameters.TreeNode] = []

  ///
  /// The number of prediction dimensions or classes in the model.
  ///
  /// All instances of ``evaluationIndex`` in a leaf node
  /// must be less than this value,
  /// and the number of values in the ``basePredictionValue`` field
  /// must be equal to this value.
  ///
  /// For regression,
  /// this is the dimension of the prediction.
  /// For classification,
  /// this is the number of classes.
  public var numPredictionDimensions: UInt64 = 0

  ///
  /// The base prediction value.
  ///
  /// The number of values in this must match
  /// the default values of the tree model.
  public var basePredictionValue: [Double] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public struct TreeNode {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var treeID: UInt64 = 0

    public var nodeID: UInt64 = 0

    ///
    /// The branch mode parameters.
    ///
    /// If branch is false,
    /// then the parameters in this section must be filled in
    /// to determine how the branching functions.
    public var nodeBehavior: TreeEnsembleParameters.TreeNode.TreeNodeBehavior = .branchOnValueLessThanEqual

    ///
    /// If the node behavior mode is a branch mode,
    /// then these values must be filled in.
    public var branchFeatureIndex: UInt64 = 0

    public var branchFeatureValue: Double = 0

    public var trueChildNodeID: UInt64 = 0

    public var falseChildNodeID: UInt64 = 0

    public var missingValueTracksTrueChild: Bool = false

    public var evaluationInfo: [TreeEnsembleParameters.TreeNode.EvaluationInfo] = []

    ///
    /// The relative hit rate of a node for optimization purposes.
    ///
    /// This value has no effect on the accuracy of the result;
    /// it allows the tree to optimize for frequent branches.
    /// The value is relative,
    /// compared to the hit rates of other branch nodes.
    ///
    /// You typically use a proportion of training samples
    /// that reached this node
    /// or some similar metric to derive this value.
    public var relativeHitRate: Double = 0

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public enum TreeNodeBehavior: SwiftProtobuf.Enum {
      public typealias RawValue = Int
      case branchOnValueLessThanEqual // = 0
      case branchOnValueLessThan // = 1
      case branchOnValueGreaterThanEqual // = 2
      case branchOnValueGreaterThan // = 3
      case branchOnValueEqual // = 4
      case branchOnValueNotEqual // = 5
      case leafNode // = 6
      case UNRECOGNIZED(Int)

      public init() {
        self = .branchOnValueLessThanEqual
      }

      public init?(rawValue: Int) {
        switch rawValue {
        case 0: self = .branchOnValueLessThanEqual
        case 1: self = .branchOnValueLessThan
        case 2: self = .branchOnValueGreaterThanEqual
        case 3: self = .branchOnValueGreaterThan
        case 4: self = .branchOnValueEqual
        case 5: self = .branchOnValueNotEqual
        case 6: self = .leafNode
        default: self = .UNRECOGNIZED(rawValue)
        }
      }

      public var rawValue: Int {
        switch self {
        case .branchOnValueLessThanEqual: return 0
        case .branchOnValueLessThan: return 1
        case .branchOnValueGreaterThanEqual: return 2
        case .branchOnValueGreaterThan: return 3
        case .branchOnValueEqual: return 4
        case .branchOnValueNotEqual: return 5
        case .leafNode: return 6
        case .UNRECOGNIZED(let i): return i
        }
      }

    }

    ///
    /// The leaf mode.
    ///
    /// If ``nodeBahavior`` == ``LeafNode``,
    /// then the evaluationValue is added to the base prediction value
    /// in order to get the final prediction.
    /// To support multiclass classification
    /// as well as regression and binary classification,
    /// the evaluation value is encoded here as a sparse vector,
    /// with evaluationIndex being the index of the base vector
    /// that evaluation value is added to.
    /// In the single class case,
    /// it is expected that evaluationIndex is exactly 0.
    public struct EvaluationInfo {
      // SwiftProtobuf.Message conformance is added in an extension below. See the
      // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
      // methods supported on all messages.

      public var evaluationIndex: UInt64 = 0

      public var evaluationValue: Double = 0

      public var unknownFields = SwiftProtobuf.UnknownStorage()

      public init() {}
    }

    public init() {}
  }

  public init() {}
}

#if swift(>=4.2)

extension TreeEnsembleParameters.TreeNode.TreeNodeBehavior: CaseIterable {
  // The compiler won't synthesize support with the UNRECOGNIZED case.
  public static var allCases: [TreeEnsembleParameters.TreeNode.TreeNodeBehavior] = [
    .branchOnValueLessThanEqual,
    .branchOnValueLessThan,
    .branchOnValueGreaterThanEqual,
    .branchOnValueGreaterThan,
    .branchOnValueEqual,
    .branchOnValueNotEqual,
    .leafNode,
  ]
}

#endif  // swift(>=4.2)

///
/// A tree ensemble classifier.
public struct TreeEnsembleClassifier {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var treeEnsemble: TreeEnsembleParameters {
    get {return _treeEnsemble ?? TreeEnsembleParameters()}
    set {_treeEnsemble = newValue}
  }
  /// Returns true if `treeEnsemble` has been explicitly set.
  public var hasTreeEnsemble: Bool {return self._treeEnsemble != nil}
  /// Clears the value of `treeEnsemble`. Subsequent reads from it will return its default value.
  public mutating func clearTreeEnsemble() {self._treeEnsemble = nil}

  public var postEvaluationTransform: TreeEnsemblePostEvaluationTransform = .noTransform

  /// Required class label mapping
  public var classLabels: TreeEnsembleClassifier.OneOf_ClassLabels? = nil

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

  /// Required class label mapping
  public enum OneOf_ClassLabels: Equatable {
    case stringClassLabels(StringVector)
    case int64ClassLabels(Int64Vector)

  #if !swift(>=4.1)
    public static func ==(lhs: TreeEnsembleClassifier.OneOf_ClassLabels, rhs: TreeEnsembleClassifier.OneOf_ClassLabels) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.stringClassLabels, .stringClassLabels): return {
        guard case .stringClassLabels(let l) = lhs, case .stringClassLabels(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      case (.int64ClassLabels, .int64ClassLabels): return {
        guard case .int64ClassLabels(let l) = lhs, case .int64ClassLabels(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      default: return false
      }
    }
  #endif
  }

  public init() {}

  fileprivate var _treeEnsemble: TreeEnsembleParameters? = nil
}

///
/// A tree ensemble regressor.
public struct TreeEnsembleRegressor {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var treeEnsemble: TreeEnsembleParameters {
    get {return _treeEnsemble ?? TreeEnsembleParameters()}
    set {_treeEnsemble = newValue}
  }
  /// Returns true if `treeEnsemble` has been explicitly set.
  public var hasTreeEnsemble: Bool {return self._treeEnsemble != nil}
  /// Clears the value of `treeEnsemble`. Subsequent reads from it will return its default value.
  public mutating func clearTreeEnsemble() {self._treeEnsemble = nil}

  public var postEvaluationTransform: TreeEnsemblePostEvaluationTransform = .noTransform

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _treeEnsemble: TreeEnsembleParameters? = nil
}

#if swift(>=5.5) && canImport(_Concurrency)
extension TreeEnsemblePostEvaluationTransform: @unchecked Sendable {}
extension TreeEnsembleParameters: @unchecked Sendable {}
extension TreeEnsembleParameters.TreeNode: @unchecked Sendable {}
extension TreeEnsembleParameters.TreeNode.TreeNodeBehavior: @unchecked Sendable {}
extension TreeEnsembleParameters.TreeNode.EvaluationInfo: @unchecked Sendable {}
extension TreeEnsembleClassifier: @unchecked Sendable {}
extension TreeEnsembleClassifier.OneOf_ClassLabels: @unchecked Sendable {}
extension TreeEnsembleRegressor: @unchecked Sendable {}
#endif  // swift(>=5.5) && canImport(_Concurrency)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension TreeEnsemblePostEvaluationTransform: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "NoTransform"),
    1: .same(proto: "Classification_SoftMax"),
    2: .same(proto: "Regression_Logistic"),
    3: .same(proto: "Classification_SoftMaxWithZeroClassReference"),
  ]
}

extension TreeEnsembleParameters: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".TreeEnsembleParameters"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "nodes"),
    2: .same(proto: "numPredictionDimensions"),
    3: .same(proto: "basePredictionValue"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedMessageField(value: &self.nodes) }()
      case 2: try { try decoder.decodeSingularUInt64Field(value: &self.numPredictionDimensions) }()
      case 3: try { try decoder.decodeRepeatedDoubleField(value: &self.basePredictionValue) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.nodes.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodes, fieldNumber: 1)
    }
    if self.numPredictionDimensions != 0 {
      try visitor.visitSingularUInt64Field(value: self.numPredictionDimensions, fieldNumber: 2)
    }
    if !self.basePredictionValue.isEmpty {
      try visitor.visitPackedDoubleField(value: self.basePredictionValue, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: TreeEnsembleParameters, rhs: TreeEnsembleParameters) -> Bool {
    if lhs.nodes != rhs.nodes {return false}
    if lhs.numPredictionDimensions != rhs.numPredictionDimensions {return false}
    if lhs.basePredictionValue != rhs.basePredictionValue {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TreeEnsembleParameters.TreeNode: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = TreeEnsembleParameters.protoMessageName + ".TreeNode"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "treeId"),
    2: .same(proto: "nodeId"),
    3: .same(proto: "nodeBehavior"),
    10: .same(proto: "branchFeatureIndex"),
    11: .same(proto: "branchFeatureValue"),
    12: .same(proto: "trueChildNodeId"),
    13: .same(proto: "falseChildNodeId"),
    14: .same(proto: "missingValueTracksTrueChild"),
    20: .same(proto: "evaluationInfo"),
    30: .same(proto: "relativeHitRate"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularUInt64Field(value: &self.treeID) }()
      case 2: try { try decoder.decodeSingularUInt64Field(value: &self.nodeID) }()
      case 3: try { try decoder.decodeSingularEnumField(value: &self.nodeBehavior) }()
      case 10: try { try decoder.decodeSingularUInt64Field(value: &self.branchFeatureIndex) }()
      case 11: try { try decoder.decodeSingularDoubleField(value: &self.branchFeatureValue) }()
      case 12: try { try decoder.decodeSingularUInt64Field(value: &self.trueChildNodeID) }()
      case 13: try { try decoder.decodeSingularUInt64Field(value: &self.falseChildNodeID) }()
      case 14: try { try decoder.decodeSingularBoolField(value: &self.missingValueTracksTrueChild) }()
      case 20: try { try decoder.decodeRepeatedMessageField(value: &self.evaluationInfo) }()
      case 30: try { try decoder.decodeSingularDoubleField(value: &self.relativeHitRate) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.treeID != 0 {
      try visitor.visitSingularUInt64Field(value: self.treeID, fieldNumber: 1)
    }
    if self.nodeID != 0 {
      try visitor.visitSingularUInt64Field(value: self.nodeID, fieldNumber: 2)
    }
    if self.nodeBehavior != .branchOnValueLessThanEqual {
      try visitor.visitSingularEnumField(value: self.nodeBehavior, fieldNumber: 3)
    }
    if self.branchFeatureIndex != 0 {
      try visitor.visitSingularUInt64Field(value: self.branchFeatureIndex, fieldNumber: 10)
    }
    if self.branchFeatureValue != 0 {
      try visitor.visitSingularDoubleField(value: self.branchFeatureValue, fieldNumber: 11)
    }
    if self.trueChildNodeID != 0 {
      try visitor.visitSingularUInt64Field(value: self.trueChildNodeID, fieldNumber: 12)
    }
    if self.falseChildNodeID != 0 {
      try visitor.visitSingularUInt64Field(value: self.falseChildNodeID, fieldNumber: 13)
    }
    if self.missingValueTracksTrueChild != false {
      try visitor.visitSingularBoolField(value: self.missingValueTracksTrueChild, fieldNumber: 14)
    }
    if !self.evaluationInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.evaluationInfo, fieldNumber: 20)
    }
    if self.relativeHitRate != 0 {
      try visitor.visitSingularDoubleField(value: self.relativeHitRate, fieldNumber: 30)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: TreeEnsembleParameters.TreeNode, rhs: TreeEnsembleParameters.TreeNode) -> Bool {
    if lhs.treeID != rhs.treeID {return false}
    if lhs.nodeID != rhs.nodeID {return false}
    if lhs.nodeBehavior != rhs.nodeBehavior {return false}
    if lhs.branchFeatureIndex != rhs.branchFeatureIndex {return false}
    if lhs.branchFeatureValue != rhs.branchFeatureValue {return false}
    if lhs.trueChildNodeID != rhs.trueChildNodeID {return false}
    if lhs.falseChildNodeID != rhs.falseChildNodeID {return false}
    if lhs.missingValueTracksTrueChild != rhs.missingValueTracksTrueChild {return false}
    if lhs.evaluationInfo != rhs.evaluationInfo {return false}
    if lhs.relativeHitRate != rhs.relativeHitRate {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TreeEnsembleParameters.TreeNode.TreeNodeBehavior: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "BranchOnValueLessThanEqual"),
    1: .same(proto: "BranchOnValueLessThan"),
    2: .same(proto: "BranchOnValueGreaterThanEqual"),
    3: .same(proto: "BranchOnValueGreaterThan"),
    4: .same(proto: "BranchOnValueEqual"),
    5: .same(proto: "BranchOnValueNotEqual"),
    6: .same(proto: "LeafNode"),
  ]
}

extension TreeEnsembleParameters.TreeNode.EvaluationInfo: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = TreeEnsembleParameters.TreeNode.protoMessageName + ".EvaluationInfo"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "evaluationIndex"),
    2: .same(proto: "evaluationValue"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularUInt64Field(value: &self.evaluationIndex) }()
      case 2: try { try decoder.decodeSingularDoubleField(value: &self.evaluationValue) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.evaluationIndex != 0 {
      try visitor.visitSingularUInt64Field(value: self.evaluationIndex, fieldNumber: 1)
    }
    if self.evaluationValue != 0 {
      try visitor.visitSingularDoubleField(value: self.evaluationValue, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: TreeEnsembleParameters.TreeNode.EvaluationInfo, rhs: TreeEnsembleParameters.TreeNode.EvaluationInfo) -> Bool {
    if lhs.evaluationIndex != rhs.evaluationIndex {return false}
    if lhs.evaluationValue != rhs.evaluationValue {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TreeEnsembleClassifier: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".TreeEnsembleClassifier"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "treeEnsemble"),
    2: .same(proto: "postEvaluationTransform"),
    100: .same(proto: "stringClassLabels"),
    101: .same(proto: "int64ClassLabels"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._treeEnsemble) }()
      case 2: try { try decoder.decodeSingularEnumField(value: &self.postEvaluationTransform) }()
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
    try { if let v = self._treeEnsemble {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    if self.postEvaluationTransform != .noTransform {
      try visitor.visitSingularEnumField(value: self.postEvaluationTransform, fieldNumber: 2)
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

  public static func ==(lhs: TreeEnsembleClassifier, rhs: TreeEnsembleClassifier) -> Bool {
    if lhs._treeEnsemble != rhs._treeEnsemble {return false}
    if lhs.postEvaluationTransform != rhs.postEvaluationTransform {return false}
    if lhs.classLabels != rhs.classLabels {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TreeEnsembleRegressor: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".TreeEnsembleRegressor"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "treeEnsemble"),
    2: .same(proto: "postEvaluationTransform"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._treeEnsemble) }()
      case 2: try { try decoder.decodeSingularEnumField(value: &self.postEvaluationTransform) }()
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    try { if let v = self._treeEnsemble {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    if self.postEvaluationTransform != .noTransform {
      try visitor.visitSingularEnumField(value: self.postEvaluationTransform, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public static func ==(lhs: TreeEnsembleRegressor, rhs: TreeEnsembleRegressor) -> Bool {
    if lhs._treeEnsemble != rhs._treeEnsemble {return false}
    if lhs.postEvaluationTransform != rhs.postEvaluationTransform {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
