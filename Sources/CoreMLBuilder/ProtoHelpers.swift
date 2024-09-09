// The Swift Programming Language
// https://docs.swift.org/swift-book

import CoreMLProto
import Foundation

extension Model {
  public init(
    description: ModelDescription,
    mlProgram: MILSpec_Program,
    specificationVersion: Int32 = 7
  ) {
    self.init()
    self.specificationVersion = specificationVersion
    self.description_p = description
    self.mlProgram = mlProgram
  }

  public init(
    description: ModelDescription,
    neuralNetwork: NeuralNetwork,
    specificationVersion: Int32 = 7
  ) {
    self.init()
    self.specificationVersion = specificationVersion
    self.description_p = description
    self.neuralNetwork = neuralNetwork
  }
}

extension ModelDescription {
  public init(
    functions: [FunctionDescription] = [],
    input: [FeatureDescription] = [],
    output: [FeatureDescription] = []
  ) {
    self.init()
    self.functions = functions
    self.input = input
    self.output = output
  }
}

extension FeatureType {
  public init(multiArray shape: [Int64], dataType: ArrayFeatureType.ArrayDataType) {
    self.init()
    self.multiArrayType = ArrayFeatureType()
    self.multiArrayType.shape = shape
    self.multiArrayType.dataType = dataType
  }
}

extension FeatureDescription {
  public init(name: String, type: FeatureType) {
    self.init()
    self.name = name
    self.type = type
  }
}

extension MILSpec_Program {
  public init(
    version: Int64, functions: [String: MILSpec_Function], attributes: [String: MILSpec_Value]
  ) {
    self.init()
    self.version = version
    self.functions = functions
    self.attributes = attributes
  }
}

extension MILSpec_Function {
  public init(
    inputs: [MILSpec_NamedValueType] = [],
    opset: String = "",
    blockSpecializations: [String: MILSpec_Block] = [:],
    attributes: [String: MILSpec_Value] = [:]
  ) {
    self.init()
    self.inputs = inputs
    self.opset = opset
    self.blockSpecializations = blockSpecializations
    self.attributes = attributes
  }
}

extension MILSpec_Block {
  public init(
    inputs: [MILSpec_NamedValueType] = [],
    outputs: [String] = [],
    operations: [MILSpec_Operation] = [],
    attributes: [String: MILSpec_Value] = [:]
  ) {
    self.init()
    self.inputs = inputs
    self.outputs = outputs
    self.operations = operations
    self.attributes = attributes
  }
}

extension MILSpec_NamedValueType {
  public init(
    name: String = "",
    type: MILSpec_ValueType
  ) {
    self.init()
    self.name = name
    self.type = type
  }
}

extension MILSpec_ValueType {
  public init(tensorType: MILSpec_TensorType) {
    self.init()
    self.tensorType = tensorType
  }

  public init(dictionaryType: MILSpec_DictionaryType) {
    self.init()
    self.dictionaryType = dictionaryType
  }

  public init(tensorType: MILSpec_DataType, shape: [UInt64]) {
    self.init(
      tensorType: MILSpec_TensorType(
        dataType: tensorType,
        rank: Int64(shape.count),
        dimensions: shape.map({ x in MILSpec_Dimension(constant: x) })
      ))
  }
}

extension MILSpec_DictionaryType {
  public init(keyType: MILSpec_ValueType, valueType: MILSpec_ValueType) {
    self.init()
    self.keyType = keyType
    self.valueType = valueType
  }
}

extension MILSpec_TensorType {
  public init(
    dataType: MILSpec_DataType = .unusedType,
    rank: Int64 = 0,
    dimensions: [MILSpec_Dimension] = [],
    attributes: [String: MILSpec_Value] = [:]
  ) {
    self.init()
    self.dataType = dataType
    self.rank = rank
    self.dimensions = dimensions
    self.attributes = attributes
  }
}

extension MILSpec_Dimension {
  public init(constant: UInt64) {
    self.init()
    self.constant = ConstantDimension()
    self.constant.size = constant
  }
}

extension MILSpec_Value {
  public init(type: MILSpec_ValueType, immediateValue: MILSpec_Value.ImmediateValue) {
    self.init()
    self.type = type
    self.immediateValue = immediateValue
  }

  public init(immediateStringDict: [String: String]) {
    self.init()
    self.type = MILSpec_ValueType(
      dictionaryType: MILSpec_DictionaryType(
        keyType: MILSpec_ValueType(
          tensorType: MILSpec_TensorType(dataType: .string)
        ),
        valueType: MILSpec_ValueType(
          tensorType: MILSpec_TensorType(dataType: .string)
        )
      )
    )
    let values = immediateStringDict.map { (k, v) in
      MILSpec_DictionaryValue.KeyValuePair(
        key: MILSpec_Value(immediateString: k),
        value: MILSpec_Value(immediateString: v)
      )
    }
    self.immediateValue = ImmediateValue(dictionary: MILSpec_DictionaryValue(values: values))
  }

  public init(immediateString: String) {
    self.init(
      type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .string)),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(strings: [immediateString]))
    )
  }

  public init(immediateFloat: Float) {
    self.init(
      type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .float32)),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(floats: [immediateFloat]))
    )
  }

  public init(immediateInt: Int32) {
    self.init(
      type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .int32)),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(ints: [immediateInt]))
    )
  }

  public init(immediateInts: [Int32]) {
    self.init(
      type: MILSpec_ValueType(
        tensorType: MILSpec_TensorType(
          dataType: .int32,
          rank: 1,
          dimensions: [MILSpec_Dimension(constant: UInt64(immediateInts.count))]
        )
      ),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(ints: immediateInts))
    )
  }

  public init(immediateLongInt: Int64) {
    self.init(
      type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .int64)),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(longInts: [immediateLongInt]))
    )
  }

  public init(immediateBool: Bool) {
    self.init(
      type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .bool)),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(bools: [immediateBool]))
    )
  }

  public init(immediateDouble: Double) {
    self.init(
      type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .float64)),
      immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(doubles: [immediateDouble]))
    )
  }
}

extension MILSpec_Value.ImmediateValue {
  public init(tensor: MILSpec_TensorValue) {
    self.init()
    self.tensor = tensor
  }

  public init(dictionary: MILSpec_DictionaryValue) {
    self.init()
    self.dictionary = dictionary
  }
}

extension MILSpec_TensorValue {
  public init(floats: [Float]) {
    self.init()
    var repeatedFloats = MILSpec_TensorValue.RepeatedFloats()
    repeatedFloats.values = floats
    self.floats = repeatedFloats
  }

  public init(ints: [Int32]) {
    self.init()
    var repeatedInts = MILSpec_TensorValue.RepeatedInts()
    repeatedInts.values = ints
    self.ints = repeatedInts
  }

  public init(bools: [Bool]) {
    self.init()
    var repeatedBools = MILSpec_TensorValue.RepeatedBools()
    repeatedBools.values = bools
    self.bools = repeatedBools
  }

  public init(strings: [String]) {
    self.init()
    var repeatedStrings = MILSpec_TensorValue.RepeatedStrings()
    repeatedStrings.values = strings
    self.strings = repeatedStrings
  }

  public init(longInts: [Int64]) {
    self.init()
    var repeatedLongInts = MILSpec_TensorValue.RepeatedLongInts()
    repeatedLongInts.values = longInts
    self.longInts = repeatedLongInts
  }

  public init(doubles: [Double]) {
    self.init()
    var repeatedDoubles = MILSpec_TensorValue.RepeatedDoubles()
    repeatedDoubles.values = doubles
    self.doubles = repeatedDoubles
  }

  public init(bytes: Data) {
    self.init()
    var repeatedBytes = MILSpec_TensorValue.RepeatedBytes()
    repeatedBytes.values = bytes
    self.bytes = repeatedBytes
  }
}

public protocol ToSpecValue {
  func toSpecValue() -> MILSpec_Value
}

extension String: ToSpecValue {
  public func toSpecValue() -> MILSpec_Value {
    return MILSpec_Value(immediateString: self)
  }
}

extension Int32: ToSpecValue {
  public func toSpecValue() -> MILSpec_Value {
    return MILSpec_Value(immediateInt: self)
  }
}

extension Int: ToSpecValue {
  public func toSpecValue() -> MILSpec_Value {
    return MILSpec_Value(immediateInt: Int32(self))
  }
}

extension MILSpec_DictionaryValue {
  public init(values: [MILSpec_DictionaryValue.KeyValuePair] = []) {
    self.init()
    self.values = values
  }
}

extension MILSpec_DictionaryValue.KeyValuePair {
  public init(key: MILSpec_Value, value: MILSpec_Value) {
    self.init()
    self.key = key
    self.value = value
  }
}

extension MILSpec_Operation {
  public init(
    type: String = "",
    inputs: [String: MILSpec_Argument] = [:],
    outputs: [MILSpec_NamedValueType] = [],
    blocks: [MILSpec_Block] = [],
    attributes: [String: MILSpec_Value] = [:]
  ) {
    self.init()
    self.type = type
    self.inputs = inputs
    self.outputs = outputs
    self.blocks = blocks
    self.attributes = attributes
  }

  public init(
    constWithName name: String,
    opName: String,
    value: MILSpec_Value
  ) {
    self.init(
      type: "const",
      outputs: [
        MILSpec_NamedValueType(name: name, type: value.type)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName),
        "val": value,
      ]
    )
  }

  public init(
    cast input: some ToBinding,
    typeName: some ToBinding,
    opName: String,
    outName: String,
    outType: MILSpec_ValueType
  ) {
    self.init(
      type: "cast",
      inputs: [
        "dtype": MILSpec_Argument(arguments: [typeName.toBinding()]),
        "x": MILSpec_Argument(arguments: [input.toBinding()]),
      ],
      outputs: [
        MILSpec_NamedValueType(name: outName, type: outType)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName)
      ]
    )
  }

  public init(
    conv x: some ToBinding,
    bias: some ToBinding,
    dilations: some ToBinding,
    groups: some ToBinding,
    pad: some ToBinding,
    padType: some ToBinding,
    strides: some ToBinding,
    weight: some ToBinding,
    outName: String,
    outType: MILSpec_ValueType,
    opName: String
  ) {
    self.init(
      type: "conv",
      inputs: [
        "bias": MILSpec_Argument(arguments: [bias.toBinding()]),
        "dilations": MILSpec_Argument(arguments: [dilations.toBinding()]),
        "groups": MILSpec_Argument(arguments: [groups.toBinding()]),
        "pad": MILSpec_Argument(arguments: [pad.toBinding()]),
        "pad_type": MILSpec_Argument(arguments: [padType.toBinding()]),
        "strides": MILSpec_Argument(arguments: [strides.toBinding()]),
        "weight": MILSpec_Argument(arguments: [weight.toBinding()]),
        "x": MILSpec_Argument(arguments: [x.toBinding()]),
      ],
      outputs: [
        MILSpec_NamedValueType(name: outName, type: outType)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName)
      ]
    )
  }

  public init(
    linear x: some ToBinding,
    weight: some ToBinding,
    bias: some ToBinding,
    outName: String,
    outType: MILSpec_ValueType,
    opName: String
  ) {
    self.init(
      type: "linear",
      inputs: [
        "bias": MILSpec_Argument(arguments: [bias.toBinding()]),
        "weight": MILSpec_Argument(arguments: [weight.toBinding()]),
        "x": MILSpec_Argument(arguments: [x.toBinding()]),
      ],
      outputs: [
        MILSpec_NamedValueType(name: outName, type: outType)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName)
      ]
    )
  }

  public init(
    matmul x: some ToBinding,
    y: some ToBinding,
    transposeX: some ToBinding,
    transposeY: some ToBinding,
    outName: String,
    outType: MILSpec_ValueType,
    opName: String
  ) {
    self.init(
      type: "matmul",
      inputs: [
        "transpose_x": MILSpec_Argument(arguments: [transposeX.toBinding()]),
        "transpose_y": MILSpec_Argument(arguments: [transposeY.toBinding()]),
        "x": MILSpec_Argument(arguments: [x.toBinding()]),
        "y": MILSpec_Argument(arguments: [y.toBinding()]),
      ],
      outputs: [
        MILSpec_NamedValueType(name: outName, type: outType)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName)
      ]
    )
  }

  public init(
    transpose input: some ToBinding,
    perm: [Int32],
    opName: String,
    outName: String,
    outType: MILSpec_ValueType
  ) {
    self.init(
      type: "transpose",
      inputs: [
        "perm": MILSpec_Argument(arguments: [
          MILSpec_Value(immediateInts: perm).toBinding()
        ]),
        "x": MILSpec_Argument(arguments: [input.toBinding()]),
      ],
      outputs: [
        MILSpec_NamedValueType(name: outName, type: outType)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName)
      ]
    )
  }

  public init(
    transpose input: some ToBinding,
    perm: some ToBinding,
    opName: String,
    outName: String,
    outType: MILSpec_ValueType
  ) {
    self.init(
      type: "transpose",
      inputs: [
        "perm": MILSpec_Argument(arguments: [perm.toBinding()]),
        "x": MILSpec_Argument(arguments: [input.toBinding()]),
      ],
      outputs: [
        MILSpec_NamedValueType(name: outName, type: outType)
      ],
      attributes: [
        "name": MILSpec_Value(immediateString: opName)
      ]
    )
  }
}

extension MILSpec_Argument {
  public init(arguments: [MILSpec_Argument.Binding] = []) {
    self.init()
    self.arguments = arguments
  }
}

extension MILSpec_Argument.Binding {

  /// Initializes a `Binding` with a name.
  public init(name: String) {
    self.init()
    self.name = name
  }

  /// Initializes a `Binding` with a `MILSpec_Value`.
  public init(value: MILSpec_Value) {
    self.init()
    self.value = value
  }
}

public protocol ToBinding {
  func toBinding() -> MILSpec_Argument.Binding
}

extension MILSpec_Value: ToBinding {
  public func toBinding() -> MILSpec_Argument.Binding {
    return MILSpec_Argument.Binding(value: self)
  }
}

extension String: ToBinding {
  public func toBinding() -> MILSpec_Argument.Binding {
    return MILSpec_Argument.Binding(name: self)
  }
}

extension Bool: ToBinding {
  public func toBinding() -> MILSpec_Argument.Binding {
    return MILSpec_Argument.Binding(value: MILSpec_Value(immediateBool: self))
  }
}

extension NeuralNetwork {
  public init(
    layers: [NeuralNetworkLayer] = [],
    preprocessing: [NeuralNetworkPreprocessing] = [],
    arrayInputShapeMapping: NeuralNetworkMultiArrayShapeMapping = .rank5ArrayMapping,
    imageInputShapeMapping: NeuralNetworkImageShapeMapping = .rank5ImageMapping,
    updateParams: NetworkUpdateParameters? = nil
  ) {
    self.init()
    self.layers = layers
    self.preprocessing = preprocessing
    self.arrayInputShapeMapping = arrayInputShapeMapping
    self.imageInputShapeMapping = imageInputShapeMapping
    if let updateParams = updateParams {
      self.updateParams = updateParams
    }
  }
}

extension NeuralNetworkLayer {
  public init(
    convolution: ConvolutionLayerParams,
    name: String,
    input: [String],
    output: [String],
    inputTensor: [Tensor],
    outputTensor: [Tensor],
    isUpdatable: Bool
  ) {
    self.init()
    self.convolution = convolution
    self.name = name
    self.input = input
    self.output = output
    self.inputTensor = inputTensor
    self.outputTensor = outputTensor
    self.isUpdatable = isUpdatable
  }
}

extension ConvolutionLayerParams {
  public init(
    outputChannels: UInt64 = 0,
    kernelChannels: UInt64 = 0,
    nGroups: UInt64 = 1,
    kernelSize: [UInt64] = [3, 3],
    stride: [UInt64] = [1, 1],
    dilationFactor: [UInt64] = [1, 1],
    convolutionPaddingType: OneOf_ConvolutionPaddingType? = nil,
    isDeconvolution: Bool = false,
    hasBias: Bool = false,
    weights: WeightParams? = nil,
    bias: WeightParams? = nil,
    outputShape: [UInt64] = []
  ) {
    self.init()
    self.outputChannels = outputChannels
    self.kernelChannels = kernelChannels
    self.nGroups = nGroups
    self.kernelSize = kernelSize
    self.stride = stride
    self.dilationFactor = dilationFactor
    self.convolutionPaddingType =
      convolutionPaddingType
      ?? .valid(
        ValidPadding(
          paddingAmounts: BorderAmounts(
            borderAmounts: [
              BorderAmounts.EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
              BorderAmounts.EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
            ]
          )
        )
      )
    self.isDeconvolution = isDeconvolution
    self.hasBias_p = hasBias
    self.weights = weights ?? WeightParams()
    self.bias = bias ?? WeightParams()
    self.outputShape = outputShape
  }
}

extension BorderAmounts {
  public init(borderAmounts: [BorderAmounts.EdgeSizes]) {
    self.init()
    self.borderAmounts = borderAmounts
  }
}

extension BorderAmounts.EdgeSizes {
  public init(startEdgeSize: UInt64 = 0, endEdgeSize: UInt64 = 0) {
    self.init()
    self.startEdgeSize = startEdgeSize
    self.endEdgeSize = endEdgeSize
  }
}

extension ValidPadding {
  public init(paddingAmounts: BorderAmounts) {
    self.init()
    self.paddingAmounts = paddingAmounts
  }
}

extension Tensor {
  public init(dimValue: [Int64]) {
    self.init()
    self.rank = UInt32(dimValue.count)
    self.dimValue = dimValue
  }
}

extension WeightParams {
  public init(floatValue: [Float], isUpdatable: Bool = false) {
    self.init()
    self.floatValue = floatValue
    self.isUpdatable = isUpdatable
  }

  public init(float16Value: Data, isUpdatable: Bool = false) {
    self.init()
    self.float16Value = float16Value
    self.isUpdatable = isUpdatable
  }
}
