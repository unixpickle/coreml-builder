// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import CoreMLProto

public extension Model {
    init(
        description: ModelDescription,
        mlProgram: MILSpec_Program,
        specificationVersion: Int32 = 7
    ) {
        self.init()
        self.specificationVersion = specificationVersion
        self.description_p = description
        self.mlProgram = mlProgram
    }
}

public extension ModelDescription {
    init(
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

public extension FeatureType {
    init(multiArray shape: [Int64], dataType: ArrayFeatureType.ArrayDataType) {
        self.init()
        self.multiArrayType = ArrayFeatureType()
        self.multiArrayType.shape = shape
        self.multiArrayType.dataType = dataType
    }
}

public extension FeatureDescription {
    init(name: String, type: FeatureType) {
        self.init()
        self.name = name
        self.type = type
    }
}

public extension MILSpec_Program {
    init(version: Int64, functions: [String: MILSpec_Function], attributes: [String: MILSpec_Value]) {
        self.init()
        self.version = version
        self.functions = functions
        self.attributes = attributes
    }
}

public extension MILSpec_Function {
    init(
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

public extension MILSpec_Block {
    init(
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

public extension MILSpec_NamedValueType {
    init(
        name: String = "",
        type: MILSpec_ValueType
    ) {
        self.init()
        self.name = name
        self.type = type
    }
}

public extension MILSpec_ValueType {
    init(tensorType: MILSpec_TensorType) {
        self.init()
        self.tensorType = tensorType
    }

    init(dictionaryType: MILSpec_DictionaryType) {
        self.init()
        self.dictionaryType = dictionaryType
    }
}

public extension MILSpec_DictionaryType {
    init(keyType: MILSpec_ValueType, valueType: MILSpec_ValueType) {
        self.init()
        self.keyType = keyType
        self.valueType = valueType
    }
}

public extension MILSpec_TensorType {
    init(
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

public extension MILSpec_Dimension {
    init(constant: UInt64) {
        self.init()
        self.constant = ConstantDimension()
        self.constant.size = constant
    }
}

public extension MILSpec_Value {
    init(type: MILSpec_ValueType, immediateValue: MILSpec_Value.ImmediateValue) {
        self.init()
        self.type = type
        self.immediateValue = immediateValue
    }

    init(immediateStringDict: [String: String]) {
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

    init(immediateString: String) {
        self.init(
            type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .string)),
            immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(strings: [immediateString]))
        )
    }

    init(immediateFloat: Float) {
        self.init(
            type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .float32)),
            immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(floats: [immediateFloat]))
        )
    }

    init(immediateInt: Int32) {
        self.init(
            type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .int32)),
            immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(ints: [immediateInt]))
        )
    }

    init(immediateInts: [Int32]) {
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

    init(immediateLongInt: Int64) {
        self.init(
            type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .int64)),
            immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(longInts: [immediateLongInt]))
        )
    }

    init(immediateBool: Bool) {
        self.init(
            type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .bool)),
            immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(bools: [immediateBool]))
        )
    }

    init(immediateDouble: Double) {
        self.init(
            type: MILSpec_ValueType(tensorType: MILSpec_TensorType(dataType: .float64)),
            immediateValue: ImmediateValue(tensor: MILSpec_TensorValue(doubles: [immediateDouble]))
        )
    }
}

public extension MILSpec_Value.ImmediateValue {
    init(tensor: MILSpec_TensorValue) {
        self.init()
        self.tensor = tensor
    }

    init(dictionary: MILSpec_DictionaryValue) {
        self.init()
        self.dictionary = dictionary
    }
}

public extension MILSpec_TensorValue {
    init(floats: [Float]) {
        self.init()
        var repeatedFloats = MILSpec_TensorValue.RepeatedFloats()
        repeatedFloats.values = floats
        self.floats = repeatedFloats
    }

    init(ints: [Int32]) {
        self.init()
        var repeatedInts = MILSpec_TensorValue.RepeatedInts()
        repeatedInts.values = ints
        self.ints = repeatedInts
    }

    init(bools: [Bool]) {
        self.init()
        var repeatedBools = MILSpec_TensorValue.RepeatedBools()
        repeatedBools.values = bools
        self.bools = repeatedBools
    }

    init(strings: [String]) {
        self.init()
        var repeatedStrings = MILSpec_TensorValue.RepeatedStrings()
        repeatedStrings.values = strings
        self.strings = repeatedStrings
    }

    init(longInts: [Int64]) {
        self.init()
        var repeatedLongInts = MILSpec_TensorValue.RepeatedLongInts()
        repeatedLongInts.values = longInts
        self.longInts = repeatedLongInts
    }

    init(doubles: [Double]) {
        self.init()
        var repeatedDoubles = MILSpec_TensorValue.RepeatedDoubles()
        repeatedDoubles.values = doubles
        self.doubles = repeatedDoubles
    }

    init(bytes: Data) {
        self.init()
        var repeatedBytes = MILSpec_TensorValue.RepeatedBytes()
        repeatedBytes.values = bytes
        self.bytes = repeatedBytes
    }
}

public extension MILSpec_DictionaryValue {
    init(values: [MILSpec_DictionaryValue.KeyValuePair] = []) {
        self.init()
        self.values = values
    }
}

public extension MILSpec_DictionaryValue.KeyValuePair {
    init(key: MILSpec_Value, value: MILSpec_Value) {
        self.init()
        self.key = key
        self.value = value
    }
}

public extension MILSpec_Operation {
    init(
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
}

public extension MILSpec_Argument {
    init(arguments: [MILSpec_Argument.Binding] = []) {
        self.init()
        self.arguments = arguments
    }
}

public extension MILSpec_Argument.Binding {

    /// Initializes a `Binding` with a name.
    init(name: String) {
        self.init()
        self.name = name
    }

    /// Initializes a `Binding` with a `MILSpec_Value`.
    init(value: MILSpec_Value) {
        self.init()
        self.value = value
    }
}

func convLayer() {
    var model = Model()
    model.specificationVersion = 7
    model.description_p = ModelDescription()
    model.description_p.input = [
        
    ]
}
