import CoreMLProto
import CoreML

public struct Conv2D {
    public let batch: Int64?
    public let channels: Int64
    public let height: Int64
    public let width: Int64
    public let outChannels: Int64
    public let kernelSize: Int64
    public let dtype: DType

    public var inShape: [UInt64] {
        if let batch = batch {
            [
                UInt64(batch),
                UInt64(channels),
                UInt64(height),
                UInt64(width),
            ]
        } else {
            [
                UInt64(channels),
                UInt64(height),
                UInt64(width),
            ]
        }
    }

    public var outShape: [UInt64] {
        if let batch = batch {
            [
                UInt64(batch),
                UInt64(outChannels),
                UInt64(height - kernelSize + 1),
                UInt64(width - kernelSize + 1),
            ]
        } else {
            [
                UInt64(outChannels),
                UInt64(height - kernelSize + 1),
                UInt64(width - kernelSize + 1),
            ]
        }
    }

    private var arrayDataType: ArrayFeatureType.ArrayDataType {
        switch dtype {
        case .float16: .float16
        case .float32: .float32
        }
    }

    private var milDataType: MILSpec_DataType {
        switch dtype {
        case .float16: .float16
        case .float32: .float32
        }
    }

    public init(
        batch: Int64? = nil,
        channels: Int64,
        height: Int64,
        width: Int64,
        outChannels: Int64,
        kernelSize: Int64,
        dtype: DType = .float16
    ) {
        self.batch = batch
        self.channels = channels
        self.height = height
        self.width = width
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dtype = dtype
    }

    public func model(asNeuralNetwork: Bool) async throws -> MLModel {
        let spec = self.spec(asNeuralNetwork: asNeuralNetwork)
        let data = try spec.serializedData()
        let asset = try MLModelAsset(specification: data)
        return try await MLModel.load(asset: asset, configuration: MLModelConfiguration())
    }

    private func spec(asNeuralNetwork: Bool) -> Model {
        if asNeuralNetwork {
            Model(
                description: self.modelDescription(),
                neuralNetwork: self.neuralNetwork()
            )
        } else {
            Model(
                description: self.modelDescription(),
                mlProgram: self.mlProgram()
            )
        }
    }

    private func modelDescription() -> ModelDescription {
        ModelDescription(
            input: [
                FeatureDescription(
                    name: "input",
                    type: FeatureType(multiArray: inShape.map({Int64($0)}), dataType: arrayDataType)
                ),
            ],
            output: [
                FeatureDescription(
                    name: "output",
                    type: FeatureType(multiArray: outShape.map({Int64($0)}), dataType: arrayDataType)
                ),
            ]
        )
    }

    private func mlProgram() -> MILSpec_Program {
        MILSpec_Program(
            version: 1,
            functions: ["main": MILSpec_Function(
                inputs: [
                    MILSpec_NamedValueType(
                        name: "input",
                        type: MILSpec_ValueType(
                            tensorType: milDataType,
                            shape: inShape
                        )
                    ),
                ],
                opset: "CoreML6",
                blockSpecializations: ["CoreML6" : MILSpec_Block(
                    outputs: ["output"],
                    operations: [
                        MILSpec_Operation(
                            constWithName: "w",
                            opName: "declare_w",
                            value: MILSpec_Value(
                                type: MILSpec_ValueType(
                                    tensorType: milDataType,
                                    shape: [
                                        UInt64(outChannels),
                                        UInt64(channels),
                                        UInt64(kernelSize),
                                        UInt64(kernelSize),
                                    ]
                                ),
                                immediateValue: MILSpec_Value.ImmediateValue(
                                    tensor: createTensorValue(size: outChannels * channels * kernelSize * kernelSize)
                                )
                            )
                        ),
                        MILSpec_Operation(
                            constWithName: "b",
                            opName: "declare_b",
                            value: MILSpec_Value(
                                type: MILSpec_ValueType(
                                    tensorType: milDataType,
                                    shape: [UInt64(outChannels)]
                                ),
                                immediateValue: MILSpec_Value.ImmediateValue(
                                    tensor: createTensorValue(size: outChannels)
                                )
                            )
                        ),
                        MILSpec_Operation(
                            conv: "input",
                            bias: "b",
                            dilations: MILSpec_Value(immediateInts: [1, 1]),
                            groups: 1.toSpecValue(),
                            pad: MILSpec_Value(immediateInts: [0, 0, 0, 0]),
                            padType: "custom".toSpecValue(),
                            strides: MILSpec_Value(immediateInts: [1, 1]),
                            weight: "w",
                            outName: "output",
                            outType: MILSpec_ValueType(tensorType: milDataType, shape: outShape),
                            opName: "apply_conv"
                        )
                    ]
                )]
            )],
            attributes: ["buildInfo": MILSpec_Value(immediateStringDict: [
                "coremltools-version": "7.2",
                "coremltools-component-torch": "2.2.0",
                "coremltools-source-dialect": "TorchScript",
            ])]
        )
    }

    private func createTensorValue(size: Int64) -> MILSpec_TensorValue {
        switch dtype {
        case .float16:
            MILSpec_TensorValue(bytes: Data([UInt8](repeating: 0, count: Int(size * 2))))
        case .float32:
            MILSpec_TensorValue(floats: [Float](repeating: 0, count: Int(size)))
        }
    }

    private func neuralNetwork() -> NeuralNetwork {
        NeuralNetwork(
            layers: [
                NeuralNetworkLayer(
                    convolution: ConvolutionLayerParams(
                        outputChannels: UInt64(outChannels),
                        kernelChannels: UInt64(channels),
                        nGroups: 1,
                        kernelSize: [UInt64(kernelSize), UInt64(kernelSize)],
                        stride: [1, 1],
                        dilationFactor: [1, 1],
                        hasBias: true,
                        weights: createWeightParams(size: outChannels * channels * kernelSize * kernelSize),
                        bias: createWeightParams(size: outChannels),
                        outputShape: outShape
                    ),
                    name: "conv",
                    input: ["input"],
                    output: ["output"],
                    inputTensor: [Tensor(dimValue: inShape.map({Int64($0)}))],
                    outputTensor: [Tensor(dimValue: outShape.map({Int64($0)}))],
                    isUpdatable: true
                )
            ],
            updateParams: NetworkUpdateParameters()
        )
    }

    private func createWeightParams(size: Int64) -> WeightParams {
        switch dtype {
        case .float16:
            WeightParams(
                float16Value: Data([UInt8](repeating: 0, count: Int(2 * size))),
                isUpdatable: true
            )
        case .float32:
            WeightParams(
                floatValue: [Float](repeating: 0, count: Int(size)),
                isUpdatable: true
            )
        }
    }
}
