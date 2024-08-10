import CoreMLProto
import CoreML

struct Conv2D {
    let batch: Int64
    let channels: Int64
    let height: Int64
    let width: Int64
    let outChannels: Int64
    let kernelSize: Int64

    var inShape: [UInt64] {
        [
            UInt64(batch),
            UInt64(channels),
            UInt64(height),
            UInt64(width),
        ]
    }

    var outShape: [UInt64] {
        [
            UInt64(batch),
            UInt64(outChannels),
            UInt64(height - kernelSize + 1),
            UInt64(width - kernelSize + 1),
        ]
    }

    func model() async throws -> MLModel {
        let spec = self.spec()
        let data = try spec.serializedData()
        let asset = try MLModelAsset(specification: data)
        return try await MLModel.load(asset: asset, configuration: MLModelConfiguration())
    }

    func spec() -> Model {
        Model(
            description: ModelDescription(
                input: [
                    FeatureDescription(
                        name: "input",
                        type: FeatureType(multiArray: inShape.map({Int64($0)}), dataType: .float32)
                    ),
                ],
                output: [
                    FeatureDescription(
                        name: "output",
                        type: FeatureType(multiArray: outShape.map({Int64($0)}), dataType: .float16)
                    ),
                ]
            ),
            mlProgram: MILSpec_Program(
                version: 1,
                functions: ["main": MILSpec_Function(
                    inputs: [
                        MILSpec_NamedValueType(
                            name: "input",
                            type: MILSpec_ValueType(
                                tensorType: .float32,
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
                                        tensorType: .float16,
                                        shape: [
                                            UInt64(outChannels),
                                            UInt64(channels),
                                            UInt64(kernelSize),
                                            UInt64(kernelSize),
                                        ]
                                    ),
                                    immediateValue: MILSpec_Value.ImmediateValue(
                                        tensor: MILSpec_TensorValue(
                                            bytes: Data([UInt8](
                                                repeating: 0,
                                                count: Int(2 * outChannels * channels * kernelSize * kernelSize)
                                            ))
                                        )
                                    )
                                )
                            ),
                            MILSpec_Operation(
                                constWithName: "b",
                                opName: "declare_b",
                                value: MILSpec_Value(
                                    type: MILSpec_ValueType(
                                        tensorType: .float16,
                                        shape: [UInt64(outChannels)]
                                    ),
                                    immediateValue: MILSpec_Value.ImmediateValue(
                                        tensor: MILSpec_TensorValue(
                                            bytes: Data([UInt8](
                                                repeating: 0,
                                                count: Int(outChannels * 2)
                                            ))
                                        )
                                    )
                                )
                            ),
                            MILSpec_Operation(
                                cast: "input",
                                typeName: "fp16".toSpecValue(),
                                opName: "cast_input",
                                outName: "input_as_fp16",
                                outType: MILSpec_ValueType(
                                    tensorType: .float16,
                                    shape: inShape
                                )
                            ),
                            MILSpec_Operation(
                                conv: "input_as_fp16",
                                bias: "b",
                                dilations: MILSpec_Value(immediateInts: [1, 1]),
                                groups: 1.toSpecValue(),
                                pad: MILSpec_Value(immediateInts: [0, 0, 0, 0]),
                                padType: "custom".toSpecValue(),
                                strides: MILSpec_Value(immediateInts: [1, 1]),
                                weight: "w",
                                outName: "output",
                                outType: MILSpec_ValueType(tensorType: .float16, shape: outShape),
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
        )
    }
}
