import CoreML
import CoreMLProto

public struct Linear {
  public let batch: Int64
  public let inSize: Int64
  public let outSize: Int64
  public let weight: TensorData
  public let bias: TensorData

  public var inShape: [UInt64] {
    [UInt64(batch), UInt64(inSize)]
  }

  public var outShape: [UInt64] {
    [UInt64(batch), UInt64(outSize)]
  }

  private var arrayDataType: ArrayFeatureType.ArrayDataType {
    weight.dtype.arrayDataType
  }

  private var milDataType: MILSpec_DataType {
    weight.dtype.milDataType
  }

  public init(
    batch: Int64,
    inSize: Int64,
    outSize: Int64,
    weight: TensorData,
    bias: TensorData
  ) {
    self.batch = batch
    self.inSize = inSize
    self.outSize = outSize
    self.weight = weight
    self.bias = bias
    assert(weight.dtype == bias.dtype)
  }

  public func model() async throws -> MLModel {
    let spec = self.spec()
    let data = try spec.serializedData()
    let asset = try MLModelAsset(specification: data)
    return try await MLModel.load(asset: asset, configuration: MLModelConfiguration())
  }

  private func spec() -> Model {
    Model(
      description: self.modelDescription(),
      mlProgram: self.mlProgram()
    )
  }

  private func modelDescription() -> ModelDescription {
    ModelDescription(
      input: [
        FeatureDescription(
          name: "input",
          type: FeatureType(multiArray: inShape.map({ Int64($0) }), dataType: arrayDataType)
        )
      ],
      output: [
        FeatureDescription(
          name: "output",
          type: FeatureType(multiArray: outShape.map({ Int64($0) }), dataType: arrayDataType)
        )
      ]
    )
  }

  private func mlProgram() -> MILSpec_Program {
    MILSpec_Program(
      version: 1,
      functions: [
        "main": MILSpec_Function(
          inputs: [
            MILSpec_NamedValueType(
              name: "input",
              type: MILSpec_ValueType(
                tensorType: milDataType,
                shape: inShape
              )
            )
          ],
          opset: "CoreML6",
          blockSpecializations: [
            "CoreML6": MILSpec_Block(
              outputs: ["output"],
              operations: [
                MILSpec_Operation(
                  constWithName: "w",
                  opName: "declare_w",
                  value: MILSpec_Value(
                    type: MILSpec_ValueType(
                      tensorType: milDataType,
                      shape: [
                        UInt64(outSize),
                        UInt64(inSize),
                      ]
                    ),
                    immediateValue: MILSpec_Value.ImmediateValue(
                      tensor: weight.tensorValue
                    )
                  )
                ),
                MILSpec_Operation(
                  constWithName: "b",
                  opName: "declare_b",
                  value: MILSpec_Value(
                    type: MILSpec_ValueType(
                      tensorType: milDataType,
                      shape: [UInt64(outSize)]
                    ),
                    immediateValue: MILSpec_Value.ImmediateValue(
                      tensor: bias.tensorValue
                    )
                  )
                ),
                MILSpec_Operation(
                  linear: "input",
                  weight: "w",
                  bias: "b",
                  outName: "output",
                  outType: MILSpec_ValueType(tensorType: milDataType, shape: outShape),
                  opName: "apply_linear"
                ),
              ]
            )
          ]
        )
      ],
      attributes: [
        "buildInfo": MILSpec_Value(immediateStringDict: [
          "coremltools-version": "7.2",
          "coremltools-component-torch": "2.2.0",
          "coremltools-source-dialect": "TorchScript",
        ])
      ]
    )
  }
}
