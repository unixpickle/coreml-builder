import CoreML
import CoreMLProto

public struct Matmul {
  public let xShape: (Int64, Int64)
  public let yShape: (Int64, Int64)
  public let transposeX: Bool
  public let transposeY: Bool
  public let x: TensorData?
  public let y: TensorData?
  public let dtype: DType

  public var outShape: (Int64, Int64) {
    (transposeX ? xShape.1 : xShape.0, transposeY ? yShape.0 : yShape.1)
  }

  private var arrayDataType: ArrayFeatureType.ArrayDataType {
    dtype.arrayDataType
  }

  private var milDataType: MILSpec_DataType {
    dtype.milDataType
  }

  public init(
    xShape: (Int64, Int64),
    yShape: (Int64, Int64),
    transposeX: Bool,
    transposeY: Bool,
    x: TensorData? = nil,
    y: TensorData? = nil,
    dtype: DType = .float16
  ) {
    self.xShape = xShape
    self.yShape = yShape
    self.transposeX = transposeX
    self.transposeY = transposeY
    self.x = x
    self.y = y
    if let x = x {
      assert(x.dtype == dtype)
    }
    if let y = y {
      assert(y.dtype == dtype)
    }
    self.dtype = dtype
  }

  public func model(computeUnits: MLComputeUnits = .all) async throws -> MLModel {
    let spec = self.spec()
    let data = try spec.serializedData()
    let asset = try MLModelAsset(specification: data)
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    return try await MLModel.load(asset: asset, configuration: config)
  }

  private func spec() -> Model {
    Model(
      description: self.modelDescription(),
      mlProgram: self.mlProgram()
    )
  }

  private func modelDescription() -> ModelDescription {
    ModelDescription(
      input: (x != nil
        ? []
        : [
          FeatureDescription(
            name: "x",
            type: FeatureType(
              multiArray: [Int64(xShape.0), Int64(xShape.1)], dataType: arrayDataType)
          )
        ])
        + (y != nil
          ? []
          : [
            FeatureDescription(
              name: "y",
              type: FeatureType(
                multiArray: [Int64(yShape.0), Int64(yShape.1)], dataType: arrayDataType)
            )
          ]),
      output: [
        FeatureDescription(
          name: "output",
          type: FeatureType(multiArray: [outShape.0, outShape.1], dataType: arrayDataType)
        )
      ]
    )
  }

  private func mlProgram() -> MILSpec_Program {
    let xType = MILSpec_ValueType(
      tensorType: milDataType,
      shape: [UInt64(xShape.0), UInt64(xShape.1)]
    )
    let yType = MILSpec_ValueType(
      tensorType: milDataType,
      shape: [UInt64(yShape.0), UInt64(yShape.1)]
    )

    return MILSpec_Program(
      version: 1,
      functions: [
        "main": MILSpec_Function(
          inputs: (x != nil
            ? []
            : [
              MILSpec_NamedValueType(name: "x", type: xType)
            ])
            + (y != nil
              ? []
              : [
                MILSpec_NamedValueType(name: "y", type: yType)
              ]),
          opset: "CoreML6",
          blockSpecializations: [
            "CoreML6": MILSpec_Block(
              outputs: ["output"],
              operations: (declareInput("x", shape: xShape, data: x)
                + declareInput("y", shape: yShape, data: y) + [
                  MILSpec_Operation(
                    matmul: "x",
                    y: "y",
                    transposeX: transposeX,
                    transposeY: transposeY,
                    outName: "output",
                    outType: MILSpec_ValueType(
                      tensorType: milDataType,
                      shape: [UInt64(outShape.0), UInt64(outShape.1)]
                    ),
                    opName: "apply_matmul"
                  )
                ])
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

  private func declareInput(_ name: String, shape: (Int64, Int64), data: TensorData?)
    -> [MILSpec_Operation]
  {
    guard let data = data else {
      return []
    }
    return [
      MILSpec_Operation(
        constWithName: name,
        opName: "declare_\(name)",
        value: MILSpec_Value(
          type: MILSpec_ValueType(
            tensorType: milDataType,
            shape: [UInt64(shape.0), UInt64(shape.1)]
          ),
          immediateValue: MILSpec_Value.ImmediateValue(
            tensor: data.tensorValue
          )
        )
      )
    ]
  }
}
