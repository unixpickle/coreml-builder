import CoreML
import CoreMLProto

public enum DType {
  case float32
  case float16

  public var arrayDataType: ArrayFeatureType.ArrayDataType {
    switch self {
    case .float16: .float16
    case .float32: .float32
    }
  }

  public var milDataType: MILSpec_DataType {
    switch self {
    case .float16: .float16
    case .float32: .float32
    }
  }

  public var coreMLType: MLMultiArrayDataType {
    switch self {
    case .float16: .float16
    case .float32: .float32
    }
  }
}

public enum TensorData {
  case float16(Data)
  case float32([Float])

  public var dtype: DType {
    switch self {
    case .float16:
      .float16
    case .float32:
      .float32
    }
  }

  public var tensorValue: MILSpec_TensorValue {
    switch self {
    case .float16(let data):
      MILSpec_TensorValue(bytes: data)
    case .float32(let arr):
      MILSpec_TensorValue(floats: arr)
    }
  }

  public var weightParams: WeightParams {
    switch self {
    case .float16(let data):
      WeightParams(
        float16Value: data,
        isUpdatable: true
      )
    case .float32(let arr):
      WeightParams(
        floatValue: arr,
        isUpdatable: true
      )
    }
  }

  public static func zeros(dtype: DType, size: Int) -> TensorData {
    switch dtype {
    case .float16:
      .float16(Data([UInt8](repeating: 0, count: size * 2)))
    case .float32:
      .float32([Float](repeating: 0, count: size))
    }
  }
}
