import CoreMLProto
import CoreML

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