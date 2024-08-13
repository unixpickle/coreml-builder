import Foundation
import CoreMLBuilder
import CoreML

typealias MatmulBackend = (Int64, DType) async throws -> (MLModel, any MLFeatureProvider)

@main
struct Main {
    static func main() async {
        let backends = backends()
        for size in [Int64]([2048, 4096, 4096+2048, 8192, 16384]) {
            print("matrix size: \(size)")
            for dtype in [DType.float16, DType.float32] {
                print(" dtype \(dtype)")
                for (name, backend) in backends {
                    print("  backend: \(name)")
                    do {
                        let t1 = DispatchTime.now()
                        let (model, features) = try await backend(size, dtype)
                        let t2 = DispatchTime.now()

                        // Warmup
                        try model.prediction(from: features)

                        let t3 = DispatchTime.now()
                        try model.prediction(from: features)
                        let t4 = DispatchTime.now()
                        let runNanos = Double(t4.uptimeNanoseconds - t3.uptimeNanoseconds)
                        let totalNanos = runNanos + Double(t2.uptimeNanoseconds - t1.uptimeNanoseconds)
                        print(
                            "   => \(round(2 * pow(Double(size), 3) / runNanos)) GFLOPs" +
                            " (with setup: \(round(2 * pow(Double(size), 3) / totalNanos)) GFLOPs)"
                        )
                    } catch {
                        print("   => error: \(error)")
                    }
                }
            }
        }
    }
}

func backends() -> [String : MatmulBackend] {
    return [
        "conv": convMatmul(asNN: false),
        "conv_nn": convMatmul(asNN: true),
        "conv_dynamic": dynamicConvMatmul(),
        "linear": linearMatmul(),
        "matmul_op": matmulOpMatmul(),
    ]
}

func convMatmul(asNN: Bool) -> MatmulBackend {
    { size, dtype in
        let conv = Conv2D(
            batch: asNN ? nil : 1,
            channels: size,
            height: 1,
            width: size,
            outChannels: size,
            kernelSize: 1,
            dtype: dtype,
            weight: TensorData.zeros(dtype: dtype, size: Int(size * size)),
            bias: TensorData.zeros(dtype: dtype, size: Int(size))
        )
        let model = try await conv.model(asNeuralNetwork: asNN)
        let arr = try MLMultiArray(
            shape: (asNN ? [] : [1]) + [NSNumber(value: size), 1, NSNumber(value: size)],
            dataType: dtype.coreMLType
        )
        let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input": MLFeatureValue(multiArray: arr)
        ])
        return (model, featureProvider)
    }
}

func dynamicConvMatmul() -> MatmulBackend {
    { size, dtype in
        let conv = Conv2D(
            batch: 1,
            channels: size,
            height: 1,
            width: size,
            outChannels: size,
            kernelSize: 1,
            dtype: dtype
        )
        let model = try await conv.model(asNeuralNetwork: false)
        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: size), 1, NSNumber(value: size)],
            dataType: dtype.coreMLType
        )
        let w = try MLMultiArray(
            shape: [NSNumber(value: size), NSNumber(value: size), 1, 1],
            dataType: dtype.coreMLType
        )
        let b = try MLMultiArray(
            shape: [NSNumber(value: size)],
            dataType: dtype.coreMLType
        )
        let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input": MLFeatureValue(multiArray: arr),
            "w": MLFeatureValue(multiArray: w),
            "b": MLFeatureValue(multiArray: b)
        ])
        return (model, featureProvider)
    }
}

func linearMatmul() -> MatmulBackend {
    { size, dtype in
        let lin = Linear(
            batch: size,
            inSize: size,
            outSize: size,
            weight: TensorData.zeros(dtype: dtype, size: Int(size * size)),
            bias: TensorData.zeros(dtype: dtype, size: Int(size))
        )
        let model = try await lin.model()
        let arr = try MLMultiArray(
            shape: [NSNumber(value: size), NSNumber(value: size)],
            dataType: dtype.coreMLType
        )
        let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input": MLFeatureValue(multiArray: arr)
        ])
        return (model, featureProvider)
    }
}

func matmulOpMatmul() -> MatmulBackend {
    { size, dtype in
        let matmul = Matmul(
            xShape: (size, size),
            yShape: (size, size),
            transposeX: false,
            transposeY: false,
            dtype: dtype
        )
        let model = try await matmul.model()
        let x = try MLMultiArray(
            shape: [NSNumber(value: size), NSNumber(value: size)],
            dataType: dtype.coreMLType
        )
        let y = try MLMultiArray(
            shape: [NSNumber(value: size), NSNumber(value: size)],
            dataType: dtype.coreMLType
        )
        let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: x),
            "y": MLFeatureValue(multiArray: y)
        ])
        return (model, featureProvider)
    }
}
