import Foundation
import CoreMLBuilder
import CoreML

@main
struct Main {
    static func main() async {
        for size in [Int64]([2048, 4096, 4096+2048, 8192, 16384]) {
            print("matrix size: \(size)")
            for dtype in [DType.float16, DType.float32] {
                print(" dtype \(dtype)")
                for asNN in [false, true] {
                    print("  conv (asNN=\(asNN))")
                    let conv = Conv2D(
                        batch: asNN ? nil : 1,
                        channels: size,
                        height: 1,
                        width: size,
                        outChannels: size,
                        kernelSize: 1,
                        dtype: dtype
                    )
                    do {
                        let model = try await conv.model(asNeuralNetwork: asNN)
                        let arr = try MLMultiArray(
                            shape: (asNN ? [] : [1]) + [NSNumber(value: size), 1, NSNumber(value: size)],
                            dataType: dtype.coreMLType
                        )
                        let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                            "input": MLFeatureValue(multiArray: arr)
                        ])

                        // Warmup
                        try model.prediction(from: featureProvider)

                        let t1 = DispatchTime.now()
                        try model.prediction(from: featureProvider)
                        let t2 = DispatchTime.now()
                        let elapsedNanos = Double(t2.uptimeNanoseconds - t1.uptimeNanoseconds)
                        print("   => \(2 * pow(Double(size), 3) / elapsedNanos) GFLOPs")
                    } catch {
                        print("   => error: \(error)")
                    }
                }
                print("  linear")
                let lin = Linear(
                    batch: size,
                    inSize: size,
                    outSize: size,
                    dtype: dtype
                )
                do {
                    let model = try await lin.model()
                    let arr = try MLMultiArray(
                        shape: [NSNumber(value: size), NSNumber(value: size)],
                        dataType: dtype.coreMLType
                    )
                    let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                        "input": MLFeatureValue(multiArray: arr)
                    ])

                    // Warmup
                    try model.prediction(from: featureProvider)

                    let t1 = DispatchTime.now()
                    try model.prediction(from: featureProvider)
                    let t2 = DispatchTime.now()
                    let elapsedNanos = Double(t2.uptimeNanoseconds - t1.uptimeNanoseconds)
                    print("   => \(2 * pow(Double(size), 3) / elapsedNanos) GFLOPs")
                } catch {
                    print("   => error: \(error)")
                }
            }
        }
    }
}
