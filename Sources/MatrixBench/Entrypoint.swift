import Foundation
import CoreMLBuilder
import CoreML

@main
struct Main {
    static func main() async {
        for size in [Int64]([2048, 4096, 4096+2048, 8192, 16384]) {
            print("matrix size: \(size)")
            for asNN in [false, true] {
                let conv = Conv2D(
                    batch: asNN ? nil : 1,
                    channels: size,
                    height: 1,
                    width: size,
                    outChannels: size,
                    kernelSize: 1
                )
                do {
                    let model = try await conv.model(asNeuralNetwork: asNN)
                    let arr = try MLMultiArray(shape: (asNN ? [] : [1]) + [NSNumber(value: size), 1, NSNumber(value: size)], dataType: .float16)
                    let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                        "input": MLFeatureValue(multiArray: arr)
                    ])

                    // Warmup
                    try model.prediction(from: featureProvider)

                    let t1 = DispatchTime.now()
                    try model.prediction(from: featureProvider)
                    let t2 = DispatchTime.now()
                    let elapsedNanos = Double(t2.uptimeNanoseconds - t1.uptimeNanoseconds)
                    print(" => \(2 * pow(Double(size), 3) / elapsedNanos) GFLOPs (asNN=\(asNN))")
                } catch {
                    print(" => error (asNN=\(asNN)): \(error)")
                }
            }
        }
    }
}
