// Implement a fast 32k by 32k matmul using multiple CoreML graphs
// from different threads at once.
//
// On my M2 Max, a single thread gets 3227 GFLOPS, while two threads
// gets 6350 GFLOPS and four threads get 7418 GFLOPS.
// Utilization of the ANE according to powermetrics never surpasses
// about 55%, which is surprisingly accurate for these results.

import Foundation
import CoreMLBuilder
import CoreML

@main
struct Main {
    static func main() async {
        let dtype = DType.float16
        let blockSize = 4096
        let maxConcurrency = 4
        let numBlocks = 8
        let totalSize = blockSize * numBlocks

        do {
            print("allocating inputs...")
            let blocksX: [MLMultiArray]
            let blocksY: [MLMultiArray]
            blocksX = try (0..<(numBlocks*numBlocks)).map { _ in
                try MLMultiArray(
                    shape: [NSNumber(value: blockSize), NSNumber(value: blockSize)],
                    dataType: dtype.coreMLType
                )
            }
            blocksY = try (0..<(numBlocks*numBlocks)).map { _ in
                try MLMultiArray(
                    shape: [NSNumber(value: blockSize), NSNumber(value: blockSize)],
                    dataType: dtype.coreMLType
                )
            }

            print("creating matmul model...")
            let matmul = Matmul(
                xShape: (Int64(blockSize), Int64(blockSize)),
                yShape: (Int64(blockSize), Int64(blockSize)),
                transposeX: false,
                transposeY: false,
                dtype: dtype
            )

            print("dispatching workers...")
            let semaphore = DispatchSemaphore(value: maxConcurrency)
            let t1 = DispatchTime.now()
            let group = DispatchGroup()
            for i in 0..<numBlocks {
                for j in 0..<numBlocks {
                    group.enter()
                    let model = try await matmul.model()
                    DispatchQueue.global().async {
                        semaphore.wait()
                        defer {
                            group.leave()
                            semaphore.signal()
                        }
                        do {
                            // Note that we would typically accumulate these values
                            // here, but we don't do that here to avoid complexity.
                            for k in 0..<numBlocks {
                                let feats: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                                    "x": MLFeatureValue(multiArray: blocksX[i*numBlocks + k]),
                                    "y": MLFeatureValue(multiArray: blocksY[j + numBlocks*k])
                                ])
                                try model.prediction(from: feats)
                            }
                        } catch {
                            print("error in background worker! \(error)")
                        }
                    }
                }
            }
            await withCheckedContinuation { continuation in
                group.notify(queue: .global()) {
                    continuation.resume()
                }
            }
            let t2 = DispatchTime.now()
            let nanos = Double(t2.uptimeNanoseconds - t1.uptimeNanoseconds)
            let flops = pow(Double(totalSize), 3) * 2
            print("GFLOPS: \(flops / nanos)")
        } catch {
            print("error: \(error)")
        }
    }
}
