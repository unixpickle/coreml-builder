// Implement a fast 32k by 32k matmul using multiple CoreML graphs
// from different threads at once.
//
// On my M2 Max, a single thread gets 3227 GFLOPS, while two threads
// gets 6350 GFLOPS and four threads get 7418 GFLOPS.
// Utilization of the ANE according to powermetrics never surpasses
// about 55%, which is surprisingly accurate for these results.

import ArgumentParser
@preconcurrency import CoreML
import CoreMLBuilder
import Foundation

@main
struct BlockANEMatmul: ParsableCommand {
  @Option(help: "Size of smaller matrices (rows).")
  var blockRows: Int = 4096

  @Option(help: "Size of smaller matrices (columns).")
  var blockCols: Int = 4096

  @Option(help: "Block column of right-hand-size matrix.")
  var blockOut: Int = 4096

  @Option(help: "Size of full, large matrix (input dim).")
  var outRows: Int = 32768

  @Option(help: "Size of full, large matrix (inner dim).")
  var innerDim: Int = 32768

  @Option(help: "Size of full, large matrix (output dim).")
  var outCols: Int = 32768

  @Option(help: "Worker threads to run at once.")
  var concurrency: Int = 4

  mutating func run() throws {
    let sem = DispatchSemaphore(value: 0)
    let unmutable = self
    Task {
      defer {
        sem.signal()
      }
      do {
        try await unmutable.runAsync()
      } catch {
        print("error: \(error)")
      }
    }
    sem.wait()
  }

  func runAsync() async throws {
    let dtype = DType.float16
    assert(outRows % blockRows == 0, "blockRows \(blockRows) must divide outRows \(outRows)")
    assert(innerDim % blockCols == 0, "blockCols \(blockCols) must divide innerDim \(innerDim)")
    assert(outCols % blockOut == 0, "blockOut \(blockOut) must divide outCols \(outCols)")
    let aRowBlocks = outRows / blockRows
    let aColBlocks = innerDim / blockCols
    let bRowBlocks = innerDim / blockCols
    let bColBlocks = outCols / blockOut
    print("allocating inputs...")
    let blocksX: [MLMultiArray]
    let blocksY: [MLMultiArray]
    blocksX = try (0..<(aRowBlocks * aColBlocks)).map { _ in
      try MLMultiArray(
        shape: [NSNumber(value: blockRows), NSNumber(value: blockCols)],
        dataType: dtype.coreMLType
      )
    }
    blocksY = try (0..<(bRowBlocks * bColBlocks)).map { _ in
      try MLMultiArray(
        shape: [NSNumber(value: blockCols), NSNumber(value: blockOut)],
        dataType: dtype.coreMLType
      )
    }

    print("creating matmul model...")
    let matmul = Matmul(
      xShape: (Int64(blockRows), Int64(blockCols)),
      yShape: (Int64(blockCols), Int64(blockOut)),
      transposeX: false,
      transposeY: false,
      dtype: dtype
    )
    let models = try await Queue(matmul: matmul, concurrency: concurrency)

    print("dispatching workers...")
    let t1 = DispatchTime.now()
    let group = DispatchGroup()
    for i in 0..<aRowBlocks {
      for j in 0..<bColBlocks {
        group.enter()
        DispatchQueue.global().async {
          let model = models.get()
          defer {
            models.put(model)
            group.leave()
          }
          do {
            // Note that we would typically accumulate these values
            // here, but we don't do that here to avoid complexity.
            for k in 0..<aColBlocks {
              let feats: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: blocksX[i * aColBlocks + k]),
                "y": MLFeatureValue(multiArray: blocksY[j + bColBlocks * k]),
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
    let flops = Double(outRows * innerDim * outCols) * 2
    print("GFLOPS: \(flops / nanos)")
  }
}

final class Queue: @unchecked Sendable {
  private let queue: DispatchQueue
  private var free: [MLModel] = []
  private var sem: DispatchSemaphore

  init(matmul: Matmul, concurrency: Int) async throws {
    queue = DispatchQueue(label: "serial")
    for _ in 0..<concurrency {
      free.append(try await matmul.model(computeUnits: .cpuAndNeuralEngine))
    }
    sem = DispatchSemaphore(value: concurrency)
  }

  func get() -> MLModel {
    sem.wait()
    return DispatchQueue.main.sync {
      return free.popLast()!
    }
  }

  func put(_ model: MLModel) {
    DispatchQueue.main.sync {
      free.append(model)
    }
    sem.signal()
  }
}
