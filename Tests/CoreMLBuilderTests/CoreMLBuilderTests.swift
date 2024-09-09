import CoreML
import XCTest

@testable import CoreMLBuilder

final class CoreMLBuilderTests: XCTestCase {
  func testConvModel() async throws {
    for asNeuralNetwork in [false, true] {
      let conv = Conv2D(
        batch: asNeuralNetwork ? nil : 1,
        channels: 8192,
        height: 1,
        width: 512,
        outChannels: 512,
        kernelSize: 1,
        weight: TensorData.zeros(dtype: .float16, size: 8192 * 512),
        bias: TensorData.zeros(dtype: .float16, size: 512)
      )
      let model = try await conv.model(asNeuralNetwork: asNeuralNetwork)
      let arr = try MLMultiArray(
        shape: asNeuralNetwork ? [8192, 1, 512] : [1, 8192, 1, 512], dataType: .float16)
      let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
        "input": MLFeatureValue(multiArray: arr)
      ])
      let result = try model.prediction(from: featureProvider)
      XCTAssertEqual(
        result.featureValue(for: "output")?.multiArrayValue?.shape,
        asNeuralNetwork ? [512, 1, 512] : [1, 512, 1, 512]
      )
    }
  }

  func testConvModelPassParams() async throws {
    let conv = Conv2D(
      batch: 1,
      channels: 8192,
      height: 1,
      width: 512,
      outChannels: 512,
      kernelSize: 1
    )
    let model = try await conv.model(asNeuralNetwork: false)
    let arr = try MLMultiArray(shape: [1, 8192, 1, 512], dataType: .float16)
    let w = try MLMultiArray(shape: [512, 8192, 1, 1], dataType: .float16)
    let b = try MLMultiArray(shape: [512], dataType: .float16)
    let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
      "input": MLFeatureValue(multiArray: arr),
      "w": MLFeatureValue(multiArray: w),
      "b": MLFeatureValue(multiArray: b),
    ])
    let result = try model.prediction(from: featureProvider)
    XCTAssertEqual(
      result.featureValue(for: "output")?.multiArrayValue?.shape,
      [1, 512, 1, 512]
    )
  }
}
