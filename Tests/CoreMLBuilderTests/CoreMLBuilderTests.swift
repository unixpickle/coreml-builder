import XCTest
import CoreML
@testable import CoreMLBuilder

final class CoreMLBuilderTests: XCTestCase {
    func testConvModel() async throws {
        let conv = Conv2D(
            batch: 1,
            channels: 8192,
            height: 1,
            width: 512,
            outChannels: 512,
            kernelSize: 1
        )
        let model = try await conv.model()
        let arr = try MLMultiArray(shape: [1, 8192, 1, 512], dataType: .float32)
        let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input": MLFeatureValue(multiArray: arr)
        ])
        let result = try model.prediction(from: featureProvider)
        XCTAssertEqual(result.featureValue(for: "output")?.multiArrayValue?.shape, [1, 512, 1, 512])
    }
}
