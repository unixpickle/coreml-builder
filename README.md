# coreml-builder

At the moment, Apple's CoreML library is the main way to run models on the Apple Neural Engine. It would be nice if we could just call APIs within it directly, such as conv2d() or matmul(), but this is not how it works. Instead, the CoreML framework itself does not _create_ models -- for that, you typically need something like [coremltools](https://github.com/apple/coremltools).

The goal of this project is to provide high-level APIs to construct CoreML models within a Swift program. The end-goal would be to even expose high-level APIs like `matmul()` which, under the hood, create a new model and execute it with CoreML.
