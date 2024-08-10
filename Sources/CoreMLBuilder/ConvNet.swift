import CoreMLProto

func convolution2D() -> Model {
    Model(
        description: ModelDescription(
            input: [
                FeatureDescription(
                    name: "input",
                    type: FeatureType(multiArray: [1, 8192, 1, 512], dataType: .float32)
                ),
            ],
            output: [
                FeatureDescription(
                    name: "output",
                    type: FeatureType(multiArray: [1, 512, 1, 512], dataType: .float16)
                ),
            ]
        ),
        mlProgram: MILSpec_Program(
            version: 1,
            functions: ["main": MILSpec_Function(
                inputs: [],
                opset: "CoreML6",
                blockSpecializations: ["CoreML6" : MILSpec_Block(
                    inputs: [],
                    outputs: [],
                    operations: []
                )]
            )],
            attributes: ["buildInfo": MILSpec_Value(immediateStringDict: [
                "coremltools-version": "7.2",
                "coremltools-component-torch": "2.2.0",
                "coremltools-source-dialect": "TorchScript",
            ])]
        )
    )
}