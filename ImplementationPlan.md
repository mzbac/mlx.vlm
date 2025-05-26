# Implementation Plan for `MLXVLM` Library

**Overall Goal:** Develop a Swift package named `MLXVLM` for loading and running Vision Language Models (VLMs) using the MLX Swift framework. The library should support various VLMs, handle multi-modal inputs (text, images, videos) efficiently, and integrate seamlessly with `MLXLMCommon` for shared LLM functionalities.

**Core Principles:**

- **Modularity:** Each VLM will have its dedicated components (model, processor, configurations).
- **Extensibility:** The design should make it straightforward to add support for new VLMs.
- **Clarity:** Code should be well-structured and readable.
- **Efficiency:** Leverage MLX for performance and manage memory for media processing.
- **`MLXLMCommon` Integration:** Utilize base classes and protocols from `MLXLMCommon` (e.g., `LanguageModel`, `ModelFactory`, `UserInputProcessor`).

---

**Phase 1: `MLXVLM` Package Initialization and Core Definitions**

**Step 1.0: Initialize the `MLXVLM` Swift Package**

- **Objective:** Create the basic SPM project structure for the new library.
- **Agent Instruction (to User/Copilot):**
  1.  "Create a new directory named `MLXVLM` (this will be the root of your package)."
  2.  "Navigate into this `MLXVLM` directory and execute `swift package init --type library --name MLXVLM`."
  3.  "Open the generated `Package.swift` file."
  4.  "Declare the Swift tools version at the top, e.g., `// swift-tools-version:5.9`."
  5.  "Define the `platforms` for your package, e.g., `.macOS(.v13)`."
  6.  "In the `dependencies` array of the `Package` object, add the following:"
      - `.package(url: "https://github.com/ml-explore/mlx-swift.git", branch: "main")` (or a specific version tag)
      - `.package(url: "https://github.com/huggingface/swift-tokenizers.git", branch: "main")` (or a specific version tag)
      - "If `MLXLMCommon` is a local package: `.package(name: "MLXLMCommon", path: "../MLXLMCommon")` (adjust path as needed). If it's a remote package, add its URL and version."
  7.  "In the `targets` array, for the `.library(name: "MLXVLM", targets: ["MLXVLM"])` target, add its `dependencies`:"
      - `.product(name: "MLX", package: "mlx-swift")`
      - `.product(name: "MLXNN", package: "mlx-swift")`
      - `.product(name: "MLXOptimizers", package: "mlx-swift")`
      - `.product(name: "MLXRandom", package: "mlx-swift")`
      - `.product(name: "MLXFast", package: "mlx-swift")`
      - `.product(name: "Tokenizers", package: "swift-tokenizers")`
      - `.product(name: "Hub", package: "swift-tokenizers")` (or `HubApi` if it's a separate product from its package)
      - `"MLXLMCommon"`
  8.  "Also in `targets`, ensure there's a `.testTarget(name: "MLXVLMTests", dependencies: ["MLXVLM"])`."
- **Verification Command (Agent):** `"In the 'MLXVLM' root directory, run 'swift build'."`

**Step 1.1: Define Core `VLMModel` Protocol & `VLMError` Enumeration**

- **Objective:** Establish the central VLM protocol and error types specific to VLM operations.
- **Agent Instruction (to User/Copilot):**

  1.  "Create a new file `Sources/MLXVLM/VLMModel.swift`."
  2.  "Inside `VLMModel.swift`, add the following code. This protocol will be implemented by all VLM classes:"

      ```swift
      import MLX // From mlx-swift
      import MLXLMCommon // Your common library

      /// Marker protocol for Vision Language Models, inheriting capabilities
      /// from LanguageModel (for text generation) and LoRAModel (for LoRA support).
      public protocol VLMModel: LanguageModel, LoRAModel {
          // Currently, no additional VLM-specific methods are mandated by this protocol itself.
          // Model-specific multi-modal handling will primarily reside in their
          // 'prepare' method implementation and internal architecture.
      }
      ```

  3.  "Create a new file `Sources/MLXVLM/VLMError.swift`."
  4.  "Inside `VLMError.swift`, define the VLM-specific error enumeration:"

      ```swift
      import Foundation

      public enum VLMError: LocalizedError {
          case imageRequired
          case maskRequired // If relevant for specific VLM designs
          case singleImageAllowed
          case singleVideoAllowed
          case singleMediaTypeAllowed
          case imageProcessingFailure(String)
          case videoProcessingFailure(String)
          case processing(String) // Generic processing error

          public var errorDescription: String? {
              switch self {
              case .imageRequired:
                  return NSLocalizedString("An image is required for this VLM operation.", comment: "VLMError imageRequired")
              case .maskRequired:
                  return NSLocalizedString("An image mask is required for this VLM operation.", comment: "VLMError maskRequired")
              case .singleImageAllowed:
                  return NSLocalizedString("Only a single image is allowed for this VLM operation.", comment: "VLMError singleImageAllowed")
              case .singleVideoAllowed:
                  return NSLocalizedString("Only a single video is allowed for this VLM operation.", comment: "VLMError singleVideoAllowed")
              case .singleMediaTypeAllowed:
                  return NSLocalizedString("Only a single media type (image or video) is allowed for this VLM operation.", comment: "VLMError singleMediaTypeAllowed")
              case .imageProcessingFailure(let details):
                  return String(format: NSLocalizedString("Failed to process the image: %@", comment: "VLMError imageProcessingFailure"), details)
              case .videoProcessingFailure(let details):
                  return String(format: NSLocalizedString("Failed to process the video: %@", comment: "VLMError videoProcessingFailure"), details)
              case .processing(let details):
                  return String(format: NSLocalizedString("VLM Processing error: %@", comment: "VLMError processing"), details)
              }
          }
      }
      ```

- **Verification Command (Agent):** `"Run 'swift build' in the 'MLXVLM' directory."`

**Step 1.2: Implement `MediaProcessing.swift` for VLM Media Handling**

- **Objective:** Provide a suite of utilities for image and video manipulation common to VLMs.
- **Agent Instruction (to User/Copilot):**
  1.  "Create `Sources/MLXVLM/MediaProcessing.swift`."
  2.  "Import `AVFoundation`, `CoreImage.CIFilterBuiltins`, `MLX`, and `MLXLMCommon` (for `UserInput.Processing`, `VideoFrame`, `ProcessedFrames`)."
  3.  "Define `public struct VideoFrame { public let frame: CIImage; public let timeStamp: CMTime; /* init */ }` and `public struct ProcessedFrames { public let frames: [MLXArray]; public let timestamps: [CMTime]; public let totalDuration: CMTime; /* init */ }`."
  4.  "Implement a static `CIContext` instance: `private let ciContext = CIContext()`."
  5.  "Within an `public enum MediaProcessing`, implement the following static methods. For the detailed logic of each method, refer to `mlx-swift-examples/Libraries/MLXVLM/MediaProcessing.swift`:"
      - `inSRGBToneCurveSpace(_ image: CIImage) -> CIImage`
      - `inLinearToneCurveSpace(_ image: CIImage) -> CIImage`
      - `bestFit(_ size: CGSize, in other: CGSize) -> CGSize`
      - `bestFitScale(_ size: CGSize, in other: CGSize) -> CGFloat`
      - `aspectRatioForResample(_ image: CIImage, size: CGSize) -> Float`
      - `resampleLanczos(_ image: CIImage, to size: CGSize) -> CIImage`
      - `resampleBicubic(_ image: CIImage, to size: CGSize) -> CIImage`
      - `normalize(_ image: CIImage, mean: (CGFloat, CGFloat, CGFloat), std: (CGFloat, CGFloat, CGFloat)) -> CIImage`
      - `asMLXArray(_ image: CIImage, colorSpace: CGColorSpace? = nil) -> MLXArray` (Crucial: ensure output format is typically `[1, C, H, W]`, and handle RGBA to RGB if necessary).
      - `rectSmallerOrEqual(_ extent: CGRect, size: CGSize) -> Bool`
      - `centerCrop(_ extent: CGRect, size: CGSize) -> CGRect`
      - `centerCrop(_ image: CIImage, size: CGSize) -> CIImage`
      - `fitIn(_ size: CGSize, shortestEdge: Int) -> CGSize`
      - `fitIn(_ size: CGSize, longestEdge: Int) -> CGSize`
      - `apply(_ image: CIImage, processing: UserInput.Processing?) -> CIImage` (Uses `UserInput.Processing` from `MLXLMCommon`).
      - `asCIImageSequence(_ asset: AVAsset, samplesPerSecond: Int) async throws -> [CIImage]`
      - `asProcessedSequence(_ asset: AVAsset, samplesPerSecond: Int, frameProcessing: @escaping (VideoFrame) throws -> VideoFrame = { $0 }) async throws -> ProcessedFrames`
      - `asProcessedSequence(_ asset: AVAsset, maxFrames: Int, targetFPS: @escaping (CMTime) -> Double, frameProcessing: @escaping (VideoFrame) throws -> VideoFrame = { $0 }) async throws -> ProcessedFrames`
  6.  "Define a `public extension CIImage` with convenience methods: `resampled(to:method:)`, `toSRGB()`, `toLinear()`, `normalized(mean:std:)`, `asMLXArray()` that call the corresponding `MediaProcessing` static methods."
- **Verification Command (Agent):** `"Run 'swift build'."`
- **Testing Guidance (Agent instructs User/Copilot):**
  1.  "Create `Tests/MLXVLMTests/MediaProcessingTests.swift`."
  2.  "Write unit tests for: `MediaProcessing.bestFitScale`, `MediaProcessing.resampleBicubic` (check output dimensions), `MediaProcessing.normalize` (conceptual check or sample pixel if feasible), and `MediaProcessing.asMLXArray` (check output shape and dtype for a small test `CIImage`)."
  3.  **Verification Command (Agent):** `"Run 'swift test' in the 'MLXVLM' directory."`

---

**Phase 2: VLM Factory, Registries, and Core Loading Infrastructure**

**Step 2.0: Define `BaseProcessorConfiguration`**

- **Objective:** A standard structure for decoding `processor_class` from a VLM's `preprocessor_config.json`.
- **Agent Instruction (to User/Copilot):**

  1.  "Create `Sources/MLXVLM/VLMFactoryInfrastructure.swift`."
  2.  "Add the following struct definition to `VLMFactoryInfrastructure.swift`:"

      ```swift
      import Foundation

      /// Configuration to determine the processor class from preprocessor_config.json
      public struct BaseProcessorConfiguration: Codable, Sendable {
          public let processorClass: String

          enum CodingKeys: String, CodingKey {
              case processorClass = "processor_class"
          }
      }
      ```

- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 2.1: Implement `VLMTypeRegistry`**

- **Objective:** A registry for mapping VLM `model_type` strings (from `config.json`) to their respective model class initializers.
- **Agent Instruction (to User/Copilot):**
  1.  "In `VLMFactoryInfrastructure.swift`, define `public class VLMTypeRegistry: ModelTypeRegistry` (subclassing from `MLXLMCommon.ModelTypeRegistry`)."
  2.  "Include the private helper function for creating model instances:
      `private func createVLM<C: Codable, M>(_ configurationType: C.Type, _ modelInit: @escaping (C) -> M) -> (URL) throws -> M { /* decode JSON from URL into C, then call modelInit */ }`."
  3.  "Implement `public static let shared = VLMTypeRegistry(creators: all())`."
  4.  "Implement an empty `private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] { return [:] }`. This dictionary will be populated as specific VLMs are added."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 2.2: Implement `VLMProcessorTypeRegistry`**

- **Objective:** A registry for mapping VLM `processor_class` strings (from `preprocessor_config.json`) to their processor class initializers.
- **Agent Instruction (to User/Copilot):**
  1.  "In `VLMFactoryInfrastructure.swift`, define `public class VLMProcessorTypeRegistry: ProcessorTypeRegistry` (subclassing from `MLXLMCommon.ProcessorTypeRegistry`)."
  2.  "Include the private helper function for creating processor instances:
      `private func createVLMProcessor<C: Codable, P>(_ configurationType: C.Type, _ processorInit: @escaping (C, any Tokenizer) -> P) -> (URL, any Tokenizer) throws -> P { /* decode JSON from URL into C, then call processorInit with config and tokenizer */ }`."
  3.  "Implement `public static let shared = VLMProcessorTypeRegistry(creators: all())`."
  4.  "Implement an empty `private static func all() -> [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor] { return [:] }`. This will be populated later."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 2.3: Implement `VLMRegistry` (for `ModelConfiguration` instances)**

- **Objective:** Central place for predefined `MLXLMCommon.ModelConfiguration` instances for known VLM checkpoints.
- **Agent Instruction (to User/Copilot):**
  1.  "In `VLMFactoryInfrastructure.swift`, define `public class VLMRegistry: AbstractModelRegistry` (subclassing `MLXLMCommon.AbstractModelRegistry`)."
  2.  "Implement `public static let shared = VLMRegistry(modelConfigurations: all())`."
  3.  "Implement `static public func all() -> [ModelConfiguration] { return [] }`. Later, this will be populated with known VLM configurations (e.g., default prompts, specific tokenizer IDs) using `MLXLMCommon.ModelConfiguration`."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 2.4: Implement `VLMModelFactory`**

- **Objective:** The central factory for orchestrating the loading of VLMs.
- **Agent Instruction (to User/Copilot):**
  1.  "Create `Sources/MLXVLM/VLMModelFactory.swift`."
  2.  "Define `public class VLMModelFactory: ModelFactory` (subclassing `MLXLMCommon.ModelFactory`). Import `Hub`, `MLX`, `MLXLMCommon`, `Tokenizers`."
  3.  "Implement the initializer: `public init(typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry, modelRegistry: AbstractModelRegistry)`."
  4.  "Implement `public static let shared` using `VLMTypeRegistry.shared`, `VLMProcessorTypeRegistry.shared`, and `VLMRegistry.shared` (these are the VLM-specific registries)."
  5.  "Declare `public let typeRegistry: ModelTypeRegistry`, `public let processorRegistry: ProcessorTypeRegistry`, `public let modelRegistry: AbstractModelRegistry`."
  6.  "Implement `public func _load(hub: HubApi, configuration: ModelConfiguration, progressHandler: @Sendable @escaping (Progress) -> Void) async throws -> ModelContext`. The logic is as follows:"
      - "Call `downloadModel(hub: configuration: progressHandler:)` (from `MLXLMCommon`) to get `modelDirectory`."
      - "Construct `configurationURL = modelDirectory.appending(component: "config.json")`."
      - "Decode `MLXLMCommon.BaseConfiguration` from `configurationURL` to get `modelType` and `perLayerQuantization`."
      - "Use `(self.typeRegistry as! VLMTypeRegistry).createModel(configuration: configurationURL, modelType: baseConfig.modelType)` to instantiate the `VLMModel`. (Cast is okay here as we initialize with `VLMTypeRegistry`)."
      - "Call `loadWeights(modelDirectory: modelDirectory, model: model, perLayerQuantization: baseConfig.perLayerQuantization)` (from `MLXLMCommon`) to load weights into the model."
      - "Call `loadTokenizer(configuration: configuration, hub: hub)` (from `MLXLMCommon`) to get the `Tokenizer`."
      - "Construct `processorConfigurationURL = modelDirectory.appending(component: "preprocessor_config.json")`."
      - "Decode `BaseProcessorConfiguration` (from Step 2.0) from `processorConfigurationURL` to get `processorClass`."
      - "Use `(self.processorRegistry as! VLMProcessorTypeRegistry).createModel(configuration: processorConfigurationURL, processorType: baseProcessorConfig.processorClass, tokenizer: tokenizer)` to instantiate the `UserInputProcessor`."
      - "Return a new `ModelContext` (from `MLXLMCommon`) populated with `configuration`, `model`, `processor`, and `tokenizer`."
- **Verification Command (Agent):** `"Run 'swift build'."`

---

**Phase 3: Implementing Specific VLMs**

_(The following steps should be repeated for each VLM, e.g., PaliGemma, Idefics3. Use `<YourVLM>` as a placeholder for the specific VLM's name.)_

**Step 3.X.0: Create Model-Specific Directory**

- **Agent Instruction (to User/Copilot):**
  1.  "Create a new directory: `Sources/MLXVLM/Models/<YourVLM>` (e.g., `Sources/MLXVLM/Models/PaliGemma`)."

**Step 3.X.1: Implement `<YourVLM>Configuration.swift` for Model and Processor**

- **Objective:** Define Codable structs for the VLM's `config.json` and `preprocessor_config.json`.
- **Agent Instruction (to User/Copilot):**
  1.  "In `Sources/MLXVLM/Models/<YourVLM>/`, create `<YourVLM>Configuration.swift`."
  2.  "Define `public struct <YourVLM>ModelConfig: Codable, Sendable { ... }`. Its properties must match the structure of the specific VLM's `config.json`. This often includes nested structs for `text_config`, `vision_config`, etc. (Refer to `PaliGemmaConfiguration.swift` or `Idefics3Configuration.swift` in `mlx-swift-examples` for structural examples)."
  3.  "Define `public struct <YourVLM>ProcessorConfig: Codable, Sendable { ... }`. Its properties must match the VLM's `preprocessor_config.json` (e.g., image mean, std, target sizes, image sequence length, video sampling parameters). (Refer to `PaliGemmaProcessorConfiguration.swift` or `Idefics3ProcessorConfiguration.swift` in `mlx-swift-examples`)."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 3.X.2: Implement `<YourVLM>.swift` (The Model Class)**

- **Objective:** Implement the `Module` for the VLM architecture.
- **Agent Instruction (to User/Copilot):**
  1.  "In `Sources/MLXVLM/Models/<YourVLM>/`, create `<YourVLM>.swift`."
  2.  "Define `public class <YourVLM>: Module, VLMModel`. Import `MLX`, `MLXNN`, `MLXFast`, `MLXLMCommon`."
  3.  "Declare necessary sub-modules using `@ModuleInfo` (e.g., a vision encoder like `CLIPVisionModel`, a language model like `LlamaModel`, and a multimodal projector). These sub-modules can be complex and might need their own files or be nested within `<YourVLM>.swift` if simpler. Refer to the VLM's original Python architecture."
  4.  "Implement `public init(_ config: <YourVLM>ModelConfig)`. Initialize all sub-modules based on the provided configuration."
  5.  "Implement `public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult`. This is the VLM's core logic for combining modalities:"
      - "Access processed visual data: `guard let imagePixels = input.image?.pixels else { throw VLMError.imageRequired }` (or handle video from `input.video.pixels`)."
      - "Pass `imagePixels` through the VLM's vision encoder (e.g., `let imageFeatures = self.visionTower(imagePixels)`)."
      - "Project the `imageFeatures` to align dimensions with the language model's embeddings (e.g., `let projectedFeatures = self.projector(imageFeatures)`)."
      - "Get text embeddings: `let textEmbeddings = self.languageModelPart.embedTokens(input.text.tokens)`."
      - "**Combine Modalities:** This is highly VLM-specific. A common approach is to find placeholder tokens in `input.text.tokens` (e.g., an `<image>` token ID) and replace their corresponding `textEmbeddings` with the `projectedFeatures`. Alternatively, concatenate or use a more complex merging layer. The result is a single `MLXArray` of multimodal embeddings: `let multimodalEmbeddings = ...`."
      - "Feed `multimodalEmbeddings` to the language model part. Handle prompt prefilling for long sequences by evaluating in chunks (similar to `LLMModel.prepare` in `MLXLMCommon`). The `self(LMInput.Text(tokens: chunkOfMultimodalEmbeddings), cache: cache, state: nil)` call would internally use the language model part."
      - "Return `.logits(LMOutput(logits: finalLogitsFromLMPart))` or `.tokens(...)`."
  6.  "Implement `public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput`. This is for subsequent token generation. It typically calls the language model part using the `KVCache` which now contains the multimodal context from `prepare`."
  7.  "Implement `public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]`. Refer to the VLM's Python implementation or existing MLX ports for correct weight name mapping (e.g., `model.vision_tower...` to `visionTower...`) and any necessary tensor transformations (like transposing Conv2D weights for MLX compatibility)."
  8.  "Implement `LoRAModel` conformance: `public func loraLinearLayers() -> LoRALinearLayers { /* return layers for LoRA */ }`."
  9.  "Implement `KVCacheDimensionProvider` conformance: `public var kvHeads: [Int] { /* return kv_heads for each layer of the LM part */ }`."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 3.X.3: Implement `<YourVLM>Processor.swift` (Input Handling)**

- **Objective:** Create the `UserInputProcessor` for this VLM.
- **Agent Instruction (to User/Copilot):**
  1.  "In `Sources/MLXVLM/Models/<YourVLM>/`, create `<YourVLM>Processor.swift`."
  2.  "Define `public class <YourVLM>Processor: UserInputProcessor`. Import necessary modules."
  3.  "Implement `public init(_ config: <YourVLM>ProcessorConfig, tokenizer: Tokenizer)`."
  4.  "Implement `public func prepare(input: UserInput) async throws -> LMInput`:"
      - "**Text Tokenization:**
        - Use a `MessageGenerator` (e.g., `DefaultMessageGenerator` or a custom one for `<YourVLM>`) to get `[Message]` from `input.prompt`.
        - Use `tokenizer.applyChatTemplate(...)` to get `initialPromptTokens: [Int]`."
      - "**Image/Video Processing:**
        - Initialize `var allProcessedImageFeatures: [MLXArray] = []`, `var allProcessedVideoFrames: [[MLXArray]] = []`.
        - For each `userInputImage` in `input.images`:
          - `let ciImage = try userInputImage.asCIImage()`
          - `let processedCIImage = MediaProcessing.apply(ciImage, processing: input.processing)` (applies user-defined general processing).
          - Apply VLM-specific transformations using `MediaProcessing` methods and parameters from `self.config` (e.g., `resize`, `normalize`).
          - `let imageMLXArray = MediaProcessing.asMLXArray(finalCIImage)`. Add to `allProcessedImageFeatures`.
        - For each `userInputVideo` in `input.videos`:
          - `let avAsset = userInputVideo.asAVAsset()`.
          - `let processedFramesResult = try await MediaProcessing.asProcessedSequence(avAsset, samplesPerSecond: self.config.videoSampleRate, frameProcessing: { videoFrame in ... /* apply VLM-specific transforms to videoFrame.frame */ return transformedVideoFrame })`.
          - Add `processedFramesResult.frames` to `allProcessedVideoFrames`.
      - "**Visual Token Integration & Final LMInput Assembly:**
        - Determine the number of visual items (e.g., `let numImages = allProcessedImageFeatures.count`).
        - Create placeholder tokens (e.g., `Array(repeating: self.config.imageTokenID, count: numImages)`).
        - Strategically insert these placeholder tokens into `initialPromptTokens` according to `<YourVLM>`'s specific requirements for prompt structure (e.g., `text <image> text <image>`).
        - `let finalTokensArray = MLXArray(finalPromptTokens).expandedDimensions(axis: 0)`.
        - `let imagePixels = !allProcessedImageFeatures.isEmpty ? concatenated(allProcessedImageFeatures, axis: 0) : nil`. (Stack features for batching if model expects it).
        - `let videoPixels = !allProcessedVideoFrames.isEmpty ? concatenated(allProcessedVideoFrames.flatMap { $0 }, axis: 0) : nil`.
        - Create `LMInput.ProcessedImage` and `LMInput.ProcessedVideo` structs, potentially including `frames: [THW]` metadata if the model's `prepare` needs it.
        - Return `LMInput(text: .init(tokens: finalTokensArray), image: processedImageData, video: processedVideoData)`."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 3.X.4: Register `<YourVLM>` and its Processor with the Factories**

- **Objective:** Make the VLM and its processor loadable via `VLMModelFactory`.
- **Agent Instruction (to User/Copilot):**
  1.  "Open `Sources/MLXVLM/VLMFactoryInfrastructure.swift`."
  2.  "In `VLMTypeRegistry.all()`, add: `\"<model_type_string_for_YourVLM>\": createVLM(<YourVLM>ModelConfig.self, <YourVLM>.init),` (e.g., if Idefics3's `model_type` in `config.json` is `idefics3`, then `\"idefics3\": createVLM(Idefics3ModelConfig.self, Idefics3.init)`)."
  3.  "In `VLMProcessorTypeRegistry.all()`, add: `\"<ProcessorClassString_for_YourVLM>\": createVLMProcessor(<YourVLM>ProcessorConfig.self, <YourVLM>Processor.init),` (e.g., `\"Idefics3Processor\": createVLMProcessor(Idefics3ProcessorConfig.self, Idefics3Processor.init)`)."
  4.  "(Optional) In `VLMRegistry.all()`, add a predefined `MLXLMCommon.ModelConfiguration` for a known checkpoint of `<YourVLM>`, including its Hugging Face ID and a default prompt."
- **Verification Command (Agent):** `"Run 'swift build'."`

**Step 3.X.5: Implement Unit Tests for `<YourVLM>` Components**

- **Objective:** Validate the processor and basic model structure.
- **Agent Instruction (to User/Copilot):**
  1.  "Create `Tests/MLXVLMTests/<YourVLM>Tests.swift`."
  2.  "**Processor Tests for `<YourVLM>Processor`:**"
      - "Mock `Tokenizer` and initialize `<YourVLM>ProcessorConfig` with test values."
      - "Test the `prepare(input:)` method with various `UserInput` scenarios:"
        - Text-only input.
        - Input with a single image (use a fixture `CIImage` or mock `UserInput.Image.asCIImage()`). Verify the output `LMInput.image.pixels` has the expected shape and that `LMInput.text.tokens` includes correctly placed image placeholder tokens.
        - Input with multiple images (if the VLM supports it). Check feature stacking and token placement.
        - Input with a video (mock `AVAsset` or use a short fixture video file for `MediaProcessing` calls).
      - "Assert that appropriate `VLMError`s are thrown for invalid inputs (e.g., trying to pass too many images if the model doesn't support it)."
  3.  "**Model Sanitize Tests for `<YourVLM>`:**"
      - "Test `<YourVLM>.sanitize(weights:)` with a dictionary of representative Hugging Face weight names (as keys) and dummy `MLXArray`s (as values). Assert that the returned dictionary has correctly renamed keys, and that expected keys are present or absent based on the sanitization logic."
- **Verification Command (Agent):** `"Run 'swift test'."`

---

**Phase 4: End-to-End Testing, Documentation, and Finalization**

**Step 4.0: Example CLI Tool for End-to-End VLM Testing**

- **Objective:** Verify the complete VLM loading and inference pipeline with actual models.
- **Agent Instruction (to User/Copilot):**
  1.  "In the `MLXVLM` package, create a new executable target, e.g., `VLMCLI` in `Sources/VLMCLI/main.swift`."
  2.  "This tool should depend on the `MLXVLM` library."
  3.  "Implement argument parsing to accept a model ID (e.g., "mlx-community/paligemma..."), a text prompt, and path(s) to image(s)/video(s)."
  4.  "Use `VLMModelFactory.shared.loadContainer(configuration: ModelConfiguration(id: modelIDFromArg), ...)` to load the VLM."
  5.  "Construct `UserInput` from the command-line arguments."
  6.  "Inside `modelContainer.perform { context in ... }`:"
      - `let lmInput = try await context.processor.prepare(input: userInput)`
      - `let generationResult = try generate(input: lmInput, parameters: ..., context: context, didGenerate: { tokens in print(context.tokenizer.decode(tokens: tokens)); return .more })` (using `generate` from `MLXLMCommon`).
      - "Print `generationResult.output` and performance summary."
  7.  "Manually run this CLI tool with downloaded VLM weights and sample inputs to ensure the full pipeline works."
- **Verification:** Manual execution and observation of the CLI tool's output.

**Step 4.1: API Documentation (Swift-DocC)**

- **Objective:** Provide comprehensive documentation for the `MLXVLM` library's public API.
- **Agent Instruction (to User/Copilot):**
  1.  "Add Swift-DocC comments (`///`) to all public protocols, classes, structs, enums, and their public members throughout the `MLXVLM` library."
  2.  "Clearly explain the purpose of major components like `VLMModelFactory`, the `VLMModel` protocol, the role of `UserInputProcessor` for VLMs, and the structure of `LMInput` for multimodal data."
  3.  "Include brief code examples for loading a VLM and performing a basic inference."
- **Verification Command (Agent):** `"Run 'swift package generate-documentation --target MLXVLM --output-path ./docs'` (if the DocC plugin is in `Package.swift`) and review the generated HTML documentation."

**Step 4.2: Code Review, Refinement, and Performance Considerations**

- **Objective:** Finalize the library for release, ensuring code quality, consistency, and addressing any performance issues.
- **Agent Instruction (to User/Copilot):**
  1.  "Conduct a thorough review of the entire `MLXVLM` codebase. Focus on:"
      - Consistency in naming conventions and coding style.
      - Clarity and completeness of comments and documentation.
      - Robustness of error handling in all public and internal methods.
      - Efficiency of media processing steps and MLX array operations.
  2.  "Ensure all `// TODO:` comments are either resolved or documented as specific future work with clear justifications."
  3.  "Profile the example CLI tool (from Step 4.0) with different VLMs and input sizes to identify potential performance bottlenecks. Optimize critical sections if necessary, particularly in media processing and the model's `prepare` method."
- **Verification Command (Agent):** `"Run 'swift build' and 'swift test' to confirm all checks pass and the library is stable."`

This structured plan should guide the development of the `MLXVLM` library effectively, breaking down the complex task into manageable, verifiable steps.
