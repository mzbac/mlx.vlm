# Implementation Plan for `mlx-vlm` Library

**Overall Goal:** Develop a Swift package named `mlx_vlm` for loading and running Vision Language Models (VLMs) using the MLX Swift framework. The library should support various VLMs, handle multi-modal inputs (text, images, videos) efficiently, and integrate seamlessly with `MLXLMCommon` for shared LLM functionalities.

**Core Principles:**

*   **Modularity:** Each VLM will have its dedicated components (model, processor, configurations).
*   **Extensibility:** The design should make it straightforward to add support for new VLMs.
*   **Clarity:** Code should be well-structured and readable.
*   **Efficiency:** Leverage MLX for performance and manage memory for media processing.
*   **`MLXLMCommon` Integration:** Utilize base classes and protocols from `MLXLMCommon`.
*   **Reference Codebase:** The primary reference for VLM structures and porting logic is `https://github.com/ml-explore/mlx-swift-examples.git`.
*   **MLX Issues:** If MLX-specific issues arise (e.g., Metal integration, tensor operations), consult the official MLX Swift repository: `https://github.com/ml-explore/mlx-swift.git`.
*   **Verification:** Use `xcodebuild` for compilation and testing. Example build command: `xcodebuild build -scheme mlx_vlm -destination 'platform=OS X'`. Example test command: `xcodebuild test -scheme mlx_vlm-Package -destination 'platform=OS X'` (Note: Scheme names like `mlx_vlm-Package` are common for package products in Xcode; adjust if your scheme for tests is different, e.g., just `mlx_vlm` if testing the library target directly).

---

**Phase 1: `mlx_vlm` Package Initialization and Core Definitions**

**Step 1.0: Initialize the `mlx_vlm` Swift Package & Xcode Project**
-   **Objective:** Create the basic SPM project structure and generate an Xcode project.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create a new directory named `mlx_vlm_library_root` (or similar to avoid conflict with the target name if the directory is also named `mlx_vlm`)."
    2.  "Navigate into this `mlx_vlm_library_root` directory and execute `swift package init --type library --name mlx_vlm`." (This sets the library product name).
    3.  "Open the generated `Package.swift` file."
    4.  "In `Package.swift`, ensure the library name in `products` is `mlx_vlm` and the target name in `targets` is also `mlx_vlm` (by default `swift package init` will name the target `mlx_vlm` if the package name is `mlx_vlm`)."
        ```swift
        // Example snippet in Package.swift
        // ...
        products: [
            .library(
                name: "mlx_vlm", // Product name
                targets: ["mlx_vlm"]), // Target name
        ],
        // ...
        targets: [
            .target(
                name: "mlx_vlm", // Target name
                dependencies: [/* ... */]),
            .testTarget(
                name: "mlx_vlmTests", // Test target name
                dependencies: ["mlx_vlm"]),
        ]
        // ...
        ```
    5.  "Declare the Swift tools version at the top, e.g., `// swift-tools-version:5.9`."
    6.  "Define the `platforms` for your package, e.g., `.macOS(.v13)`."
    7.  "In the `dependencies` array of the `Package` object, add the following:"
        *   `.package(url: "https://github.com/ml-explore/mlx-swift.git", branch: "main")`
        *   `.package(url: "https://github.com/huggingface/swift-tokenizers.git", branch: "main")`
        *   "If `MLXLMCommon` is a local package: `.package(name: "MLXLMCommon", path: "../MLXLMCommon")`."
    8.  "In the `targets` array, for the `.target(name: "mlx_vlm", ...)` target, add its `dependencies`:"
        *   `.product(name: "MLX", package: "mlx-swift")`
        *   `.product(name: "MLXNN", package: "mlx-swift")`
        *   `.product(name: "MLXOptimizers", package: "mlx-swift")`
        *   `.product(name: "MLXRandom", package: "mlx-swift")`
        *   `.product(name: "MLXFast", package: "mlx-swift")`
        *   `.product(name: "Tokenizers", package: "swift-tokenizers")`
        *   `.product(name: "Hub", package: "swift-tokenizers")`
        *   `"MLXLMCommon"`
    9.  "In the `mlx_vlm_library_root` directory, run `swift package generate-xcodeproj`."
-   **Verification Command (Agent):** `"In the 'mlx_vlm_library_root' directory, run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'. If this fails, check MLX setup or consult https://github.com/ml-explore/mlx-swift.git."` (Ensure the scheme name `mlx_vlm` matches what Xcode generates for your library target).

**Step 1.1: Define Core `VLMModel` Protocol & `VLMError` Enumeration**
-   **Objective:** Establish the central VLM protocol and error types.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/VLMModel.swift`."
    2.  "Define `public protocol VLMModel: LanguageModel, LoRAModel {}` (importing `MLX` and `MLXLMCommon`)."
    3.  "Create `Sources/mlx_vlm/VLMError.swift`."
    4.  "Define `public enum VLMError: LocalizedError { ... }` with relevant cases and descriptions."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'. For MLX related compilation errors, refer to https://github.com/ml-explore/mlx-swift.git."`

**Step 1.2: Implement `MediaProcessing.swift` for VLM Media Handling**
-   **Objective:** Create utilities for image and video manipulation.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/MediaProcessing.swift`."
    2.  "Import `AVFoundation`, `CoreImage.CIFilterBuiltins`, `MLX`, `MLXLMCommon`."
    3.  "Define `public struct VideoFrame` and `public struct ProcessedFrames`."
    4.  "Implement `public enum MediaProcessing` with static methods (e.g., `inSRGBToneCurveSpace`, `resampleBicubic`, `normalize`, `asMLXArray`, `asProcessedSequence`). Refer to the reference `MediaProcessing.swift` at `https://github.com/ml-explore/mlx-swift-examples.git`. If details are unclear, query 'DeepWiki MCP Server'."
    5.  "Add the `public extension CIImage` with convenience wrappers."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'. Check https://github.com/ml-explore/mlx-swift.git if MLX array operations cause issues."`
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "Create `Tests/mlx_vlmTests/MediaProcessingTests.swift`."
    2.  "Write unit tests for key `MediaProcessing` functions."
    3.  **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm-Package -destination \"platform=OS X\"' (or your specific test scheme name, often just 'mlx_vlm' if testing the library target). For MLX test failures, consult https://github.com/ml-explore/mlx-swift.git."`

---

**Phase 2: VLM Factory, Registries, and Core Loading Infrastructure**

**Step 2.0: Define `BaseProcessorConfiguration`**
-   **Objective:** Common structure for `preprocessor_config.json`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/VLMFactoryInfrastructure.swift`."
    2.  "Add `public struct BaseProcessorConfiguration: Codable, Sendable { public let processorClass: String; ... }`."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'."`

**Steps 2.1 - 2.3: Implement `VLMTypeRegistry`, `VLMProcessorTypeRegistry`, `VLMRegistry`**
-   **Objective:** Create registries for VLM model types, processor types, and specific model configurations.
-   **Agent Instruction (to User/Copilot):**
    1.  "In `VLMFactoryInfrastructure.swift` (or a new `Registries.swift`):"
    2.  "Define `public class VLMTypeRegistry: ModelTypeRegistry`. Implement `shared`, empty `all()`, and the `createVLM` helper."
    3.  "Define `public class VLMProcessorTypeRegistry: ProcessorTypeRegistry`. Implement `shared`, empty `all()`, and the `createVLMProcessor` helper."
    4.  "Define `public class VLMRegistry: AbstractModelRegistry`. Implement `shared` and an empty `static public func all() -> [ModelConfiguration] { return [] }`."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'."`

**Step 2.4: Implement `VLMModelFactory`**
-   **Objective:** The main factory for loading VLMs.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/VLMModelFactory.swift`."
    2.  "Define `public class VLMModelFactory: ModelFactory`."
    3.  "Implement `init(...)` and `static let shared`."
    4.  "Implement `public func _load(...) async throws -> ModelContext`. Adapt logic from the reference `VLMModelFactory._load` in `https://github.com/ml-explore/mlx-swift-examples.git`. Ensure use of `MLXLMCommon` loading functions and VLM-specific registries."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'."`

---

**Phase 3: Implementing Specific VLMs**

*(Repeat for each VLM, e.g., PaliGemma. Replace `<YourVLM>`.)*

**Step 3.X.0: Create Model-Specific Directory**
-   **Agent Instruction:** `"Create directory Sources/mlx_vlm/Models/<YourVLM>."`

**Step 3.X.1: Implement `<YourVLM>Configuration.swift`**
-   **Objective:** Define Codable structs for the VLM's `config.json` and `preprocessor_config.json`.
-   **Agent Instruction (to User/Copilot):**
    1.  "In `Sources/mlx_vlm/Models/<YourVLM>/`, create `<YourVLM>Configuration.swift`."
    2.  "Define `public struct <YourVLM>ModelConfig: Codable, Sendable { ... }` based on `<YourVLM>`'s `config.json` from the reference repo."
    3.  "Define `public struct <YourVLM>ProcessorConfig: Codable, Sendable { ... }` based on `<YourVLM>`'s `preprocessor_config.json` from the reference."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'."`

**Step 3.X.2: Implement `<YourVLM>.swift` (The Model Class)**
-   **Objective:** Implement the VLM's neural network architecture.
-   **Agent Instruction (to User/Copilot):**
    1.  "In `Sources/mlx_vlm/Models/<YourVLM>/`, create `<YourVLM>.swift`."
    2.  "Define `public class <YourVLM>: Module, VLMModel`."
    3.  "Declare and initialize sub-modules. Refer to the architecture in the reference `<YourVLM>.swift` at `https://github.com/ml-explore/mlx-swift-examples.git`."
    4.  "Implement `public init(_ config: <YourVLM>ModelConfig)`."
    5.  "Implement `public func prepare(...)`. Adapt logic from the reference `<YourVLM>.prepare()`. For MLX issues, consult `https://github.com/ml-explore/mlx-swift.git`."
    6.  "Implement `public func callAsFunction(...)`."
    7.  "Implement `public func sanitize(weights:...)` adapting from the reference."
    8.  "Implement `loraLinearLayers()` and `kvHeads`."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'. For MLX module or tensor issues, check https://github.com/ml-explore/mlx-swift.git."`

**Step 3.X.3: Implement `<YourVLM>Processor.swift` (Input Handling)**
-   **Objective:** Create the `UserInputProcessor` for this VLM.
-   **Agent Instruction (to User/Copilot):**
    1.  "In `Sources/mlx_vlm/Models/<YourVLM>/`, create `<YourVLM>Processor.swift`."
    2.  "Define `public class <YourVLM>Processor: UserInputProcessor`."
    3.  "Implement `public init(...)`."
    4.  "Implement `public func prepare(...)`. Adapt logic from reference `<YourVLM>Processor.prepare()` at `https://github.com/ml-explore/mlx-swift-examples.git`. This includes text tokenization, media processing via `MediaProcessing`, visual token integration. For MLX array issues in media processing, consult `https://github.com/ml-explore/mlx-swift.git`."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'."`

**Step 3.X.4: Register `<YourVLM>` with Factories**
-   **Objective:** Make the VLM and its processor loadable.
-   **Agent Instruction (to User/Copilot):**
    1.  "Open `Sources/mlx_vlm/VLMFactoryInfrastructure.swift`."
    2.  "In `VLMTypeRegistry.all()`, add entry for `<YourVLM>` model type and initializer."
    3.  "In `VLMProcessorTypeRegistry.all()`, add entry for `<YourVLM>` processor class and initializer."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'."`

**Step 3.X.5: Implement Unit Tests for `<YourVLM>` Components**
-   **Objective:** Validate the processor and basic model structure.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Tests/mlx_vlmTests/<YourVLM>Tests.swift`."
    2.  "**Processor Tests:** Mock `Tokenizer`, test `prepare(input:)`."
    3.  "**Model Sanitize Tests:** Test `<YourVLM>.sanitize(weights:)`."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm-Package -destination \"platform=OS X\"' (or your test scheme, e.g., 'mlx_vlm' if testing the library target). For MLX test issues, check https://github.com/ml-explore/mlx-swift.git."`

---

**Phase 4: End-to-End Testing, Documentation, and Finalization**

**Step 4.0: Example CLI Tool for End-to-End VLM Testing**
-   **Objective:** Verify the complete VLM loading and inference pipeline.
-   **Agent Instruction (to User/Copilot):**
    1.  "In the `mlx_vlm` package, create a new executable target (e.g., `VLMRunner` in `Sources/VLMRunner/main.swift`)."
    2.  "Tool should depend on the `mlx_vlm` library product."
    3.  "Implement argument parsing."
    4.  "Use `VLMModelFactory.shared.loadContainer(...)`."
    5.  "Construct `UserInput`, call `context.processor.prepare(...)` then `generate(...)`."
    6.  "Print results. Manually run with actual VLM weights."
-   **Verification:** Manual execution. For build issues in example: `"Run 'xcodebuild build -scheme VLMRunner -destination \"platform=OS X\"'."` (Ensure `VLMRunner` is a scheme in your Xcode project).

**Step 4.1: API Documentation (Swift-DocC)**
-   **Objective:** Document the public API.
-   **Agent Instruction (to User/Copilot):**
    1.  "Add Swift-DocC comments (`///`) to all public types and members."
    2.  "Explain key components and include usage examples."
-   **Verification Command (Agent):** `"Run 'xcodebuild docbuild -scheme mlx_vlm -destination \"platform=OS X\"'` (or `swift package generate-documentation`) and review."

**Step 4.2: Code Review, Refinement, and Performance Checks**
-   **Objective:** Finalize the library.
-   **Agent Instruction (to User/Copilot):**
    1.  "Review codebase for consistency, clarity, error handling, efficiency."
    2.  "Address `// TODO:` comments."
    3.  "Profile example CLI for bottlenecks."
-   **Verification Command (Agent):** `"Run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"' and 'xcodebuild test -scheme mlx_vlm-Package -destination \"platform=OS X\"' (or your test scheme)." `