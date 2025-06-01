**Implementation Plan for `mlx-vlm` Library (Implementing `qwen2_5_vl`)**

**Overall Goal:** Develop a Swift package named `mlx_vlm` for loading and running the `qwen2_5_vl` Vision Language Model using the MLX Swift framework. The library will implement the VLM from scratch, support direct Hugging Face model loading, and use a test-first approach with layer-by-layer validation against a Python Transformers reference.

**Core Principles:**

*   **Modularity:** `qwen2_5_vl` components (Vision Encoder, Language Model, Fusion) will be distinct `MLXNN.Module`s within the `mlx_vlm` library.
*   **Extensibility:** While starting with `qwen2_5_vl`, design for potential future addition of other custom VLMs.
*   **Clarity & Testability:** Code will be well-structured, readable, and heavily unit-tested.
*   **Efficiency:** Leverage MLX for performance.
*   **Python Reference:** Python `transformers` implementation of `qwen2_5_vl` is the ground truth for architecture, weights, and intermediate outputs.
*   **DeepWiki MCP:** For `mlx-swift` specific challenges, implementation hurdles, or unexpected behavior, formulate targeted queries to DeepWiki MCP.
*   **Verification:** Use `xcodebuild` for compilation and testing.

---

**Phase 0: Project Setup & Python Reference Infrastructure**

**Step 0.0: Initialize the `mlx_vlm` Swift Package & Xcode Project**
-   **Objective:** Create the basic SPM project structure.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create a new directory named `mlx-vlm-project` (or similar, to differentiate from the target name)."
    2.  "Navigate into this `mlx-vlm-project` directory and execute `swift package init --type library --name mlx_vlm`."
    3.  "Open the generated `Package.swift` file."
    4.  "Ensure the library name in `products` is `mlx_vlm` and the target name in `targets` is `mlx_vlm`."
        ```swift
        // Example snippet in Package.swift
        products: [
            .library(name: "mlx_vlm", targets: ["mlx_vlm"]),
        ],
        targets: [
            .target(name: "mlx_vlm", dependencies: [/* ... */]),
            .testTarget(name: "mlx_vlmTests", dependencies: ["mlx_vlm"]),
        ]
        ```
    5.  "Declare Swift tools version (e.g., `// swift-tools-version:6.0`)."
    6.  "Define `platforms` (e.g., `platforms: [.macOS(.v14), .iOS(.v17)]`)."
    7.  "In `dependencies` array of `Package`, add:"
        *   `.package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.21.2"))` // Or latest compatible
        *   `.package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.20"))` // Or latest compatible
    8.  "In `targets` array, for `.target(name: "mlx_vlm", ...)` target, add `dependencies`:"
        *   `.product(name: "MLX", package: "mlx-swift")`
        *   `.product(name: "MLXNN", package: "mlx-swift")`
        *   `.product(name: "MLXRandom", package: "mlx-swift")` // For sampling
        *   `.product(name: "Transformers", package: "swift-transformers")` // For tokenizer
    9.  "In the project directory, run `swift package generate-xcodeproj`."
-   **Verification Command (Agent):** `"In the 'mlx-vlm-project' directory, run 'xcodebuild build -scheme mlx_vlm -destination \"platform=OS X\"'. If this fails, check MLX setup. For MLX issues, query DeepWiki MCP: 'Initial mlx-swift project setup and build errors for a library target on macOS.'"`

**Step 0.1: Python Reference Script Setup**
-   **Objective:** Create a Python script to extract configurations, weights, and layer outputs from the Hugging Face `qwen2_5_vl` model.
-   **Agent Instruction (to User/Copilot):**
    1.  "In a separate directory (e.g., `python_reference`), create a Python script (`extract_qwen_data.py`)."
    2.  "Install necessary Python libraries: `transformers`, `torch`, `Pillow`, `numpy`, `safetensors`, `sentencepiece`."
    3.  "Implement functions in the script to:"
        *   Load a specific `qwen2_5_vl` model (e.g., "Qwen/Qwen2.5-VL-Instruct") and its tokenizer from Hugging Face.
        *   Save the model's `config.json` (and `vision_config.json`, `text_config.json` if separate).
        *   Save the tokenizer's files (`tokenizer.json`, `tokenizer_config.json`, `merges.txt` etc.).
        *   **Crucially:** Implement a mechanism (using PyTorch forward hooks) to capture and save intermediate tensor outputs from specified layers (e.g., patch embedding, attention blocks, FFNs for both ViT and LLM, fusion layer) given sample image and text inputs. Save these as `.npy` files.
        *   Save the weights of these same layers as `.npy` or directly from `.safetensors` with clear naming.
    4.  "Run this script to generate initial reference data (config, sample tokenizer output, sample layer weights, and sample layer outputs)."
-   **Verification:** Manually inspect the generated files to ensure they seem correct and complete.

**Step 0.2: Define Core Error Types and Utility Functions**
-   **Objective:** Establish basic error handling and shared utilities within the `mlx_vlm` library.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Common/VLMError.swift`."
    2.  "Define `public enum VLMError: LocalizedError { ... }` with cases like `configFileMissing`, `weightsFileNotFound`, `tensorShapeMismatch`, `unsupportedOperation`, `pythonReferenceDataMissing`."
    3.  "Create `Sources/mlx_vlm/Common/Utilities.swift`."
    4.  "Add helper functions, e.g., for loading `.npy` files (for tests), image loading (e.g., `CGImage` to `MLXArray` - basic version initially, will be refined)."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "Create `Tests/mlx_vlmTests/CommonTests.swift`."
    2.  "Write unit tests for `.npy` loading and basic image to MLXArray conversion (if implemented)."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. For issues with basic MLXArray creation from data, query DeepWiki MCP: 'Converting [Float] to MLXArray with specific shape in mlx-swift.'"`

---

**Phase 1: Configuration, Tokenizer, and Direct Weight Loading Core**

**Step 1.0: Implement Model Configuration Structs**
-   **Objective:** Define Swift structs to hold `qwen2_5_vl` configuration, specific to this model but within the `mlx_vlm` library.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Configuration/Qwen2_5_VLConfig.swift` (organizing by model within the library)."
    2.  "Define `public struct Qwen2_5_VisionConfig: Codable { ... }`."
    3.  "Define `public struct Qwen2_5_LanguageConfig: Codable { ... }`."
    4.  "Define `public struct Qwen2_5_VL_OverallConfig: Codable { public let visionConfig: Qwen2_5_VisionConfig; public let languageConfig: Qwen2_5_LanguageConfig; /* any other top-level VLM params */ }`."
    5.  "Implement a loader function: `public static func load(from url: URL) throws -> Qwen2_5_VL_OverallConfig`."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "Create `Tests/mlx_vlmTests/Models/Qwen2_5_VL/ConfigurationTests.swift`."
    2.  "Use the `config.json` saved by your Python script. Test that `Qwen2_5_VL_OverallConfig.load()` correctly parses it."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'."`

**Step 1.1: Integrate Tokenizer**
-   **Objective:** Load and use the `qwen2_5_vl` tokenizer via `swift-transformers`, making it part of the `mlx_vlm` library's model-specific components.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Tokenization/QwenTokenizer.swift`."
    2.  "Define `public class QwenTokenizer { private let hfTokenizer: Transformers.Tokenizer; public init(tokenizerFilesAt baseURL: URL) throws { ... } }`."
    3.  "Implement `public func encode(text: String) throws -> [Int]` and `public func decode(ids: [Int]) throws -> String`."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "Create `Tests/mlx_vlmTests/Models/Qwen2_5_VL/TokenizationTests.swift`."
    2.  "Test `encode` and `decode` against Python reference outputs."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for tokenizer issues: 'Loading a local Hugging Face BPE tokenizer (Qwen-style) using swift-transformers package.'"`

**Step 1.2: Implement Direct Hugging Face Weight Loader Core**
-   **Objective:** Create a utility within `mlx_vlm` to load tensors from `.safetensors` files.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Core/Weights/SafeTensorsLoader.swift` (this can be a general utility)."
    2.  "Define `public class SafeTensorsLoader { private let tensors: [String: MLXArray]; public init(url: URL) throws { ... } }`."
    3.  "Implement `public func getTensor(named name: String) throws -> MLXArray`."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "Create `Tests/mlx_vlmTests/Core/WeightsTests.swift`."
    2.  "Test loading known tensors from a sample `.safetensors` file, verify dtype, shape, and values against Python."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for safetensor parsing: 'Parsing .safetensors file format in Swift and loading data into MLXArrays.'"`

---

**Phase 2: Vision Encoder (`Qwen2_5_VL_ViT`) Implementation (Layer by Layer)**

*   **General Python Reference Task for Phase 2:** Save weights AND output of *each sub-component* of the ViT for `qwen2_5_vl`.

**Step 2.0: Define `Qwen2_5_VL_ViT` and Basic Layers**
-   **Objective:** Implement core ViT layers as `MLXNN.Module`s within `mlx_vlm`, specific to `qwen2_5_vl`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Vision/QwenViTLayers.swift`."
    2.  "Implement `QwenVisionPatchEmbedding: MLXNN.Module`."
    3.  "Implement `QwenViTAttention: MLXNN.Module`."
    4.  "Implement `QwenViTMLP: MLXNN.Module`."
    5.  "Implement `QwenViTBlock: MLXNN.Module`."
    6.  "Each module: `init(config: Qwen2_5_VisionConfig, ...)` and `func loadWeights(loader: SafeTensorsLoader, prefix: String) throws`."
    7.  "Implement `public func callAsFunction(...)` for each."
-   **Testing Guidance (Agent instructs User/Copilot for EACH layer):**
    1.  "In `Tests/mlx_vlmTests/Models/Qwen2_5_VL/VisionLayerTests.swift`, test each layer against Python reference outputs using `MLX.allClose`."
-   **Verification Command (Agent for EACH layer):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for MLX layer implementation: 'Implementing windowed self-attention with mlx-swift tensor operations.'"`

**Step 2.1: Implement Full `Qwen2_5_VL_ViT` Module**
-   **Objective:** Assemble the `qwen2_5_vl` ViT within `mlx_vlm`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Vision/Qwen2_5_VL_ViT.swift`."
    2.  "Define `public class Qwen2_5_VL_ViT: MLXNN.Module { ... }`."
    3.  "Instantiate layers from `QwenViTLayers.swift`, add positional embeddings."
    4.  "Implement `init(config: Qwen2_5_VisionConfig)` and `public func loadWeights(loader: SafeTensorsLoader, prefix: String = "model.visual.") throws`."
    5.  "Implement `public func callAsFunction(_ image: MLXArray) -> MLXArray`."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "In `Tests/mlx_vlmTests/Models/Qwen2_5_VL/VisionTests.swift`, test the full `Qwen2_5_VL_ViT` against Python reference."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for ViT assembly: 'Best practices for nested MLXNN.Module weight loading from a SafeTensorsLoader in Swift.'"`

---

**Phase 3: Language Model (`Qwen2_5_VL_LLM`) Implementation (Layer by Layer)**

*   **General Python Reference Task for Phase 3:** Save weights AND output of *each sub-component* of the `qwen2_5_vl` LLM.

**Step 3.0: Define `Qwen2_5_VL_LLM` and Basic Layers**
-   **Objective:** Implement core LLM layers for `qwen2_5_vl` within `mlx_vlm`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Language/QwenLLMLayers.swift`."
    2.  "Implement `QwenEmbedding: MLXNN.Module`."
    3.  "Implement `QwenRMSNorm: MLXNN.Module`."
    4.  "Implement `QwenRotaryEmbedding: MLXNN.Module` (for RoPE logic)."
    5.  "Implement `QwenAttention: MLXNN.Module` (with RoPE and KV Caching)."
    6.  "Implement `QwenMLP: MLXNN.Module`."
    7.  "Implement `QwenDecoderLayer: MLXNN.Module`."
    8.  "Each module with weights: `init(config: Qwen2_5_LanguageConfig, ...)` and `func loadWeights(...)`."
    9.  "Implement `public func callAsFunction(...)` for each, managing KV cache for stateful layers."
-   **Testing Guidance (Agent instructs User/Copilot for EACH layer):**
    1.  "In `Tests/mlx_vlmTests/Models/Qwen2_5_VL/LanguageLayerTests.swift`, test each layer against Python reference outputs."
-   **Verification Command (Agent for EACH layer):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for RoPE/KV Cache: 'Implementing Rotary Positional Embedding (RoPE) application in mlx-swift' or 'Step-by-step KV cache implementation for an MLXNN.Module attention layer in Swift.'"`

**Step 3.1: Implement Full `Qwen2_5_VL_LLM` Module**
-   **Objective:** Assemble the `qwen2_5_vl` LLM within `mlx_vlm`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Language/Qwen2_5_VL_LLM.swift`."
    2.  "Define `public class Qwen2_5_VL_LLM: MLXNN.Module { ... }`."
    3.  "Instantiate layers from `QwenLLMLayers.swift`, final Norm, and LM Head."
    4.  "Implement `init(config: Qwen2_5_LanguageConfig)` and `public func loadWeights(loader: SafeTensorsLoader, prefix: String = "model.language_model.") throws`."
    5.  "Implement `public func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, caches: [KVCache]? = nil) -> (MLXArray, [KVCache])`."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "In `Tests/mlx_vlmTests/Models/Qwen2_5_VL/LanguageTests.swift`, test the full `Qwen2_5_VL_LLM` against Python reference logits."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for LLM assembly: 'Correctly initializing the LM output head in a custom mlx-swift LLM.'"`

---

**Phase 4: Multimodal Fusion, Inference Engine, and API for `qwen2_5_vl`**

**Step 4.0: Implement Multimodal Fusion Logic for `qwen2_5_vl`**
-   **Objective:** Implement how `qwen2_5_vl` combines visual and text features, as part of the `mlx_vlm`'s model-specific logic.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Core/QwenMultimodalProcessing.swift`."
    2.  "Implement the `qwen2_5_vl` specific fusion (e.g., `QwenVisionProjector: MLXNN.Module` if used, or logic for token sequence preparation)."
    3.  "Ensure `loadWeights` if fusion has learnable parameters."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "In `Tests/mlx_vlmTests/Models/Qwen2_5_VL/CoreTests.swift`."
    2.  "Test Swift fusion logic against Python reference inputs/outputs for this stage."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for fusion: 'Implementing a vision feature projector (MLP) in mlx-swift.'"`

**Step 4.1: Implement `Qwen2_5_VL_Inference` Engine**
-   **Objective:** The main class in `mlx_vlm` for orchestrating `qwen2_5_vl` inference.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create `Sources/mlx_vlm/Models/Qwen2_5_VL/Core/Qwen2_5_VL_Inference.swift`."
    2.  "Define `public class Qwen2_5_VL_Inference { ... }`."
    3.  "Properties: `visionEncoder: Qwen2_5_VL_ViT`, `languageModel: Qwen2_5_VL_LLM`, fusion components, `tokenizer: QwenTokenizer`, `config: Qwen2_5_VL_OverallConfig`."
    4.  "Implement `public init(modelDirectoryURL: URL) async throws` (loads config, tokenizer, all weights via `SafeTensorsLoader`, initializes and loads weights for ViT, LLM, Fusion)."
    5.  "Implement `public func generate(image: CGImage, prompt: String, maxTokens: Int = 200, temperature: Float = 0.7, ...) async throws -> String`."
-   **Testing Guidance (Agent instructs User/Copilot):**
    1.  "In `Tests/mlx_vlmTests/Models/Qwen2_5_VL/CoreTests.swift` or new `InferenceTests.swift`."
    2.  "Perform end-to-end test with a full `qwen2_5_vl` model download. Compare greedy search output text with Python reference."
-   **Verification Command (Agent):** `"Run 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Query DeepWiki MCP for sampling: 'Implementing top-p (nucleus) sampling for LLM generation in mlx-swift.'"`

**Step 4.2: Image Preprocessing Refinement for `qwen2_5_vl`**
-   **Objective:** Ensure image preprocessing matches `qwen2_5_vl` requirements within `mlx_vlm`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Create or refine `Sources/mlx_vlm/Models/Qwen2_5_VL/Core/QwenImageProcessor.swift`."
    2.  "Implement precise image preprocessing for `qwen2_5_vl` (resizing, normalization, channel order)."
-   **Testing Guidance:** Update ViT input tests and full inference tests. Compare preprocessed image tensor with Python's.

---

**Phase 5: Finalization, CLI Example, and Documentation for `mlx-vlm`**

**Step 5.0: Example CLI Tool (`MLXVLMRunner`) for End-to-End Usage**
-   **Objective:** Create a runnable example using the `mlx_vlm` library.
-   **Agent Instruction (to User/Copilot):**
    1.  "In the `mlx-vlm-project` package, create new executable target `MLXVLMRunner` (e.g., in `Sources/MLXVLMRunner/main.swift`)."
    2.  "Depend on the `mlx_vlm` library product."
    3.  "Implement argument parsing (model path for `qwen2_5_vl`, image path, prompt)."
    4.  "Use `Qwen2_5_VL_Inference(modelDirectoryURL: ...)` from the `mlx_vlm` library."
    5.  "Call `generate(...)` and print results."
-   **Verification:** Manually execute with a downloaded `qwen2_5_vl` model.
    `"In 'mlx-vlm-project', run 'swift run MLXVLMRunner --model-path /path/to/qwen2_5_vl_model --image-path /path/to/image.jpg --prompt \"Describe this image\"'. Query DeepWiki MCP for CLI setup: 'Setting up an executable target depending on a local library in Swift Package Manager.'"`

**Step 5.1: API Documentation (Swift-DocC) for `mlx_vlm`**
-   **Objective:** Document all public APIs in the `mlx_vlm` library, focusing on `Qwen2_5_VL_Inference`.
-   **Agent Instruction (to User/Copilot):**
    1.  "Add Swift-DocC comments (`///`) to `Qwen2_5_VL_Inference` and other key public types/methods."
-   **Verification Command (Agent):** `"Run 'xcodebuild docbuild -scheme mlx_vlm -destination \"platform=OS X\"' (or 'swift package generate-documentation') and review."`

**Step 5.2: Code Review, Refinement, Performance Checks**
-   **Objective:** Polish the `mlx_vlm` library.
-   **Agent Instruction (to User/Copilot):**
    1.  "Review codebase for clarity, consistency, error handling, efficiency."
    2.  "Address `// TODO:` comments."
    3.  "Profile `MLXVLMRunner` with Instruments if performance issues arise."
-   **Verification Command (Agent):** `"Run all tests: 'xcodebuild test -scheme mlx_vlm -destination \"platform=OS X\"'. Ensure the CLI example runs smoothly."`
