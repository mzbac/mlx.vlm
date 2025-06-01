# GitHub Copilot Instructions: `mlx-vlm` Swift Library Development

## 0. Core Development Mandates

*   **Documentation-Driven Development:** Before starting any development task, and throughout the implementation, **always refer to the `SDD.md` (Software Design Document) and `implementationPlan.md`**. These documents provide the architectural blueprint, agreed-upon design decisions, specific task breakdowns, and the overall strategic plan. Adherence ensures alignment, consistency, and a structured approach to building the library.
*   **Fidelity to Reference:** The primary goal is to create a Swift implementation that mirrors the behavior and output of the reference Python `transformers` implementation of `qwen2_5_vl`.

## I. General Swift Principles (Refer to Swift API Design Guidelines)

*   **Clarity First:** Code should be easy to read, understand, and maintain. Prioritize self-documenting code.
*   **Conciseness:** Write expressive but not overly verbose code.
*   **Safety:** Leverage Swift's type safety and error handling. Avoid runtime crashes.
*   **Performance Awareness:** Write efficient code for ML tasks, but profile before optimizing complex tensor operations. MLX's lazy evaluation is generally preferred; use `MLX.eval()` sparingly.
*   **Modularity:** Design components (ViT, LLM, layers, processors) with clear responsibilities and well-defined interfaces, guided by the `SDD.md`.

## II. Swift Language Best Practices for `mlx-vlm` (Swift 6.0+)

1.  **Type Safety & Inference:**
    *   Leverage type inference; provide explicit types for public APIs, `MLXNN.Module` properties, and where `MLXArray` shapes/dtypes improve clarity.
        ```swift
        // Good (inference is clear)
        let learningRate = 0.001
        let batchSize = 32

        // Good (explicit for clarity, API, or MLX specifics)
        public class QwenAttention: MLXNN.Module {
            let queryProjection: MLXNN.Linear
            let keyProjection: MLXNN.Linear
            // ...
            public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray { ... }
        }
        let imageTensor: MLXArray = preprocess(image) // Assuming preprocess returns MLXArray
        ```
    *   Use non-optional types by default. Optionals for genuinely absent values (e.g., `KVCache` on first pass).
    *   **Strictly avoid force unwrapping (`!`)**, especially with `MLXArray` operations or dictionary lookups for weights. Use `guard let`, `if let`, `??`.

2.  **Immutability:**
    *   Prefer `let` for configuration parameters, model dimensions, and initialized layer weights.
    *   `var` for loop counters, mutable state within functions, or `KVCache` instances.
    *   Immutable collections (`Array<MLXNN.Module>`, `Dictionary<String, MLXArray>`) for model layers/weights once loaded.

3.  **Value vs. Reference Types:**
    *   `struct` for configurations (`Qwen2_5_VisionConfig`), simple data holders.
    *   `class` for `MLXNN.Module` subclasses (as required by `MLXNN`), and for the main `Qwen2_5_VL_Inference` engine due to its stateful nature (loaded models, tokenizer).
    *   `enum` for custom errors (`VLMError`) and potentially for states if applicable.

4.  **Error Handling:**
    *   Use `VLMError` (defined in `Sources/mlx_vlm/Common/VLMError.swift`) for all project-specific errors (config loading, weight loading, tensor mismatches).
    *   Use `throw`/`try`/`catch` for all operations that can fail (file I/O, tensor loading/manipulation).
        ```swift
        // In SafeTensorsLoader.swift
        public func getTensor(named name: String) throws -> MLXArray {
            guard let tensor = self.tensors[name] else {
                throw VLMError.weightsNotFound(tensorName: name)
            }
            return tensor
        }

        // Usage
        do {
            let attentionWeights = try weightLoader.getTensor(named: "attn.q_proj.weight")
        } catch let error as VLMError {
            print("Failed to load weights: \(error.localizedDescription)")
            // Handle appropriately
        } catch {
            print("An unexpected error: \(error)")
        }
        ```
    *   Avoid `try!` and `try?` unless the error is genuinely unrecoverable in that specific context or explicitly being ignored (use `try?` with caution).

5.  **Concurrency (async/await):**
    *   Main model loading (`Qwen2_5_VL_Inference.init`) should be `async throws` due to file I/O.
    *   The `generate` method should be `async throws`.
    *   `MLXArray` operations are generally synchronous from the Swift perspective (MLX handles GPU execution).
    *   If future versions involve background data fetching/processing not directly tied to MLX's execution model, use `Task` and ensure `Sendable` conformance where needed.
        ```swift
        // In Qwen2_5_VL_Inference.swift
        public init(modelDirectoryURL: URL) async throws {
            // ... file loading for config, tokenizer, weights ...
        }

        public func generate(...) async throws -> String {
            // ...
            let visualFeatures = visionEncoder(imageTensor) // MLX sync call
            // ...
        }
        ```

6.  **Control Flow:**
    *   `guard` for early exits, especially when checking for loaded weights or valid tensor shapes.
    *   `for-in` for iterating over layers or tokens.

7.  **Functions and Methods:**
    *   Clear, descriptive names, consistent with `SDD.md`. `loadWeights`, `callAsFunction` (for `MLXNN.Module`), `preprocessImage`, `tokenizePrompt`.
    *   For `MLXNN.Module` subclasses, `callAsFunction` is the forward pass.
    *   Keep weight loading logic separate but clearly associated with its module.

8.  **Access Control:**
    *   `public` for main inference class (`Qwen2_5_VL_Inference`), core configuration structs, and any top-level functions intended for library users.
    *   `internal` (default) for most helper classes, utility functions, and individual `MLXNN.Module` layers that are not meant to be directly instantiated or used outside the library's own assembly.
    *   `private` or `fileprivate` for implementation details within a file/class.

9.  **Code Formatting and Style:**
    *   **Indentation:** 4 spaces.
    *   **Line Length:** Aim for 100-120 characters. Tensor operation chains can sometimes be longer; break them logically if it improves readability.
    *   **Whitespace:** Around operators, after commas, blank lines between methods and logical blocks within methods.
    *   **Braces:** Opening braces on the same line.

10. **Naming Conventions:**
    *   UpperCamelCase for types: `Qwen2_5_VL_Inference`, `QwenViTAttention`, `SafeTensorsLoader`, `VLMError`.
    *   lowerCamelCase for variables/properties, functions/methods: `visionEncoder`, `languageModel`, `qProjection`, `loadWeights`, `callAsFunction`.
    *   For weight names loaded from Hugging Face, use their exact names as strings, e.g., `"model.visual.patch_embed.proj.weight"`. Store these mapping prefixes carefully.

11. **Comments:**
    *   **DocC Comments (`///`):** For all public APIs (`Qwen2_5_VL_Inference`, its `init` and `generate`, public config structs).
        ```swift
        /// Initializes and loads the Qwen2.5-VL model.
        /// - Parameter modelDirectoryURL: URL to the directory containing all model files
        ///   (configs, tokenizer, .safetensors weights).
        /// - Throws: `VLMError` if loading fails.
        public init(modelDirectoryURL: URL) async throws { ... }
        ```
    *   **Explain Complex Logic:** For non-obvious tensor manipulations, specific architectural choices adapted from Python, or weight name mapping logic. Clarify how the implementation aligns with `SDD.md` or `implementationPlan.md` if not immediately obvious.
        ```swift
        // Transpose K for matrix multiplication in attention, as per reference architecture
        let K_transposed = K.transposed(0, 1) // Assuming (batch, seq, head, dim) -> (batch, head, seq, dim)

        // FIXME: This weight prefix might change if HF model structure is updated. Verify against latest HF model.
        let attentionWeightPrefix = "\(layerPrefix).attn."
        ```
    *   **MARK, FIXME, TODO:**
        ```swift
        // MARK: - Vision Encoder Layers (Refer to SDD section X.Y)

        // TODO: Implement support for top-k sampling in the generate method (as per enhancement plan in implementationPlan.md).

        // FIXME: Potential performance bottleneck in image resizing before ViT. Profile this.
        ```

12. **Project-Specific Conventions for `mlx-vlm`:**
    *   **Python Reference is King & Layer-wise Validation:**
        *   The Python `transformers` implementation of `qwen2_5_vl` is the **ground truth**. Your Swift implementation's structure, weight names, and intermediate outputs *must* be verifiable against it.
        *   **Critical Validation Step:** For every `MLXNN.Module` or significant processing block (as defined in `implementationPlan.md`):
            1.  Use the Python `transformers` implementation to process a reference input.
            2.  **Extract and save the layer-by-layer intermediate outputs** from the Python model (e.g., as `.npy` files). Also, save the exact inputs and weights if they are dynamically generated or modified in Python.
            3.  These Python-generated outputs serve as the target for your Swift implementation.
    *   **Weight Loading:**
        *   Use `SafeTensorsLoader` to load weights.
        *   Pass specific `prefix` strings to `loadWeights` methods of sub-modules to correctly scope weight names (e.g., ViT prefix: `"model.visual."`, LLM prefix: `"model.language_model."`). Verify these prefixes against the Python model structure.
    *   **Configuration:** Model dimensions and architectural details are driven by `Qwen2_5_VL_OverallConfig` loaded from `config.json`, which must be compatible with the reference Python model's configuration.
    *   **Error Handling:** Throw specific `VLMError` cases for issues related to model loading, weight finding, tensor shape mismatches, etc.
    *   **MLXArray Usage:**
        *   Be mindful of tensor shapes. Add assertions or guards for expected dimensions, cross-referencing with Python model tensor shapes.
        *   Use `MLX.eval()` sparingly, only if explicitly needed to force computation for debugging or precise comparison with Python eager execution outputs.
        *   Clean up intermediate `MLXArray`s if memory becomes an issue, though MLX has its own GC. For critical loops, consider manual disposal if absolutely necessary and profiled.
    *   **DeepWiki MCP Queries (Referencing `mlx-swift`, `mlx-swift-examples`, and `mlx-vlm`):**
        When stuck on MLX-specific implementations, refer to these repositories for existing patterns and solutions. Frame your queries considering how similar problems might have been solved in:
        *   Core `mlx-swift` library: `https://github.com/ml-explore/mlx-swift.git`
        *   Official examples: `https://github.com/ml-explore/mlx-swift-examples.git`
        *   python imp `mlx-vlm` project: `https://github.com/Blaizzy/mlx-vlm.git`

        *   Example Query (inspired by `mlx-swift-examples`):
            `"How is `MLXSequential` used to build a Vision Transformer in `mlx-swift-examples` (e.g., ViT example) and how can I adapt that pattern for the Qwen2.5-VL vision encoder, ensuring correct layer definitions as per our SDD.md?"`
        *   Example Query (referencing `mlx-swift` for core functionality):
            `"What are the best practices for implementing custom attention mechanisms using `MLXNN.Module` and low-level `MLXArray` operations in `mlx-swift` (see `mlx-swift/Sources/MLXNN/Attention.swift`) that I can apply to the Qwen2.5-VL's specific attention variant, aiming for output parity with Python?"`
        *   Example Query (for `mlx-vlm`):
            `"Given the `SafeTensorsLoader` implementation in `mlx-vlm/Sources/mlx-vlm/Common/`, how should I structure the weight loading for a new multimodal layer in `Qwen2_5_VL_LanguageModel.swift` to maintain consistency with how `QwenViT.swift` loads its weights, while ensuring all weight names match the Hugging Face model?"`
        *   Example Query (general, but implies knowledge from these repos):
            `"Optimizing `MLXArray` transpose and matmul operations in a tight loop for a custom transformer layer in Swift, similar to patterns found in `mlx-swift` or `mlx-swift-examples`, to match Python performance characteristics."`

13. **Testing (Project Specific):**
    *   **Layer-by-Layer Validation against Python Outputs:** This is paramount and directly ties into point 12.
        *   Your `Tests/mlx_vlmTests/Models/Qwen2_5_VL/{VisionLayerTests.swift, LanguageLayerTests.swift, etc.}` are critical.
        *   For each module test:
            1.  Load pre-saved inputs and **intermediate outputs generated by the Python `transformers` reference implementation** (see point 12).
            2.  Instantiate your Swift module, load corresponding weights.
            3.  Pass the same inputs to your Swift module.
            4.  **Compare the Swift module's output rigorously against the Python-generated output** using `MLX.allClose` or a similar numerical comparison utility.
    *   **End-to-End Inference Test:** `Tests/mlx_vlmTests/Models/Qwen2_5_VL/InferenceTests.swift` must validate the full pipeline against Python's output for the same image/prompt (e.g., greedy search).
    *   **Test Command:**
        ```bash
        xcodebuild test -scheme mlx_vlm -destination 'platform=OS X'
        ```
    *   Ensure test data (small `.npy` files for inputs/outputs from Python, sample configs, tiny safetensor files if feasible for isolated tests) are managed appropriately, possibly included in the test bundle or loaded from a known relative path during tests.
    *   All test development and execution must align with the testing strategy outlined in `implementationPlan.md` and `SDD.md`.
