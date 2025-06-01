## Software Design Document: Swift VLM Inference Library for Qwen2.5_VL (Independent Implementation)

**1. Introduction**

This document outlines the design for a Swift library enabling Vision Language Model (VLM) inference, specifically targeting the `qwen2_5_vl` model. The library will be built using the `mlx-swift` framework. A core objective is to implement the VLM inference logic independently of the VLM examples found in `ml-explore/mlx-swift-examples`, aiming for a robust and potentially more flexible solution. A primary feature will be the ability to directly load Hugging Face model weights (e.g., from `.safetensors` files) without requiring an intermediate conversion step. This initial version will focus on implementing the complete inference pipeline for `qwen2_5_vl`. Should issues arise during implementation, DeepWiki MCP will be queried for information on the relevant repositories (`mlx-swift`, Python `transformers`).

**2. Goals**

*   Develop a new, independent Swift library for VLM inference using `mlx-swift`.
*   Implement inference specifically for the `qwen2_5_vl` VLM.
*   Enable **direct loading of Hugging Face model weights and configurations** for `qwen2_5_vl`.
*   Provide a clear and easy-to-use API for performing inference with `qwen2_5_vl`.
*   Structure the library with custom-built VLM components (Vision Encoder, Language Model, Fusion logic) using `mlx-swift`'s `MLXNN` module.
*   Design the library in a modular way to facilitate future expansion to support other VLM models, potentially with different input modalities or structures.

**3. Non-Goals**

*   Model training or fine-tuning (initially).
*   Relying on or directly porting VLM-specific implementation logic from `ml-explore/mlx-swift-examples`. (Generic MLX usage patterns or tokenizer integrations from that repository might still serve as a general reference for interacting with `mlx-swift` or standard Swift practices).
*   Support for all VLM models available in Hugging Face Transformers in this initial version.
*   Advanced quantization or optimization techniques beyond what `mlx-swift` directly offers for inference with custom models.

**4. System Architecture**

The library will be architected with distinct, custom-built modules:

*   **Configuration Parser:** Reads and interprets Hugging Face `config.json` files (potentially separate ones for vision and language components if structured that way, or a unified one for `qwen2_5_vl`) to extract architectural parameters.
*   **Tokenizer Loader & Manager:** Integrates a Swift-based tokenizer compatible with `qwen2_5_vl`. This will likely leverage the Hugging Face `swift-transformers` package or a similar robust tokenizer solution, focusing on loading `tokenizer.json` and `tokenizer_config.json`.
*   **Weight Loader:** Responsible for loading weights directly from `.safetensors` files and mapping them to the corresponding layers in the custom-built `mlx-swift` models. This involves understanding the naming conventions in the Hugging Face `qwen2_5_vl` model.
*   **Vision Encoder (`Qwen2_5_VL_ViT`):** A custom implementation of the `qwen2_5_vl` vision transformer (ViT) architecture using `mlx-swift`'s `MLXNN.Module`. This will include specific features like dynamic resolution handling and window attention if feasible.
*   **Language Model (`Qwen2_5_VL_LLM`):** A custom implementation of the `qwen2_5_vl` language model architecture (based on Qwen2.5 LLM) using `mlx-swift`'s `MLXNN.Module`. This will include transformer blocks, attention mechanisms (e.g., MRoPE if applicable and directly implementable), and normalization layers.
*   **Multimodal Fusion Module:** Custom logic, built with `mlx-swift`, to combine the outputs of the Vision Encoder with the inputs of the Language Model, as defined by the `qwen2_5_vl` architecture.
*   **Inference Orchestrator:** Manages the end-to-end inference flow, including input preprocessing, model forwarding, and output generation/postprocessing.
*   **Input/Output Processors:** Custom routines for pre-processing input images (e.g., normalization, ensuring compatibility with the ViT, handling native resolutions) and post-processing generated text.

**Diagram:**

```
+---------------------+     +-----------------------+     +-------------------------+
|     Input Image     | --> |   Custom Vision       | --> |                         |
+---------------------+     |   Encoder (MLXNN,     |     |                         |
                            |   Qwen2.5 ViT Arch)   |     |   Custom Multimodal     | --> Output Text
+---------------------+     +-----------------------+ --> |   Fusion Module (MLX)   |
|     Input Text      | --> |   Custom Language     |     |                         |
+---------------------+     |   Model (MLXNN,       |     |                         |
                            |   Qwen2.5 LLM Arch)   |     +-------------------------+
                            +-----------------------+

+------------------------+     +---------------------+     +--------------------------+
| Configuration Parser   | --> | Hugging Face Models | <-- |      Weight Loader       |
| (config.json)          |     | (qwen2_5_vl files)  |     | (.safetensors direct)    |
+------------------------+     +---------------------+     +--------------------------+

+--------------------------+
| Tokenizer Loader/Manager |
| (e.g., swift-transformers)|
+--------------------------+
```

**5. Detailed Design**

**5.1. Model Configuration and Weight Loading (`qwen2_5_vl`)**

*   **Configuration Parsing:**
    *   The library will parse `config.json` (or relevant sub-configurations) for `qwen2_5_vl` to dynamically set up the parameters of the Vision Encoder and Language Model (e.g., hidden sizes, number of layers, attention heads, vocabulary size, image size parameters if any are fixed, patch size, etc.).
    *   This allows flexibility if minor architectural variants of `qwen2_5_vl` exist or for future models.
*   **Direct Weight Loading:**
    *   A dedicated module will load weights from `.safetensors` files.
    *   This requires a robust mechanism to map tensor names from the Hugging Face model files to the layer parameters in the custom Swift `MLXNN.Module` implementations. This mapping will be derived from analyzing the Python `transformers` implementation of `qwen2_5_vl`.
    *   The loader will handle reading tensor data and converting it to `MLXArray` instances for layer initialization.

**5.2. Tokenizer**

*   The library will integrate a tokenizer. The recommended approach is to use the **Hugging Face `swift-transformers` package** due to its aim for compatibility with Hugging Face tokenizers.
*   This involves loading `tokenizer.json` (and potentially `tokenizer_config.json`, `merges.txt` for BPE based tokenizers like Qwen's) for `qwen2_5_vl`.
*   The tokenizer must correctly handle text-to-token-ID conversion, token-ID-to-text decoding, and any special tokens specific to `qwen2_5_vl` (e.g., image placeholder tokens, instruction tokens).

**5.3. Custom Vision Encoder (`Qwen2_5_VL_ViT` - MLXNN Implementation)**

*   This module will be a `MLXNN.Module` subclass.
*   **Architecture based on `qwen2_5_vl` ViT:**
    *   **Patch Embedding:** Converts input image patches into embeddings.
    *   **Positional Embeddings:** Appropriate positional encoding for visual tokens (e.g., 2D RoPE if that's what Qwen2.5-VL uses and is implementable, or learned absolute/relative).
    *   **Transformer Blocks:** Consisting of:
        *   **Self-Attention:** Custom implementation, potentially including windowed attention if the `qwen2_5_vl` ViT uses it and it can be built with `mlx-swift` operations.
        *   **Feed-Forward Network (FFN):** Typically MLP layers with activation functions (e.g., GELU, SwiGLU).
    *   **Normalization Layers:** (e.g., `MLXNN.RMSNorm` or `MLXNN.LayerNorm` as per `qwen2_5_vl` specs).
    *   **Output Projection:** To produce the final visual embeddings.
    *   Must support dynamic image resolutions as a feature of Qwen2.5-VL.

**5.4. Custom Language Model (`Qwen2_5_VL_LLM` - MLXNN Implementation)**

*   This module will be a `MLXNN.Module` subclass.
*   **Architecture based on `qwen2_5_vl`'s LLM (Qwen2.5 base):**
    *   **Token Embedding:** Maps input token IDs to embeddings.
    *   **Positional Embeddings:** E.g., Rotary Position Embedding (RoPE), potentially Multimodal RoPE if applicable and feasible to implement for text and fused visual inputs.
    *   **Transformer Blocks:** Consisting of:
        *   **Self-Attention:** (e.g., Grouped Query Attention if used by Qwen2.5) with KV Caching support.
        *   **Feed-Forward Network (FFN):** (e.g., SwiGLU).
    *   **Normalization Layers:** (e.g., `MLXNN.RMSNorm`).
    *   **Output Head:** A linear layer projecting to the vocabulary size for token prediction.

**5.5. Custom Multimodal Fusion Module**

*   This is a critical custom component that defines how visual information is injected into the language model.
*   Implementation will precisely follow the `qwen2_5_vl` methodology. This might involve:
    *   Treating visual embeddings as special tokens in the sequence.
    *   Adding visual embeddings directly to token embeddings.
    *   Using cross-attention mechanisms between visual and text representations (if part of the architecture).
    *   The specific layer(s) and tensor operations will be built using `mlx-swift`.

**5.6. Inference Orchestrator**

*   This component will manage the overall inference process:
    1.  Accept an input image and a text prompt.
    2.  Pre-process the image (normalization, conversion to `MLXArray`).
    3.  Tokenize the text prompt.
    4.  Feed the processed image to the `Qwen2_5_VL_ViT` to obtain visual features.
    5.  Utilize the `MultimodalFusionModule` to combine visual features with tokenized text.
    6.  Perform autoregressive generation using the `Qwen2_5_VL_LLM`:
        *   Iteratively predict the next token.
        *   Employ KV caching within the LLM's attention layers for efficiency. This will require careful state management in the custom LLM implementation.
        *   Implement sampling strategies (e.g., greedy, temperature-based).
    7.  Detokenize the generated sequence of token IDs.
    8.  Post-process the output text.

**5.7. Input/Output Processing**

*   **Image Input:**
    *   Functions to load images from common formats (e.g., `CGImage` on Apple platforms, or raw pixel buffers).
    *   Preprocessing steps like normalization (to the mean/std dev expected by `qwen2_5_vl`).
    *   Conversion to `MLXArray` with the correct shape and data type.
    *   Handle dynamic resolution by ensuring the ViT and subsequent layers can adapt.
*   **Text Output:**
    *   Detokenization using the loaded tokenizer.
    *   Removal of any special tokens not meant for final output.

**6. API Design (Initial Proposal)**

```swift
import MLX // Or specific MLX modules needed
import CoreGraphics // For CGImage, or a more abstract image representation

// Configuration for the VLM model - focused on direct HF loading
public struct VLMModelLoaderConfig {
    let modelId: String // Hugging Face model ID (e.g., "Qwen/Qwen2.5-VL-Instruct")
    // OR local path to the directory containing config.json, .safetensors, tokenizer files
    let localModelPath: URL?

    // Potentially add revision/branch for HF modelId
    // let revision: String? = "main"

    public init(modelId: String, localModelPath: URL? = nil) {
        self.modelId = modelId
        self.localModelPath = localModelPath
    }
}

// Main class for VLM inference using the custom implementation
public class Qwen2_5_VL_Inference {
    private var visionEncoder: Qwen2_5_VL_ViT // Custom MLXNN Module
    private var languageModel: Qwen2_5_VL_LLM // Custom MLXNN Module
    private var fusionModule: MultimodalFusionModule // Custom MLXNN Module
    private var tokenizer: Any // Instance from swift-transformers or similar
    private var config: Any // Parsed model configuration

    public init(loaderConfig: VLMModelLoaderConfig) async throws {
        // 1. Download/Locate model files (config.json(s), .safetensors, tokenizer files)
        //    based on loaderConfig.modelId or loaderConfig.localModelPath.
        // 2. Parse model configuration(s).
        // 3. Load tokenizer (e.g., using swift-transformers).
        // 4. Instantiate custom visionEncoder, languageModel, and fusionModule using parsed config.
        // 5. Implement custom weight loading logic:
        //    - Iterate through .safetensors files.
        //    - Map tensor names to the parameters of the custom Swift MLXNN modules.
        //    - Load weights into the layers.
        // This is a from-scratch implementation, not using mlx-swift-examples factories.
    }

    public func generate(image: CGImage, prompt: String, maxTokens: Int = 200, temperature: Float = 0.6) async throws -> String {
        // 1. Preprocess CGImage -> MLXArray (normalize, resize if absolutely necessary though Qwen2.5_VL prefers native).
        // 2. Tokenize prompt using the loaded tokenizer.
        // 3. Vision encoding: Pass image MLXArray through `visionEncoder`.
        // 4. Fusion: Use `fusionModule` to combine image features and text embeddings.
        // 5. Language generation:
        //    - Autoregressively generate tokens using `languageModel`.
        //    - Implement KV caching within your custom `languageModel` for efficiency.
        //    - Apply temperature sampling.
        // 6. Detokenize generated token IDs.
        // 7. Post-process and return the string.
        return "Generated text by custom Qwen2.5_VL"
    }
}

// Define Qwen2_5_VL_ViT, Qwen2_5_VL_LLM, MultimodalFusionModule as subclasses of MLXNN.Module
// These will contain the actual layer definitions (MLXNN.Linear, MLXNN.RMSNorm, custom attention etc.)
// and the logic for loading their specific weights.

// Example (conceptual)
// public class Qwen2_5_VL_ViT: MLXNN.Module {
//     // Define layers: patch_embed, blocks (attention, mlp), norm, etc.
//     public init(config: VisionConfig) { /* ... */ }
//     public func callAsFunction(_ x: MLXArray) -> MLXArray { /* ... */ }
//     // Method to load weights for this specific module from a SafeTensorsFile and prefix
//     public func loadWeights(from safetensorFiles: [URL], config: VisionConfig) throws { /* ... */ }
// }
```

**7. Technical Challenges & Mitigation**

*   **Direct Hugging Face Model Interpretation (Weights & Config):**
    *   **Challenge:** Accurately parsing `config.json` and mapping `.safetensors` weight names/structures to the custom Swift `MLXNN.Module` layers without any intermediate Python scripts. This is the most significant challenge.
    *   **Mitigation:** Thoroughly study the `transformers` Python source code for `qwen2_5_vl` to understand its architecture, layer naming conventions, and configuration parameters. Create detailed mapping documents. Implement robust parsing and error handling for model loading. Start with a simplified version of the model if necessary and incrementally add complexity.
*   **Independent Implementation of VLM Components:**
    *   **Challenge:** Re-implementing the vision encoder, language model, and fusion logic correctly in `mlx-swift` based on the `qwen2_5_vl` architecture.
    *   **Mitigation:** Break down each component into smaller, manageable `MLXNN.Module` subclasses. Test each module individually where possible. Refer to `mlx-swift` documentation for layer APIs. Python `transformers` will be the primary architectural reference.
*   **Complex Architectural Features (e.g., Window Attention, MRoPE):**
    *   **Challenge:** Implementing specialized mechanisms like windowed attention or Multimodal RoPE in Swift using `mlx-swift`'s available operations if they are not pre-built layers.
    *   **Mitigation:** Start with standard attention/positional embeddings if the complex variants prove too difficult initially. Incrementally try to implement the more advanced features. Consult the `mlx-swift` community or file issues if fundamental operations are missing. Query DeepWiki MCP for specific implementation patterns in `mlx-swift`.
*   **Tokenizer Integration:**
    *   **Challenge:** Ensuring the chosen Swift tokenizer (e.g., from `swift-transformers`) is fully compatible and correctly configured for `qwen2_5_vl`.
    *   **Mitigation:** Test the tokenizer extensively with known inputs/outputs. Pay close attention to special tokens and vocabulary configurations.
*   **KV Caching Implementation:**
    *   **Challenge:** Correctly implementing and managing the KV cache within the custom language model for efficient autoregressive generation.
    *   **Mitigation:** Design the attention layers in the `Qwen2_5_VL_LLM` to accept and return KV cache state. Manage the cache updates carefully during the generation loop.
*   **Memory Management and Performance:**
    *   **Challenge:** VLMs are memory-intensive. Ensuring efficient memory use and acceptable performance with custom-built modules.
    *   **Mitigation:** Follow `mlx-swift` best practices for memory. Profile and optimize critical code paths. Test on target devices early.
*   **Debugging and Verification:**
    *   **Challenge:** Verifying the correctness of the custom implementation against the reference Python model without having intermediate outputs readily available from a shared framework.
    *   **Mitigation:** If possible, generate intermediate outputs from the Python `transformers` model for specific layers and compare them with the outputs of your Swift modules given the same inputs and weights. This requires careful setup but is invaluable for debugging.

**8. Testing Strategy**

*   **Unit Tests:** For individual `MLXNN.Module` subclasses (e.g., a single attention layer, an FFN block), tokenizer functions, image preprocessing routines, and weight mapping logic.
*   **Component Tests:** Test the full `Qwen2_5_VL_ViT` and `Qwen2_5_VL_LLM` with sample inputs and loaded weights, comparing outputs to reference values if obtainable.
*   **Integration Tests:**
    *   Test the complete model loading pipeline (config parsing, tokenizer loading, weight loading).
    *   Test the full `Qwen2_5_VL_Inference.generate` method with sample image-text pairs. Compare outputs against those from the Python `transformers` `qwen2_5_vl` model for correctness.
*   **Performance Benchmarks:** Measure inference latency, throughput (if applicable), and peak memory usage on target Apple Silicon hardware.

**9. Dependencies**

*   **`ml-explore/mlx-swift`:** The core MLX framework, including `MLX`, `MLXNN`, `MLXOptimizers` (though optimizers are not for this inference-focused phase).
*   **`huggingface/swift-transformers` (Recommended):** For robust and compatible tokenizer implementation.
*   **No direct dependency on `ml-explore/mlx-swift-examples` for VLM model implementation logic.**

**10. Future Considerations**

*   Extending the library to support other VLM architectures by abstracting common components.
*   Implementing model quantization loading/support if `mlx-swift` offers pathways for custom quantized models.
*   Support for batch inference if the custom design allows for it.
*   Video input capabilities, if extending to models like Qwen-VL-Chat that explicitly handle video.
