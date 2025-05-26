import MLX
import MLXLMCommon
import Foundation

/// A protocol that defines the requirements for a Vision Language Model.
///
/// VLMModel combines the capabilities of a standard language model with
/// LoRA (Low-Rank Adaptation) functionality to enable processing of both
/// textual and visual inputs.
public protocol VLMModel: LanguageModel, LoRAModel {
    // This protocol inherits all requirements from LanguageModel and LoRAModel
    // Additional VLM-specific requirements can be added here as needed
}