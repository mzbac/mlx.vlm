import Foundation

/// Represents errors that can occur when working with Vision Language Models.
///
/// `VLMError` provides specific error cases and descriptive messages for issues
/// that may arise during VLM loading, processing, or inference operations.
public enum VLMError: LocalizedError, Equatable {
    /// The model configuration is invalid or missing required fields.
    case invalidModelConfiguration(String)
    
    /// The processor configuration is invalid or missing required fields.
    case invalidProcessorConfiguration(String)
    
    /// Failed to process media input (image, video, etc.).
    case mediaProcessingFailed(String)
    
    /// Model weights are missing or incompatible with the model architecture.
    case modelWeightsMismatch(String)
    
    /// Failed to tokenize input text.
    case tokenizationFailed(String)
    
    /// The model is not compatible with the given input type.
    case incompatibleInputType(String)
    
    /// An error occurred during model inference.
    case inferenceError(String)
    
    /// The model failed to generate output for the given input.
    case generationFailed(String)
    
    /// A provided file path is invalid or a file cannot be accessed.
    case fileAccessError(String)
    
    /// A tensor operation required for the VLM failed.
    case tensorOperationFailed(String)
    
    /// An unexpected error occurred which is not covered by other error cases.
    case unknownError(String)
    
    /// A human-readable localized description of the error.
    public var errorDescription: String? {
        switch self {
        case .invalidModelConfiguration(let details):
            return "Invalid model configuration: \(details)"
        case .invalidProcessorConfiguration(let details):
            return "Invalid processor configuration: \(details)"
        case .mediaProcessingFailed(let details):
            return "Media processing failed: \(details)"
        case .modelWeightsMismatch(let details):
            return "Model weights mismatch: \(details)"
        case .tokenizationFailed(let details):
            return "Tokenization failed: \(details)"
        case .incompatibleInputType(let details):
            return "Incompatible input type: \(details)"
        case .inferenceError(let details):
            return "Inference error: \(details)"
        case .generationFailed(let details):
            return "Generation failed: \(details)"
        case .fileAccessError(let details):
            return "File access error: \(details)"
        case .tensorOperationFailed(let details):
            return "Tensor operation failed: \(details)"
        case .unknownError(let details):
            return "Unknown error: \(details)"
        }
    }
    
    /// A human-readable localized message describing what failed.
    public var failureReason: String? {
        return errorDescription
    }
    
    /// A human-readable message describing how to recover from the error.
    public var recoverySuggestion: String? {
        switch self {
        case .invalidModelConfiguration:
            return "Check that the model configuration JSON is valid and complete."
        case .invalidProcessorConfiguration:
            return "Check that the processor configuration JSON is valid and complete."
        case .mediaProcessingFailed:
            return "Verify the media file is in a supported format and not corrupted."
        case .modelWeightsMismatch:
            return "Ensure you're using the correct model weights for this model architecture."
        case .tokenizationFailed:
            return "Check if the tokenizer is properly initialized and the input text is valid."
        case .incompatibleInputType:
            return "This model does not support the provided input type. Check documentation for supported inputs."
        case .inferenceError:
            return "Try with a different input or check if the model is properly loaded."
        case .generationFailed:
            return "Try adjusting generation parameters or check if the input makes sense for this model."
        case .fileAccessError:
            return "Verify the file exists and you have proper permissions to access it."
        case .tensorOperationFailed:
            return "Check if your input tensors have the expected shapes and datatypes."
        case .unknownError:
            return "Try reloading the model or check the logs for more details."
        }
    }
}