import Testing
import CoreImage
import AVFoundation
@testable import mlx_vlm

/// Tests for MediaProcessing functionality
struct MediaProcessingTests {
    
    // MARK: - Test Setup
    
    private func createTestImage() -> CIImage {
        // Create a simple test image (10x10 red square)
        let redColor = CIColor(red: 1.0, green: 0.0, blue: 0.0)
        return CIImage(color: redColor).cropped(to: CGRect(x: 0, y: 0, width: 10, height: 10))
    }
    
    // MARK: - Image Processing Tests
    
    @Test("Test sRGB tone curve conversion")
    func testInSRGBToneCurveSpace() throws {
        let testImage = createTestImage()
        let result = MediaProcessing.inSRGBToneCurveSpace(testImage)
        
        #expect(result.extent.width == testImage.extent.width)
        #expect(result.extent.height == testImage.extent.height)
    }
    
    @Test("Test linear tone curve conversion")
    func testInLinearToneCurveSpace() throws {
        let testImage = createTestImage()
        let result = MediaProcessing.inLinearToneCurveSpace(testImage)
        
        #expect(result.extent.width == testImage.extent.width)
        #expect(result.extent.height == testImage.extent.height)
    }
    
    @Test("Test bicubic resampling")
    func testResampleBicubic() throws {
        let testImage = createTestImage()
        let targetSize = CGSize(width: 20, height: 20)
        let resampled = MediaProcessing.resampleBicubic(testImage, to: targetSize)
        
        #expect(Int(resampled.extent.width) == Int(targetSize.width))
        #expect(Int(resampled.extent.height) == Int(targetSize.height))
    }
    
    @Test("Test Lanczos resampling")
    func testResampleLanczos() throws {
        let testImage = createTestImage()
        let targetSize = CGSize(width: 20, height: 20)
        let resampled = MediaProcessing.resampleLanczos(testImage, to: targetSize)
        
        #expect(Int(resampled.extent.width) == Int(targetSize.width))
        #expect(Int(resampled.extent.height) == Int(targetSize.height))
    }
    
    @Test("Test image normalization")
    func testNormalize() throws {
        let testImage = createTestImage()
        let mean: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
        let std: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
        
        let normalized = MediaProcessing.normalize(testImage, mean: mean, std: std)
        
        #expect(normalized.extent.width == testImage.extent.width)
        #expect(normalized.extent.height == testImage.extent.height)
    }
    
    @Test("Test MLXArray conversion")
    func testAsMLXArray() throws {
        let testImage = createTestImage()
        let result = MediaProcessing.asMLXArray(testImage)
        
        // Should be [1, C, H, W] format
        #expect(result.shape.count == 4)
        #expect(result.shape[0] == 1)  // Batch size
        #expect(result.shape[1] == 3)  // Channels (RGB)
        #expect(result.shape[2] == 10) // Height
        #expect(result.shape[3] == 10) // Width
        #expect(result.dtype == .float32)
    }
    
    @Test("Test best fit calculation")
    func testBestFit() {
        let originalSize = CGSize(width: 100, height: 200)
        let targetSize = CGSize(width: 50, height: 50)
        
        let result = MediaProcessing.bestFit(originalSize, in: targetSize)
        let expectedScale = MediaProcessing.bestFitScale(originalSize, in: targetSize)
        
        #expect(result.width == round(originalSize.width * expectedScale))
        #expect(result.height == round(originalSize.height * expectedScale))
    }
    
    @Test("Test center crop")
    func testCenterCrop() {
        let testImage = createTestImage()
        let cropSize = CGSize(width: 5, height: 5)
        
        let cropped = MediaProcessing.centerCrop(testImage, size: cropSize)
        
        #expect(cropped.extent.width == cropSize.width)
        #expect(cropped.extent.height == cropSize.height)
    }
    
    @Test("Test fit in shortest edge")
    func testFitInShortestEdge() {
        let originalSize = CGSize(width: 100, height: 200)
        let shortestEdge = 50
        
        let result = MediaProcessing.fitIn(originalSize, shortestEdge: shortestEdge)
        
        // The shortest edge should be 50
        let minDimension = min(result.width, result.height)
        #expect(Int(minDimension) == shortestEdge)
        
        // Aspect ratio should be preserved
        let originalAspectRatio = originalSize.width / originalSize.height
        let resultAspectRatio = result.width / result.height
        #expect(abs(originalAspectRatio - resultAspectRatio) < 0.001)
    }
    
    @Test("Test fit in longest edge")
    func testFitInLongestEdge() {
        let originalSize = CGSize(width: 100, height: 200)
        let longestEdge = 150
        
        let result = MediaProcessing.fitIn(originalSize, longestEdge: longestEdge)
        
        // The longest edge should be <= 150
        let maxDimension = max(result.width, result.height)
        #expect(maxDimension <= CGFloat(longestEdge))
        
        // Aspect ratio should be preserved
        let originalAspectRatio = originalSize.width / originalSize.height
        let resultAspectRatio = result.width / result.height
        #expect(abs(originalAspectRatio - resultAspectRatio) < 0.001)
    }
    
    @Test("Test apply processing with nil")
    func testApplyProcessingNil() {
        let testImage = createTestImage()
        
        let result = MediaProcessing.apply(testImage, processing: nil)
        
        // Should return the original image unchanged
        #expect(result.extent.width == testImage.extent.width)
        #expect(result.extent.height == testImage.extent.height)
    }
    
    @Test("Test rect smaller or equal")
    func testRectSmallerOrEqual() {
        let smallRect = CGRect(x: 0, y: 0, width: 10, height: 10)
        let largeSize = CGSize(width: 20, height: 20)
        let smallSize = CGSize(width: 5, height: 5)
        
        #expect(MediaProcessing.rectSmallerOrEqual(smallRect, size: largeSize))
        #expect(!MediaProcessing.rectSmallerOrEqual(smallRect, size: smallSize))
    }
    
    @Test("Test aspect ratio for resample")
    func testAspectRatioForResample() {
        let testImage = createTestImage()
        let targetSize = CGSize(width: 20, height: 10)
        
        let aspectRatio = MediaProcessing.aspectRatioForResample(testImage, size: targetSize)
        
        // Should be a reasonable aspect ratio
        #expect(aspectRatio > 0)
        #expect(aspectRatio.isFinite)
    }
    
    // MARK: - CIImage Extension Tests
    
    @Test("Test CIImage resampled extension")
    func testCIImageResampled() throws {
        let testImage = createTestImage()
        let targetSize = CGSize(width: 20, height: 20)
        
        // Test bicubic resampling
        let bicubicResult = testImage.resampled(to: targetSize, method: .bicubic)
        #expect(Int(bicubicResult.extent.width) == Int(targetSize.width))
        #expect(Int(bicubicResult.extent.height) == Int(targetSize.height))
        
        // Test Lanczos resampling
        let lanczosResult = testImage.resampled(to: targetSize, method: .lanczos)
        #expect(Int(lanczosResult.extent.width) == Int(targetSize.width))
        #expect(Int(lanczosResult.extent.height) == Int(targetSize.height))
    }
    
    @Test("Test CIImage color space extensions")
    func testCIImageColorSpaceExtensions() throws {
        let testImage = createTestImage()
        
        // Test toSRGB extension
        let srgbResult = testImage.toSRGB()
        #expect(srgbResult.extent.width == testImage.extent.width)
        #expect(srgbResult.extent.height == testImage.extent.height)
        
        // Test toLinear extension
        let linearResult = testImage.toLinear()
        #expect(linearResult.extent.width == testImage.extent.width)
        #expect(linearResult.extent.height == testImage.extent.height)
    }
    
    @Test("Test CIImage normalized extension")
    func testCIImageNormalizedExtension() throws {
        let testImage = createTestImage()
        let mean: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
        let std: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
        
        let normalizedResult = testImage.normalized(mean: mean, std: std)
        #expect(normalizedResult.extent.width == testImage.extent.width)
        #expect(normalizedResult.extent.height == testImage.extent.height)
    }
    
    @Test("Test CIImage asMLXArray extension")
    func testCIImageAsMLXArrayExtension() throws {
        let testImage = createTestImage()
        
        let mlxArrayResult = testImage.asMLXArray()
        #expect(mlxArrayResult.shape.count == 4)
        #expect(mlxArrayResult.shape[0] == 1)  // Batch size
        #expect(mlxArrayResult.shape[1] == 3)  // Channels
        #expect(mlxArrayResult.shape[2] == 10) // Height
        #expect(mlxArrayResult.shape[3] == 10) // Width
    }
}