"""
Image preprocessing module for FaceAttend application
Handles grayscale conversion and histogram equalization for face images
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from src.utils.logger import get_module_logger
from src.utils.exceptions import ValidationError

class ImageProcessor:
    """Image preprocessing for face recognition"""
    
    def __init__(self):
        """Initialize the image processor"""
        self.logger = get_module_logger("ImageProcessor")
        self.logger.info("ImageProcessor initialized")
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Grayscale image
        """
        if image is None or image.size == 0:
            raise ValidationError("Empty image provided for grayscale conversion")
        
        try:
            # Check if already grayscale
            if len(image.shape) == 2:
                self.logger.debug("Image is already grayscale")
                return image.copy()
            elif len(image.shape) == 3 and image.shape[2] == 1:
                return image.squeeze()
            
            # Convert BGR to grayscale
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.logger.debug(f"Converted image to grayscale: {gray.shape}")
                return gray
            elif image.shape[2] == 4:
                # Handle BGRA
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                self.logger.debug(f"Converted BGRA image to grayscale: {gray.shape}")
                return gray
            else:
                raise ValidationError(f"Unsupported image format: {image.shape}")
                
        except Exception as e:
            self.logger.error(f"Error converting to grayscale: {str(e)}")
            raise ValidationError(f"Grayscale conversion failed: {str(e)}")
    
    def apply_histogram_equalization(self, image: np.ndarray, method: str = 'global') -> np.ndarray:
        """
        Apply histogram equalization to improve image quality
        
        Args:
            image: Input grayscale image
            method: Equalization method ('global', 'adaptive', 'clahe')
            
        Returns:
            Histogram equalized image
        """
        if image is None or image.size == 0:
            raise ValidationError("Empty image provided for histogram equalization")
        
        # Ensure image is grayscale
        if len(image.shape) != 2:
            image = self.convert_to_grayscale(image)
        
        try:
            if method == 'global':
                # Global histogram equalization
                equalized = cv2.equalizeHist(image)
                self.logger.debug("Applied global histogram equalization")
                
            elif method == 'adaptive':
                # Adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized = clahe.apply(image)
                self.logger.debug("Applied adaptive histogram equalization")
                
            elif method == 'clahe':
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                equalized = clahe.apply(image)
                self.logger.debug("Applied CLAHE histogram equalization")
                
            else:
                raise ValidationError(f"Unknown equalization method: {method}")
            
            return equalized
            
        except Exception as e:
            self.logger.error(f"Error in histogram equalization: {str(e)}")
            raise ValidationError(f"Histogram equalization failed: {str(e)}")
    
    def preprocess_face_image(self, image: np.ndarray, target_size: Tuple[int, int] = (150, 150),
                             equalization_method: str = 'clahe') -> np.ndarray:
        """
        Complete preprocessing pipeline for face images
        
        Args:
            image: Input face image
            target_size: Target size for the processed image (width, height)
            equalization_method: Histogram equalization method
            
        Returns:
            Preprocessed face image
        """
        try:
            # Step 1: Convert to grayscale
            gray_image = self.convert_to_grayscale(image)
            
            # Step 2: Resize to target size
            resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
            
            # Step 3: Apply histogram equalization
            equalized_image = self.apply_histogram_equalization(resized_image, equalization_method)
            
            # Step 4: Normalize pixel values
            normalized_image = self.normalize_image(equalized_image)
            
            self.logger.debug(f"Preprocessed face image: {normalized_image.shape}, dtype: {normalized_image.dtype}")
            return normalized_image
            
        except Exception as e:
            self.logger.error(f"Error in face image preprocessing: {str(e)}")
            raise ValidationError(f"Face image preprocessing failed: {str(e)}")
    
    def preprocess_for_recognition(self, image: np.ndarray, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Preprocess face image specifically for LBPH recognition
        
        Args:
            image: Input face image
            target_size: Target size for recognition (width, height)
            
        Returns:
            Preprocessed face image optimized for LBPH recognition
        """
        return self.preprocess_face_image(image, target_size, equalization_method='clahe')
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'none')
            
        Returns:
            Normalized image
        """
        try:
            if method == 'minmax':
                # Min-max normalization to [0, 255]
                normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                
            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(image)
                std = np.std(image)
                if std > 0:
                    normalized = ((image - mean) / std * 50 + 128).astype(np.uint8)
                    normalized = np.clip(normalized, 0, 255)
                else:
                    normalized = image.copy()
                    
            elif method == 'none':
                normalized = image.copy()
                
            else:
                raise ValidationError(f"Unknown normalization method: {method}")
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error in image normalization: {str(e)}")
            raise ValidationError(f"Image normalization failed: {str(e)}")
    
    def extract_face_region(self, image: np.ndarray, face_rect: Tuple[int, int, int, int],
                           padding: float = 0.1) -> np.ndarray:
        """
        Extract face region from image with optional padding
        
        Args:
            image: Input image
            face_rect: Face rectangle (x, y, width, height)
            padding: Padding ratio around face
            
        Returns:
            Extracted face region
        """
        try:
            x, y, w, h = face_rect
            img_h, img_w = image.shape[:2]
            
            # Calculate padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate expanded coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_w, x + w + pad_w)
            y2 = min(img_h, y + h + pad_h)
            
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                raise ValidationError("Empty face region extracted")
            
            self.logger.debug(f"Extracted face region: {face_region.shape}")
            return face_region
            
        except Exception as e:
            self.logger.error(f"Error extracting face region: {str(e)}")
            raise ValidationError(f"Face region extraction failed: {str(e)}")
    
    def enhance_face_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply additional enhancements to face image
        
        Args:
            image: Input face image
            
        Returns:
            Enhanced face image
        """
        try:
            # Ensure grayscale
            if len(image.shape) != 2:
                image = self.convert_to_grayscale(image)
            
            # Apply Gaussian blur to reduce noise
            denoised = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Apply unsharp masking for better edges
            gaussian = cv2.GaussianBlur(denoised, (9, 9), 10.0)
            enhanced = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
            
            # Clip values to valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            self.logger.debug("Applied face image enhancements")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing face image: {str(e)}")
            return image  # Return original if enhancement fails
    
    def validate_image_quality(self, image: np.ndarray, min_size: Tuple[int, int] = (50, 50),
                              min_variance: float = 100.0) -> bool:
        """
        Validate image quality for face recognition
        
        Args:
            image: Input image
            min_size: Minimum acceptable size
            min_variance: Minimum pixel variance (to detect blank images)
            
        Returns:
            True if image quality is acceptable
        """
        try:
            if image is None or image.size == 0:
                self.logger.warning("Empty image provided for quality validation")
                return False
            
            # Check size
            h, w = image.shape[:2]
            if w < min_size[0] or h < min_size[1]:
                self.logger.warning(f"Image too small: {w}x{h}, minimum: {min_size}")
                return False
            
            # Convert to grayscale for variance calculation
            if len(image.shape) == 3:
                gray = self.convert_to_grayscale(image)
            else:
                gray = image
            
            # Check variance (detect blank or very uniform images)
            variance = np.var(gray)
            if variance < min_variance:
                self.logger.warning(f"Image variance too low: {variance}, minimum: {min_variance}")
                return False
            
            # Check for extreme brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 10 or mean_brightness > 245:
                self.logger.warning(f"Extreme brightness detected: {mean_brightness}")
                return False
            
            self.logger.debug("Image quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in image quality validation: {str(e)}")
            return False
    
    def batch_preprocess_faces(self, images: List[np.ndarray], target_size: Tuple[int, int] = (150, 150)) -> List[np.ndarray]:
        """
        Preprocess a batch of face images
        
        Args:
            images: List of input face images
            target_size: Target size for processed images
            
        Returns:
            List of preprocessed face images
        """
        processed_images = []
        
        for i, image in enumerate(images):
            try:
                # Validate image quality
                if not self.validate_image_quality(image):
                    self.logger.warning(f"Skipping low quality image {i}")
                    continue
                
                # Preprocess image
                processed = self.preprocess_face_image(image, target_size)
                processed_images.append(processed)
                
            except Exception as e:
                self.logger.error(f"Error processing image {i}: {str(e)}")
                continue
        
        self.logger.info(f"Processed {len(processed_images)} out of {len(images)} images")
        return processed_images
    
    def compare_preprocessing_methods(self, image: np.ndarray) -> dict:
        """
        Compare different preprocessing methods on an image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with different preprocessing results
        """
        results = {}
        
        try:
            # Original grayscale
            gray = self.convert_to_grayscale(image)
            results['original'] = gray
            
            # Global histogram equalization
            global_eq = self.apply_histogram_equalization(gray, 'global')
            results['global_eq'] = global_eq
            
            # Adaptive histogram equalization
            adaptive_eq = self.apply_histogram_equalization(gray, 'adaptive')
            results['adaptive_eq'] = adaptive_eq
            
            # CLAHE
            clahe_eq = self.apply_histogram_equalization(gray, 'clahe')
            results['clahe_eq'] = clahe_eq
            
            # Enhanced
            enhanced = self.enhance_face_image(gray)
            results['enhanced'] = enhanced
            
            self.logger.info("Generated preprocessing method comparisons")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing comparison: {str(e)}")
            return {'original': gray} if 'gray' in locals() else {}

# Utility functions
def test_image_preprocessing(image_path: str = None):
    """
    Test image preprocessing functionality
    
    Args:
        image_path: Path to test image (optional)
    """
    import os
    
    processor = ImageProcessor()
    
    if image_path and os.path.exists(image_path):
        # Test with provided image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
    else:
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("Press SPACE to capture image for preprocessing test, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Capture Image for Preprocessing Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                image = frame.copy()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Test preprocessing methods
    try:
        results = processor.compare_preprocessing_methods(image)
        
        # Display results
        for method, processed_img in results.items():
            cv2.imshow(f'Preprocessing: {method}', processed_img)
        
        print("Press any key to close windows")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save results
        for method, processed_img in results.items():
            filename = f"preprocessing_{method}.jpg"
            cv2.imwrite(filename, processed_img)
            print(f"Saved: {filename}")
        
    except Exception as e:
        print(f"Preprocessing test failed: {str(e)}")

if __name__ == "__main__":
    test_image_preprocessing() 