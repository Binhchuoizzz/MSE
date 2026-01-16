"""
Face detection module for FaceAttend application
Implements Haar Cascade classifier for reliable face detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from src.utils.logger import get_module_logger
from src.utils.exceptions import FaceDetectionError

class FaceDetector:
    """Face detector using Haar Cascade classifiers"""
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)):
        """
        Initialize the face detector
        
        Args:
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size, smaller objects are ignored
        """
        self.logger = get_module_logger("FaceDetector")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load Haar Cascade classifiers
        self.face_cascade = None
        self.eye_cascade = None
        self.profile_cascade = None
        
        self._load_cascades()
        
        self.logger.info("FaceDetector initialized with scale_factor={}, min_neighbors={}, min_size={}".format(
            scale_factor, min_neighbors, min_size))
    
    def _load_cascades(self):
        """Load Haar Cascade classifiers"""
        try:
            # Load face cascade (frontal face)
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            if self.face_cascade.empty():
                raise FaceDetectionError("Failed to load frontal face cascade")
            
            # Load eye cascade (for additional validation)
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            # Load profile face cascade (for side faces)
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
            
            self.logger.info("Haar Cascade classifiers loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Haar Cascades: {str(e)}")
            raise FaceDetectionError(f"Cascade loading failed: {str(e)}")
    
    def detect_faces(self, frame: np.ndarray, detect_eyes: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame
        
        Args:
            frame: Input image frame
            detect_eyes: Whether to also detect eyes for validation
            
        Returns:
            List of face rectangles as (x, y, width, height) tuples
        """
        if frame is None or frame.size == 0:
            self.logger.warning("Empty frame provided for face detection")
            return []
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Detect frontal faces
            frontal_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detect profile faces
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Combine and filter faces
            all_faces = list(frontal_faces) + list(profile_faces)
            
            if len(all_faces) == 0:
                return []
            
            # Remove overlapping detections
            filtered_faces = self._filter_overlapping_faces(all_faces)
            
            # Additional validation with eye detection if requested
            if detect_eyes and len(filtered_faces) > 0:
                validated_faces = []
                for face in filtered_faces:
                    if self._validate_face_with_eyes(gray, face):
                        validated_faces.append(face)
                filtered_faces = validated_faces
            
            self.logger.debug(f"Detected {len(filtered_faces)} faces in frame")
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in filtered_faces]
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def _filter_overlapping_faces(self, faces: List[Tuple[int, int, int, int]], 
                                 overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        Filter out overlapping face detections
        
        Args:
            faces: List of face rectangles
            overlap_threshold: Minimum overlap ratio to consider faces as overlapping
            
        Returns:
            Filtered list of face rectangles
        """
        if len(faces) <= 1:
            return faces
        
        # Convert to numpy array for easier processing
        faces_array = np.array(faces)
        
        # Calculate areas
        areas = faces_array[:, 2] * faces_array[:, 3]
        
        # Sort by area (largest first)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Keep the largest face
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate overlaps with remaining faces
            remaining = indices[1:]
            overlaps = []
            
            for idx in remaining:
                overlap_ratio = self._calculate_overlap_ratio(faces_array[current], faces_array[idx])
                overlaps.append(overlap_ratio)
            
            # Remove faces that overlap significantly
            overlaps = np.array(overlaps)
            indices = remaining[overlaps < overlap_threshold]
        
        return [tuple(faces_array[i]) for i in keep]
    
    def _calculate_overlap_ratio(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """Calculate overlap ratio between two face rectangles"""
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _validate_face_with_eyes(self, gray_frame: np.ndarray, face: Tuple[int, int, int, int]) -> bool:
        """
        Validate a face detection by checking for eyes
        
        Args:
            gray_frame: Grayscale image
            face: Face rectangle (x, y, width, height)
            
        Returns:
            True if eyes are detected within the face region
        """
        try:
            x, y, w, h = face
            
            # Extract face region
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Detect eyes in the upper half of the face
            eye_region = face_roi[0:h//2, :]
            eyes = self.eye_cascade.detectMultiScale(
                eye_region,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(10, 10)
            )
            
            # Consider valid if at least one eye is detected
            return len(eyes) >= 1
            
        except Exception as e:
            self.logger.warning(f"Error in eye validation: {str(e)}")
            return True  # Assume valid if validation fails
    
    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the largest face from a list of detected faces
        
        Args:
            faces: List of face rectangles
            
        Returns:
            Largest face rectangle or None if no faces
        """
        if not faces:
            return None
        
        # Calculate areas and find the largest
        areas = [(w * h, face) for face in faces for x, y, w, h in [face]]
        largest_area, largest_face = max(areas, key=lambda x: x[0])
        
        return largest_face
    
    def is_face_centered(self, face: Tuple[int, int, int, int], frame_shape: Tuple[int, int], 
                        center_tolerance: float = 0.4) -> bool:
        """
        Check if a face is reasonably centered in the frame
        
        Args:
            face: Face rectangle (x, y, width, height)
            frame_shape: Frame dimensions (height, width)
            center_tolerance: Tolerance for center detection (0.0 to 1.0)
            
        Returns:
            True if face is centered within tolerance
        """
        x, y, w, h = face
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate frame center
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Calculate distance from center as ratio of frame size
        dx = abs(face_center_x - frame_center_x) / frame_w
        dy = abs(face_center_y - frame_center_y) / frame_h
        
        return dx <= center_tolerance and dy <= center_tolerance
    
    def is_face_good_size(self, face: Tuple[int, int, int, int], frame_shape: Tuple[int, int],
                         min_ratio: float = 0.08, max_ratio: float = 0.9) -> bool:
        """
        Check if face size is appropriate for registration
        
        Args:
            face: Face rectangle (x, y, width, height)
            frame_shape: Frame dimensions (height, width)
            min_ratio: Minimum face size ratio to frame
            max_ratio: Maximum face size ratio to frame
            
        Returns:
            True if face size is appropriate
        """
        x, y, w, h = face
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate face area ratio
        face_area = w * h
        frame_area = frame_w * frame_h
        area_ratio = face_area / frame_area
        
        return min_ratio <= area_ratio <= max_ratio
    
    def get_face_quality_score(self, face: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Get detailed face quality scores for better user feedback
        
        Args:
            face: Face rectangle (x, y, width, height)
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Dictionary with quality scores (0.0 to 1.0)
        """
        x, y, w, h = face
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate centering score
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        dx = abs(face_center_x - frame_center_x) / frame_w
        dy = abs(face_center_y - frame_center_y) / frame_h
        center_score = max(0.0, 1.0 - max(dx, dy) * 2.5)  # Convert distance to score
        
        # Calculate size score
        face_area = w * h
        frame_area = frame_w * frame_h
        area_ratio = face_area / frame_area
        
        # Optimal range is 0.15 to 0.6
        if 0.15 <= area_ratio <= 0.6:
            size_score = 1.0
        elif area_ratio < 0.15:
            size_score = max(0.0, area_ratio / 0.15)
        else:
            size_score = max(0.0, 1.0 - (area_ratio - 0.6) / 0.4)
        
        return {
            'center_score': center_score,
            'size_score': size_score,
            'overall_score': (center_score + size_score) / 2,
            'area_ratio': area_ratio,
            'center_offset_x': dx,
            'center_offset_y': dy
        }
    
    def draw_face_rectangles(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                           color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles around detected faces
        
        Args:
            frame: Input frame
            faces: List of face rectangles
            color: Rectangle color (B, G, R)
            thickness: Rectangle thickness
            
        Returns:
            Frame with face rectangles drawn
        """
        result_frame = frame.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Add face number label
            label = f"Face {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(result_frame, (x, y - 25), (x + label_size[0], y), color, -1)
            cv2.putText(result_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result_frame
    
    def update_parameters(self, scale_factor: Optional[float] = None, 
                         min_neighbors: Optional[int] = None,
                         min_size: Optional[Tuple[int, int]] = None):
        """
        Update detection parameters
        
        Args:
            scale_factor: New scale factor
            min_neighbors: New minimum neighbors
            min_size: New minimum size
        """
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if min_neighbors is not None:
            self.min_neighbors = min_neighbors
        if min_size is not None:
            self.min_size = min_size
        
        self.logger.info(f"Updated parameters: scale_factor={self.scale_factor}, "
                        f"min_neighbors={self.min_neighbors}, min_size={self.min_size}")

# Utility functions
def test_face_detection(camera_index: int = 0) -> bool:
    """
    Test face detection functionality
    
    Args:
        camera_index: Camera index to test
        
    Returns:
        True if test successful
    """
    try:
        detector = FaceDetector()
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return False
        
        print("Press 'q' to quit, 's' to save detection result")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = detector.detect_faces(frame, detect_eyes=True)
            
            # Draw face rectangles
            result_frame = detector.draw_face_rectangles(frame, faces)
            
            # Add info text
            info_text = f"Faces detected: {len(faces)}"
            cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection Test', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                cv2.imwrite('face_detection_test.jpg', result_frame)
                print("Detection result saved")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"Face detection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_face_detection() 