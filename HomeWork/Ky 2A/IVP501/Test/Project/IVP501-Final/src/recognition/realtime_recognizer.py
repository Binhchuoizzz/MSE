"""
Real-time face recognition and attendance system for FaceAttend application
Handles continuous recognition, attendance logging, and real-time feedback
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Callable, List, Tuple
from src.utils.logger import get_module_logger
from src.utils.exceptions import FaceRecognitionError, CameraError
from src.recognition.face_detector import FaceDetector
from src.recognition.lbph_recognizer import LBPHRecognizer
from src.storage.attendance_logger import AttendanceLogger
from src.storage.face_storage import FaceStorage
from src.camera.camera_manager import CameraManager

class RealtimeRecognizer:
    """Real-time face recognition and attendance system"""
    
    def __init__(self, 
                 camera_manager: CameraManager = None,
                 confidence_threshold: float = 100.0,
                 recognition_interval: float = 1.0):
        """
        Initialize the real-time recognizer
        
        Args:
            camera_manager: Camera manager instance
            confidence_threshold: Recognition confidence threshold
            recognition_interval: Minimum time between recognition attempts (seconds)
        """
        self.logger = get_module_logger("RealtimeRecognizer")
        
        # Core components
        self.camera_manager = camera_manager or CameraManager()
        self.face_detector = FaceDetector()
        self.lbph_recognizer = LBPHRecognizer(confidence_threshold=confidence_threshold)
        self.attendance_logger = AttendanceLogger()
        self.face_storage = FaceStorage()
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.recognition_interval = recognition_interval
        
        # State management
        self.is_running = False
        self.recognition_thread = None
        self.last_recognition_time = 0
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Recognition results
        self.last_recognition_result = {
            'user_id': None,
            'name': None,
            'confidence': 0.0,
            'timestamp': None,
            'faces_detected': 0
        }
        
        # Callbacks for UI updates
        self.status_callback = None
        self.recognition_callback = None
        self.frame_callback = None
        
        self.logger.info(f"RealtimeRecognizer initialized with confidence threshold: {confidence_threshold}")
    
    def set_callbacks(self, 
                     status_callback: Callable[[str], None] = None,
                     recognition_callback: Callable[[Dict], None] = None,
                     frame_callback: Callable[[np.ndarray], None] = None):
        """
        Set callback functions for UI updates
        
        Args:
            status_callback: Called when status changes
            recognition_callback: Called when recognition occurs
            frame_callback: Called with new frames
        """
        self.status_callback = status_callback
        self.recognition_callback = recognition_callback
        self.frame_callback = frame_callback
        
        self.logger.debug("Callbacks configured for real-time recognizer")
    
    def _update_status(self, status: str):
        """Update status and call callback if available"""
        self.logger.debug(f"Status update: {status}")
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {str(e)}")
    
    def _notify_recognition(self, result: Dict):
        """Notify recognition result and call callback if available"""
        self.last_recognition_result = result.copy()
        
        if self.recognition_callback:
            try:
                self.recognition_callback(result)
            except Exception as e:
                self.logger.error(f"Error in recognition callback: {str(e)}")
    
    def _update_frame(self, frame: np.ndarray):
        """Update current frame and call callback if available"""
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None
        
        if self.frame_callback:
            try:
                self.frame_callback(frame)
            except Exception as e:
                self.logger.error(f"Error in frame callback: {str(e)}")
    
    def start_recognition(self) -> bool:
        """
        Start real-time recognition and attendance capture
        
        Returns:
            True if started successfully
        """
        try:
            if self.is_running:
                self.logger.warning("Recognition already running")
                return True
            
            # Initialize camera
            if not self.camera_manager.initialize_camera():
                raise CameraError("Failed to initialize camera")
            
            # Check if model is trained
            if not self.lbph_recognizer.is_model_trained():
                self._update_status("Training model...")
                self._train_model_if_needed()
            
            # Start recognition thread
            self.is_running = True
            self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
            self.recognition_thread.start()
            
            self._update_status("Recognition started")
            self.logger.info("Real-time recognition started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recognition: {str(e)}")
            self._update_status(f"Failed to start: {str(e)}")
            return False
    
    def stop_recognition(self):
        """Stop real-time recognition"""
        try:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Wait for recognition thread to finish
            if self.recognition_thread and self.recognition_thread.is_alive():
                self.recognition_thread.join(timeout=2.0)
            
            # Release camera
            self.camera_manager.release_camera()
            
            self._update_status("Recognition stopped")
            self.logger.info("Real-time recognition stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping recognition: {str(e)}")
    
    def _train_model_if_needed(self) -> bool:
        """Train the model if it hasn't been trained yet"""
        try:
            users = self.face_storage.list_users()
            if len(users) == 0:
                self._update_status("No registered users found")
                return False
            
            # Train the model
            training_result = self.lbph_recognizer.train_model()
            
            if training_result['success']:
                self._update_status(f"Model trained: {training_result['users_count']} users")
                return True
            else:
                self._update_status("Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            self._update_status(f"Training error: {str(e)}")
            return False
    
    def _recognition_loop(self):
        """Main recognition loop running in separate thread"""
        self.logger.info("Recognition loop started")
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera_manager.capture_frame()
                
                if frame is None:
                    self._update_status("No camera frame")
                    time.sleep(0.1)
                    continue
                
                # Update frame for UI
                self._update_frame(frame)
                
                # Check if enough time has passed since last recognition
                current_time = time.time()
                if current_time - self.last_recognition_time < self.recognition_interval:
                    time.sleep(0.05)
                    continue
                
                # Perform face detection and recognition
                self._process_frame(frame)
                self.last_recognition_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in recognition loop: {str(e)}")
                self._update_status(f"Recognition error: {str(e)}")
                time.sleep(1.0)
        
        self.logger.info("Recognition loop ended")
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame for face detection and recognition"""
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame, detect_eyes=True)
            
            recognition_result = {
                'user_id': None,
                'name': None,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'faces_detected': len(faces)
            }
            
            if len(faces) == 0:
                self._update_status("No faces detected")
                self._notify_recognition(recognition_result)
                return
            
            if len(faces) > 1:
                self._update_status("Multiple faces detected")
                self._notify_recognition(recognition_result)
                return
            
            # Single face detected - proceed with recognition
            face_rect = faces[0]
            x, y, w, h = face_rect
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Perform recognition
            user_id, confidence = self.lbph_recognizer.recognize_face(face_region)
            
            if user_id:
                # Get user info
                user_info = self.face_storage.get_user_info(user_id)
                user_name = user_info.get('name', 'Unknown') if user_info else 'Unknown'
                
                # Log attendance
                attendance_logged = self.attendance_logger.log_attendance(
                    user_id, user_name, confidence
                )
                
                recognition_result.update({
                    'user_id': user_id,
                    'name': user_name,
                    'confidence': confidence
                })
                
                if attendance_logged:
                    self._update_status(f"Attendance logged: {user_name} ({confidence:.1f})")
                else:
                    self._update_status(f"Recognized: {user_name} (duplicate entry)")
                
                self.logger.info(f"Recognition successful: {user_name} (confidence: {confidence:.2f})")
            else:
                self._update_status(f"Unknown person (confidence: {confidence:.1f})")
                recognition_result['confidence'] = confidence
            
            self._notify_recognition(recognition_result)
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            self._update_status(f"Processing error: {str(e)}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame with face detection overlay"""
        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return None
                
                frame = self.current_frame.copy()
            
            # Add face detection overlay
            faces = self.face_detector.detect_faces(frame)
            if faces:
                frame_with_overlay = self.face_detector.draw_face_rectangles(frame, faces)
                
                # Add recognition info overlay
                if self.last_recognition_result['user_id']:
                    result = self.last_recognition_result
                    text = f"{result['name']} ({result['confidence']:.1f})"
                    cv2.putText(frame_with_overlay, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame_with_overlay
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error getting current frame: {str(e)}")
            return None
    
    def retrain_model(self) -> bool:
        """Retrain the recognition model with current registered users"""
        try:
            self._update_status("Retraining model...")
            
            training_result = self.lbph_recognizer.train_model()
            
            if training_result['success']:
                self._update_status(f"Model retrained: {training_result['users_count']} users")
                self.logger.info("Model retrained successfully")
                return True
            else:
                self._update_status("Model retraining failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Model retraining failed: {str(e)}")
            self._update_status(f"Retraining error: {str(e)}")
            return False
    
    def update_confidence_threshold(self, threshold: float):
        """Update the recognition confidence threshold"""
        self.confidence_threshold = threshold
        self.lbph_recognizer.set_confidence_threshold(threshold)
        self.logger.info(f"Confidence threshold updated to {threshold}")
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition statistics"""
        try:
            model_info = self.lbph_recognizer.get_model_info()
            attendance_stats = self.attendance_logger.get_attendance_statistics()
            
            return {
                'model_info': model_info,
                'attendance_stats': attendance_stats,
                'last_recognition': self.last_recognition_result,
                'is_running': self.is_running
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recognition stats: {str(e)}")
            return {}
    
    def is_recognition_running(self) -> bool:
        """Check if recognition is currently running"""
        return self.is_running


def test_realtime_recognizer():
    """Test function for real-time recognizer"""
    try:
        # Initialize recognizer
        recognizer = RealtimeRecognizer()
        
        # Test basic functionality
        print("Testing real-time recognizer...")
        
        # Check model status
        stats = recognizer.get_recognition_stats()
        print(f"Model trained: {stats.get('model_info', {}).get('is_trained', False)}")
        print(f"Registered users: {stats.get('model_info', {}).get('users_count', 0)}")
        
        # Note: Actual recognition testing would require camera and registered users
        print("Real-time recognizer test completed (basic functionality)")
        
        return True
        
    except Exception as e:
        print(f"Realtime Recognizer test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_realtime_recognizer() 