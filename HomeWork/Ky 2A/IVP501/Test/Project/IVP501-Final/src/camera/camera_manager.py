"""
Camera management module for FaceAttend application
Handles webcam initialization, configuration, and frame capture
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable
from src.utils.logger import get_module_logger
from src.utils.exceptions import CameraError

class CameraManager:
    """Manages webcam operations for the FaceAttend application"""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize the camera manager
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_recording = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.logger = get_module_logger("CameraManager")
        
        # Camera properties
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        self.logger.info(f"CameraManager initialized with camera index {camera_index}")
    
    def initialize_camera(self) -> bool:
        """
        Initialize the camera
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.logger.info("Initializing camera...")
            
            # Release any existing camera
            if self.cap is not None:
                self.cap.release()
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise CameraError(f"Cannot open camera with index {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test camera by reading a frame
            ret, frame = self.cap.read()
            if not ret:
                raise CameraError("Cannot read frame from camera")
            
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def is_camera_available(self) -> bool:
        """
        Check if camera is available and working
        
        Returns:
            True if camera is available, False otherwise
        """
        if self.cap is None:
            return False
        
        return self.cap.isOpened()
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera
        
        Returns:
            Captured frame as numpy array, None if failed
        """
        if not self.is_camera_available():
            self.logger.warning("Camera not available for frame capture")
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                return frame
            else:
                self.logger.warning("Failed to capture frame")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {str(e)}")
            return None
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get a frame from the camera (OpenCV-style interface)
        
        Returns:
            Tuple of (success, frame) like cv2.VideoCapture.read()
        """
        frame = self.capture_frame()
        if frame is not None:
            return True, frame
        else:
            return False, None
    
    def start_recording(self, frame_callback: Optional[Callable] = None):
        """
        Start continuous frame capture in a separate thread
        
        Args:
            frame_callback: Optional callback function to process each frame
        """
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return
        
        if not self.is_camera_available():
            self.logger.error("Cannot start recording - camera not available")
            return
        
        self.is_recording = True
        self.logger.info("Starting camera recording")
        
        def recording_loop():
            while self.is_recording:
                frame = self.capture_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    # Call frame callback if provided
                    if frame_callback:
                        try:
                            frame_callback(frame)
                        except Exception as e:
                            self.logger.error(f"Error in frame callback: {str(e)}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(1/self.fps)
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=recording_loop, daemon=True)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop continuous frame capture"""
        if self.is_recording:
            self.logger.info("Stopping camera recording")
            self.is_recording = False
            
            # Wait for recording thread to finish
            if hasattr(self, 'recording_thread'):
                self.recording_thread.join(timeout=2.0)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame
        
        Returns:
            Most recent frame as numpy array, None if not available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def get_camera_properties(self) -> dict:
        """
        Get current camera properties
        
        Returns:
            Dictionary of camera properties
        """
        if not self.is_camera_available():
            return {}
        
        try:
            properties = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            }
            return properties
        except Exception as e:
            self.logger.error(f"Error getting camera properties: {str(e)}")
            return {}
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            True if resolution set successfully, False otherwise
        """
        if not self.is_camera_available():
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify the resolution was set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.frame_width = actual_width
            self.frame_height = actual_height
            
            self.logger.info(f"Resolution set to {actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            return False
    
    def release(self):
        """Release camera resources"""
        self.logger.info("Releasing camera resources")
        
        # Stop recording if active
        self.stop_recording()
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.logger.info("Camera resources released")
    
    def release_camera(self):
        """Alias for release() method for compatibility"""
        self.release()
    
    def __del__(self):
        """Destructor to ensure camera is properly released"""
        self.release()

def test_camera(camera_index: int = 0) -> Tuple[bool, str]:
    """
    Test camera functionality
    
    Args:
        camera_index: Index of camera to test
        
    Returns:
        Tuple of (success, message)
    """
    try:
        camera = CameraManager(camera_index)
        
        if not camera.initialize_camera():
            return False, f"Failed to initialize camera {camera_index}"
        
        frame = camera.capture_frame()
        camera.release()
        
        if frame is not None:
            return True, f"Camera {camera_index} is working correctly"
        else:
            return False, f"Camera {camera_index} failed to capture frame"
            
    except Exception as e:
        return False, f"Camera test failed: {str(e)}" 