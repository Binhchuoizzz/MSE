"""
Camera display widget for FaceAttend application
Provides video stream display functionality in Tkinter
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from typing import Optional, Callable
from src.camera.camera_manager import CameraManager
from src.utils.logger import get_module_logger
from src.utils.exceptions import UIError, CameraError

class CameraWidget(ttk.Frame):
    """Widget for displaying camera feed in Tkinter"""
    
    def __init__(self, parent, width: int = 640, height: int = 480, camera_index: int = 0):
        """
        Initialize the camera widget
        
        Args:
            parent: Parent widget
            width: Display width
            height: Display height
            camera_index: Camera index to use
        """
        super().__init__(parent)
        
        self.logger = get_module_logger("CameraWidget")
        self.width = width
        self.height = height
        self.camera_index = camera_index
        
        # Camera manager
        self.camera_manager = CameraManager(camera_index)
        self.is_streaming = False
        self.stream_thread = None
        
        # Frame callback
        self.frame_callback = None
        
        # UI components
        self.create_widgets()
        
        self.logger.info(f"CameraWidget initialized with size {width}x{height}")
    
    def create_widgets(self):
        """Create the widget components"""
        # Configure grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Video display label
        self.video_label = ttk.Label(self, text="Camera Feed", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(self)
        control_frame.grid(row=1, column=0, pady=5)
        
        # Control buttons
        self.start_btn = ttk.Button(
            control_frame, 
            text="Start Camera", 
            command=self.start_stream
        )
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(
            control_frame, 
            text="Stop Camera", 
            command=self.stop_stream,
            state=tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Camera not started")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Set initial placeholder
        self.set_placeholder_image()
    
    def set_placeholder_image(self):
        """Set a placeholder image when camera is not active"""
        # Create a placeholder image
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        placeholder.fill(50)  # Dark gray background
        
        # Add text
        cv2.putText(
            placeholder, 
            "Camera Feed", 
            (self.width//2 - 80, self.height//2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        self.display_frame(placeholder)
    
    def display_frame(self, frame: np.ndarray):
        """
        Display a frame in the widget
        
        Args:
            frame: Frame to display as numpy array
        """
        try:
            # Resize frame to fit widget
            frame_resized = cv2.resize(frame, (self.width, self.height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.logger.error(f"Error displaying frame: {str(e)}")
    
    def start_stream(self):
        """Start the camera stream"""
        try:
            self.logger.info("Starting camera stream")
            
            # Initialize camera
            if not self.camera_manager.initialize_camera():
                raise CameraError("Failed to initialize camera")
            
            # Start streaming
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            # Update UI
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.status_var.set("Camera streaming")
            
            self.logger.info("Camera stream started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start camera stream: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.set_placeholder_image()
    
    def stop_stream(self):
        """Stop the camera stream"""
        try:
            self.logger.info("Stopping camera stream")
            
            # Stop streaming
            self.is_streaming = False
            
            # Wait for stream thread to finish
            if self.stream_thread:
                self.stream_thread.join(timeout=2.0)
            
            # Release camera
            self.camera_manager.release()
            
            # Update UI
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.status_var.set("Camera stopped")
            
            # Set placeholder
            self.set_placeholder_image()
            
            self.logger.info("Camera stream stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping camera stream: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def _stream_loop(self):
        """Main streaming loop (runs in separate thread)"""
        while self.is_streaming:
            try:
                # Capture frame
                frame = self.camera_manager.capture_frame()
                
                if frame is not None:
                    # Call frame callback if set
                    if self.frame_callback:
                        try:
                            processed_frame = self.frame_callback(frame)
                            if processed_frame is not None:
                                frame = processed_frame
                        except Exception as e:
                            self.logger.error(f"Error in frame callback: {str(e)}")
                    
                    # Display frame (must be done in main thread)
                    self.after_idle(lambda f=frame: self.display_frame(f))
                
                # Control frame rate
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                self.logger.error(f"Error in stream loop: {str(e)}")
                self.after_idle(lambda: self.status_var.set(f"Stream error: {str(e)}"))
                break
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], Optional[np.ndarray]]):
        """
        Set a frame callback function
        
        Args:
            callback: Function that takes a frame and optionally returns a processed frame
        """
        self.frame_callback = callback
        self.logger.debug("Frame callback set")
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the display with an external frame
        
        Args:
            frame: Frame to display
        """
        if frame is not None:
            self.display_frame(frame)
    
    def capture_image(self) -> Optional[np.ndarray]:
        """
        Capture a single image from the camera
        
        Returns:
            Captured frame as numpy array, None if failed
        """
        if not self.is_streaming:
            self.logger.warning("Cannot capture image - camera not streaming")
            return None
        
        return self.camera_manager.capture_frame()
    
    def get_camera_properties(self) -> dict:
        """Get current camera properties"""
        return self.camera_manager.get_camera_properties()
    
    def is_camera_active(self) -> bool:
        """Check if camera is currently active"""
        return self.is_streaming and self.camera_manager.is_camera_available()
    
    def destroy(self):
        """Clean up resources when widget is destroyed"""
        self.logger.info("Cleaning up CameraWidget")
        self.stop_stream()
        super().destroy()

# Test function for the widget
def test_camera_widget():
    """Test the camera widget functionality"""
    root = tk.Tk()
    root.title("Camera Widget Test")
    root.geometry("700x600")
    
    # Create camera widget
    camera_widget = CameraWidget(root, width=640, height=480)
    camera_widget.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    # Add close button
    close_btn = ttk.Button(root, text="Close", command=root.quit)
    close_btn.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    test_camera_widget() 