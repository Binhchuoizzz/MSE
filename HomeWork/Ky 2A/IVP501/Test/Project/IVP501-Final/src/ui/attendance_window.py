"""
Attendance capture window for FaceAttend application
Provides real-time face recognition and attendance logging interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from datetime import datetime, date
from typing import Dict, Optional
from src.utils.logger import get_module_logger
from src.utils.exceptions import UIError, CameraError
from src.recognition.realtime_recognizer import RealtimeRecognizer
from src.ui.camera_widget import CameraWidget

class AttendanceWindow:
    """Attendance capture window with real-time recognition"""
    
    def __init__(self, parent=None):
        """
        Initialize the attendance window
        
        Args:
            parent: Parent window (optional)
        """
        self.logger = get_module_logger("AttendanceWindow")
        self.parent = parent
        
        # Initialize components
        self.realtime_recognizer = RealtimeRecognizer(confidence_threshold=80.0)
        
        # Window state
        self.window = None
        self.is_window_open = False
        
        # UI components
        self.camera_widget = None
        self.status_label = None
        self.recognition_info_frame = None
        self.start_stop_button = None
        self.threshold_scale = None
        self.attendance_listbox = None
        
        # Recognition state
        self.is_recognition_active = False
        self.last_recognition_result = None
        
        self.logger.info("AttendanceWindow initialized")
    
    def show_window(self):
        """Show the attendance window"""
        try:
            if self.is_window_open:
                if self.window:
                    self.window.lift()
                    self.window.focus_set()
                return
            
            self._create_window()
            self._setup_ui()
            self._setup_recognizer_callbacks()
            
            self.is_window_open = True
            self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
            
            self.logger.info("Attendance window shown")
            
        except Exception as e:
            self.logger.error(f"Failed to show attendance window: {str(e)}")
            raise UIError(f"Attendance window creation failed: {str(e)}")
    
    def _create_window(self):
        """Create the main window"""
        self.window = tk.Toplevel() if self.parent else tk.Tk()
        self.window.title("FaceAttend - Attendance Capture")
        self.window.geometry("1000x700")
        self.window.configure(bg='#f0f0f0')
        
        # Configure window properties
        self.window.resizable(True, True)
        self.window.minsize(800, 600)
        
        # Center window on screen
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.window.winfo_screenheight() // 2) - (700 // 2)
        self.window.geometry(f"1000x700+{x}+{y}")
    
    def _setup_ui(self):
        """Set up the user interface"""
        try:
            # Create main frame
            main_frame = ttk.Frame(self.window, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            self.window.columnconfigure(0, weight=1)
            self.window.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(1, weight=1)
            
            # Title
            title_label = ttk.Label(main_frame, text="Real-Time Attendance Capture", 
                                  font=("Arial", 16, "bold"))
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
            
            # Left panel - Camera and controls
            left_frame = ttk.Frame(main_frame)
            left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
            left_frame.columnconfigure(0, weight=1)
            left_frame.rowconfigure(1, weight=1)
            
            # Camera controls
            controls_frame = ttk.LabelFrame(left_frame, text="Camera Controls", padding="10")
            controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            controls_frame.columnconfigure(1, weight=1)
            
            # Start/Stop button
            self.start_stop_button = ttk.Button(controls_frame, text="Start Recognition", 
                                              command=self._toggle_recognition)
            self.start_stop_button.grid(row=0, column=0, padx=(0, 10))
            
            # Confidence threshold
            ttk.Label(controls_frame, text="Confidence Threshold:").grid(row=0, column=1, sticky=tk.W)
            self.threshold_scale = ttk.Scale(controls_frame, from_=50, to=150, 
                                           orient=tk.HORIZONTAL, length=200,
                                           command=self._on_threshold_change)
            self.threshold_scale.set(80)
            self.threshold_scale.grid(row=0, column=2, padx=(5, 0))
            
            # Threshold value label
            self.threshold_value_label = ttk.Label(controls_frame, text="80.0")
            self.threshold_value_label.grid(row=0, column=3, padx=(5, 0))
            
            # Camera display
            camera_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="5")
            camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            camera_frame.columnconfigure(0, weight=1)
            camera_frame.rowconfigure(0, weight=1)
            
            # Camera widget
            self.camera_widget = CameraWidget(camera_frame, width=640, height=480)
            self.camera_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Set up camera widget callback for frame updates
            self.camera_widget.set_frame_callback(self._camera_frame_callback)
            
            # Status display
            status_frame = ttk.LabelFrame(left_frame, text="Status", padding="10")
            status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
            status_frame.columnconfigure(0, weight=1)
            
            self.status_label = ttk.Label(status_frame, text="Ready to start recognition", 
                                        font=("Arial", 10), foreground="blue")
            self.status_label.grid(row=0, column=0, sticky=tk.W)
            
            # Right panel - Recognition info and attendance log
            right_frame = ttk.Frame(main_frame)
            right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
            right_frame.columnconfigure(0, weight=1)
            right_frame.rowconfigure(1, weight=1)
            
            # Recognition info
            self._setup_recognition_info_panel(right_frame)
            
            # Attendance log
            self._setup_attendance_log_panel(right_frame)
            
            self.logger.debug("UI setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up UI: {str(e)}")
            raise UIError(f"UI setup failed: {str(e)}")
    
    def _setup_recognition_info_panel(self, parent):
        """Set up the recognition information panel"""
        self.recognition_info_frame = ttk.LabelFrame(parent, text="Recognition Information", padding="10")
        self.recognition_info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.recognition_info_frame.columnconfigure(1, weight=1)
        
        # User info
        ttk.Label(self.recognition_info_frame, text="Recognized User:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.user_name_label = ttk.Label(self.recognition_info_frame, text="None", 
                                       font=("Arial", 12), foreground="gray")
        self.user_name_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Confidence
        ttk.Label(self.recognition_info_frame, text="Confidence:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky=tk.W)
        self.confidence_label = ttk.Label(self.recognition_info_frame, text="0.0", 
                                        font=("Arial", 11), foreground="gray")
        self.confidence_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Faces detected
        ttk.Label(self.recognition_info_frame, text="Faces Detected:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W)
        self.faces_count_label = ttk.Label(self.recognition_info_frame, text="0", 
                                         font=("Arial", 11), foreground="gray")
        self.faces_count_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Last recognition time
        ttk.Label(self.recognition_info_frame, text="Last Recognition:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky=tk.W)
        self.last_recognition_label = ttk.Label(self.recognition_info_frame, text="Never", 
                                              font=("Arial", 9), foreground="gray")
        self.last_recognition_label.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
        # Retrain model button
        retrain_button = ttk.Button(self.recognition_info_frame, text="Retrain Model", 
                                  command=self._retrain_model)
        retrain_button.grid(row=4, column=0, columnspan=2, pady=(10, 0))
    
    def _setup_attendance_log_panel(self, parent):
        """Set up the attendance log panel"""
        log_frame = ttk.LabelFrame(parent, text="Today's Attendance", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Attendance listbox with scrollbar
        list_frame = ttk.Frame(log_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.attendance_listbox = tk.Listbox(list_frame, font=("Courier", 9))
        self.attendance_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.attendance_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.attendance_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Refresh button
        refresh_button = ttk.Button(log_frame, text="Refresh Log", command=self._refresh_attendance_log)
        refresh_button.grid(row=1, column=0, pady=(10, 0))
        
        # Load initial attendance log
        self._refresh_attendance_log()
    
    def _setup_recognizer_callbacks(self):
        """Set up callbacks for the real-time recognizer"""
        self.realtime_recognizer.set_callbacks(
            status_callback=self._on_status_update,
            recognition_callback=self._on_recognition_result,
            frame_callback=self._on_frame_update
        )
    
    def _toggle_recognition(self):
        """Toggle recognition on/off"""
        try:
            if not self.is_recognition_active:
                # Start recognition
                if self.realtime_recognizer.start_recognition():
                    self.is_recognition_active = True
                    self.start_stop_button.configure(text="Stop Recognition")
                    self.camera_widget.start_stream()  # Start camera widget stream
                    self.logger.info("Recognition started")
                else:
                    messagebox.showerror("Error", "Failed to start recognition. Please check camera connection.")
            else:
                # Stop recognition
                self.realtime_recognizer.stop_recognition()
                self.camera_widget.stop_stream()  # Stop camera widget stream
                self.is_recognition_active = False
                self.start_stop_button.configure(text="Start Recognition")
                self._update_status("Recognition stopped")
                self.logger.info("Recognition stopped")
                
        except Exception as e:
            self.logger.error(f"Error toggling recognition: {str(e)}")
            messagebox.showerror("Error", f"Recognition error: {str(e)}")
    
    def _on_threshold_change(self, value):
        """Handle confidence threshold change"""
        threshold = float(value)
        if hasattr(self, 'threshold_value_label') and self.threshold_value_label:
            self.threshold_value_label.configure(text=f"{threshold:.1f}")
        if self.realtime_recognizer:
            self.realtime_recognizer.update_confidence_threshold(threshold)
    
    def _on_status_update(self, status: str):
        """Handle status updates from recognizer"""
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.configure(text=status)
    
    def _on_recognition_result(self, result: Dict):
        """Handle recognition results from recognizer"""
        try:
            self.last_recognition_result = result
            
            # Update recognition info display
            if result.get('user_id'):
                self.user_name_label.configure(text=result['name'], foreground="green")
                self.confidence_label.configure(text=f"{result['confidence']:.1f}", foreground="green")
            else:
                self.user_name_label.configure(text="Unknown", foreground="red")
                self.confidence_label.configure(text=f"{result.get('confidence', 0):.1f}", foreground="red")
            
            self.faces_count_label.configure(text=str(result.get('faces_detected', 0)))
            
            # Update last recognition time
            if result.get('timestamp'):
                try:
                    timestamp = datetime.fromisoformat(result['timestamp'])
                    time_str = timestamp.strftime("%H:%M:%S")
                    self.last_recognition_label.configure(text=time_str)
                except:
                    self.last_recognition_label.configure(text="Now")
            
            # Refresh attendance log if new attendance was logged
            if result.get('user_id'):
                self._refresh_attendance_log()
                
        except Exception as e:
            self.logger.error(f"Error handling recognition result: {str(e)}")
    
    def _on_frame_update(self, frame: np.ndarray):
        """Handle frame updates from recognizer"""
        if self.camera_widget and frame is not None:
            self.camera_widget.update_frame(frame)
    
    def _retrain_model(self):
        """Retrain the recognition model"""
        try:
            if messagebox.askyesno("Retrain Model", 
                                 "Retrain the recognition model with current registered users?"):
                self._update_status("Retraining model...")
                
                # Retrain in a separate thread to avoid UI freezing
                threading.Thread(target=self._retrain_model_thread, daemon=True).start()
                
        except Exception as e:
            self.logger.error(f"Error initiating model retraining: {str(e)}")
            messagebox.showerror("Error", f"Retraining error: {str(e)}")
    
    def _retrain_model_thread(self):
        """Retrain model in separate thread"""
        try:
            success = self.realtime_recognizer.retrain_model()
            
            if success:
                self.window.after(0, lambda: messagebox.showinfo("Success", "Model retrained successfully!"))
            else:
                self.window.after(0, lambda: messagebox.showerror("Error", "Model retraining failed."))
                
        except Exception as e:
            self.logger.error(f"Error in model retraining thread: {str(e)}")
            self.window.after(0, lambda: messagebox.showerror("Error", f"Retraining error: {str(e)}"))
    
    def _refresh_attendance_log(self):
        """Refresh the attendance log display"""
        try:
            if not self.attendance_listbox:
                return
            
            # Clear current list
            self.attendance_listbox.delete(0, tk.END)
            
            # Get today's attendance
            attendance_records = self.realtime_recognizer.attendance_logger.get_daily_attendance()
            
            if not attendance_records:
                self.attendance_listbox.insert(0, "No attendance records for today")
                return
            
            # Add header
            self.attendance_listbox.insert(0, "Time     | Name            | Confidence")
            self.attendance_listbox.insert(1, "-" * 45)
            
            # Add attendance records
            for record in reversed(attendance_records[-20:]):  # Show last 20 records
                time_str = record.get('time', 'Unknown')
                name = record.get('name', 'Unknown')[:15]  # Truncate long names
                confidence = record.get('confidence', '0.0')
                
                line = f"{time_str} | {name:<15} | {confidence}"
                self.attendance_listbox.insert(tk.END, line)
            
            # Scroll to bottom
            self.attendance_listbox.see(tk.END)
            
        except Exception as e:
            self.logger.error(f"Error refreshing attendance log: {str(e)}")
    
    def _update_status(self, status: str):
        """Update status label"""
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.configure(text=status)
    
    def _on_window_close(self):
        """Handle window close event"""
        try:
            # Stop recognition if running
            if self.is_recognition_active:
                self.realtime_recognizer.stop_recognition()
                self.camera_widget.stop_stream()
            
            # Clean up camera widget
            if self.camera_widget:
                self.camera_widget.destroy()
            
            self.is_window_open = False
            
            if self.window:
                self.window.destroy()
                self.window = None
            
            self.logger.info("Attendance window closed")
            
        except Exception as e:
            self.logger.error(f"Error closing attendance window: {str(e)}")
    
    def _camera_frame_callback(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Callback for camera widget frame updates"""
        # Get processed frame from recognizer if available
        processed_frame = self.realtime_recognizer.get_current_frame()
        if processed_frame is not None:
            return processed_frame
        return frame


def test_attendance_window():
    """Test function for attendance window"""
    try:
        # Create and show attendance window
        app = AttendanceWindow()
        app.show_window()
        
        # Start the main loop
        if app.window:
            app.window.mainloop()
        
        return True
        
    except Exception as e:
        print(f"Attendance Window test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_attendance_window() 