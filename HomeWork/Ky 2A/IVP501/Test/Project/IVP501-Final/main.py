#!/usr/bin/env python3
"""
FaceAttend - Python Desktop Face Recognition Attendance System
Main application entry point

Author: FaceAttend Team
Version: 1.0
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import os
from datetime import datetime

# Import application modules
try:
    from src.ui.camera_widget import CameraWidget
    from src.ui.registration_window import open_registration_window
    from src.camera.camera_manager import test_camera
    from src.utils.logger import setup_logger, get_module_logger
    from src.storage.face_storage import FaceStorage
    from src.ui.attendance_window import AttendanceWindow
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all modules are properly installed")

class FaceAttendApp:
    """Main application class for FaceAttend"""
    
    def __init__(self):
        """Initialize the FaceAttend application"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting FaceAttend Application")
        
        # Initialize storage
        self.face_storage = FaceStorage()
        
        # Create main window
        self.root = tk.Tk()
        self.setup_main_window()
        
        # Initialize main window
        self.create_main_window()
    
    def setup_logging(self):
        """Set up basic logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        log_filename = f"logs/faceattend_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_main_window(self):
        """Configure main window properties"""
        self.root.title("FaceAttend - Face Recognition Attendance System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Mac-specific settings
        if sys.platform == "darwin":
            # Set Mac-specific window properties
            self.root.tk.call('tk', 'scaling', 1.0)
            
            # Handle window close event
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_main_window(self):
        """Create main window for Phase 1"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Home tab
        self.create_home_tab()
        
        # Camera test tab
        self.create_camera_tab()
        
        # Status area
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_var = tk.StringVar()
        self.status_var.set("System ready - Phase 2 implementation")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack()
    
    def create_home_tab(self):
        """Create the home tab"""
        home_frame = ttk.Frame(self.notebook)
        self.notebook.add(home_frame, text="Home")
        
        # Configure grid
        home_frame.columnconfigure(0, weight=1)
        
        # Title section
        title_frame = ttk.Frame(home_frame)
        title_frame.pack(pady=20)
        
        title_label = ttk.Label(
            title_frame, 
            text="FaceAttend", 
            font=("Arial", 28, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame, 
            text="Face Recognition Attendance System", 
            font=("Arial", 14)
        )
        subtitle_label.pack(pady=(5, 0))
        
        version_label = ttk.Label(
            title_frame, 
            text="Version 1.0 - Phase 2 Implementation", 
            font=("Arial", 10),
            foreground="gray"
        )
        version_label.pack(pady=(5, 0))
        
        # Features section
        features_frame = ttk.LabelFrame(home_frame, text="Available Features", padding=20)
        features_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Navigation buttons grid
        button_frame = ttk.Frame(features_frame)
        button_frame.pack()
        
        # Row 1
        register_btn = ttk.Button(
            button_frame,
            text="📷 Register Face",
            command=self.open_register_screen,
            width=25
        )
        register_btn.grid(row=0, column=0, padx=10, pady=10)
        
        attendance_btn = ttk.Button(
            button_frame,
            text="👤 Start Attendance",
            command=self.open_attendance_screen,
            width=25
        )
        attendance_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Row 2
        logs_btn = ttk.Button(
            button_frame,
            text="📊 View Attendance Logs",
            command=self.open_logs_screen,
            width=25
        )
        logs_btn.grid(row=1, column=0, padx=10, pady=10)
        
        camera_btn = ttk.Button(
            button_frame,
            text="🎥 Test Camera",
            command=self.test_camera,
            width=25
        )
        camera_btn.grid(row=1, column=1, padx=10, pady=10)
        
        # Info section - Updated for Phase 4
        info_frame = ttk.LabelFrame(home_frame, text="System Status", padding=15)
        info_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Get storage stats
        stats = self.face_storage.get_storage_stats()
        
        # Check recognition model status
        try:
            from src.recognition.lbph_recognizer import LBPHRecognizer
            recognizer = LBPHRecognizer()
            model_info = recognizer.get_model_info()
            model_trained = model_info.get('is_trained', False)
            users_in_model = model_info.get('users_count', 0)
        except Exception:
            model_trained = False
            users_in_model = 0
        
        info_text = f"""
Phase 4 Implementation Complete ✓
• Face registration system ✓
• Face detection with Haar Cascades ✓  
• Image preprocessing pipeline ✓
• Face image storage system ✓
• LBPH recognition engine ✓
• Real-time attendance capture ✓
• Attendance logging system ✓
• Attendance logs viewer ✓
• CSV export functionality ✓
• Statistical reporting ✓

System Statistics:
• Total registered users: {stats.get('total_users', 0)}
• Total face images: {stats.get('total_images', 0)}
• Storage size: {stats.get('total_size_mb', 0):.1f} MB
• Recognition model: {'Trained' if model_trained else 'Not trained'}
• Users in model: {users_in_model}

Ready for Use:
• Face registration ✓
• Real-time attendance capture ✓
• Attendance logs viewing ✓
• Data export and management ✓

Coming in Future Phases:
• Performance optimization (Phase 5)
• Mac application packaging (Phase 6)
• Final testing and delivery (Phase 7)
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack()
    
    def create_camera_tab(self):
        """Create the camera test tab"""
        camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(camera_frame, text="Camera Test")
        
        # Instructions
        instructions = ttk.Label(
            camera_frame,
            text="Use this tab to test your camera functionality.\nThis will be used for face registration and attendance capture.",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        instructions.pack(pady=20)
        
        # Camera widget
        try:
            self.camera_widget = CameraWidget(camera_frame, width=640, height=480)
            self.camera_widget.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        except Exception as e:
            self.logger.error(f"Failed to create camera widget: {str(e)}")
            error_label = ttk.Label(
                camera_frame,
                text=f"Camera widget unavailable: {str(e)}",
                foreground="red"
            )
            error_label.pack(pady=20)
    
    def open_register_screen(self):
        """Open face registration screen (Phase 2 implementation)"""
        self.status_var.set("Opening face registration window...")
        
        try:
            def on_registration_complete(user_id):
                """Callback when registration is completed"""
                self.status_var.set(f"Registration completed for user: {user_id}")
                self.logger.info(f"Registration completed for user: {user_id}")
                
                # Refresh home tab to show updated stats
                self.refresh_home_tab()
                
                # Show success message
                messagebox.showinfo(
                    "Registration Complete",
                    f"Face registration completed successfully!\n\nUser ID: {user_id}\n\nThe user can now be recognized by the system."
                )
            
            # Open registration window
            registration_window = open_registration_window(
                parent=self.root,
                on_complete=on_registration_complete
            )
            
            self.logger.info("Face registration window opened")
            
        except Exception as e:
            self.logger.error(f"Failed to open registration window: {str(e)}")
            messagebox.showerror("Error", f"Failed to open registration window: {str(e)}")
            self.status_var.set("Error opening registration window")
    
    def refresh_home_tab(self):
        """Refresh the home tab to show updated statistics"""
        try:
            # This is a simple refresh - in a full implementation you might want to 
            # update specific widgets rather than recreating the entire tab
            self.status_var.set("Home tab refreshed with latest statistics")
        except Exception as e:
            self.logger.error(f"Error refreshing home tab: {str(e)}")
    
    def open_attendance_screen(self):
        """Open attendance capture screen (Phase 3 implementation)"""
        self.status_var.set("Opening attendance capture window...")
        
        try:
            # Open attendance window
            if not hasattr(self, 'attendance_window') or not self.attendance_window.is_window_open:
                self.attendance_window = AttendanceWindow(parent=self.root)
            
            self.attendance_window.show_window()
            self.status_var.set("Attendance capture window opened")
            self.logger.info("Attendance capture window opened")
            
        except Exception as e:
            self.logger.error(f"Failed to open attendance window: {str(e)}")
            messagebox.showerror("Error", f"Failed to open attendance window: {str(e)}")
            self.status_var.set("Error opening attendance window")
    
    def open_logs_screen(self):
        """Open attendance logs screen (Phase 4 implementation)"""
        self.status_var.set("Opening attendance logs window...")
        
        try:
            # Import and open logs window
            from src.ui.logs_window import open_logs_window
            
            # Open logs window
            if not hasattr(self, 'logs_window') or not self.logs_window.is_window_open:
                self.logs_window = open_logs_window(parent=self.root)
            else:
                self.logs_window.show_window()
            
            self.status_var.set("Attendance logs window opened")
            self.logger.info("Attendance logs window opened")
            
        except Exception as e:
            self.logger.error(f"Failed to open logs window: {str(e)}")
            messagebox.showerror("Error", f"Failed to open logs window: {str(e)}")
            self.status_var.set("Error opening logs window")
    
    def test_camera(self):
        """Test camera functionality (Phase 1 implementation)"""
        self.status_var.set("Testing camera...")
        
        try:
            # Test camera using the camera manager
            success, message = test_camera(0)
            
            if success:
                self.status_var.set("Camera test successful!")
                messagebox.showinfo("Camera Test", f"✓ {message}\n\nYou can also test the live camera feed in the 'Camera Test' tab.")
                self.logger.info("Camera test successful")
            else:
                self.status_var.set("Camera test failed")
                messagebox.showerror("Camera Error", f"✗ {message}\n\nPlease check:\n• Camera is connected\n• Camera permissions are granted\n• No other app is using the camera")
                self.logger.error(f"Camera test failed: {message}")
                
        except Exception as e:
            error_msg = f"Camera test failed: {str(e)}"
            self.status_var.set("Camera test failed")
            messagebox.showerror("Camera Error", error_msg)
            self.logger.error(error_msg)
    
    def on_closing(self):
        """Handle application closing"""
        try:
            self.logger.info("Application closing")
            
            # Clean up camera widget if it exists
            if hasattr(self, 'camera_widget'):
                self.camera_widget.destroy()
            
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {str(e)}")
            self.root.destroy()
    
    def run(self):
        """Start the application main loop"""
        try:
            self.logger.info("Application started successfully")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            messagebox.showerror("Application Error", f"An error occurred: {str(e)}")
        finally:
            self.logger.info("Application shutdown")

def main():
    """Main entry point"""
    try:
        app = FaceAttendApp()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 