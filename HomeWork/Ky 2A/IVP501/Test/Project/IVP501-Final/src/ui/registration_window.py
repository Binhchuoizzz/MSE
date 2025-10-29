"""
Face registration window for FaceAttend application
Provides UI for capturing and storing face images for new users
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
from threading import Thread, Event
import time
from typing import List, Optional, Callable
from src.ui.camera_widget import CameraWidget
from src.recognition.face_detector import FaceDetector
from src.recognition.image_processor import ImageProcessor
from src.storage.face_storage import FaceStorage
from src.utils.logger import get_module_logger
from src.utils.exceptions import UIError, FaceDetectionError, StorageError

class RegistrationWindow:
    """Face registration window with live camera feed and image capture"""
    
    def __init__(self, parent=None, on_complete_callback: Callable = None):
        """
        Initialize the registration window
        
        Args:
            parent: Parent window
            on_complete_callback: Callback function called when registration is complete
        """
        self.logger = get_module_logger("RegistrationWindow")
        self.parent = parent
        self.on_complete_callback = on_complete_callback
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()
        self.face_storage = FaceStorage()
        
        # Registration state
        self.current_user_id = None
        self.captured_images = []
        self.target_image_count = 8
        self.registration_active = False
        self.auto_capture_enabled = False
        self.auto_complete_enabled = False
        self.capture_event = Event()
        
        # Create window
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.setup_window()
        self.create_widgets()
        
        self.logger.info("Registration window initialized")
    
    def setup_window(self):
        """Configure window properties"""
        self.window.title("Face Registration - FaceAttend")
        self.window.geometry("800x700")
        self.window.resizable(True, True)
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.window.winfo_screenheight() // 2) - (700 // 2)
        self.window.geometry(f"800x700+{x}+{y}")
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        """Create and layout all widgets"""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Face Registration", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # User setup tab
        self.create_user_setup_tab()
        
        # Capture tab
        self.create_capture_tab()
        
        # Review tab
        self.create_review_tab()
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Initially disable tabs
        self.notebook.tab(1, state='disabled')  # Capture tab
        self.notebook.tab(2, state='disabled')  # Review tab
    
    def create_user_setup_tab(self):
        """Create user setup tab"""
        setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(setup_frame, text="User Setup")
        
        # Instructions
        instructions = ttk.Label(
            setup_frame,
            text="Enter user information to begin face registration",
            font=("Arial", 12)
        )
        instructions.pack(pady=20)
        
        # User info frame
        info_frame = ttk.LabelFrame(setup_frame, text="User Information", padding=20)
        info_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Name entry
        ttk.Label(info_frame, text="Full Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(info_frame, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Add validation trace to help debug
        def name_changed(*args):
            current_value = self.name_var.get()
            self.logger.debug(f"Name field changed: '{current_value}'")
        
        self.name_var.trace('w', name_changed)
        
        # Set initial focus to name entry
        self.name_entry.focus_set()
        
        # User ID entry (optional)
        ttk.Label(info_frame, text="User ID (optional):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.user_id_var = tk.StringVar()
        self.user_id_entry = ttk.Entry(info_frame, textvariable=self.user_id_var, width=30)
        self.user_id_entry.grid(row=1, column=1, padx=(10, 0), pady=5)
        
        # Help text
        help_text = ttk.Label(
            info_frame,
            text="Leave User ID blank to auto-generate from name and timestamp",
            font=("Arial", 9),
            foreground="gray"
        )
        help_text.grid(row=2, column=0, columnspan=2, pady=(5, 0))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(setup_frame, text="Capture Settings", padding=20)
        settings_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Number of images
        ttk.Label(settings_frame, text="Number of images to capture:").grid(row=0, column=0, sticky=tk.W)
        self.image_count_var = tk.IntVar(value=self.target_image_count)
        image_count_spin = ttk.Spinbox(
            settings_frame, 
            from_=5, 
            to=15, 
            textvariable=self.image_count_var,
            width=10
        )
        image_count_spin.grid(row=0, column=1, padx=(10, 0))
        
        # Auto capture checkbox
        self.auto_capture_var = tk.BooleanVar(value=True)
        auto_capture_check = ttk.Checkbutton(
            settings_frame,
            text="Enable auto-capture (recommended)",
            variable=self.auto_capture_var
        )
        auto_capture_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Auto completion checkbox
        self.auto_complete_var = tk.BooleanVar(value=True)
        auto_complete_check = ttk.Checkbutton(
            settings_frame,
            text="Auto-advance to review when all images captured",
            variable=self.auto_complete_var
        )
        auto_complete_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Start button
        self.start_button = ttk.Button(
            setup_frame,
            text="Start Registration",
            command=self.start_registration,
            style="Accent.TButton"
        )
        self.start_button.pack(pady=30)
        
        # Add debug/test button for troubleshooting
        debug_button = ttk.Button(
            setup_frame,
            text="🔍 Test Name Entry",
            command=self.test_name_entry
        )
        debug_button.pack(pady=(0, 10))
        
        # Bind Enter key to start registration
        self.name_entry.bind('<Return>', lambda e: self.start_registration())
        self.user_id_entry.bind('<Return>', lambda e: self.start_registration())
    
    def create_capture_tab(self):
        """Create image capture tab"""
        capture_frame = ttk.Frame(self.notebook)
        self.notebook.add(capture_frame, text="Capture Images")
        
        # Instructions
        self.capture_instructions = ttk.Label(
            capture_frame,
            text="Position your face in the camera view. Images will be captured automatically.",
            font=("Arial", 12)
        )
        self.capture_instructions.pack(pady=10)
        
        # Camera frame
        camera_frame = ttk.Frame(capture_frame)
        camera_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Camera widget
        self.camera_widget = CameraWidget(camera_frame, width=640, height=480)
        self.camera_widget.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        # Set frame callback for face detection
        self.camera_widget.set_frame_callback(self.process_camera_frame)
        
        # Control panel
        control_panel = ttk.Frame(camera_frame)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Progress info
        progress_frame = ttk.LabelFrame(control_panel, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="0 / 0")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 14, "bold"))
        progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Capture controls
        controls_frame = ttk.LabelFrame(control_panel, text="Capture Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.manual_capture_button = ttk.Button(
            controls_frame,
            text="📷 Capture Now",
            command=self.manual_capture,
            state=tk.DISABLED
        )
        self.manual_capture_button.pack(fill=tk.X, pady=2)
        
        # Add force capture button for debugging
        self.force_capture_button = ttk.Button(
            controls_frame,
            text="⚡ Force Capture",
            command=self.force_capture,
            state=tk.DISABLED
        )
        self.force_capture_button.pack(fill=tk.X, pady=2)
        
        self.skip_button = ttk.Button(
            controls_frame,
            text="Skip This Image",
            command=self.skip_image,
            state=tk.DISABLED
        )
        self.skip_button.pack(fill=tk.X, pady=2)
        
        self.restart_button = ttk.Button(
            controls_frame,
            text="Restart Capture",
            command=self.restart_capture
        )
        self.restart_button.pack(fill=tk.X, pady=2)
        
        # Face detection info
        detection_frame = ttk.LabelFrame(control_panel, text="Detection Status", padding=10)
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.face_status_var = tk.StringVar(value="No face detected")
        face_status_label = ttk.Label(detection_frame, textvariable=self.face_status_var)
        face_status_label.pack()
        
        self.quality_status_var = tk.StringVar(value="")
        quality_status_label = ttk.Label(detection_frame, textvariable=self.quality_status_var)
        quality_status_label.pack()
        
        # Debug info panel
        debug_frame = ttk.LabelFrame(control_panel, text="Debug Info", padding=5)
        debug_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.debug_info_var = tk.StringVar(value="Starting...")
        debug_info_label = ttk.Label(debug_frame, textvariable=self.debug_info_var, font=("Courier", 8))
        debug_info_label.pack()
        
        # Add checkbox to enable/disable debug info
        self.show_debug_var = tk.BooleanVar(value=False)
        debug_check = ttk.Checkbutton(
            debug_frame,
            text="Show detailed debug info",
            variable=self.show_debug_var
        )
        debug_check.pack()
        
        # Captured images preview
        preview_frame = ttk.LabelFrame(control_panel, text="Captured Images", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for thumbnails
        canvas = tk.Canvas(preview_frame, width=150, height=200)
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=canvas.yview)
        self.thumbnail_frame = ttk.Frame(canvas)
        
        self.thumbnail_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.thumbnail_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Complete button
        self.complete_capture_button = ttk.Button(
            capture_frame,
            text="Complete Registration",
            command=self.complete_registration,
            state=tk.DISABLED,
            style="Accent.TButton"
        )
        self.complete_capture_button.pack(pady=10)
    
    def create_review_tab(self):
        """Create review tab"""
        review_frame = ttk.Frame(self.notebook)
        self.notebook.add(review_frame, text="Review & Save")
        
        # Instructions
        instructions = ttk.Label(
            review_frame,
            text="Review captured images and save the registration",
            font=("Arial", 12)
        )
        instructions.pack(pady=10)
        
        # Review content will be populated dynamically
        self.review_content_frame = ttk.Frame(review_frame)
        self.review_content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready to start registration")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Close button
        close_button = ttk.Button(
            status_frame,
            text="Cancel",
            command=self.on_close
        )
        close_button.pack(side=tk.RIGHT)
    
    def start_registration(self):
        """Start the face registration process"""
        try:
            # Force focus update to ensure Entry widgets have their values set
            self.window.update_idletasks()
            
            # Validate input with better debugging
            name = self.name_var.get().strip()
            
            # Debug logging to help identify the issue
            self.logger.info(f"Name validation: Raw value='{self.name_var.get()}', Stripped='{name}', Length={len(name)}")
            
            # Also try getting the value directly from the Entry widget as fallback
            if not name:
                try:
                    name_from_entry = self.name_entry.get().strip()
                    self.logger.info(f"Fallback name from entry: '{name_from_entry}'")
                    if name_from_entry:
                        name = name_from_entry
                        self.name_var.set(name)  # Update the StringVar
                except Exception as entry_error:
                    self.logger.error(f"Error getting name from entry widget: {entry_error}")
            
            if not name:
                self.logger.warning("Name validation failed - no name provided")
                messagebox.showerror("Error", f"Please enter a name.\n\nDebug info:\n- StringVar value: '{self.name_var.get()}'\n- Entry widget value: '{self.name_entry.get() if hasattr(self, 'name_entry') else 'N/A'}'")
                # Set focus to the name entry to help user
                self.name_entry.focus_set()
                return
            
            user_id = self.user_id_var.get().strip() or None
            
            # Update settings
            self.target_image_count = self.image_count_var.get()
            self.auto_capture_enabled = self.auto_capture_var.get()
            self.auto_complete_enabled = self.auto_complete_var.get()
            
            # Create user
            self.current_user_id = self.face_storage.create_user(name, user_id)
            
            # Reset capture state
            self.captured_images = []
            self.registration_active = True
            
            # Update UI
            self.progress_bar.configure(maximum=self.target_image_count)
            self.progress_var.set(f"0 / {self.target_image_count}")
            
            # Enable capture tab
            self.notebook.tab(1, state='normal')
            self.notebook.select(1)
            
            # Start camera
            self.camera_widget.start_stream()
            
            # Update controls
            self.manual_capture_button.configure(state=tk.NORMAL)
            self.force_capture_button.configure(state=tk.NORMAL)
            self.skip_button.configure(state=tk.NORMAL)
            
            self.status_var.set(f"Registration started for {name} (ID: {self.current_user_id})")
            self.logger.info(f"Started registration for user: {name} (ID: {self.current_user_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to start registration: {str(e)}")
            messagebox.showerror("Error", f"Failed to start registration: {str(e)}")
    
    def test_name_entry(self):
        """Test method to debug name entry issues"""
        try:
            # Update UI to ensure latest values
            self.window.update_idletasks()
            
            # Get values from both sources
            stringvar_value = self.name_var.get()
            entry_value = self.name_entry.get()
            
            # Test results
            results = f"""Name Entry Debug Results:
            
StringVar value: '{stringvar_value}'
Entry widget value: '{entry_value}'
StringVar length: {len(stringvar_value)}
Entry length: {len(entry_value)}
StringVar stripped: '{stringvar_value.strip()}'
Entry stripped: '{entry_value.strip()}'

Both values match: {stringvar_value == entry_value}
StringVar is empty: {not stringvar_value.strip()}
Entry is empty: {not entry_value.strip()}

Widget exists: {hasattr(self, 'name_entry')}
StringVar exists: {hasattr(self, 'name_var')}
            """
            
            self.logger.info(f"Name entry test results: {results}")
            messagebox.showinfo("Name Entry Test", results)
            
        except Exception as e:
            error_msg = f"Error testing name entry: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Test Error", error_msg)
    
    def process_camera_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process camera frame for face detection and auto-capture
        
        Args:
            frame: Camera frame
            
        Returns:
            Processed frame with face detection overlay
        """
        if not self.registration_active:
            return frame
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame, detect_eyes=True)
            
            # Draw face rectangles
            processed_frame = frame.copy()
            
            if len(faces) == 0:
                self.face_status_var.set("No face detected")
                self.quality_status_var.set("Please position your face in view")
                
                # Draw instruction text
                cv2.putText(
                    processed_frame, 
                    "Please position your face in view", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                
            elif len(faces) == 1:
                face = faces[0]
                x, y, w, h = face
                
                # Get detailed quality scores
                quality_scores = self.face_detector.get_face_quality_score(face, frame.shape)
                
                # Check face quality with more lenient thresholds
                is_centered = quality_scores['center_score'] > 0.6  # Lowered threshold
                is_good_size = quality_scores['size_score'] > 0.7   # Lowered threshold
                overall_quality = quality_scores['overall_score']
                
                if overall_quality > 0.7:  # More lenient overall threshold
                    # Excellent face detected
                    self.face_status_var.set(f"Excellent face detected (Quality: {overall_quality:.0%})")
                    self.quality_status_var.set("Ready to capture!")
                    
                    # Draw green rectangle
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Auto capture if enabled
                    if self.auto_capture_enabled and len(self.captured_images) < self.target_image_count:
                        self.auto_capture_image(frame, face)
                        
                elif overall_quality > 0.5:  # Good enough for capture
                    # Good face detected
                    self.face_status_var.set(f"Good face detected (Quality: {overall_quality:.0%})")
                    self.quality_status_var.set("Capturing in next few seconds...")
                    
                    # Draw yellow-green rectangle
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 200, 100), 3)
                    
                    # Auto capture with slightly longer delay
                    if self.auto_capture_enabled and len(self.captured_images) < self.target_image_count:
                        self.auto_capture_image(frame, face, min_delay=1.5)
                        
                else:
                    # Face needs adjustment
                    status_parts = []
                    if quality_scores['center_score'] < 0.6:
                        if quality_scores['center_offset_x'] > quality_scores['center_offset_y']:
                            status_parts.append(f"move {'left' if quality_scores['center_offset_x'] > 0 else 'right'}")
                        else:
                            status_parts.append(f"move {'up' if quality_scores['center_offset_y'] > 0 else 'down'}")
                    
                    if quality_scores['size_score'] < 0.7:
                        if quality_scores['area_ratio'] < 0.15:
                            status_parts.append("move closer")
                        else:
                            status_parts.append("move back")
                    
                    self.face_status_var.set(f"Face detected (Quality: {overall_quality:.0%})")
                    self.quality_status_var.set(f"Please {', '.join(status_parts) if status_parts else 'hold still'}")
                    
                    # Draw yellow rectangle
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Add quality indicator on frame
                quality_text = f"Quality: {overall_quality:.0%}"
                cv2.putText(
                    processed_frame,
                    quality_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Update debug info if enabled
                if self.show_debug_var.get():
                    debug_text = (
                        f"Center: {quality_scores['center_score']:.2f} "
                        f"Size: {quality_scores['size_score']:.2f}\n"
                        f"Area: {quality_scores['area_ratio']:.3f} "
                        f"OffsetX: {quality_scores['center_offset_x']:.3f}"
                    )
                    self.debug_info_var.set(debug_text)
                else:
                    self.debug_info_var.set("Enable debug for details")
                
            else:
                self.face_status_var.set("Multiple faces detected")
                self.quality_status_var.set("Please ensure only one face is visible")
                
                # Draw red rectangles for all faces
                for face in faces:
                    x, y, w, h = face
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Update debug info
                if self.show_debug_var.get():
                    self.debug_info_var.set(f"Found {len(faces)} faces")
                
            # General debug info when no faces detected
            if len(faces) == 0 and self.show_debug_var.get():
                self.debug_info_var.set("No faces detected")
            
            # Draw progress info
            progress_text = f"Captured: {len(self.captured_images)} / {self.target_image_count}"
            cv2.putText(
                processed_frame,
                progress_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Show completion status if all images captured
            if len(self.captured_images) >= self.target_image_count:
                completion_text = "ALL IMAGES CAPTURED! 🎉"
                text_size = cv2.getTextSize(completion_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 50
                
                # Draw background rectangle
                cv2.rectangle(processed_frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(
                    processed_frame,
                    completion_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3
                )
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {str(e)}")
            return frame
    
    def auto_capture_image(self, frame: np.ndarray, face: tuple, min_delay: float = 1.0):
        """
        Auto-capture image with timing delay
        
        Args:
            frame: Camera frame
            face: Detected face rectangle
            min_delay: Minimum delay between captures in seconds
        """
        # Implement capture delay to avoid rapid captures
        current_time = time.time()
        if not hasattr(self, 'last_capture_time'):
            self.last_capture_time = 0
        
        if current_time - self.last_capture_time > min_delay:  # Reduced from 2.0 seconds
            self.capture_face_image(frame, face)
            self.last_capture_time = current_time
    
    def manual_capture(self):
        """Manually capture current camera frame with lenient validation"""
        try:
            frame = self.camera_widget.capture_image()
            if frame is not None:
                faces = self.face_detector.detect_faces(frame, detect_eyes=False)  # Skip eye detection for manual
                if len(faces) >= 1:
                    # Use the largest face if multiple faces detected
                    largest_face = self.face_detector.get_largest_face(faces)
                    self.capture_face_image(frame, largest_face, manual=True)
                    self.status_var.set("Manual capture successful!")
                else:
                    messagebox.showwarning("Warning", "No face detected in current frame.\nTry the 'Force Capture' button to capture anyway.")
            else:
                messagebox.showerror("Error", "Failed to capture image from camera")
                
        except Exception as e:
            self.logger.error(f"Manual capture failed: {str(e)}")
            messagebox.showerror("Error", f"Capture failed: {str(e)}")
    
    def force_capture(self):
        """Force capture regardless of face detection (for debugging)"""
        try:
            frame = self.camera_widget.capture_image()
            if frame is not None:
                # Create a dummy face rectangle covering center of frame
                h, w = frame.shape[:2]
                dummy_face = (w//4, h//4, w//2, h//2)  # Center quarter of frame
                
                self.capture_face_image(frame, dummy_face, manual=True, force=True)
                self.status_var.set("Force capture successful!")
                messagebox.showinfo("Force Capture", "Image captured without face validation.\nThis helps when face detection is having issues.")
            else:
                messagebox.showerror("Error", "Failed to capture image from camera")
                
        except Exception as e:
            self.logger.error(f"Force capture failed: {str(e)}")
            messagebox.showerror("Error", f"Force capture failed: {str(e)}")
    
    def capture_face_image(self, frame: np.ndarray, face: tuple, manual: bool = False, force: bool = False):
        """
        Capture and process a face image
        
        Args:
            frame: Camera frame
            face: Face rectangle (x, y, w, h)
            manual: Whether this is a manual capture
            force: Whether to skip all quality validations
        """
        try:
            # Extract face region
            face_image = self.image_processor.extract_face_region(frame, face, padding=0.2)
            
            # Validate image quality (skip for force captures)
            if not force and not self.image_processor.validate_image_quality(face_image, min_variance=50.0):  # Lowered threshold
                if manual:
                    # Show warning but still allow manual capture
                    result = messagebox.askyesno(
                        "Low Quality Image",
                        "The captured image quality is low. Do you want to capture it anyway?"
                    )
                    if not result:
                        return
                else:
                    self.logger.warning("Captured image failed quality validation")
                    return
            
            # Preprocess face image
            processed_face = self.image_processor.preprocess_face_image(face_image)
            
            # Add to captured images
            self.captured_images.append({
                'original': face_image,
                'processed': processed_face,
                'face_rect': face,
                'timestamp': time.time(),
                'manual': manual,
                'force': force
            })
            
            # Update UI
            self.update_progress()
            self.add_thumbnail(face_image)
            
            capture_type = "force" if force else ("manual" if manual else "auto")
            self.logger.info(f"Captured image {len(self.captured_images)} of {self.target_image_count} ({capture_type})")
            
            # Check if complete
            if len(self.captured_images) >= self.target_image_count:
                self.complete_capture_button.configure(state=tk.NORMAL)
                
                # Show completion message
                self.face_status_var.set("🎉 All images captured!")
                
                # Check if auto-completion is enabled
                if self.auto_complete_enabled:
                    # Start countdown timer
                    self.start_completion_countdown()
                    self.quality_status_var.set("Auto-advancing to review...")
                else:
                    # Manual completion only
                    self.status_var.set("All images captured! Click 'Complete Registration' to proceed.")
                    self.quality_status_var.set("Ready to proceed manually")
                
                # Stop auto-capture to prevent additional captures
                self.auto_capture_enabled = False
                
        except Exception as e:
            self.logger.error(f"Failed to capture face image: {str(e)}")
            if manual or force:
                messagebox.showerror("Error", f"Capture failed: {str(e)}")
    
    def add_thumbnail(self, image: np.ndarray):
        """Add thumbnail to preview panel"""
        try:
            # Resize for thumbnail
            thumbnail = cv2.resize(image, (80, 80))
            
            # Convert to PhotoImage
            from PIL import Image, ImageTk
            thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(thumbnail_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Create thumbnail label
            thumbnail_label = ttk.Label(self.thumbnail_frame, image=photo)
            thumbnail_label.image = photo  # Keep reference
            thumbnail_label.pack(pady=2)
            
        except Exception as e:
            self.logger.error(f"Failed to add thumbnail: {str(e)}")
    
    def update_progress(self):
        """Update progress indicators"""
        count = len(self.captured_images)
        self.progress_var.set(f"{count} / {self.target_image_count}")
        self.progress_bar.configure(value=count)
    
    def skip_image(self):
        """Skip current image and continue"""
        if len(self.captured_images) > 0:
            self.captured_images.pop()
            self.update_progress()
            # Remove last thumbnail
            thumbnails = self.thumbnail_frame.winfo_children()
            if thumbnails:
                thumbnails[-1].destroy()
    
    def restart_capture(self):
        """Restart the capture process"""
        self.captured_images = []
        self.update_progress()
        
        # Clear thumbnails
        for child in self.thumbnail_frame.winfo_children():
            child.destroy()
        
        self.complete_capture_button.configure(state=tk.DISABLED)
        self.status_var.set("Capture restarted")
    
    def complete_registration(self):
        """Complete the registration process"""
        try:
            # Cancel auto-completion countdown if running
            self.cancel_auto_completion()
            
            if len(self.captured_images) < 5:
                messagebox.showerror("Error", "At least 5 images are required for registration")
                return
            
            # Save images to storage
            face_images = [img['original'] for img in self.captured_images]
            saved_files = self.face_storage.save_multiple_face_images(self.current_user_id, face_images)
            
            # Update status
            self.status_var.set(f"Registration completed! Saved {len(saved_files)} images.")
            
            # Enable review tab
            self.notebook.tab(2, state='normal')
            self.create_review_content()
            self.notebook.select(2)
            
            # Stop camera
            self.camera_widget.stop_stream()
            self.registration_active = False
            
            self.logger.info(f"Registration completed for user {self.current_user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to complete registration: {str(e)}")
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
    
    def auto_complete_registration(self):
        """Automatically complete registration and move to review stage"""
        try:
            # Check if we still have enough images (user might have deleted some)
            if len(self.captured_images) < 5:
                self.status_var.set("Not enough images. Please capture more or click 'Complete Registration' manually.")
                return
            
            # Perform the completion automatically
            self.complete_registration()
            
            # Show success message
            messagebox.showinfo(
                "Auto-Complete", 
                f"🎉 Registration completed automatically!\n\n"
                f"Captured {len(self.captured_images)} images successfully.\n"
                f"You can now review and finalize the registration."
            )
            
        except Exception as e:
            self.logger.error(f"Auto-completion failed: {str(e)}")
            # Fall back to manual completion
            self.status_var.set("Auto-completion failed. Please click 'Complete Registration' manually.")
            messagebox.showwarning(
                "Auto-Complete Failed", 
                f"Automatic completion encountered an issue: {str(e)}\n\n"
                f"Please click 'Complete Registration' button manually to proceed."
            )
    
    def start_completion_countdown(self, countdown_seconds: int = 5):
        """Start countdown timer for auto-completion"""
        self.completion_countdown = countdown_seconds
        self.update_countdown_display()
    
    def update_countdown_display(self):
        """Update the countdown display"""
        if self.completion_countdown > 0:
            self.status_var.set(f"🎉 All images captured! Auto-advancing to review in {self.completion_countdown} seconds... (Click 'Complete Registration' to skip)")
            self.completion_countdown -= 1
            # Schedule next countdown update
            self.window.after(1000, self.update_countdown_display)
        else:
            # Countdown finished, auto-complete
            self.auto_complete_registration()
    
    def cancel_auto_completion(self):
        """Cancel the auto-completion countdown"""
        if hasattr(self, 'completion_countdown'):
            self.completion_countdown = 0
            self.status_var.set("Auto-completion cancelled. Click 'Complete Registration' when ready.")
    
    def create_review_content(self):
        """Create review tab content"""
        # Clear existing content
        for child in self.review_content_frame.winfo_children():
            child.destroy()
        
        # User info summary
        if self.current_user_id:
            user_info = self.face_storage.get_user_info(self.current_user_id)
            
            info_frame = ttk.LabelFrame(self.review_content_frame, text="Registration Summary", padding=20)
            info_frame.pack(fill=tk.X, pady=(0, 20))
            
            ttk.Label(info_frame, text=f"Name: {user_info['name']}", font=("Arial", 12)).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"User ID: {user_info['user_id']}", font=("Arial", 12)).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Images Captured: {user_info['image_count']}", font=("Arial", 12)).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Registration Date: {user_info['created_at'][:10]}", font=("Arial", 12)).pack(anchor=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(self.review_content_frame)
        button_frame.pack(pady=20)
        
        finish_button = ttk.Button(
            button_frame,
            text="Finish Registration",
            command=self.finish_registration,
            style="Accent.TButton"
        )
        finish_button.pack(side=tk.LEFT, padx=(0, 10))
        
        register_another_button = ttk.Button(
            button_frame,
            text="Register Another User",
            command=self.register_another_user
        )
        register_another_button.pack(side=tk.LEFT)
    
    def finish_registration(self):
        """Finish registration and close window"""
        if self.on_complete_callback:
            try:
                self.on_complete_callback(self.current_user_id)
            except Exception as e:
                self.logger.error(f"Error in completion callback: {str(e)}")
        
        self.on_close()
    
    def register_another_user(self):
        """Start registration for another user"""
        # Reset state
        self.current_user_id = None
        self.captured_images = []
        self.registration_active = False
        
        # Reset UI
        self.name_var.set("")
        self.user_id_var.set("")
        self.notebook.tab(1, state='disabled')
        self.notebook.tab(2, state='disabled')
        self.notebook.select(0)
        
        # Clear thumbnails
        for child in self.thumbnail_frame.winfo_children():
            child.destroy()
        
        self.status_var.set("Ready for new registration")
    
    def on_close(self):
        """Handle window close event"""
        try:
            # Stop camera if running
            if hasattr(self, 'camera_widget'):
                self.camera_widget.stop_stream()
            
            self.registration_active = False
            self.logger.info("Registration window closed")
            self.window.destroy()
            
        except Exception as e:
            self.logger.error(f"Error closing registration window: {str(e)}")
            self.window.destroy()

def open_registration_window(parent=None, on_complete=None):
    """
    Open face registration window
    
    Args:
        parent: Parent window
        on_complete: Callback function for registration completion
        
    Returns:
        Registration window instance
    """
    return RegistrationWindow(parent, on_complete)

if __name__ == "__main__":
    # Test the registration window
    root = tk.Tk()
    root.withdraw()  # Hide root window
    
    def on_registration_complete(user_id):
        print(f"Registration completed for user: {user_id}")
    
    registration_window = RegistrationWindow(on_complete_callback=on_registration_complete)
    root.mainloop() 