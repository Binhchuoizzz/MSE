#!/usr/bin/env python3
"""
Test script for face registration fixes
This script helps verify that the improved registration system works correctly
"""

import sys
import tkinter as tk
from tkinter import messagebox
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from src.ui.registration_window import RegistrationWindow
    from src.recognition.face_detector import FaceDetector
    from src.storage.face_storage import FaceStorage
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running this from the project root directory")
    sys.exit(1)

def test_face_detector():
    """Test the improved face detector"""
    print("Testing Face Detector improvements...")
    
    detector = FaceDetector()
    
    # Test with relaxed parameters
    print(f"Face detection parameters:")
    print(f"  Scale factor: {detector.scale_factor}")
    print(f"  Min neighbors: {detector.min_neighbors}")
    print(f"  Min size: {detector.min_size}")
    
    # Test centering tolerance
    test_face = (100, 100, 100, 100)  # Example face rectangle
    test_frame_shape = (480, 640)     # Example frame shape
    
    is_centered_old = detector.is_face_centered(test_face, test_frame_shape, center_tolerance=0.3)
    is_centered_new = detector.is_face_centered(test_face, test_frame_shape, center_tolerance=0.4)
    
    print(f"  Face centering (old tolerance): {is_centered_old}")
    print(f"  Face centering (new tolerance): {is_centered_new}")
    
    # Test quality scores
    quality_scores = detector.get_face_quality_score(test_face, test_frame_shape)
    print(f"  Quality scores: {quality_scores}")
    
    print("✓ Face detector test completed\n")

def test_storage_system():
    """Test the storage system"""
    print("Testing Storage System...")
    
    storage = FaceStorage()
    stats = storage.get_storage_stats()
    
    print(f"Storage statistics:")
    print(f"  Total users: {stats.get('total_users', 0)}")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"  Storage path: {stats.get('storage_path', 'Unknown')}")
    print(f"  Total size: {stats.get('total_size_mb', 0):.1f} MB")
    
    # Test validation
    report = storage.validate_storage_integrity()
    print(f"  Storage integrity: {'✓ Valid' if report['valid'] else '✗ Issues found'}")
    
    if report['issues']:
        print(f"  Issues: {len(report['issues'])}")
        for issue in report['issues'][:3]:  # Show first 3 issues
            print(f"    - {issue}")
    
    print("✓ Storage system test completed\n")

def on_registration_complete(user_id):
    """Callback for when registration completes"""
    print(f"✓ Registration completed for user: {user_id}")
    messagebox.showinfo("Success", f"Registration test completed successfully!\n\nUser ID: {user_id}")

def main():
    """Main test function"""
    print("="*60)
    print("FaceAttend Registration System Test")
    print("="*60)
    print()
    
    # Test individual components
    test_face_detector()
    test_storage_system()
    
    # Test the registration window
    print("Testing Registration Window...")
    print("This will open the improved registration window with:")
    print("  • More lenient face detection (40% center tolerance)")
    print("  • Relaxed size requirements (8%-90% face area)")
    print("  • Quality score feedback (0-100%)")
    print("  • Manual capture button (works with any face)")
    print("  • Force capture button (works without face detection)")
    print("  • Debug info panel (shows detection details)")
    print("  • Faster auto-capture (1-1.5 second delays)")
    print()
    
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Open registration window
        registration_window = RegistrationWindow(
            parent=None,  # No parent, so it creates its own window
            on_complete_callback=on_registration_complete
        )
        
        print("Registration window opened. Please test the following:")
        print("1. Position your face in the camera view")
        print("2. Check the Quality percentage displayed")
        print("3. Enable 'Show detailed debug info' to see exact scores")
        print("4. Try manual capture if auto-capture doesn't work")
        print("5. Try force capture if face detection fails completely")
        print()
        print("The system should now be much more responsive!")
        print("Press Ctrl+C to exit this test when done.")
        
        # Start the event loop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            root.quit()
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    main() 