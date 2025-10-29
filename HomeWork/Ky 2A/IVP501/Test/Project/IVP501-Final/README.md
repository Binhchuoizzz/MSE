FaceAttend — Python Desktop Face Recognition Attendance System (Classical Version)

Author: [Your Name]
Date: [Today’s Date]
Version: 1.0

⸻

1. Objective

Develop a Python-based desktop application for Mac that captures user attendance via face recognition, using classical algorithms (no deep learning, machine learning, or AI). The system should maximize accuracy within classical constraints and provide a user-friendly Tkinter-based interface.

⸻

2. Scope

2.1 In Scope
• Face registration via webcam.
• Automatic face detection and recognition for attendance logging.
• Use of classical algorithms (LBPH, Eigenfaces, Fisherfaces — prioritize LBPH).
• Data storage in local filesystem:
• Store registered face images.
• Store attendance logs in CSV format.
• Tkinter-based UI with basic interaction:
• Face registration.
• Attendance capture.
• Attendance log viewing/export.
• Single device operation (Mac).
• Installation-based distribution.
• Grayscale processing and histogram equalization for accuracy improvement.

2.2 Out of Scope
• Deep learning, machine learning, or AI-based face recognition.
• Web-based or cloud-based solutions.
• User authentication/security.
• Admin dashboards.
• Database storage (No SQL/NoSQL DBMS).
• Offline mode (the app can be online but doesn’t depend on online resources).
• Backup/recovery mechanisms.

⸻

3. Functional Requirements

ID Requirement
FR-1 The system shall allow users to register faces by capturing multiple (5–10) images through webcam.
FR-2 The system shall preprocess images to grayscale and apply histogram equalization to improve image quality.
FR-3 The system shall use LBPH for face recognition with optional fallbacks to Eigenfaces or Fisherfaces.
FR-4 The system shall automatically detect faces from webcam stream using Haar Cascades.
FR-5 The system shall automatically recognize faces without manual triggering during attendance capture.
FR-6 The system shall store face images in organized folders in the local filesystem.
FR-7 The system shall store attendance logs (timestamp + user ID) in a CSV file daily.
FR-8 The system shall allow users to view attendance logs in the UI.
FR-9 The system shall allow users to export attendance logs as CSV files.
FR-10 The system shall work exclusively on Mac platforms.
FR-11 The system shall provide simple error messages (e.g., “No face detected”, “Face not recognized”).

⸻

4. Non-Functional Requirements

ID Requirement
NFR-1 The application shall provide 90%+ accuracy in controlled lighting conditions using classical algorithms.
NFR-2 The system shall start up in under 5 seconds.
NFR-3 The application shall process a face detection/recognition within 1 second.
NFR-4 The system shall have a minimalist UI via Tkinter.
NFR-5 The application shall not store data externally — all storage is local.
NFR-6 The application shall maintain a confidence threshold for recognition to reduce false positives (configurable, e.g., 50–80).
NFR-7 Face images and logs shall be stored in a structured filesystem layout.
NFR-8 The system shall provide a simple installer (using PyInstaller or similar) for deployment on Mac.
NFR-9 The system shall be lightweight, not consuming more than 100MB of RAM during idle operations.

⸻

5. User Interface

UI ID Feature
UI-1 Home Screen: Buttons to Register Face, Start Attendance, View Attendance Logs.
UI-2 Face Registration Screen: Webcam live feed, Capture multiple images with slight pose/lighting variations.
UI-3 Attendance Capture Screen: Webcam live feed with real-time scanning; show name/ID upon successful recognition.
UI-4 Logs Screen: Display attendance records and allow export to CSV.
UI-5 Status Area: Show live status messages (e.g., “Scanning for faces…”, “Face recognized: User 001”).

⸻

6. Technical Design

6.1 Architecture Overview
• UI Layer: Tkinter GUI.
• Face Detection: Haar Cascade Classifier (OpenCV).
• Face Recognition: LBPH Recognizer (OpenCV).
• Preprocessing:
• Convert images to grayscale.
• Apply histogram equalization.
• Storage:
• Face images: faces/<user_id>/img1.jpg, img2.jpg, ...
• Attendance logs: attendance_logs/YYYY-MM-DD.csv
• Installer: PyInstaller to generate .app for Mac.

6.2 Directory Structure

FaceAttend/
├── faces/
│ ├── 001/
│ │ ├── img1.jpg
│ │ ├── img2.jpg
├── attendance_logs/
│ ├── 2025-05-31.csv
├── trainer.yml # LBPH trained model
├── main.py # Application entry point
├── README.md

⸻

7. Acceptance Criteria

ID Acceptance Test
AC-1 A user can register their face with 5+ sample images.
AC-2 A user can have their attendance marked by facing the webcam, without needing to press any buttons.
AC-3 Attendance logs are created and stored correctly in CSV format daily.
AC-4 The system recognizes registered users with a minimum of 90% accuracy under normal lighting.
AC-5 The user interface is usable and does not freeze during webcam operation.
AC-6 The app can be installed and launched on Mac without requiring terminal commands.

⸻

8. Milestones

Milestone Target Date Description
M1 +1 Week Set up basic Tkinter GUI + Webcam access
M2 +2 Weeks Implement face registration (capture + save)
M3 +3 Weeks Implement face recognition + attendance logging
M4 +4 Weeks Build basic log viewing + CSV export
M5 +5 Weeks Accuracy tuning (threshold testing, environment stabilization)
M6 +6 Weeks Build installer for Mac using PyInstaller
M7 +7 Weeks Final testing, documentation, and delivery

⸻

9. Risks and Mitigation

Risk Impact Mitigation
Poor lighting/pose causes low recognition accuracy High Encourage consistent lighting; implement multiple sample captures and preprocessing.
Haar Cascade may have false detections Medium Fine-tune cascade parameters and limit frame capture rate to avoid overloading.
LBPH model may not scale with hundreds of users Medium Limit to <100 users for first version; explore hybrid approaches later if needed.
Tkinter UI limitations for complex UX Low Minimal UI is sufficient for V1; revisit UX improvements in later versions.

⸻

10. Glossary
    • LBPH: Local Binary Pattern Histograms — a classical face recognition algorithm robust to lighting changes.
    • Haar Cascade: A classical object detection method useful for detecting faces.
    • Tkinter: Python’s standard GUI package.
    • CSV: Comma-Separated Values — file format for attendance log export.

⸻

End of Document
