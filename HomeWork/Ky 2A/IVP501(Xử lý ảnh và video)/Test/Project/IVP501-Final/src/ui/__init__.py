# UI package for FaceAttend application 

from .camera_widget import CameraWidget
from .registration_window import RegistrationWindow, open_registration_window
from .attendance_window import AttendanceWindow
from .logs_window import LogsWindow, open_logs_window

__all__ = [
    'CameraWidget',
    'RegistrationWindow', 
    'open_registration_window',
    'AttendanceWindow',
    'LogsWindow',
    'open_logs_window'
] 