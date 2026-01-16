"""
Custom exception classes for FaceAttend application
Provides specific error types for different components
"""

class FaceAttendError(Exception):
    """Base exception class for FaceAttend application"""
    pass

class CameraError(FaceAttendError):
    """Exception raised for camera-related errors"""
    pass

class FaceDetectionError(FaceAttendError):
    """Exception raised for face detection errors"""
    pass

class FaceRecognitionError(FaceAttendError):
    """Exception raised for face recognition errors"""
    pass

class StorageError(FaceAttendError):
    """Exception raised for storage-related errors"""
    pass

class UIError(FaceAttendError):
    """Exception raised for UI-related errors"""
    pass

class ConfigurationError(FaceAttendError):
    """Exception raised for configuration errors"""
    pass

class ValidationError(FaceAttendError):
    """Exception raised for data validation errors"""
    pass 