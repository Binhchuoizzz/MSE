"""
Logging utility for FaceAttend application
Provides centralized logging configuration and management
"""

import logging
import os
from datetime import datetime
from typing import Optional

class FaceAttendLogger:
    """Custom logger class for FaceAttend application"""
    
    def __init__(self, name: str = "FaceAttend", log_dir: str = "logs"):
        """
        Initialize the logger
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return self.logger
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for all logs
        log_filename = os.path.join(
            self.log_dir, 
            f"faceattend_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Error file handler
        error_filename = os.path.join(
            self.log_dir,
            f"faceattend_errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler = logging.FileHandler(error_filename)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)
        
        return self.logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False):
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info)

def setup_logger(name: str = "FaceAttend", log_dir: str = "logs") -> logging.Logger:
    """
    Set up and return a configured logger
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    logger_instance = FaceAttendLogger(name, log_dir)
    return logger_instance.get_logger()

def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f"FaceAttend.{module_name}")

# Global logger instance
_global_logger = None

def get_global_logger() -> logging.Logger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger 