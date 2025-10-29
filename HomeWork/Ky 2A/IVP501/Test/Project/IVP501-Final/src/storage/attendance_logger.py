"""
Attendance logging module for FaceAttend application
Handles attendance data logging, CSV management, and attendance records
"""

import csv
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from src.utils.logger import get_module_logger
from src.utils.exceptions import StorageError, ValidationError

class AttendanceLogger:
    """Manages attendance logging and CSV file operations"""
    
    def __init__(self, base_directory: str = "attendance_logs"):
        """
        Initialize the attendance logger
        
        Args:
            base_directory: Base directory for storing attendance logs
        """
        self.logger = get_module_logger("AttendanceLogger")
        self.base_directory = Path(base_directory)
        
        # Ensure base directory exists
        self._initialize_storage()
        
        # CSV headers
        self.csv_headers = ['timestamp', 'user_id', 'name', 'confidence', 'date', 'time']
        
        self.logger.info(f"AttendanceLogger initialized with base directory: {self.base_directory}")
    
    def _initialize_storage(self):
        """Initialize storage directory structure"""
        try:
            self.base_directory.mkdir(parents=True, exist_ok=True)
            self.logger.info("Attendance logs directory initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize attendance storage: {str(e)}")
            raise StorageError(f"Attendance storage initialization failed: {str(e)}")
    
    def _get_daily_log_path(self, target_date: date = None) -> Path:
        """
        Get the file path for a daily log file
        
        Args:
            target_date: Target date (defaults to today)
            
        Returns:
            Path to the daily log file
        """
        if target_date is None:
            target_date = date.today()
        
        filename = f"{target_date.strftime('%Y-%m-%d')}.csv"
        return self.base_directory / filename
    
    def _ensure_daily_log_exists(self, target_date: date = None) -> Path:
        """
        Ensure daily log file exists with proper headers
        
        Args:
            target_date: Target date (defaults to today)
            
        Returns:
            Path to the daily log file
        """
        log_path = self._get_daily_log_path(target_date)
        
        try:
            if not log_path.exists():
                # Create new log file with headers
                with open(log_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.csv_headers)
                
                self.logger.info(f"Created new daily log file: {log_path}")
            
            return log_path
            
        except Exception as e:
            self.logger.error(f"Failed to create daily log file: {str(e)}")
            raise StorageError(f"Daily log file creation failed: {str(e)}")
    
    def log_attendance(self, user_id: str, name: str, confidence: float = 0.0) -> bool:
        """
        Log an attendance entry
        
        Args:
            user_id: User ID
            name: User name
            confidence: Recognition confidence score
            
        Returns:
            True if attendance was logged successfully
        """
        try:
            # Validate inputs
            if not user_id or not name:
                raise ValidationError("User ID and name are required")
            
            # Check for duplicate entry (within 5 minutes)
            if self._is_duplicate_entry(user_id):
                self.logger.warning(f"Duplicate attendance entry prevented for user {user_id}")
                return False
            
            # Prepare attendance record
            now = datetime.now()
            attendance_record = {
                'timestamp': now.isoformat(),
                'user_id': user_id,
                'name': name,
                'confidence': f"{confidence:.2f}",
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S')
            }
            
            # Ensure daily log file exists
            log_path = self._ensure_daily_log_exists(now.date())
            
            # Append attendance record
            with open(log_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                writer.writerow(attendance_record)
            
            self.logger.info(f"Logged attendance for {name} (ID: {user_id}) with confidence {confidence:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log attendance: {str(e)}")
            return False
    
    def _is_duplicate_entry(self, user_id: str, time_window_minutes: int = 5) -> bool:
        """
        Check if there's a duplicate entry within the time window
        
        Args:
            user_id: User ID to check
            time_window_minutes: Time window in minutes to check for duplicates
            
        Returns:
            True if duplicate entry exists
        """
        try:
            log_path = self._get_daily_log_path()
            
            if not log_path.exists():
                return False
            
            # Read today's log
            df = pd.read_csv(log_path)
            
            if df.empty:
                return False
            
            # Filter for the specific user
            user_entries = df[df['user_id'] == user_id]
            
            if user_entries.empty:
                return False
            
            # Check the most recent entry
            latest_entry = user_entries.iloc[-1]
            latest_timestamp = datetime.fromisoformat(latest_entry['timestamp'])
            
            # Check if within time window
            time_diff = datetime.now() - latest_timestamp
            return time_diff.total_seconds() < (time_window_minutes * 60)
            
        except Exception as e:
            self.logger.error(f"Error checking for duplicate entry: {str(e)}")
            return False  # Allow logging if check fails
    
    def get_daily_attendance(self, target_date: date = None) -> List[Dict]:
        """
        Get attendance records for a specific date
        
        Args:
            target_date: Target date (defaults to today)
            
        Returns:
            List of attendance records
        """
        try:
            log_path = self._get_daily_log_path(target_date)
            
            if not log_path.exists():
                self.logger.info(f"No attendance log found for {target_date}")
                return []
            
            attendance_records = []
            with open(log_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    attendance_records.append(row)
            
            self.logger.debug(f"Retrieved {len(attendance_records)} attendance records for {target_date}")
            return attendance_records
            
        except Exception as e:
            self.logger.error(f"Failed to get daily attendance: {str(e)}")
            return []
    
    def get_attendance_range(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Get attendance records for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of attendance records
        """
        try:
            all_records = []
            current_date = start_date
            
            while current_date <= end_date:
                daily_records = self.get_daily_attendance(current_date)
                all_records.extend(daily_records)
                current_date = current_date + timedelta(days=1)
            
            self.logger.info(f"Retrieved {len(all_records)} attendance records from {start_date} to {end_date}")
            return all_records
            
        except Exception as e:
            self.logger.error(f"Failed to get attendance range: {str(e)}")
            return []
    
    def get_user_attendance(self, user_id: str, start_date: date = None, end_date: date = None) -> List[Dict]:
        """
        Get attendance records for a specific user
        
        Args:
            user_id: User ID
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            List of attendance records for the user
        """
        try:
            if end_date is None:
                end_date = date.today()
            
            if start_date is None:
                start_date = date.today() - timedelta(days=30)
            
            all_records = self.get_attendance_range(start_date, end_date)
            user_records = [record for record in all_records if record['user_id'] == user_id]
            
            self.logger.debug(f"Retrieved {len(user_records)} attendance records for user {user_id}")
            return user_records
            
        except Exception as e:
            self.logger.error(f"Failed to get user attendance: {str(e)}")
            return []
    
    def export_attendance_csv(self, output_path: str, start_date: date = None, end_date: date = None) -> bool:
        """
        Export attendance records to a CSV file
        
        Args:
            output_path: Output file path
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            True if export was successful
        """
        try:
            if end_date is None:
                end_date = date.today()
            
            if start_date is None:
                start_date = date.today() - timedelta(days=30)
            
            # Get attendance records
            records = self.get_attendance_range(start_date, end_date)
            
            if not records:
                self.logger.warning("No attendance records found for export")
                return False
            
            # Write to CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                writer.writeheader()
                writer.writerows(records)
            
            self.logger.info(f"Exported {len(records)} attendance records to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export attendance CSV: {str(e)}")
            return False
    
    def get_attendance_statistics(self, start_date: date = None, end_date: date = None) -> Dict:
        """
        Get attendance statistics for a date range
        
        Args:
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            Dictionary with attendance statistics
        """
        try:
            if end_date is None:
                end_date = date.today()
            
            if start_date is None:
                start_date = date.today() - timedelta(days=30)
            
            records = self.get_attendance_range(start_date, end_date)
            
            if not records:
                return {
                    'total_entries': 0,
                    'unique_users': 0,
                    'date_range': f"{start_date} to {end_date}",
                    'daily_averages': {}
                }
            
            # Calculate statistics
            unique_users = set(record['user_id'] for record in records)
            daily_counts = {}
            
            for record in records:
                record_date = record['date']
                daily_counts[record_date] = daily_counts.get(record_date, 0) + 1
            
            avg_daily_attendance = sum(daily_counts.values()) / len(daily_counts) if daily_counts else 0
            
            stats = {
                'total_entries': len(records),
                'unique_users': len(unique_users),
                'date_range': f"{start_date} to {end_date}",
                'daily_averages': daily_counts,
                'avg_daily_attendance': round(avg_daily_attendance, 2)
            }
            
            self.logger.debug(f"Calculated attendance statistics: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attendance statistics: {str(e)}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old log files
        
        Args:
            days_to_keep: Number of days to keep (default 90 days)
            
        Returns:
            Number of files deleted
        """
        try:
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for log_file in self.base_directory.glob("*.csv"):
                try:
                    # Extract date from filename
                    file_date = datetime.strptime(log_file.stem, '%Y-%m-%d').date()
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted old log file: {log_file}")
                        
                except ValueError:
                    # Skip files that don't match the date format
                    continue
            
            self.logger.info(f"Cleaned up {deleted_count} old log files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {str(e)}")
            return 0


def test_attendance_logger():
    """Test function for attendance logger"""
    try:
        # Initialize logger
        logger = AttendanceLogger()
        
        # Test logging attendance
        success = logger.log_attendance("test_user_001", "Test User", 85.5)
        print(f"Attendance logging test: {'PASSED' if success else 'FAILED'}")
        
        # Test getting daily attendance
        daily_records = logger.get_daily_attendance()
        print(f"Daily attendance records: {len(daily_records)}")
        
        # Test statistics
        stats = logger.get_attendance_statistics()
        print(f"Attendance statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Attendance Logger test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_attendance_logger() 