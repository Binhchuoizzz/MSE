"""
Face image storage module for FaceAttend application
Handles face image storage, directory management, and file organization
"""

import os
import shutil
import json
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from src.utils.logger import get_module_logger
from src.utils.exceptions import StorageError, ValidationError

class FaceStorage:
    """Manages face image storage and organization"""
    
    def __init__(self, base_directory: str = "faces"):
        """
        Initialize the face storage manager
        
        Args:
            base_directory: Base directory for storing face images
        """
        self.logger = get_module_logger("FaceStorage")
        self.base_directory = Path(base_directory)
        self.user_data_file = self.base_directory / "users.json"
        
        # Ensure base directory exists
        self._initialize_storage()
        
        # Load existing user data
        self.user_data = self._load_user_data()
        
        self.logger.info(f"FaceStorage initialized with base directory: {self.base_directory}")
    
    def _initialize_storage(self):
        """Initialize storage directory structure"""
        try:
            # Create base directory
            self.base_directory.mkdir(parents=True, exist_ok=True)
            
            # Create metadata directory
            metadata_dir = self.base_directory / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            self.logger.info("Storage directory structure initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {str(e)}")
            raise StorageError(f"Storage initialization failed: {str(e)}")
    
    def _load_user_data(self) -> Dict:
        """Load user data from JSON file"""
        try:
            if self.user_data_file.exists():
                with open(self.user_data_file, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded user data for {len(data)} users")
                return data
            else:
                self.logger.info("No existing user data found, starting fresh")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to load user data: {str(e)}")
            return {}
    
    def _save_user_data(self):
        """Save user data to JSON file"""
        try:
            # Create backup if file exists
            if self.user_data_file.exists():
                backup_file = self.user_data_file.with_suffix('.json.bak')
                shutil.copy2(self.user_data_file, backup_file)
            
            # Save current data
            with open(self.user_data_file, 'w') as f:
                json.dump(self.user_data, f, indent=2, default=str)
            
            self.logger.debug("User data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save user data: {str(e)}")
            raise StorageError(f"Failed to save user data: {str(e)}")
    
    def generate_user_id(self, name: str) -> str:
        """
        Generate a unique user ID
        
        Args:
            name: User name
            
        Returns:
            Unique user ID
        """
        # Clean name for use in ID
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_').lower()
        
        # Generate base ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_id = f"{clean_name}_{timestamp}"
        
        # Ensure uniqueness
        user_id = base_id
        counter = 1
        while self.user_exists(user_id):
            user_id = f"{base_id}_{counter}"
            counter += 1
        
        self.logger.info(f"Generated user ID: {user_id}")
        return user_id
    
    def user_exists(self, user_id: str) -> bool:
        """
        Check if a user exists
        
        Args:
            user_id: User ID to check
            
        Returns:
            True if user exists
        """
        return user_id in self.user_data
    
    def create_user(self, name: str, user_id: str = None) -> str:
        """
        Create a new user and their storage directory
        
        Args:
            name: User name
            user_id: Optional custom user ID
            
        Returns:
            Created user ID
        """
        try:
            # Generate or validate user ID
            if user_id is None:
                user_id = self.generate_user_id(name)
            elif self.user_exists(user_id):
                raise ValidationError(f"User ID '{user_id}' already exists")
            
            # Create user directory
            user_dir = self.base_directory / user_id
            user_dir.mkdir(exist_ok=True)
            
            # Create user metadata
            user_metadata = {
                'name': name,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'image_count': 0,
                'images': []
            }
            
            # Add to user data
            self.user_data[user_id] = user_metadata
            
            # Save user data
            self._save_user_data()
            
            self.logger.info(f"Created new user: {name} (ID: {user_id})")
            return user_id
            
        except Exception as e:
            self.logger.error(f"Failed to create user: {str(e)}")
            raise StorageError(f"User creation failed: {str(e)}")
    
    def save_face_image(self, user_id: str, image: np.ndarray, image_index: int = None) -> str:
        """
        Save a face image for a user
        
        Args:
            user_id: User ID
            image: Face image to save
            image_index: Optional image index (auto-generated if None)
            
        Returns:
            Saved image filename
        """
        try:
            if not self.user_exists(user_id):
                raise ValidationError(f"User '{user_id}' does not exist")
            
            # Get user directory
            user_dir = self.base_directory / user_id
            
            # Generate image filename
            if image_index is None:
                image_index = len(self.user_data[user_id]['images']) + 1
            
            filename = f"img_{image_index:02d}.jpg"
            filepath = user_dir / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), image)
            if not success:
                raise StorageError(f"Failed to save image to {filepath}")
            
            # Update user metadata
            image_metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'index': image_index,
                'saved_at': datetime.now().isoformat(),
                'file_size': filepath.stat().st_size
            }
            
            self.user_data[user_id]['images'].append(image_metadata)
            self.user_data[user_id]['image_count'] = len(self.user_data[user_id]['images'])
            self.user_data[user_id]['updated_at'] = datetime.now().isoformat()
            
            # Save user data
            self._save_user_data()
            
            self.logger.info(f"Saved face image for user {user_id}: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save face image: {str(e)}")
            raise StorageError(f"Face image save failed: {str(e)}")
    
    def save_multiple_face_images(self, user_id: str, images: List[np.ndarray]) -> List[str]:
        """
        Save multiple face images for a user
        
        Args:
            user_id: User ID
            images: List of face images to save
            
        Returns:
            List of saved image filenames
        """
        saved_filenames = []
        
        try:
            for i, image in enumerate(images):
                try:
                    filename = self.save_face_image(user_id, image)
                    saved_filenames.append(filename)
                except Exception as e:
                    self.logger.error(f"Failed to save image {i} for user {user_id}: {str(e)}")
                    continue
            
            self.logger.info(f"Saved {len(saved_filenames)} out of {len(images)} images for user {user_id}")
            return saved_filenames
            
        except Exception as e:
            self.logger.error(f"Failed to save multiple images: {str(e)}")
            raise StorageError(f"Multiple image save failed: {str(e)}")
    
    def load_user_images(self, user_id: str) -> List[np.ndarray]:
        """
        Load all face images for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of loaded face images
        """
        try:
            if not self.user_exists(user_id):
                raise ValidationError(f"User '{user_id}' does not exist")
            
            images = []
            user_images = self.user_data[user_id]['images']
            
            for img_metadata in user_images:
                filepath = Path(img_metadata['filepath'])
                if filepath.exists():
                    image = cv2.imread(str(filepath))
                    if image is not None:
                        images.append(image)
                    else:
                        self.logger.warning(f"Could not load image: {filepath}")
                else:
                    self.logger.warning(f"Image file not found: {filepath}")
            
            self.logger.info(f"Loaded {len(images)} images for user {user_id}")
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to load user images: {str(e)}")
            raise StorageError(f"Image loading failed: {str(e)}")
    
    def get_user_info(self, user_id: str) -> Dict:
        """
        Get user information
        
        Args:
            user_id: User ID
            
        Returns:
            User information dictionary
        """
        if not self.user_exists(user_id):
            raise ValidationError(f"User '{user_id}' does not exist")
        
        return self.user_data[user_id].copy()
    
    def list_users(self) -> List[Dict]:
        """
        List all users
        
        Returns:
            List of user information dictionaries
        """
        return [
            {
                'user_id': user_id,
                'name': data['name'],
                'image_count': data['image_count'],
                'created_at': data['created_at']
            }
            for user_id, data in self.user_data.items()
        ]
    
    def delete_user(self, user_id: str, confirm: bool = False) -> bool:
        """
        Delete a user and all their data
        
        Args:
            user_id: User ID to delete
            confirm: Confirmation flag for safety
            
        Returns:
            True if deletion successful
        """
        try:
            if not confirm:
                raise ValidationError("Deletion requires confirmation")
            
            if not self.user_exists(user_id):
                raise ValidationError(f"User '{user_id}' does not exist")
            
            # Delete user directory
            user_dir = self.base_directory / user_id
            if user_dir.exists():
                shutil.rmtree(user_dir)
            
            # Remove from user data
            del self.user_data[user_id]
            
            # Save updated user data
            self._save_user_data()
            
            self.logger.info(f"Deleted user: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete user: {str(e)}")
            raise StorageError(f"User deletion failed: {str(e)}")
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        """
        Clean up orphaned files and directories
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {'deleted_dirs': 0, 'deleted_files': 0, 'errors': 0}
        
        try:
            # Get all directories in base directory
            for item in self.base_directory.iterdir():
                if item.is_dir() and item.name not in ['metadata'] and item.name not in self.user_data:
                    try:
                        shutil.rmtree(item)
                        stats['deleted_dirs'] += 1
                        self.logger.info(f"Deleted orphaned directory: {item}")
                    except Exception as e:
                        stats['errors'] += 1
                        self.logger.error(f"Failed to delete orphaned directory {item}: {str(e)}")
            
            # Clean up broken image references
            for user_id, user_data in self.user_data.items():
                valid_images = []
                for img_metadata in user_data['images']:
                    filepath = Path(img_metadata['filepath'])
                    if filepath.exists():
                        valid_images.append(img_metadata)
                    else:
                        stats['deleted_files'] += 1
                        self.logger.info(f"Removed broken image reference: {filepath}")
                
                if len(valid_images) != len(user_data['images']):
                    self.user_data[user_id]['images'] = valid_images
                    self.user_data[user_id]['image_count'] = len(valid_images)
            
            # Save updated data
            self._save_user_data()
            
            self.logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            stats['errors'] += 1
            return stats
    
    def validate_storage_integrity(self) -> Dict[str, any]:
        """
        Validate storage integrity
        
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'issues': [],
            'user_count': len(self.user_data),
            'total_images': 0,
            'missing_files': 0,
            'orphaned_dirs': 0
        }
        
        try:
            # Check each user
            for user_id, user_data in self.user_data.items():
                user_dir = self.base_directory / user_id
                
                # Check if user directory exists
                if not user_dir.exists():
                    report['issues'].append(f"Missing directory for user: {user_id}")
                    report['valid'] = False
                    continue
                
                # Check each image
                for img_metadata in user_data['images']:
                    report['total_images'] += 1
                    filepath = Path(img_metadata['filepath'])
                    
                    if not filepath.exists():
                        report['missing_files'] += 1
                        report['issues'].append(f"Missing image file: {filepath}")
                        report['valid'] = False
            
            # Check for orphaned directories
            for item in self.base_directory.iterdir():
                if item.is_dir() and item.name not in ['metadata'] and item.name not in self.user_data:
                    report['orphaned_dirs'] += 1
                    report['issues'].append(f"Orphaned directory: {item}")
            
            self.logger.info(f"Storage validation completed: {report}")
            return report
            
        except Exception as e:
            self.logger.error(f"Storage validation failed: {str(e)}")
            report['valid'] = False
            report['issues'].append(f"Validation error: {str(e)}")
            return report
    
    def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage statistics
        
        Returns:
            Storage statistics dictionary
        """
        try:
            stats = {
                'total_users': len(self.user_data),
                'total_images': sum(user['image_count'] for user in self.user_data.values()),
                'storage_path': str(self.base_directory),
                'last_updated': max(
                    [datetime.fromisoformat(user['updated_at']) for user in self.user_data.values()],
                    default=datetime.now()
                ).isoformat() if self.user_data else None
            }
            
            # Calculate total storage size
            total_size = 0
            for user_data in self.user_data.values():
                for img_metadata in user_data['images']:
                    filepath = Path(img_metadata['filepath'])
                    if filepath.exists():
                        total_size += filepath.stat().st_size
            
            stats['total_size_bytes'] = total_size
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    def export_user_data(self, output_file: str = None) -> str:
        """
        Export user data to JSON file
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Export file path
        """
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"faceattend_users_export_{timestamp}.json"
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'version': '1.0',
                'users': self.user_data,
                'stats': self.get_storage_stats()
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"User data exported to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export user data: {str(e)}")
            raise StorageError(f"Export failed: {str(e)}")

# Utility functions
def test_face_storage():
    """Test face storage functionality"""
    import tempfile
    
    # Create temporary storage for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FaceStorage(temp_dir)
        
        print("Testing face storage...")
        
        # Test user creation
        user_id = storage.create_user("Test User")
        print(f"Created user: {user_id}")
        
        # Test image storage
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        filename = storage.save_face_image(user_id, test_image)
        print(f"Saved image: {filename}")
        
        # Test image loading
        images = storage.load_user_images(user_id)
        print(f"Loaded {len(images)} images")
        
        # Test storage stats
        stats = storage.get_storage_stats()
        print(f"Storage stats: {stats}")
        
        # Test validation
        report = storage.validate_storage_integrity()
        print(f"Validation report: {report}")
        
        print("Face storage test completed successfully!")

if __name__ == "__main__":
    test_face_storage() 