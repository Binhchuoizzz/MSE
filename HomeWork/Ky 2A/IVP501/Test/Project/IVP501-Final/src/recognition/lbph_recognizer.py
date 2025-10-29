"""
LBPH (Local Binary Pattern Histogram) face recognition module for FaceAttend application
Implements face recognition using OpenCV's LBPH recognizer
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from src.utils.logger import get_module_logger
from src.utils.exceptions import FaceRecognitionError, ValidationError
from src.storage.face_storage import FaceStorage
from src.recognition.image_processor import ImageProcessor

class LBPHRecognizer:
    """LBPH Face Recognition Engine"""
    
    def __init__(self, 
                 model_path: str = "trainer.yml",
                 confidence_threshold: float = 100.0,
                 face_storage: FaceStorage = None):
        """
        Initialize the LBPH recognizer
        
        Args:
            model_path: Path to save/load the trained model
            confidence_threshold: Recognition confidence threshold (lower is better)
            face_storage: Face storage instance for loading training data
        """
        self.logger = get_module_logger("LBPHRecognizer")
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,           # Radius of LBP pattern
            neighbors=8,        # Number of neighbors
            grid_x=8,          # Number of cells in horizontal direction
            grid_y=8           # Number of cells in vertical direction
        )
        
        # Face storage and image processor
        self.face_storage = face_storage or FaceStorage()
        self.image_processor = ImageProcessor()
        
        # User label mapping
        self.user_labels = {}  # user_id -> numeric_label
        self.label_users = {}  # numeric_label -> user_id
        self.next_label = 1
        
        # Model metadata
        self.model_metadata = {
            'created_at': None,
            'updated_at': None,
            'training_images': 0,
            'users_count': 0,
            'version': '1.0'
        }
        
        # Load existing model if available
        self._load_model_if_exists()
        
        self.logger.info(f"LBPHRecognizer initialized with confidence threshold: {confidence_threshold}")
    
    def _load_model_if_exists(self):
        """Load existing model and metadata if available"""
        try:
            if self.model_path.exists():
                self.recognizer.read(str(self.model_path))
                
                # Load metadata
                metadata_path = self.model_path.with_suffix('.metadata.pkl')
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        saved_data = pickle.load(f)
                        self.user_labels = saved_data.get('user_labels', {})
                        self.label_users = saved_data.get('label_users', {})
                        self.next_label = saved_data.get('next_label', 1)
                        self.model_metadata = saved_data.get('metadata', self.model_metadata)
                
                self.logger.info(f"Loaded existing model with {len(self.user_labels)} users")
            else:
                self.logger.info("No existing model found, starting fresh")
                
        except Exception as e:
            self.logger.warning(f"Failed to load existing model: {str(e)}")
            self._reset_model()
    
    def _reset_model(self):
        """Reset the model to initial state"""
        self.user_labels.clear()
        self.label_users.clear()
        self.next_label = 1
        self.model_metadata = {
            'created_at': None,
            'updated_at': None,
            'training_images': 0,
            'users_count': 0,
            'version': '1.0'
        }
        self.logger.info("Model reset to initial state")
    
    def _assign_user_label(self, user_id: str) -> int:
        """
        Assign a numeric label to a user
        
        Args:
            user_id: User ID
            
        Returns:
            Numeric label for the user
        """
        if user_id not in self.user_labels:
            label = self.next_label
            self.user_labels[user_id] = label
            self.label_users[label] = user_id
            self.next_label += 1
            self.logger.debug(f"Assigned label {label} to user {user_id}")
            return label
        
        return self.user_labels[user_id]
    
    def _load_training_data(self, user_ids: List[str] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load training data from face storage
        
        Args:
            user_ids: Optional list of specific user IDs to load (loads all if None)
            
        Returns:
            Tuple of (face_images, labels)
        """
        faces = []
        labels = []
        
        try:
            # Get user list
            if user_ids is None:
                users = self.face_storage.list_users()
                user_ids = [user['user_id'] for user in users]
            
            self.logger.info(f"Loading training data for {len(user_ids)} users")
            
            for user_id in user_ids:
                try:
                    # Get user label
                    label = self._assign_user_label(user_id)
                    
                    # Load user images
                    user_images = self.face_storage.load_user_images(user_id)
                    
                    if not user_images:
                        self.logger.warning(f"No images found for user {user_id}")
                        continue
                    
                    # Process images
                    for image in user_images:
                        if image is not None:
                            # Preprocess image
                            processed_image = self.image_processor.preprocess_for_recognition(image)
                            if processed_image is not None:
                                faces.append(processed_image)
                                labels.append(label)
                    
                    self.logger.debug(f"Loaded {len(user_images)} images for user {user_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load data for user {user_id}: {str(e)}")
                    continue
            
            self.logger.info(f"Loaded {len(faces)} training images from {len(user_ids)} users")
            return faces, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {str(e)}")
            raise FaceRecognitionError(f"Training data loading failed: {str(e)}")
    
    def train_model(self, user_ids: List[str] = None, save_model: bool = True) -> Dict[str, Any]:
        """
        Train the LBPH recognition model
        
        Args:
            user_ids: Optional list of specific user IDs to train on
            save_model: Whether to save the trained model
            
        Returns:
            Training results and statistics
        """
        try:
            self.logger.info("Starting model training...")
            
            # Load training data
            faces, labels = self._load_training_data(user_ids)
            
            if len(faces) == 0:
                raise FaceRecognitionError("No training data available")
            
            if len(faces) < 2:
                raise FaceRecognitionError("At least 2 face images required for training")
            
            # Convert to numpy arrays
            faces_array = np.array(faces)
            labels_array = np.array(labels)
            
            # Train the recognizer
            start_time = datetime.now()
            self.recognizer.train(faces_array, labels_array)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Update metadata
            self.model_metadata.update({
                'updated_at': datetime.now().isoformat(),
                'training_images': len(faces),
                'users_count': len(set(labels)),
                'training_time_seconds': training_time
            })
            
            if self.model_metadata['created_at'] is None:
                self.model_metadata['created_at'] = self.model_metadata['updated_at']
            
            # Save model if requested
            if save_model:
                self.save_model()
            
            training_stats = {
                'success': True,
                'training_images': len(faces),
                'users_count': len(set(labels)),
                'training_time': training_time,
                'model_path': str(self.model_path)
            }
            
            self.logger.info(f"Model training completed successfully. "
                           f"Images: {len(faces)}, Users: {len(set(labels))}, "
                           f"Time: {training_time:.2f}s")
            
            return training_stats
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise FaceRecognitionError(f"Training failed: {str(e)}")
    
    def update_model_with_new_user(self, user_id: str) -> Dict[str, Any]:
        """
        Update the model by retraining with a new user's data
        
        Args:
            user_id: ID of the new user to add
            
        Returns:
            Update results
        """
        try:
            self.logger.info(f"Updating model with new user: {user_id}")
            
            if not self.face_storage.user_exists(user_id):
                raise ValidationError(f"User {user_id} does not exist")
            
            # Retrain the entire model (LBPH doesn't support incremental training)
            return self.train_model(save_model=True)
            
        except Exception as e:
            self.logger.error(f"Failed to update model with new user: {str(e)}")
            raise FaceRecognitionError(f"Model update failed: {str(e)}")
    
    def save_model(self):
        """Save the trained model and metadata to disk"""
        try:
            # Save model
            self.recognizer.save(str(self.model_path))
            
            # Save metadata
            metadata_path = self.model_path.with_suffix('.metadata.pkl')
            metadata_to_save = {
                'user_labels': self.user_labels,
                'label_users': self.label_users,
                'next_label': self.next_label,
                'metadata': self.model_metadata
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_to_save, f)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise FaceRecognitionError(f"Model saving failed: {str(e)}")
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face in the given image
        
        Args:
            face_image: Face image to recognize
            
        Returns:
            Tuple of (user_id, confidence) or (None, confidence) if not recognized
        """
        try:
            if face_image is None or face_image.size == 0:
                return None, float('inf')
            
            # Preprocess image
            processed_image = self.image_processor.preprocess_for_recognition(face_image)
            if processed_image is None:
                return None, float('inf')
            
            # Perform recognition
            label, confidence = self.recognizer.predict(processed_image)
            
            # Check confidence threshold
            if confidence <= self.confidence_threshold:
                user_id = self.label_users.get(label, None)
                if user_id:
                    self.logger.debug(f"Recognized user {user_id} with confidence {confidence:.2f}")
                    return user_id, confidence
            
            self.logger.debug(f"Face not recognized (confidence: {confidence:.2f})")
            return None, confidence
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {str(e)}")
            return None, float('inf')
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set the recognition confidence threshold
        
        Args:
            threshold: New confidence threshold (lower is more strict)
        """
        self.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold updated to {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Model information dictionary
        """
        return {
            'model_path': str(self.model_path),
            'confidence_threshold': self.confidence_threshold,
            'users_count': len(self.user_labels),
            'user_ids': list(self.user_labels.keys()),
            'is_trained': len(self.user_labels) > 0,
            'metadata': self.model_metadata.copy()
        }
    
    def is_model_trained(self) -> bool:
        """Check if the model has been trained"""
        return len(self.user_labels) > 0 and self.model_metadata.get('training_images', 0) > 0


def test_lbph_recognizer():
    """Test function for LBPH recognizer"""
    try:
        # Initialize recognizer
        recognizer = LBPHRecognizer()
        
        # Check if model exists
        model_info = recognizer.get_model_info()
        print(f"Model trained: {model_info['is_trained']}")
        print(f"Users count: {model_info['users_count']}")
        
        if model_info['users_count'] > 0:
            print(f"Registered users: {model_info['user_ids']}")
        
        return True
        
    except Exception as e:
        print(f"LBPH Recognizer test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_lbph_recognizer() 