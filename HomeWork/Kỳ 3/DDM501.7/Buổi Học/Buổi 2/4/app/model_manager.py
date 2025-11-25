import mlflow
import mlflow.sklearn
from typing import Optional, Dict, Any
import logging
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, 
                 mlflow_uri: str = None,
                 model_name: str = None,
                 model_version: Optional[str] = None):
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", 'http://localhost:5001')
        self.model_name = model_name or os.getenv("MODEL_NAME", 'iris_classifier')
        self.model_version = model_version or os.getenv("MODEL_VERSION", 'latest')
        self.model = None
        self.model_info = {}
        logger.info(f'Initializing ModelManager with: MLFlow URI: {self.mlflow_uri}, Model Name: {self.model_name}, Model Version: {self.model_version}')

        mlflow.set_tracking_uri(self.mlflow_uri)
        logger.info(f'Mlflow URI: {self.mlflow_uri}')
    
    def load_model(self):
        model_uri = f'models:/{self.model_name}/{self.model_version}'
        logger.info(f'Loading model from: {model_uri}')

        self.model = mlflow.sklearn.load_model(model_uri)
        client = mlflow.MlflowClient(tracking_uri=self.mlflow_uri)
        version_details = client.get_model_version(
            name=self.model_name,
            version=self.model_version,
        ) 
        self.model_info = {
            'name': version_details.name,
            'version': version_details.version,
            'run_id': version_details.run_id,
            'status': version_details.status,
            'creation_timestamp': version_details.creation_timestamp,
            'last_updated_timestamp': version_details.last_updated_timestamp,
            'description': version_details.description,
            'tags': version_details.tags,
            'run_link': version_details.run_link,
        }
        logger.info(f'Model loaded successfully: {self.model_info}')
        return True
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError('Model not loaded')
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        classes = self.model.classes_.tolist()
        prob_dict = {
            cls: float(prob) for cls, prob in zip(classes, probabilities[0])
        }
        return {
            'prediction': str(predictions[0]),
            'probability': float(max(probabilities[0])),
            'all_posibilities': prob_dict,
            'model_version': self.model_version,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info

model_manager = None
def get_model_manager() -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    model_manager.load_model()
    return model_manager