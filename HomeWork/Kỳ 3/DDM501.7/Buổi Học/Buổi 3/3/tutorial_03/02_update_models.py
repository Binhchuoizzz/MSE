"""
    This script is to demonstrate how to update the model version and tags.
    We will use the mlflow client to update the model version and tags.
"""
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv(dotenv_path=".env")

print(f"MLFLOW_TRACKING_URI: {os.environ['MLFLOW_TRACKING_URI']}")
# initialize the mlflow client
client = MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])

# update the model version and tags
MODEL_NAME = "breast_cancer-predictor"
version = "3"
description = """
This model is trained on the breast cancer dataset and is used to predict the species of an breast cancer.
Model information:
- Algorithm: Random Forest
- n_estimators: 100
- Random state: 42
- Accuracy: 0.956
Training date: 2025-11-11
Dataset: iris
"""
client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description=description,
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key='algorithm',
    value='random_forest',
)