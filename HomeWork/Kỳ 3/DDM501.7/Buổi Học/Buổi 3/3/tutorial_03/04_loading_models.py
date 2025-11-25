"""
    This script is to demonstrate how to load the model from the alias and version.
"""
import os
from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc
from utils.data import get_data


load_dotenv(dotenv_path=".env")
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
MODEL_NAME = "breast_cancer-predictor"
tag = "dev"
version = "5"

## Load model from alias
dev_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@{tag}"
)
print(f"Loaded model from {tag} successfully")

X_test, _, _, _ = get_data()
print(dev_model.predict(X_test))

## Load model from version
dev_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{version}"
)
print(f"Loaded model from {version} successfully")

_, X_test, _, _ = get_data()
print(dev_model.predict(X_test))