import os
import numpy as np
import mlflow
import mlflow.pyfunc
from utils.data import get_data
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
MODEL_NAME = "breast_cancer-predictor"
tag = "dev"
version = "5"

def get_model_for_prediction(model_name: str, traffic_split=0.9):
    """
    This function is to get the model for prediction.
    Args:
        model_name: str
        traffic_split: float
    Returns:
        model: mlflow.pyfunc.Model
        alias: str
    """
    if np.random.random() < traffic_split:
        alias = "prod"
    else:
        alias = "dev"
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model, alias

if __name__ == "__main__":
    print('Simulation....')
    for i in range(10):
        model, alias = get_model_for_prediction(MODEL_NAME, 0.9)
        print(f"Model {alias} is used for prediction")