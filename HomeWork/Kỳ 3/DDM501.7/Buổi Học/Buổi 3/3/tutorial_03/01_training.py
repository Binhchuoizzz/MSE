"""
    This script is to demonstrate how to train a model and log it to mlflow.
    In this script, we set the tags, params, metrics, model and signature for the run.
    Let check the mlflow UI to see the results.
"""
import os
import random
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.data import get_data


MODEL_NAME = "breast_cancer-predictor"
EXPERIMENT_NAME = "breast_cancer-experiment-demo"

load_dotenv(dotenv_path=".env")
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment(EXPERIMENT_NAME)

# get data
X_train, X_test, y_train, y_test = get_data()
# initialize experiment

with mlflow.start_run(run_name="breast_cancer_training_v1"):
    mlflow.set_tags(
        {
            'owner': 'dsteam',
            'algorithm': 'random_forest',
            'dataset': 'breast_cancer',
            'version': '1',
        }
    )

    mlflow.log_params(
        {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 100
        }
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )
    print(f"Model registered successfully: {MODEL_NAME}")
    print(f"Version 1: accuracy={acc}")

with mlflow.start_run(run_name="breast_cancer_training_v2"):
    mlflow.set_tags(
        {
            'owner': 'dsteam',
            'algorithm': 'random_forest',
            'dataset': 'breast_cancer',
            'version': '1',
        }
    )
    mlflow.log_params(
        {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 200,
        }
    )
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Create signature
    signature = infer_signature(X_test, predictions)
    input_example = X_test[:5] # Let get some sample data

    # Let log the metrics, model and signature
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,  # This is the signature of the model
        input_example=input_example,  # This is the input example of the model
        registered_model_name=MODEL_NAME,
    )
    print(f"Model registered successfully: {MODEL_NAME}")
    print(f"Version 2: accuracy={acc}")