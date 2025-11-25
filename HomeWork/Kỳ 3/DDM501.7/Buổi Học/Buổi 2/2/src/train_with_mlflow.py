import joblib
import os
import random
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

random.seed(42)

# Set tracking URI to project root directory
# To make sure the mlruns/models will be created in the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"file://{project_root}/mlruns")


tags = {
    'owner': 'ds_team',
    'model_type': 'RandomForestClassifier',
    'dataset': 'breast_cancer',
    'dataset_version': '1.0',
    'environment': 'development',
    'training_type': 'baseline'
}


def load_data(test_size=0.2, random_state=42):
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                test_size=test_size, random_state=random_state,
                stratify=y)
    return X_train, X_test, y_train, y_test


def run_experiment():
    mlflow.set_experiment("iris_classification/random_forest/baseline")
    with mlflow.start_run(run_name='random_forest_baseline_01'):
        # Set the tags
        mlflow.set_tags(tags)

        # Load the data
        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, y_test = load_data(test_size, random_state)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Log the scaler
        mlflow.log_param('scaler', 'StandardScaler')

        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': random_state
        }

        # Log the parameters
        mlflow.log_params(params)

        # Train the model
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log the metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path='model'
        )
        # Log the scaler
        os.makedirs(f'{project_root}/models/preprocessor', exist_ok=True)
        scaler_path = os.path.join(f'{project_root}/models/preprocessor', 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(
            scaler_path,
            artifact_path='preprocessor'
        )

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Run Name: {mlflow.active_run().info.run_name}")
        print('--------------------------------')
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print('--------------------------------')

        # Return the run ID in order to register the model later
        return mlflow.active_run().info.run_id


def register_model(run_id, model_name='brest_cancer_predictor'):
    # Set tracking URI to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")

    # Register the model
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )


def load_model_from_run(run_id):
    # Load the model from the run
    model = mlflow.sklearn.load_model(
        model_uri=f"runs:/{run_id}/model"
    )
    return model


def load_model_from_registry(model_name='brest_cancer_predictor', version=1):
    # Load the model from the registry
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/{version}"
    )
    return model


def load_scaler_from_run(run_id):
    # Download the artifact and get the local path
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessor/scaler.pkl"
    )
    
    # Load the scaler using joblib
    scaler = joblib.load(artifact_path)
    return scaler


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return accuracy, f1, precision, recall

if __name__ == '__main__':
    run_id = run_experiment()
    register_model(run_id)

    # Option 1: Load the model from the run. Uncomment this to load the model from the run.
    # model = load_model_from_run(run_id)

    # Option 2: Load the model from the registry
    model = load_model_from_registry(model_name='brest_cancer_predictor')

    # Load the scaler from the run
    scaler = load_scaler_from_run(run_id)

    # Load new test data and preprocess with the loaded scaler
    _, X_test, _, y_test = load_data(test_size=0.2, random_state=0)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate model with scaled data
    evaluate_model(model, X_test_scaled, y_test)