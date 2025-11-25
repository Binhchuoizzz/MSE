from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')

def run_simple_experiment(**context):
    """Run a simple ML experiment with MLFlow tracking"""
    print("Starting simple ML experiment...")
    
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    experiment_name = "simple_experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    # Generate sample data
    print("Generating data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Try different hyperparameters
    hyperparams_list = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 15},
    ]
    
    best_accuracy = 0
    best_run_id = None
    
    for params in hyperparams_list:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            print(f"\n Training with params: {params}")
            
            # Train model
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log to MLFlow
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Log model
            mlflow.sklearn.log_model(
                 sk_model=model,
                 artifact_path="model"
            )
            
            print(f" Run completed!")
            print(f" Run ID: {run.info.run_id}")
            print(f" Accuracy: {accuracy:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = run.info.run_id
    
    print(f"\n Best model:")
    print(f"   Run ID: {best_run_id}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    
    print(f"\n View results in MLFlow UI: {MLFLOW_TRACKING_URI}")
    
    return {
        'best_run_id': best_run_id,
        'best_accuracy': best_accuracy,
        'total_runs': len(hyperparams_list)
    }

# DAG definition
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='simple_mlflow_experiment',
    default_args=default_args,
    description='Simple MLFlow experiment to learn the basics',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'mlflow', 'tutorial'],
) as dag:
    
    task_experiment = PythonOperator(
        task_id='run_experiment',
        python_callable=run_simple_experiment,
        provide_context=True,
    )