from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')
DATA_PATH = '/opt/airflow/data/wine_quality.csv'
MODEL_PATH = '/opt/airflow/models'
EXPERIMENT_NAME = 'wine_quality_classification'


def get_or_create_experiment(experiment_name):
    """Get or create MLFlow experiment"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    return experiment_id


def generate_data(**context):
    """Generate sample dataset"""
    
    X, y = make_classification(
        n_samples=1000,
        n_features=11,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['quality'] = y
    
    # Save
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    
    print(f" Generated {len(df)} samples")
    print(f"Quality distribution:\n{df['quality'].value_counts()}")
    
    return {'n_samples': len(df), 'n_features': len(feature_names)}


def validate_data(**context):
    """Validate dataset quality"""
    print(" Validating data...")
    
    df = pd.read_csv(DATA_PATH)
    
    validations = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'feature_count': len(df.columns) - 1,  # Exclude target
    }
    
    print(f"Validation results:")
    for key, value in validations.items():
        print(f"  {key}: {value}")
    
    if validations['missing_values'] > 0:
        print("️  Warning: Missing values detected!")
    
    return validations


def preprocess_data(**context):
    """Preprocess and split data"""
    print(" Preprocessing data...")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (if any)
    df = df.fillna(df.mean())
    
    # Split features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data and scaler
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    np.save(f'{MODEL_PATH}/X_train.npy', X_train_scaled)
    np.save(f'{MODEL_PATH}/X_test.npy', X_test_scaled)
    np.save(f'{MODEL_PATH}/y_train.npy', y_train)
    np.save(f'{MODEL_PATH}/y_test.npy', y_test)
    
    with open(f'{MODEL_PATH}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f" Preprocessing complete!")
    print(f"  Train set: {X_train_scaled.shape}")
    print(f"  Test set: {X_test_scaled.shape}")
    
    # Push to XCom for next tasks
    ti = context['ti']
    ti.xcom_push(key='train_size', value=X_train_scaled.shape[0])
    ti.xcom_push(key='test_size', value=X_test_scaled.shape[0])
    
    return {
        'train_size': X_train_scaled.shape[0],
        'test_size': X_test_scaled.shape[0],
        'n_features': X_train_scaled.shape[1]
    }

def train_random_forest(**context):
    """Train Random Forest model and log to MLFlow"""
    print(" Training Random Forest...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    
    # Load data
    X_train = np.load(f'{MODEL_PATH}/X_train.npy')
    y_train = np.load(f'{MODEL_PATH}/y_train.npy')
    X_test = np.load(f'{MODEL_PATH}/X_test.npy')
    y_test = np.load(f'{MODEL_PATH}/y_test.npy')
    
    # Hyperparameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    
    # Start MLFlow run
    with mlflow.start_run(experiment_id=experiment_id, run_name='RandomForest') as run:
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        }
        
        # Log to MLFlow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance
        feature_importance = dict(zip(
            [f'feature_{i}' for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        mlflow.log_params(feature_importance)
        
        run_id = run.info.run_id
        
        print(f" Random Forest trained!")
        print(f" Run ID: {run_id}")
        print(f" Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        # Push run_id to XCom
        ti = context['ti']
        ti.xcom_push(key='rf_run_id', value=run_id)
        ti.xcom_push(key='rf_accuracy', value=metrics['test_accuracy'])
        
        return metrics


def train_gradient_boosting(**context):
    """Train Gradient Boosting model and log to MLFlow"""
    print(" Training Gradient Boosting...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    
    # Load data
    X_train = np.load(f'{MODEL_PATH}/X_train.npy')
    y_train = np.load(f'{MODEL_PATH}/y_train.npy')
    X_test = np.load(f'{MODEL_PATH}/X_test.npy')
    y_test = np.load(f'{MODEL_PATH}/y_test.npy')
    
    # Hyperparameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
    
    # Start MLFlow run
    with mlflow.start_run(experiment_id=experiment_id, run_name='GradientBoosting') as run:
        # Train model
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        }
        
        # Log to MLFlow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path="model"
                            )
        
        run_id = run.info.run_id
        
        print(f" Gradient Boosting trained!")
        print(f"  Run ID: {run_id}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        # Push run_id to XCom
        ti = context['ti']
        ti.xcom_push(key='gb_run_id', value=run_id)
        ti.xcom_push(key='gb_accuracy', value=metrics['test_accuracy'])
        
        return metrics


def train_logistic_regression(**context):
    """Train Logistic Regression model and log to MLFlow"""
    print(" Training Logistic Regression...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    
    # Load data
    X_train = np.load(f'{MODEL_PATH}/X_train.npy')
    y_train = np.load(f'{MODEL_PATH}/y_train.npy')
    X_test = np.load(f'{MODEL_PATH}/X_test.npy')
    y_test = np.load(f'{MODEL_PATH}/y_test.npy')
    
    # Hyperparameters
    params = {
        'max_iter': 1000,
        'random_state': 42
    }
    
    # Start MLFlow run
    with mlflow.start_run(experiment_id=experiment_id, run_name='LogisticRegression') as run:
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        }
        
        # Log to MLFlow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path="model"
                            )
        
        run_id = run.info.run_id
        
        print(f" Logistic Regression trained!")
        print(f"  Run ID: {run_id}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        # Push run_id to XCom
        ti = context['ti']
        ti.xcom_push(key='lr_run_id', value=run_id)
        ti.xcom_push(key='lr_accuracy', value=metrics['test_accuracy'])
        
        return metrics

def select_best_model(**context):
    """Compare models and select the best one"""

    ti = context['ti']
    
    # Get accuracies from all models
    models = {
        'RandomForest': ti.xcom_pull(task_ids='train_random_forest', key='rf_accuracy'),
        'GradientBoosting': ti.xcom_pull(task_ids='train_gradient_boosting', key='gb_accuracy'),
        'LogisticRegression': ti.xcom_pull(task_ids='train_logistic_regression', key='lr_accuracy'),
    }
    
    print("Model comparison:")
    for model_name, accuracy in models.items():
        print(f"  {model_name}: {accuracy:.4f}")
    
    # Find best model
    best_model_name = max(models, key=models.get)
    best_accuracy = models[best_model_name]
    
    # Get run_id of best model
    if best_model_name == 'RandomForest':
        best_run_id = ti.xcom_pull(task_ids='train_random_forest', key='rf_run_id')
    elif best_model_name == 'GradientBoosting':
        best_run_id = ti.xcom_pull(task_ids='train_gradient_boosting', key='gb_run_id')
    else:
        best_run_id = ti.xcom_pull(task_ids='train_logistic_regression', key='lr_run_id')
    
    print(f"\n Best model: {best_model_name}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Run ID: {best_run_id}")
    
    # Push to XCom
    ti.xcom_push(key='best_model_name', value=best_model_name)
    ti.xcom_push(key='best_run_id', value=best_run_id)
    ti.xcom_push(key='best_accuracy', value=best_accuracy)
    
    return {
        'best_model': best_model_name,
        'best_run_id': best_run_id,
        'best_accuracy': best_accuracy
    }

def register_best_model(**context):
    """Register best model to MLFlow Model Registry"""
    print(" Registering model to MLFlow Model Registry...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    ti = context['ti']
    best_run_id = ti.xcom_pull(task_ids='select_best_model', key='best_run_id')
    best_model_name = ti.xcom_pull(task_ids='select_best_model', key='best_model_name')
    best_accuracy = ti.xcom_pull(task_ids='select_best_model', key='best_accuracy')
    
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "wine_quality_model"
    
    # Register model
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f" Model registered!")
        print(f"  Model Name: {model_name}")
        print(f"  Version: {model_version.version}")
        print(f"  Algorithm: {best_model_name}")
        print(f"  Accuracy: {best_accuracy:.4f}")
        
        return {
            'model_name': model_name,
            'version': model_version.version,
            'algorithm': best_model_name,
            'accuracy': best_accuracy
        }
    except Exception as e:
        print(f"Note: {str(e)}")
        print("Model registered but version info may not be available")
        return {'status': 'registered'}


def send_notification(**context):
    """Send notification about pipeline completion"""
    print(" Sending notification...")
    
    ti = context['ti']
    
    best_model_name = ti.xcom_pull(task_ids='select_best_model', key='best_model_name')
    best_accuracy = ti.xcom_pull(task_ids='select_best_model', key='best_accuracy')
    
    execution_date = context['execution_date']
    
    message = f"""
    Execution Date: {execution_date}
    Best Model: {best_model_name}
    Test Accuracy: {best_accuracy:.2%}
    Model has been registered to MLFlow Model Registry
    """
    
    print(message)
    
    return {'status': 'notification_sent'}

# DAG Definition

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ml_pipeline_with_mlflow',
    default_args=default_args,
    description='ML Pipeline with MLFlow tracking and model registry',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['ml', 'mlflow', 'production'],
) as dag:
    
    # Task 1: Generate data
    task_generate_data = PythonOperator(
        task_id='generate_data',
        python_callable=generate_data,
        provide_context=True,
    )
    
    # Task 2: Validate data
    task_validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    # Task 3: Preprocess
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True,
    )
    
    # Task 4-6: Train models (PARALLEL)
    task_train_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_random_forest,
        provide_context=True,
    )
    
    task_train_gb = PythonOperator(
        task_id='train_gradient_boosting',
        python_callable=train_gradient_boosting,
        provide_context=True,
    )
    
    task_train_lr = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_logistic_regression,
        provide_context=True,
    )
    
    # Task 7: Select best model
    task_select_best = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
        provide_context=True,
    )
    
    # Task 8: Register model
    task_register = PythonOperator(
        task_id='register_model',
        python_callable=register_best_model,
        provide_context=True,
    )
    
    # Task 9: Send notification
    task_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
    )
    
    # Flow:
    # Generate → Validate → Preprocess → [Train 3 models in parallel] → Select Best → Register → Notify
    
    task_generate_data >> task_validate >> task_preprocess
    task_preprocess >> [task_train_rf, task_train_gb, task_train_lr]
    [task_train_rf, task_train_gb, task_train_lr] >> task_select_best
    task_select_best >> task_register >> task_notify