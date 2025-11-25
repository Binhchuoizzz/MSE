import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import json
import os

from utils import (
    split_data,
    scale_data,
    calculate_metrics,
    print_metrics
)


class Experiment:
    def __init__(self, name, output_dir='results/experiments'):
        self.name = name
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.id = f'{name}_{self.timestamp}'

        self.experiment_dir = os.path.join(output_dir, self.id)
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def run(self, model, model_name, params):
        print(f'Running {model_name} experiment...')

        data = load_breast_cancer()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        model.fit(X_train_scaled, y_train)
        metrics = calculate_metrics(y_test, model.predict(X_test_scaled))
        print_metrics(metrics, model_name)

        model_path = os.path.join(self.experiment_dir, f'{model_name}.pkl')
        joblib.dump(model, model_path)
        print(f'Model saved to {model_path}')

        scaler_path = os.path.join(self.experiment_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f'Scaler saved to {scaler_path}')

        metadata = {
            'experiment_id': self.id,
            'experiment_name': self.name,
            'timestamp': self.timestamp,
            'model_name': model_name,
            'metrics': {k: v for k, v in metrics.items()},
            'model_path': model_path,
            'scaler_path': scaler_path,
        }
        metadata_path = os.path.join(self.experiment_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return metrics, model
    

def fit_rf_model(n_estimators=100):
    experiment = Experiment('random_forest_experiment')
    params = {
        'n_estimators': n_estimators,
        'max_depth': None,
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    metrics, trained_model = experiment.run(model, 'Random Forest', params)
    return metrics, trained_model


if __name__ == '__main__':
    print('Running Random Forest experiment...')
    metrics_1, model_01 = fit_rf_model(n_estimators=100)
    time.sleep(1)
    metrics_2, model_02 = fit_rf_model(n_estimators=200)

    print('Comparing models...')
    print(f'Model 01: n_est=100 Accuracy: {metrics_1["accuracy"]}')
    print(f'Model 02: n_est=200 Accuracy: {metrics_2["accuracy"]}')