import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed
np.random.seed(42)

# Generate synthetic wine quality dataset
n_samples = 1000

# Generate features
X, y = make_classification(
    n_samples=n_samples,
    n_features=11,
    n_informative=8,
    n_redundant=2,
    n_classes=3,  # Quality: Low (0), Medium (1), High (2)
    random_state=42
)

# Create DataFrame
feature_names = [
    'fixed_acidity',
    'volatile_acidity', 
    'citric_acid',
    'residual_sugar',
    'chlorides',
    'free_sulfur_dioxide',
    'total_sulfur_dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

df = pd.DataFrame(X, columns=feature_names)
df['quality'] = y  # 0, 1, 2 (Low, Medium, High quality)

# Scale features to realistic ranges
df['fixed_acidity'] = (df['fixed_acidity'] - df['fixed_acidity'].min()) / \
                       (df['fixed_acidity'].max() - df['fixed_acidity'].min()) * 10 + 4
df['volatile_acidity'] = (df['volatile_acidity'] - df['volatile_acidity'].min()) / \
                          (df['volatile_acidity'].max() - df['volatile_acidity'].min()) * 1.2 + 0.1
df['citric_acid'] = (df['citric_acid'] - df['citric_acid'].min()) / \
                     (df['citric_acid'].max() - df['citric_acid'].min()) * 0.8
df['residual_sugar'] = (df['residual_sugar'] - df['residual_sugar'].min()) / \
                        (df['residual_sugar'].max() - df['residual_sugar'].min()) * 15 + 0.5
df['chlorides'] = (df['chlorides'] - df['chlorides'].min()) / \
                   (df['chlorides'].max() - df['chlorides'].min()) * 0.5 + 0.01
df['free_sulfur_dioxide'] = (df['free_sulfur_dioxide'] - df['free_sulfur_dioxide'].min()) / \
                             (df['free_sulfur_dioxide'].max() - df['free_sulfur_dioxide'].min()) * 60 + 1
df['total_sulfur_dioxide'] = (df['total_sulfur_dioxide'] - df['total_sulfur_dioxide'].min()) / \
                              (df['total_sulfur_dioxide'].max() - df['total_sulfur_dioxide'].min()) * 250 + 6
df['density'] = (df['density'] - df['density'].min()) / \
                 (df['density'].max() - df['density'].min()) * 0.02 + 0.99
df['pH'] = (df['pH'] - df['pH'].min()) / \
            (df['pH'].max() - df['pH'].min()) * 1.5 + 2.7
df['sulphates'] = (df['sulphates'] - df['sulphates'].min()) / \
                   (df['sulphates'].max() - df['sulphates'].min()) * 1.5 + 0.3
df['alcohol'] = (df['alcohol'] - df['alcohol'].min()) / \
                 (df['alcohol'].max() - df['alcohol'].min()) * 7 + 8

# Save dataset
DATA_PATH = os.getenv('AIRFLOW_PROJ_DIR', '.') + '/data/wine_quality.csv'
print(f"Saving dataset to: {DATA_PATH}")
df.to_csv(DATA_PATH, index=False)
print(f"Dataset created: {len(df)} samples")
print(f"Features: {list(df.columns)}")
print(f"\nQuality distribution:")
print(df['quality'].value_counts().sort_index())
print(f"\nSample data:")
print(df.head())