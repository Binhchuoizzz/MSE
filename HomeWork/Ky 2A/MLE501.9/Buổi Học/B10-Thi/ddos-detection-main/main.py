# ================================================================================
# DDoS DETECTION SYSTEM - NETWORK INTRUSION ANALYSIS
# ================================================================================
# This system processes network traffic data to detect DDoS attacks using
# machine learning techniques with advanced preprocessing and feature engineering.
# ================================================================================

import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

# ================================================================================
# STEP 1: DATA ACQUISITION FROM KAGGLE
# ================================================================================
# Download the network intrusion dataset containing multiple CSV files with
# different types of network traffic (normal and various attack patterns)

print("🔄 STEP 1: Starting data acquisition from Kaggle...")
print("Start downloading dataset...")

dataset_path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")

print("Download successfully.")
print("Path to dataset: ", dataset_path)

def load_network_intrusion_dataset(dataset_directory):
    """
    STEP 1A: Load and combine multiple CSV files from the network intrusion dataset
    - Identifies all CSV files in the dataset directory
    - Reads each file and extracts metadata (shape, labels)
    - Combines all dataframes into a single unified dataset
    """
    file_list = os.listdir(dataset_directory)
    csv_file_list = [file for file in file_list if file.endswith(".csv")]
    print("List of csv files: ", csv_file_list)

    dataframe_list = []

    # Read each CSV file and collect metadata
    for csv_file in csv_file_list:
        file_path = os.path.join(dataset_directory, csv_file)
        print("Start reading file: ", csv_file)
        try:
            current_dataframe = pd.read_csv(file_path)
            print("Shape: ", current_dataframe.shape)
            print("Columns: ", current_dataframe.shape[1])

            # Extract label distribution for each file
            if 'Label' in current_dataframe.columns:
                label_distribution = current_dataframe['Label'].value_counts()
                print("Labels: ", list(label_distribution.index))
            dataframe_list.append(current_dataframe)
        except Exception as error:
            print("Error: ", error)

    # Combine all dataframes into single dataset
    if dataframe_list:
        combined_network_data = pd.concat(dataframe_list, ignore_index=True)
        print("Shape: ", combined_network_data.shape)
        print("Columns: ", combined_network_data.columns)

        if 'Label' in combined_network_data.columns:
            print(combined_network_data['Label'].value_counts())
        return combined_network_data
    else:
        raise ValueError('Error: can not read csv files!')

# ================================================================================
# STEP 2: OUTLIER DETECTION AND HANDLING
# ================================================================================
# Network traffic data contains extreme values that can skew model performance.
# This step detects and caps outliers using statistical methods.

def detect_and_cap_outliers(network_dataframe, detection_method='iqr', outlier_action='cap'):
    """
    STEP 2A: Detect outliers using IQR or Z-score methods
    STEP 2B: Cap extreme values to reasonable bounds instead of removing data
    STEP 2C: Provide detailed statistics on outliers found and handled
    """
    print("🔍 Starting outlier detection...")
    numeric_column_list = network_dataframe.select_dtypes(include=[np.number]).columns
    numeric_column_list = [column for column in numeric_column_list if column not in ['Label', 'Binary_Label']]
    
    for column_name in numeric_column_list:
        if detection_method == 'iqr':
            # STEP 2A1: Calculate IQR bounds for outlier detection
            first_quartile = network_dataframe[column_name].quantile(0.25)
            third_quartile = network_dataframe[column_name].quantile(0.75)
            interquartile_range = third_quartile - first_quartile
            lower_outlier_bound = first_quartile - 1.5 * interquartile_range
            upper_outlier_bound = third_quartile + 1.5 * interquartile_range
            
            # STEP 2A2: Identify outliers beyond IQR bounds
            outlier_mask = (network_dataframe[column_name] < lower_outlier_bound) | (network_dataframe[column_name] > upper_outlier_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0 and outlier_action == 'cap':
                # STEP 2B1: Cap outliers to bounds instead of removing
                network_dataframe[column_name] = np.clip(network_dataframe[column_name], lower_outlier_bound, upper_outlier_bound)
                print(f"   📊 {column_name}: Capped {outlier_count} outliers ({outlier_count/len(network_dataframe)*100:.2f}%)")
        
        elif detection_method == 'zscore':
            # STEP 2A4: Alternative Z-score method for outlier detection
            column_mean = network_dataframe[column_name].mean()
            column_std = network_dataframe[column_name].std()
            z_score_values = np.abs((network_dataframe[column_name] - column_mean) / column_std)
            outlier_mask = z_score_values > 3
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0 and outlier_action == 'cap':
                # STEP 2B2: Cap using Z-score bounds
                lower_zscore_bound = column_mean - 3 * column_std
                upper_zscore_bound = column_mean + 3 * column_std
                network_dataframe[column_name] = np.clip(network_dataframe[column_name], lower_zscore_bound, upper_zscore_bound)
                print(f"   📊 {column_name}: Capped {outlier_count} outliers ({outlier_count/len(network_dataframe)*100:.2f}%)")
    
    print(f"✅ Outlier detection complete. Shape maintained: {network_dataframe.shape[0]} rows")
    return network_dataframe

# ================================================================================
# STEP 3: NETWORK SECURITY FEATURE ENGINEERING
# ================================================================================
# Create domain-specific features that capture network attack patterns and 
# traffic characteristics important for DDoS detection.

def engineer_network_security_features(network_dataframe):
    """
    STEP 3A: Create ratio features between related network metrics
    STEP 3B: Generate rate features (bytes/duration, packets/time)
    STEP 3C: Build flow asymmetry features (forward vs backward traffic)
    STEP 3D: Add port classification features (well-known, registered, dynamic)
    STEP 3E: Compute statistical aggregation features
    """
    print("🚀 Creating network-specific features...")
    original_feature_count = network_dataframe.shape[1]
    
    column_names = network_dataframe.columns.tolist()
    
    # STEP 3A: Create ratio features between related network metrics
    feature_ratio_pairs = [
        ('Fwd Packets/s', 'Bwd Packets/s'),
        ('Flow Bytes/s', 'Flow Packets/s'),
        ('Fwd Packet Length Max', 'Fwd Packet Length Min'),
        ('Bwd Packet Length Max', 'Bwd Packet Length Min'),
        ('Total Fwd Packets', 'Total Backward Packets'),
        ('Fwd Header Length', 'Bwd Header Length'),
        ('Fwd Packet Length Mean', 'Bwd Packet Length Mean'),
        ('Subflow Fwd Bytes', 'Subflow Bwd Bytes')
    ]
    
    # STEP 3A1: Generate ratio features for traffic comparison
    for first_column, second_column in feature_ratio_pairs:
        if first_column in column_names and second_column in column_names:
            network_dataframe[f'{first_column}_to_{second_column}_ratio'] = network_dataframe[first_column] / (network_dataframe[second_column] + 1e-6)
    
    # STEP 3B: Create rate features (metrics per unit time)
    duration_column_names = [column for column in column_names if 'duration' in column.lower() or 'time' in column.lower()]
    byte_column_names = [column for column in column_names if 'byte' in column.lower()]
    
    # STEP 3B1: Generate rate features for temporal analysis
    for duration_column in duration_column_names:
        for byte_column in byte_column_names:
            if duration_column in column_names and byte_column in column_names:
                network_dataframe[f'{byte_column}_per_duration'] = network_dataframe[byte_column] / (network_dataframe[duration_column] + 1e-6)
                break
    
    # STEP 3C: Statistical aggregation features for traffic patterns
    forward_traffic_columns = [column for column in column_names if 'fwd' in column.lower() and network_dataframe[column].dtype in ['int64', 'float64']]
    backward_traffic_columns = [column for column in column_names if 'bwd' in column.lower() and network_dataframe[column].dtype in ['int64', 'float64']]
    
    # STEP 3C1: Forward traffic aggregations
    if len(forward_traffic_columns) > 2:
        network_dataframe['forward_total_activity'] = network_dataframe[forward_traffic_columns].sum(axis=1)
        network_dataframe['forward_mean_activity'] = network_dataframe[forward_traffic_columns].mean(axis=1)
    
    # STEP 3C2: Backward traffic aggregations
    if len(backward_traffic_columns) > 2:
        network_dataframe['backward_total_activity'] = network_dataframe[backward_traffic_columns].sum(axis=1)
        network_dataframe['backward_mean_activity'] = network_dataframe[backward_traffic_columns].mean(axis=1)
    
    # STEP 3C3: Flow asymmetry features (directional traffic imbalance)
    if len(forward_traffic_columns) > 0 and len(backward_traffic_columns) > 0:
        forward_traffic_sum = network_dataframe[forward_traffic_columns].sum(axis=1)
        backward_traffic_sum = network_dataframe[backward_traffic_columns].sum(axis=1)
        network_dataframe['traffic_flow_asymmetry'] = (forward_traffic_sum - backward_traffic_sum) / (forward_traffic_sum + backward_traffic_sum + 1e-6)
    
    # STEP 3D: Port-based security features
    port_column_names = [column for column in column_names if 'port' in column.lower()]
    for port_column in port_column_names:
        if port_column in column_names and network_dataframe[port_column].dtype in ['int64', 'float64']:
            # STEP 3D1: Classify ports by type (security relevance)
            network_dataframe[f'{port_column}_is_wellknown'] = (network_dataframe[port_column] <= 1023).astype(int)
            network_dataframe[f'{port_column}_is_registered'] = ((network_dataframe[port_column] > 1023) & (network_dataframe[port_column] <= 49151)).astype(int)
            network_dataframe[f'{port_column}_is_dynamic'] = (network_dataframe[port_column] > 49151).astype(int)
            break
    
    total_new_features = network_dataframe.shape[1] - original_feature_count
    print(f"✅ Created {total_new_features} network-specific features")
    return network_dataframe

# ================================================================================
# STEP 4: CORRELATION ANALYSIS AND REDUNDANCY REMOVAL
# ================================================================================
# Network features often have high correlation. This step removes redundant
# features to improve model performance and reduce computational complexity.

def remove_highly_correlated_features(feature_matrix, correlation_threshold=0.95):
    """
    STEP 4A: Calculate correlation matrix for all features
    STEP 4B: Identify feature pairs with correlation above threshold
    STEP 4C: Remove redundant features while preserving information
    STEP 4D: Provide detailed correlation analysis report
    """
    print("🔗 Analyzing feature correlations...")
    
    # STEP 4A: Calculate absolute correlation matrix
    correlation_matrix = feature_matrix.corr().abs()
    
    # STEP 4B: Create upper triangle mask to avoid duplicate pairs
    upper_triangle_matrix = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # STEP 4B1: Find highly correlated feature pairs
    highly_correlated_pairs = []
    features_to_remove = set()
    
    for column_name in upper_triangle_matrix.columns:
        correlated_feature_list = upper_triangle_matrix.index[upper_triangle_matrix[column_name] > correlation_threshold].tolist()
        if correlated_feature_list:
            for correlated_feature in correlated_feature_list:
                highly_correlated_pairs.append((column_name, correlated_feature, upper_triangle_matrix.loc[correlated_feature, column_name]))
                # STEP 4C: Keep first feature, remove second (arbitrary but consistent)
                features_to_remove.add(correlated_feature)
    
    # STEP 4C1: Remove redundant features
    features_to_remove_list = list(features_to_remove)
    reduced_feature_matrix = feature_matrix.drop(columns=features_to_remove_list)
    
    # STEP 4D: Provide correlation analysis report
    print(f"📊 Found {len(highly_correlated_pairs)} highly correlated pairs (threshold: {correlation_threshold})")
    print(f"📉 Removed {len(features_to_remove_list)} redundant features")
    print(f"📈 Features reduced from {feature_matrix.shape[1]} to {reduced_feature_matrix.shape[1]}")
    
    if len(highly_correlated_pairs) > 0:
        print("Top 5 most correlated pairs:")
        sorted_correlation_pairs = sorted(highly_correlated_pairs, key=lambda x: x[2], reverse=True)[:5]
        for feature_one, feature_two, correlation_value in sorted_correlation_pairs:
            print(f"   • {feature_one} ↔ {feature_two}: {correlation_value:.3f}")
    
    return reduced_feature_matrix

# ================================================================================
# STEP 5: COMPREHENSIVE DATA PREPROCESSING
# ================================================================================
# Main preprocessing pipeline that coordinates all data cleaning and enhancement steps.

def preprocess_network_data(network_dataframe):
    """
    STEP 5A: Clean column names and basic data formatting
    STEP 5B: Apply outlier detection and handling
    STEP 5C: Engineer network security features
    STEP 5D: Handle missing values and infinity values
    STEP 5E: Create binary attack/benign labels
    """
    print("🔄 STEP 5: Starting comprehensive data preprocessing...")
    processed_data = network_dataframe.copy()

    # STEP 5A: Clean column names (remove extra spaces)
    processed_data.columns = processed_data.columns.str.strip()

    # STEP 5B: Apply outlier detection and capping
    processed_data = detect_and_cap_outliers(processed_data, detection_method='iqr', outlier_action='cap')

    # STEP 5C: Engineer network security features
    processed_data = engineer_network_security_features(processed_data)

    # STEP 5D1: Handle infinity values (replace with NaN)
    numeric_column_list = processed_data.select_dtypes(include=[np.number]).columns
    processed_data[numeric_column_list] = processed_data[numeric_column_list].replace([np.inf, -np.inf], np.nan)

    # STEP 5D2: Handle missing values with median imputation
    for column_name in numeric_column_list:
        if processed_data[column_name].isnull().sum() > 0:
            processed_data[column_name] = processed_data[column_name].fillna(processed_data[column_name].median())

    # STEP 5D3: Handle missing values in categorical columns
    object_column_list = processed_data.select_dtypes(include=['object']).columns
    for column_name in object_column_list:
        if column_name != 'Label':
            processed_data[column_name] = processed_data[column_name].fillna(processed_data[column_name].mode()[0] if not processed_data[column_name].mode().empty else 'Unknown')

    # STEP 5E: Create binary labels for classification (0=benign, 1=attack)
    processed_data['Binary_Label'] = processed_data['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)
    print("After preprocessing:")
    print("Shape: ", processed_data.shape)
    print("Binary Label:")
    print(processed_data["Binary_Label"].value_counts())
    print("Attack Rate: ", processed_data['Binary_Label'].mean())
    return processed_data

# ================================================================================
# STEP 6: ENSEMBLE FEATURE SELECTION
# ================================================================================
# Select optimal features using multiple methods to ensure robust feature selection
# that captures different types of relationships between features and target.

def select_best_features_ensemble(preprocessed_network_data, number_of_features=30):
    """
    STEP 6A: Remove highly correlated features
    STEP 6B: Remove zero-variance features
    STEP 6C: Apply F-test statistical feature selection
    STEP 6D: Apply mutual information feature selection
    STEP 6E: Apply tree-based feature importance
    STEP 6F: Combine all methods using weighted ensemble scoring
    STEP 6G: Select top features based on ensemble scores
    """
    print("🎯 STEP 6: Starting ensemble feature selection...")

    # STEP 6A: Separate features from target variable
    feature_matrix = preprocessed_network_data.drop(['Label', 'Binary_Label'], axis=1)
    target_labels = preprocessed_network_data['Binary_Label']

    print("Initial number of features: ", feature_matrix.shape[1])

    # STEP 6A1: Remove highly correlated features first
    decorrelated_feature_matrix = remove_highly_correlated_features(feature_matrix, correlation_threshold=0.95)

    # STEP 6B: Remove features with zero variance
    variance_threshold_selector = VarianceThreshold(threshold=0)
    variance_threshold_selector.fit(decorrelated_feature_matrix)
    selected_variance_columns = decorrelated_feature_matrix.columns[variance_threshold_selector.get_support()]
    print("After removing zero variance: ", len(selected_variance_columns))

    print("🧠 Applying ensemble feature selection...")
    
    # STEP 6C: F-test statistical significance method
    f_test_selector = SelectKBest(score_func=f_classif, k=min(number_of_features, len(selected_variance_columns)))
    f_test_selector.fit(decorrelated_feature_matrix[selected_variance_columns], target_labels)
    f_test_scores = f_test_selector.scores_

    # STEP 6D: Mutual information method (captures non-linear relationships)
    mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=min(number_of_features, len(selected_variance_columns)))
    mutual_info_selector.fit(decorrelated_feature_matrix[selected_variance_columns], target_labels)
    mutual_info_scores = mutual_info_selector.scores_

    # STEP 6E: Tree-based feature importance method
    random_forest_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    random_forest_selector.fit(decorrelated_feature_matrix[selected_variance_columns], target_labels)
    tree_importance_scores = random_forest_selector.feature_importances_

    # STEP 6F: Normalize all scores to 0-1 range for fair combination
    normalized_f_test_scores = (f_test_scores - f_test_scores.min()) / (f_test_scores.max() - f_test_scores.min() + 1e-6)
    normalized_mutual_info_scores = (mutual_info_scores - mutual_info_scores.min()) / (mutual_info_scores.max() - mutual_info_scores.min() + 1e-6)
    normalized_tree_scores = (tree_importance_scores - tree_importance_scores.min()) / (tree_importance_scores.max() - tree_importance_scores.min() + 1e-6)

    # STEP 6F1: Create weighted ensemble score (F-test 40%, MI 30%, Tree 30%)
    ensemble_feature_scores = 0.4 * normalized_f_test_scores + 0.3 * normalized_mutual_info_scores + 0.3 * normalized_tree_scores

    # STEP 6G: Select top features based on ensemble scores
    top_feature_indices = np.argsort(ensemble_feature_scores)[-number_of_features:]
    selected_feature_names = selected_variance_columns[top_feature_indices].tolist()

    print("Final chosen features: ", len(selected_feature_names))

    # STEP 6G1: Display top features by importance
    print("🏆 Top 10 features by ensemble scoring:")
    feature_importance_ranking = list(zip(selected_feature_names, ensemble_feature_scores[top_feature_indices]))
    feature_importance_ranking.sort(key=lambda x: x[1], reverse=True)

    for rank_index, (feature_name, importance_score) in enumerate(feature_importance_ranking[:10]):
        print(f"{rank_index+1:2d}. {feature_name}: {importance_score:.3f}")

    # STEP 6G2: Return final selected feature matrix
    final_selected_features = decorrelated_feature_matrix[selected_variance_columns].iloc[:, top_feature_indices]

    return final_selected_features, selected_feature_names, target_labels

# ================================================================================
# STEP 7: DATA STANDARDIZATION AND CLASS BALANCING
# ================================================================================
# Prepare final dataset for machine learning by standardizing features and 
# balancing class distribution to prevent bias toward majority class.

def standardize_and_balance_data(feature_matrix, target_labels, balancing_method="undersample", random_seed=42):
    """
    STEP 7A: Standardize features using StandardScaler
    STEP 7B: Separate data by class (benign vs attack)
    STEP 7C: Apply chosen balancing method (undersample/oversample)
    STEP 7D: Shuffle final balanced dataset
    STEP 7E: Return ML-ready feature matrix and labels
    """
    print("📏 STEP 7: Starting data standardization and balancing...")

    # STEP 7A: Standardize features to zero mean and unit variance
    feature_scaler = StandardScaler()
    scaled_feature_matrix = feature_scaler.fit_transform(feature_matrix)

    print("⚖️ Start balancing data ...")
    combined_scaled_dataset = pd.DataFrame(scaled_feature_matrix)
    combined_scaled_dataset['target'] = target_labels

    # STEP 7B: Separate samples by class
    benign_samples = combined_scaled_dataset[combined_scaled_dataset['target']==0]
    attack_samples = combined_scaled_dataset[combined_scaled_dataset['target']==1]

    print(f"Class 0 (BENIGN): {len(benign_samples):,}")
    print(f"Class 1 (ATTACK): {len(attack_samples):,}")

    # STEP 7C: Apply balancing strategy
    if balancing_method == "undersample":
        # STEP 7C1: Undersample majority class to minority class size
        minimum_class_size = min(len(benign_samples), len(attack_samples))
        final_sample_size = min(minimum_class_size, 100000)  # Cap at 100k for memory

        resampled_benign_data = resample(benign_samples, replace=False, n_samples=final_sample_size, random_state=random_seed)
        resampled_attack_data = resample(attack_samples, replace=False, n_samples=final_sample_size, random_state=random_seed)

        balanced_dataset = pd.concat([resampled_benign_data, resampled_attack_data])

    elif balancing_method == "oversample":
        # STEP 7C2: Oversample minority class to majority class size
        maximum_class_size = max(len(benign_samples), len(attack_samples))
        final_sample_size = min(maximum_class_size, 50000)  # Cap at 50k for memory

        if len(benign_samples) < len(attack_samples):
            resampled_benign_data = resample(benign_samples, replace=True, n_samples=final_sample_size, random_state=random_seed)
            resampled_attack_data = resample(attack_samples, replace=False, n_samples=final_sample_size, random_state=random_seed)
        else:
            resampled_benign_data = resample(benign_samples, replace=False, n_samples=final_sample_size, random_state=random_seed)
            resampled_attack_data = resample(attack_samples, replace=True, n_samples=final_sample_size, random_state=random_seed)
        
        balanced_dataset = pd.concat([resampled_benign_data, resampled_attack_data])

    # STEP 7D: Shuffle balanced dataset for random distribution
    shuffled_balanced_dataset = balanced_dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # STEP 7E: Extract final feature matrix and target labels
    final_feature_matrix = shuffled_balanced_dataset.drop('target', axis=1).values
    final_target_labels = shuffled_balanced_dataset['target'].values

    print(f"After balancing: {pd.Series(final_target_labels).value_counts()}")
    return final_feature_matrix, final_target_labels

# ================================================================================
# MAIN EXECUTION PIPELINE
# ================================================================================
# Execute the complete DDoS detection preprocessing pipeline step by step

print("🚀 Starting DDoS Detection System Pipeline...")
print("=" * 80)

# STEP 1: Load network intrusion dataset from Kaggle
network_data = load_network_intrusion_dataset(dataset_path)
print("✅ STEP 1 COMPLETED: Data acquisition successful")
print("=" * 80)

# STEP 2-5: Comprehensive preprocessing (outliers, features, cleaning)
preprocessed_network_data = preprocess_network_data(network_data)
print("✅ STEP 2-5 COMPLETED: Data preprocessing successful")
print("=" * 80)

# STEP 6: Select optimal features using ensemble methods
selected_features_matrix, selected_feature_names, target_labels = select_best_features_ensemble(preprocessed_network_data, number_of_features=30)
print("✅ STEP 6 COMPLETED: Feature selection successful")
print("=" * 80)

# STEP 7: Standardize and balance data for ML training
balanced_feature_matrix, balanced_target_labels = standardize_and_balance_data(selected_features_matrix, target_labels, balancing_method="undersample")
print("✅ STEP 7 COMPLETED: Data standardization and balancing successful")
print("=" * 80)

# FINAL SUMMARY
print("🎉 PIPELINE COMPLETED: Enhanced preprocessing and feature selection completed!")
print(f"📊 Final dataset shape: {balanced_feature_matrix.shape}")
print(f"📋 Selected features: {len(selected_feature_names)}")
print("🚀 Dataset is now ready for model training!")
print("=" * 80)