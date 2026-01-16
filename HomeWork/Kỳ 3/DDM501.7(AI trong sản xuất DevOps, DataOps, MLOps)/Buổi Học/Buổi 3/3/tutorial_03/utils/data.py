from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """
    This function is to load the breast cancer dataset.
    Returns:
        X: numpy array
        y: numpy array
    """
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    This function is to split the data into training and testing sets.
    Args:
        X: numpy array
        y: numpy array
        test_size: float
        random_state: int
    Returns:
        X_train: numpy array
        X_test: numpy array
        y_train: numpy array
        y_test: numpy array
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def get_data(test_size=0.2, random_state=42):
    """
    This function is to get the data from the dataset.
    Args:
        test_size: float
        random_state: int
    Returns:
        X_train: numpy array
        X_test: numpy array
        y_train: numpy array
        y_test: numpy array
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test