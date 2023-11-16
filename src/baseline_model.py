import fire

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_baseline_model_error(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Returns the mean absolute error of a baseline model that always predicts the mean of the target
    """
    baseline_pred = np.array([y_test.mean() for _ in range(len(y_test))])
    baseline_error = mean_absolute_error(y_test, baseline_pred)

    return baseline_error # type: ignore

def train_baseline(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Trains a tree-based model using the features and target passed as arguments
    """
    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Baseline performance
    baseline_error = get_baseline_model_error(X_test, y_test)
    print(f'Baseline error: {baseline_error}')

if __name__ == '__main__':
    fire.Fire(train_baseline)