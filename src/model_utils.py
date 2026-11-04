"""
model_utils.py
--------------
Shared helpers for model training:
  * time-based train/test split
  * preprocessing pipeline (encoding + scaling) that is fitted ONLY on training data
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from data_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES, FEATURES

# Last 12 months are kept as the TEST set (data the model has never seen)
TEST_START = "2017-10-01"
RANDOM_STATE = 42


def time_split(df: pd.DataFrame, target: str):
    """Split by date (not randomly), because we want to predict the FUTURE.
    Random splitting would let the model 'peek' at hours right next to the test hours."""
    df = df.sort_values("date_time")
    train = df[df["date_time"] < TEST_START]
    test = df[df["date_time"] >= TEST_START]
    return train[FEATURES], test[FEATURES], train[target], test[target], train, test


def build_preprocessor() -> ColumnTransformer:
    """Encoding + scaling.
    - Numeric columns  -> StandardScaler (mean 0, std 1)  - needed for Linear/Logistic/KNN/SVM
    - Categorical      -> OneHotEncoder (hour 0..23 becomes 24 yes/no columns)
    - Binary 0/1 flags -> passed through unchanged
    Because this sits INSIDE a Pipeline, it is fitted on training folds only -> no data leakage."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])


# 3-fold time-series cross-validation: each fold trains on the past and validates on the next period
CV = TimeSeriesSplit(n_splits=3)
