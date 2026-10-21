# src/preprocess.py
"""Preprocessing pipeline for the churn dataset.

The pipeline is built with scikit‑learn's :class:`ColumnTransformer` and
handles:
* One‑hot encoding of categorical columns (dropping the first level to avoid
  collinearity),
* Standard scaling of numerical columns,
* Optional SMOTE oversampling on the training split.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# The columns are defined here so they can be imported elsewhere
NUMERIC_COLS = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "estimated_salary",
    "balance_salary_ratio",
    "is_zero_balance",
    "tenure_age_ratio",
    "credit_score_age_ratio",
]
CATEGORICAL_COLS = ["country", "gender", "credit_card", "active_member"]

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), NUMERIC_COLS),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            CATEGORICAL_COLS,
        ),
    ]
)


def fit_transform_train(X_train: pd.DataFrame, y_train: pd.Series, apply_smote: bool = True):
    """Fit the preprocessing pipeline on the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix.
    y_train : pd.Series
        Target vector.
    apply_smote : bool, default=True
        Whether to apply SMOTE after scaling/encoding.

    Returns
    -------
    X_processed, y_processed, fitted_preprocess
        Processed feature matrix, possibly oversampled target, and the fitted
        ``ColumnTransformer``.
    """
    X_processed = preprocess.fit_transform(X_train)
    if apply_smote:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_processed, y_train)
        return X_res, y_res, preprocess
    else:
        return X_processed, y_train, preprocess


def transform_test(X_test: pd.DataFrame, fitted_preprocess: ColumnTransformer):
    """Transform the test set using a previously fitted preprocessing pipeline."""
    return fitted_preprocess.transform(X_test)
