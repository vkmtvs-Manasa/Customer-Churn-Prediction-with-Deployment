# src/data_loader.py
"""Data loading and basic feature engineering for the churn dataset.

The raw CSV (bank_customer_churn.csv) is stored in ``data/raw``. This module
* reads the CSV,
* drops any rows with missing values (none expected),
* creates a few simple engineered features that improve model performance,
* and returns a cleaned :class:`pandas.DataFrame` ready for further processing.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bank_customer_churn.csv"


def load_data() -> pd.DataFrame:
    """Load the raw churn CSV and perform lightweight preprocessing.

    Returns
    -------
    pd.DataFrame
        A dataframe with numeric and categorical columns ready for the
        preprocessing pipeline.
    """
    df = pd.read_csv(DATA_PATH)
    # Ensure the expected column names are present (rename if needed)
    rename_map = {
        "CustomerId": "customer_id",
        "CreditScore": "credit_score",
        "Geography": "country",
        "Gender": "gender",
        "Age": "age",
        "Tenure": "tenure",
        "Balance": "balance",
        "NumOfProducts": "products_number",
        "HasCrCard": "credit_card",
        "IsActiveMember": "active_member",
        "EstimatedSalary": "estimated_salary",
        "Exited": "churn",
    }
    df = df.rename(columns=rename_map)
    # Drop columns we do not need for modeling (e.g., customer_id)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    # Engineered features
    df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1e-5)
    df["is_zero_balance"] = (df["balance"] == 0).astype(int)
    df["tenure_age_ratio"] = df["tenure"] / (df["age"] + 1e-5)
    df["credit_score_age_ratio"] = df["credit_score"] / (df["age"] + 1e-5)
    # Ensure correct dtypes
    cat_cols = ["country", "gender", "credit_card", "active_member"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df
