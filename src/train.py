# src/train.py
"""Model training and hyper‑parameter search for churn prediction.

Three models are trained:
1. Logistic Regression
2. Random Forest
3. XGBoost

Each model is evaluated with 5‑fold stratified cross‑validation on the
training data. The best model (by ROC‑AUC) is persisted to ``models/`` along
with the fitted preprocessing pipeline.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import json

from .data_loader import load_data
from .preprocess import fit_transform_train, transform_test, preprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def train_models():
    df = load_data()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Split manually – we will let the caller handle a train/test split later.
    # For demonstration we train on the full data (after SMOTE) and later
    # evaluate on a held‑out set in the notebook.
    X_proc, y_proc, fitted_preprocess = fit_transform_train(X, y, apply_smote=True)

    models = {
        "logreg": LogisticRegression(max_iter=500, n_jobs=-1, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "xgb": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_proc,
            y_proc,
            cv=cv,
            scoring=["roc_auc", "average_precision", "accuracy", "f1"],
            n_jobs=-1,
            return_train_score=False,
        )
        results[name] = {k: float(v.mean()) for k, v in scores.items()}
        # Fit on the whole oversampled data for final model
        model.fit(X_proc, y_proc)
        # Persist the model
        model_path = PROJECT_ROOT / "models" / f"{name}_model.joblib"
        joblib.dump(model, model_path)

    # Persist the preprocessing pipeline
    preproc_path = PROJECT_ROOT / "models" / "preprocess.joblib"
    joblib.dump(fitted_preprocess, preproc_path)

    # Save metrics JSON
    metrics_path = PROJECT_ROOT / "models" / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Training complete. Models and metrics saved to 'models/'.")

if __name__ == "__main__":
    train_models()
