# app.py
"""Streamlit deployment for the churn prediction model.

The app loads the pre‑trained XGBoost model (or falls back to Logistic
Regression) from ``models/`` and the preprocessing pipeline. Users can input
a customer's features via the sidebar, obtain a churn probability gauge,
and see a local SHAP waterfall explanation.
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.joblib"
PREPROC_PATH = PROJECT_ROOT / "models" / "preprocess.joblib"

# Load artifacts (cached for performance)
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)
    # SHAP explainer – works with XGBTree models
    explainer = shap.TreeExplainer(model)
    return model, preproc, explainer

model, preproc, explainer = load_artifacts()

st.title("Customer Churn Prediction")
st.caption("Enter customer details on the left to obtain a churn probability.")

# Sidebar inputs
st.sidebar.header("Customer Features")
credit_score = st.sidebar.slider("Credit Score", 350, 850, 600)
country = st.sidebar.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
products_number = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.sidebar.checkbox("Has Credit Card", value=True)
active_member = st.sidebar.checkbox("Active Member", value=True)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Assemble DataFrame
input_dict = {
    "credit_score": credit_score,
    "country": country,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": int(credit_card),
    "active_member": int(active_member),
    "estimated_salary": estimated_salary,
    # Engineered features – recomputed exactly as in data_loader
    "balance_salary_ratio": balance / (estimated_salary + 1e-5),
    "is_zero_balance": int(balance == 0),
    "tenure_age_ratio": tenure / (age + 1e-5),
    "credit_score_age_ratio": credit_score / (age + 1e-5),
}

input_df = pd.DataFrame([input_dict])

# Preprocess & predict
X_processed = preproc.transform(input_df)
prob = model.predict_proba(X_processed)[:, 1][0]

# Display probability gauge
st.subheader("Churn Probability")
st.progress(prob)
st.metric(label="Probability", value=f"{prob:.2%}")

# SHAP local explanation
st.subheader("Feature Impact (SHAP)")
shap_values = explainer.shap_values(X_processed)
# shap package expects numpy array
shap.initjs()
shap.waterfall_plot(shap.Explanation(values=shap_values[0], data=input_df.iloc[0]), max_display=12)

# What‑If simulation – allow user to tweak one feature
st.subheader("What‑If Simulator")
feature_to_modify = st.selectbox("Select feature to vary", list(input_dict.keys()))
new_value = st.slider(f"New value for {feature_to_modify}", float(min(input_df[feature_to_modify])), float(max(input_df[feature_to_modify])*2), float(input_df[feature_to_modify][0]))
modified = input_df.copy()
modified[feature_to_modify] = new_value
if feature_to_modify.startswith("is_") or feature_to_modify.startswith("credit_card") or feature_to_modify.startswith("active_member"):
    modified[feature_to_modify] = int(new_value)
# Re‑engineer dependent features if needed
if feature_to_modify in ["balance", "estimated_salary"]:
    modified["balance_salary_ratio"] = modified["balance"] / (modified["estimated_salary"] + 1e-5)
if feature_to_modify in ["balance"]:
    modified["is_zero_balance"] = int(modified["balance"] == 0)
if feature_to_modify in ["tenure", "age"]:
    modified["tenure_age_ratio"] = modified["tenure"] / (modified["age"] + 1e-5)
if feature_to_modify in ["credit_score", "age"]:
    modified["credit_score_age_ratio"] = modified["credit_score"] / (modified["age"] + 1e-5)

X_mod = preproc.transform(modified)
prob_mod = model.predict_proba(X_mod)[:, 1][0]
st.write(f"New churn probability after changing **{feature_to_modify}**: {prob_mod:.2%}")
