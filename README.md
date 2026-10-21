# README.md

## Customer Churn Prediction with Deployment

This repository contains a **full‑stack machine‑learning portfolio project** that demonstrates the end‑to‑end workflow for a churn prediction use‑case:

1. **Data ingestion & cleaning** – `src/data_loader.py` loads the Kaggle Telco/Bank churn CSV, performs basic cleaning and engineering of a few useful features.
2. **Pre‑processing pipeline** – `src/preprocess.py` builds a `ColumnTransformer` that scales numeric columns, one‑hot encodes categoricals, and optionally applies **SMOTE** to balance the training set.
3. **Model training & benchmarking** – `src/train.py` trains three classifiers (Logistic Regression, Random Forest, XGBoost) with 5‑fold stratified CV, stores the best models and a JSON summary of metrics under `models/`.
4. **Evaluation notebook** – `notebooks/evaluation.ipynb` loads the saved artifacts, visualises ROC/PR curves, confusion matrices and prints a comparative metrics table.
5. **Explainability** – SHAP is installed and can be incorporated in the notebook or Streamlit UI to highlight feature impact.
6. **Interactive deployment** – `app.py` is a Streamlit app that lets a user input a customer’s attributes, returns a churn probability gauge, shows a local SHAP waterfall plot, and offers a *What‑If* simulator.
7. **Reproducibility** – `requirements.txt` pins all needed packages (compatible NumPy < 2 for the current environment).

### Quick‑start
```bash
# Clone the repo
git clone https://github.com/vkmtvs-Manasa/Customer-Churn-Prediction-with-Deployment.git
cd Customer-Churn-Prediction-with-Deployment

# (Optional) create a virtual environment
python -m venv venv && source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline (produces models and metrics)
python src/train.py

# Launch the Streamlit UI
streamlit run app.py
```

### Project structure
```
customer-churn-prediction/
├─ data/
│   └─ raw/                 # original CSV (bank_customer_churn.csv)
├─ models/                  # trained .joblib files + metrics JSON
├─ notebooks/
│   └─ evaluation.ipynb    # evaluation & visualisation
├─ reports/figures/         # generated plots (saved by notebook)
├─ src/
│   ├─ __init__.py
│   ├─ data_loader.py
│   ├─ preprocess.py
│   └─ train.py
├─ app.py                   # Streamlit deployment app
├─ .gitignore
├─ requirements.txt
└─ README.md
```

Feel free to explore the notebook for deeper analysis or customise the Streamlit UI for your own use‑case.
