import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# -------------------- Page config --------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 - Classification Model Comparison")

# -------------------- Robust paths --------------------
# Works whether app.py is in repo root OR inside a subfolder
BASE_DIR = Path(__file__).resolve().parent

def find_project_root(start: Path) -> Path:
    """
    Walk up a few levels to find the repo root that contains /model and /metrics.
    """
    cur = start
    for _ in range(5):
        if (cur / "model").exists() and (cur / "metrics").exists():
            return cur
        cur = cur.parent
    return start  # fallback

PROJECT_DIR = find_project_root(BASE_DIR)
MODEL_DIR = PROJECT_DIR / "model"
METRICS_DIR = PROJECT_DIR / "metrics"
COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"

# -------------------- Sidebar: Upload + selection --------------------
st.sidebar.header("Run Model on Uploaded Test Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (test data)",
    type=["csv"],
    help="Upload a test CSV that includes the same columns used in training. For Adult dataset, include 'income' label."
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
run_btn = st.sidebar.button("Run Evaluation")

# -------------------- Load comparison table --------------------
st.subheader("Comparison Table (All 6 Models)")

if COMPARISON_CSV.exists():
    cmp_df = pd.read_csv(COMPARISON_CSV)
    st.dataframe(cmp_df, use_container_width=True)
else:
    st.warning(
        f"Missing file: {COMPARISON_CSV}\n\n"
        "Create and upload it to metrics/model_comparison.csv"
    )

st.divider()

# -------------------- Cached model loader --------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def get_loaded_model(name: str):
    path = MODEL_DIR / MODEL_FILES[name]
    if not path.exists():
        return None, path
    return load_model(str(path)), path

# -------------------- Helper: normalize Adult labels --------------------
def normalize_income_labels(y: pd.Series) -> pd.Series:
    """
    Convert typical Adult labels to {0,1}.
    Accepts: <=50K, >50K, <=50K., >50K. and trims whitespace.
    """
    s = y.astype(str).str.strip()
    mapping = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}
    out = s.map(mapping)
    return out

def pick_default_target(columns):
    # Adult dataset target is usually "income"
    if "income" in columns:
        return "income"
    # common alternatives
    for c in ["target", "label", "class", "salary"]:
        if c in columns:
            return c
    return columns[-1]  # fallback

# -------------------- Main flow --------------------
if uploaded_file is None:
    st.info("Upload a CSV file to run predictions and show confusion matrix/report.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# Choose target (default auto-detected)
default_target = pick_default_target(df.columns)
target_col = st.sidebar.selectbox(
    "Select target column (label)",
    df.columns,
    index=list(df.columns).index(default_target)
)

# If Adult dataset detected, guide user strongly
if "income" in df.columns and target_col != "income":
    st.sidebar.warning("For Adult dataset, label should be 'income'. Select 'income' to avoid feature mismatch errors.")

# Load model
model, model_path = get_loaded_model(model_name)
if model is None:
    st.error(f"Model file not found: {model_path}\n\nUpload it into the model/ folder in GitHub.")
    st.stop()

# Only run evaluation when button clicked
if not run_btn:
    st.info("Select model and target column, then click Run Evaluation.")
    st.stop()

# -------------------- Prepare X and y --------------------
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in uploaded CSV.")
    st.stop()

X = df.drop(columns=[target_col])
y_raw = df[target_col]

# Convert y to numeric if needed
if target_col == "income":
    y = normalize_income_labels(y_raw)
    if y.isna().any():
        st.error("Income labels not recognized. Expected <=50K / >50K (with or without trailing dot).")
        st.stop()
    y = y.astype(int)
else:
    # If user selects non-income target, try to coerce to int
    try:
        y = y_raw.astype(int)
    except Exception:
        st.error(
            "Selected target column is not numeric and not 'income'.\n"
            "For Adult dataset, please select 'income' as the label."
        )
        st.stop()

# -------------------- Predict --------------------
# Important: model is likely a Pipeline (preprocess + classifier)
try:
    y_pred = model.predict(X)
except Exception as e:
    st.error(
        "Model prediction failed. This usually happens when your uploaded CSV columns "
        "don't match training features.\n\n"
        f"Details: {type(e).__name__}: {e}"
    )
    st.stop()

# Predict probabilities (for AUC) if available
y_prob = None
try:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # scale for AUC usage
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
except Exception:
    y_prob = None

# -------------------- Metrics --------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
mcc = matthews_corrcoef(y, y_pred)
auc = roc_auc_score(y, y_prob) if y_prob is not None else np.nan

st.subheader(f"Evaluation Results: {model_name}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
c3.metric("Precision", f"{precision:.4f}")
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

st.divider()

# -------------------- Confusion Matrix + Report --------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification Report")
st.text(classification_report(y, y_pred, zero_division=0))

# Optional: allow download of predictions
st.subheader("Download Predictions (Optional)")
pred_out = df.copy()
pred_out["predicted_label"] = y_pred
st.download_button(
    "Download predictions as CSV",
    data=pred_out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)
