import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 - Classification Model Comparison")

# ---------- Load saved assets ----------
st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "KNN": joblib.load("model/knn.joblib"),
        "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "Random Forest": joblib.load("model/random_forest.joblib"),
        "XGBoost": joblib.load("model/xgboost.joblib"),
    }
    return models

@st.cache_data
def load_comparison_table():
    return pd.read_csv("metrics/model_comparison.csv")

def plot_cm(cm, labels):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    return fig

models = load_models()
comparison_df = load_comparison_table()

# ---------- Sidebar ----------
st.sidebar.header("Run Model on Uploaded Test Data")
uploaded = st.sidebar.file_uploader("Upload CSV (test data)", type=["csv"])
model_name = st.sidebar.selectbox("Select model", list(models.keys()))
run_btn = st.sidebar.button("Run Evaluation")

# ---------- Main: show comparison table ----------
st.subheader("Comparison Table (All 6 Models)")
st.dataframe(comparison_df, use_container_width=True)

st.divider()

if uploaded is None:
    st.info("Upload a CSV file to run predictions and show confusion matrix/report.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Uploaded Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# Choose target column
target_col = st.sidebar.selectbox("Select target column (label)", df.columns)

if run_btn:
    X = df.drop(columns=[target_col])
    y_true_raw = df[target_col]

    # Convert label to 0/1 if needed (Adult dataset style)
    if y_true_raw.dtype == "object":
        y_true = y_true_raw.map({"<=50K": 0, ">50K": 1})
        if y_true.isna().any():
            st.error("Target column has unexpected labels. Ensure it uses <=50K and >50K (or provide numeric 0/1).")
            st.stop()
    else:
        y_true = y_true_raw

    model = models[model_name]

    y_pred = model.predict(X)

    st.subheader(f"Results for: {model_name}")

    cm = confusion_matrix(y_true, y_pred)
    labels = ["0", "1"]

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_cm(cm, labels))

    with col2:
        st.text(classification_report(y_true, y_pred, zero_division=0))
