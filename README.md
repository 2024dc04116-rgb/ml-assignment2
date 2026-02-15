# ML Assignment 2 - Classification Models (Adult Census Income)

## a) Problem Statement
Build and compare multiple classification models to predict whether a person’s annual income is **>50K** or **<=50K** using demographic and work-related attributes. Evaluate each model using Accuracy, AUC, Precision, Recall, F1 score, and MCC. Deploy a Streamlit app that allows selecting a model and evaluating on uploaded test data.

## b) Dataset Description (Adult Census Income)
**Dataset:** Adult Census Income dataset (Kaggle)  
**Target column:** income (<=50K or >50K)  
**Features:** 14 columns (mix of numerical and categorical)  
**Rows:** 48,842 (after cleaning may reduce slightly)

Preprocessing performed:
- Replaced " ?" with missing values
- Dropped missing rows
- One-hot encoded categorical features
- Scaled numeric features where required (Logistic Regression, KNN)
- Train-test split with stratification

## c) Models Used (6 models)
1. Logistic Regression  
2. Decision Tree  
3. KNN  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

### Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.854291 | 0.904345 | 0.739366 | 0.609694 | 0.668298 | 0.580438 |
| Decision Tree | 0.854445 | 0.892159 | 0.768631 | 0.565689 | 0.651727 | 0.572958 |
| KNN | 0.840626 | 0.891503 | 0.689828 | 0.614158 | 0.649798 | 0.548634 |
| Naive Bayes | 0.794411 | 0.873075 | 0.555394 | 0.732143 | 0.631637 | 0.501801 |
| Random Forest | 0.857362 | 0.910960 | 0.772379 | 0.577806 | 0.661073 | 0.582790 |
| XGBoost | 0.869799 | 0.925107 | 0.772727 | 0.650510 | 0.706371 | 0.627330 |

### Observations (one per model)

| Model | Observation |
|---|---|
| Logistic Regression | Strong baseline with balanced metrics and good generalization. |
| Decision Tree | Higher precision but lower recall; tends to be conservative and may overfit without constraints. |
| KNN | Moderate performance; sensitive to high-dimensional one-hot encoded feature space. |
| Naive Bayes | Fast and simple, but weaker accuracy due to feature independence assumption. |
| Random Forest | Improves over single tree by bagging; provides strong AUC and stable performance. |
| XGBoost | Best overall model across Accuracy, AUC, F1, and MCC due to boosting and capturing feature interactions. |

## How to Run
### 1) Install dependencies
```bash
pip install -r requirements.txt
