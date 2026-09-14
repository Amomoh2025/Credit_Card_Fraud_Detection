# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions, built as a learning exercise in handling severely imbalanced classification problems.

## Problem

Given a transaction's features, predict whether it's fraudulent. The dataset is highly imbalanced — only ~0.17% of transactions are fraudulent — which makes this a good case study in why standard accuracy is a misleading metric and how to properly evaluate and train models under class imbalance.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions made by European cardholders, with 492 labeled as fraud. Features `V1`–`V28` are anonymized via PCA to protect cardholder identity; `Time` and `Amount` are the only non-transformed features.

## Approach

1. **EDA** — examined class imbalance and the distributions of `Amount` and `Time`
2. **Cleaning** — removed 1,081 duplicate rows
3. **Feature scaling** — standardized `Amount` and `Time` to match the scale of the PCA components
4. **Train/test split** — 80/20, stratified to preserve the fraud ratio in both sets
5. **SMOTE** — applied to the training set only, to synthetically balance the fraud class without touching the test set
6. **Models trained**: Logistic Regression, Decision Tree, Random Forest
7. **Threshold tuning** — explored whether adjusting Logistic Regression's decision threshold could close the performance gap with Random Forest

## Results

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| Precision (fraud) | 0.05 | 0.40 | **0.92** |
| Recall (fraud) | **0.87** | 0.71 | 0.76 |
| F1-score (fraud) | 0.10 | 0.51 | **0.83** |
| ROC-AUC | **0.962** | 0.852 | 0.944 |

**Random Forest performed best overall**, achieving far fewer false positives (6 vs. 1,479 for Logistic Regression) while still catching 76% of fraud cases. Logistic Regression had higher raw recall and ROC-AUC, but threshold tuning showed it couldn't match Random Forest's precision even at aggressive cutoffs — suggesting the gap comes from the underlying decision boundary, not just threshold calibration.

The most predictive features were `V14`, `V10`, `V17`, and `V4` — consistent with other analyses of this dataset. Because features are PCA-anonymized, their real-world meaning is unknown.

## Setup

```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in a `data/` folder, then open `fraud_detection.ipynb`.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn