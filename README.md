# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)

> An end-to-end Machine Learning system that detects fraudulent credit card transactions in real time using ensemble models, SMOTE for imbalance handling, and a virtual transaction simulator.

---

## 📌 Problem Statement

Credit card fraud causes **$32+ billion** in losses annually. Traditional rule-based systems struggle with novel fraud patterns. This project builds an ML-based detection system capable of:

- Identifying fraud with **high recall** (catching most frauds)
- Minimizing **false positives** (not blocking legitimate transactions)
- Running predictions on a **simulated real-time transaction stream**

---

## ✅ Solution

| Component | Detail |
|---|---|
| Dataset | Kaggle Credit Card Fraud Detection (284,807 transactions) |
| Imbalance Handling | SMOTE (Synthetic Minority Oversampling Technique) |
| Models | Random Forest + XGBoost |
| Evaluation | Precision, Recall, F1, ROC-AUC |
| Simulation | Virtual 60-transaction fraud detection demo |
| Alerts | Flagged fraud saved to CSV |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — ML pipeline, preprocessing, evaluation
- **imbalanced-learn** — SMOTE
- **XGBoost** — Gradient boosting model
- **Matplotlib, Seaborn** — Visualization
- **Joblib** — Model persistence

---

## 📁 Project Structure