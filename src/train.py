"""
train.py
Handles model training with SMOTE for imbalance correction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split into train and test sets."""
    print(f"[INFO] Splitting data: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class ratio in both splits
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to balance the training set.
    IMPORTANT: Only apply SMOTE to training data, NEVER to test data.
    """
    print("[INFO] Applying SMOTE to handle class imbalance...")
    print(f"[INFO] Before SMOTE - Fraud: {y_train.sum()} | Legit: {(y_train==0).sum()}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE  - Fraud: {y_resampled.sum()} | Legit: {(y_resampled==0).sum()}")
    return X_resampled, y_resampled


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    print("[INFO] Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    print("[INFO] Random Forest training complete.")
    return rf_model


def train_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    """Train an XGBoost classifier."""
    print("[INFO] Training XGBoost model...")

    # Calculate scale_pos_weight for XGBoost imbalance handling
    # (even after SMOTE, good to set this)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    print("[INFO] XGBoost training complete.")
    return xgb_model


def save_model(model, filepath: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved to {filepath}")


def load_model(filepath: str):
    """Load a saved model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    model = joblib.load(filepath)
    print(f"[INFO] Model loaded from {filepath}")
    return model