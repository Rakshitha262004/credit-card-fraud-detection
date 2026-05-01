"""
preprocess.py
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card dataset from CSV."""
    print(f"[INFO] Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Please download from: "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """Print a full data quality report."""
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    print(f"Shape            : {df.shape}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")
    print(f"\nClass Distribution:")
    counts = df['Class'].value_counts()
    total = len(df)
    print(f"  Legitimate (0) : {counts[0]:,} ({counts[0]/total*100:.2f}%)")
    print(f"  Fraudulent (1) : {counts[1]:,} ({counts[1]/total*100:.2f}%)")
    print("="*50 + "\n")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    print("[INFO] Cleaning data...")
    original_len = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"[INFO] Removed {original_len - len(df)} duplicate rows.")

    # Drop nulls (dataset is clean, but good practice)
    df = df.dropna()

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    print("[INFO] Engineering features...")

    # Log transform Amount (reduces skewness)
    df['Amount_Log'] = np.log1p(df['Amount'])

    # Extract hour of day from Time (Time is seconds from first transaction)
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24

    # Amount category bins
    df['Amount_Bin'] = pd.cut(
        df['Amount'],
        bins=[-1, 10, 100, 500, 1000, df['Amount'].max()],
        labels=['micro', 'small', 'medium', 'large', 'very_large']
    )

    # One-hot encode the Amount_Bin
    df = pd.get_dummies(df, columns=['Amount_Bin'], drop_first=False)

    return df


def scale_features(df: pd.DataFrame) -> tuple:
    """Scale Amount and Time columns. Returns df and scaler."""
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

    # Drop original unscaled columns we've now replaced or engineered
    df = df.drop(columns=['Time', 'Amount'])

    return df, scaler


def get_features_and_target(df: pd.DataFrame) -> tuple:
    """Split into feature matrix X and target vector y."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y