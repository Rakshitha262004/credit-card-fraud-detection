"""
simulate.py
Virtual fraud simulation - generates synthetic transaction scenarios
to demonstrate how the model catches fraud in "real-time".
"""

import numpy as np
import pandas as pd


def generate_synthetic_transactions(n_legit: int = 50, n_fraud: int = 10,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transactions (legit + fraud) to simulate
    a real-time transaction stream.

    Fraud patterns encoded:
    - Very high or very low amounts
    - Transactions at unusual hours (2–4 AM)
    - Extreme PCA feature values
    """
    np.random.seed(random_state)

    # ── Legitimate transactions ──────────────────────────────────
    legit_records = {
        'Amount': np.random.lognormal(mean=3.5, sigma=1.0, size=n_legit),
        'Hour':   np.random.choice(range(8, 22), size=n_legit),  # Normal hours
        'Class':  np.zeros(n_legit, dtype=int)
    }
    # Random PCA features (V1-V28) for legit — near zero, low variance
    for v in range(1, 29):
        legit_records[f'V{v}'] = np.random.normal(0, 0.5, size=n_legit)

    legit_df = pd.DataFrame(legit_records)

    # ── Fraudulent transactions ──────────────────────────────────
    fraud_records = {
        'Amount': np.random.choice(
            # Fraud often = very small amounts (testing card) or very large
            np.concatenate([
                np.random.uniform(0.1, 2.0, size=n_fraud // 2),
                np.random.uniform(800, 2500, size=n_fraud - n_fraud // 2)
            ])
        ),
        'Hour':   np.random.choice([1, 2, 3, 4, 23], size=n_fraud),  # Late night
        'Class':  np.ones(n_fraud, dtype=int)
    }
    # PCA features for fraud — more extreme values
    for v in range(1, 29):
        fraud_records[f'V{v}'] = np.random.normal(0, 3.0, size=n_fraud)

    fraud_df = pd.DataFrame(fraud_records)

    # ── Combine and shuffle ──────────────────────────────────────
    combined = pd.concat([legit_df, fraud_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Add engineered features
    combined['Amount_Log'] = np.log1p(combined['Amount'])
    combined['Amount_Scaled'] = (combined['Amount'] - combined['Amount'].mean()) / combined['Amount'].std()

    # Amount bins (dummy columns to match training features)
    combined['Amount_Bin_micro'] = (combined['Amount'] <= 10).astype(int)
    combined['Amount_Bin_small'] = ((combined['Amount'] > 10) & (combined['Amount'] <= 100)).astype(int)
    combined['Amount_Bin_medium'] = ((combined['Amount'] > 100) & (combined['Amount'] <= 500)).astype(int)
    combined['Amount_Bin_large'] = ((combined['Amount'] > 500) & (combined['Amount'] <= 1000)).astype(int)
    combined['Amount_Bin_very_large'] = (combined['Amount'] > 1000).astype(int)

    # Time-related
    combined['Time_Scaled'] = (combined['Hour'] - combined['Hour'].mean()) / combined['Hour'].std()

    return combined


def run_simulation(model, feature_columns: list,
                   n_legit: int = 50, n_fraud: int = 10) -> pd.DataFrame:
    """
    Simulate a transaction stream and run fraud detection.
    Returns a DataFrame with predictions and fraud alerts.
    """
    print("\n" + "="*55)
    print("  VIRTUAL FRAUD DETECTION SIMULATION")
    print("="*55)
    print(f"  Simulating {n_legit} legitimate + {n_fraud} fraudulent transactions")
    print("="*55)

    # Generate synthetic data
    sim_data = generate_synthetic_transactions(n_legit=n_legit, n_fraud=n_fraud)

    # Align columns with training feature columns
    # Add any missing columns as zeros
    for col in feature_columns:
        if col not in sim_data.columns:
            sim_data[col] = 0

    X_sim = sim_data[feature_columns]
    y_true = sim_data['Class']

    # Run predictions
    y_pred = model.predict(X_sim)
    y_prob = model.predict_proba(X_sim)[:, 1]

    # Build result DataFrame
    results = sim_data[['Amount', 'Hour', 'Class']].copy()
    results['Fraud_Probability'] = (y_prob * 100).round(2)
    results['Predicted'] = y_pred
    results['Alert'] = results['Predicted'].apply(
        lambda x: '🚨 FRAUD ALERT' if x == 1 else '✅ Approved'
    )
    results['Actual'] = results['Class'].apply(
        lambda x: 'FRAUD' if x == 1 else 'LEGIT'
    )

    # Print simulation output
    print(f"\n{'TXN':>4} {'Amount':>10} {'Hour':>6} {'Fraud%':>8} {'Alert':<20} {'Actual'}")
    print("-"*65)
    for i, row in results.iterrows():
        marker = " ◄ MISS" if (row['Predicted'] != row['Class']) else ""
        print(f"{i+1:>4} ${row['Amount']:>9.2f} {int(row['Hour']):>5}h "
              f"{row['Fraud_Probability']:>7.1f}% {row['Alert']:<20} "
              f"[{row['Actual']}]{marker}")

    # Summary
    fraud_caught = ((y_pred == 1) & (y_true == 1)).sum()
    fraud_total  = (y_true == 1).sum()
    false_alarms = ((y_pred == 1) & (y_true == 0)).sum()

    print("\n" + "="*55)
    print(f"  SIMULATION SUMMARY")
    print(f"  Total Transactions  : {len(results)}")
    print(f"  Fraud Caught        : {fraud_caught} / {fraud_total}")
    print(f"  False Alarms        : {false_alarms}")
    print(f"  Detection Rate      : {fraud_caught/fraud_total*100:.1f}%")
    print("="*55 + "\n")

    return results