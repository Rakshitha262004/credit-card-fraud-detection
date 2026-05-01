"""
alerts.py
Generate and save fraud alert reports.
"""

import pandas as pd
import os
from datetime import datetime


def generate_alert_report(predictions_df: pd.DataFrame,
                           save_path: str = "outputs/fraud_alerts.csv") -> None:
    """
    Filter flagged fraud transactions and save as alert CSV.
    """
    fraud_alerts = predictions_df[predictions_df['Predicted'] == 1].copy()
    fraud_alerts['Alert_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fraud_alerts.to_csv(save_path, index=False)

    print(f"[ALERT] {len(fraud_alerts)} fraud alert(s) saved to {save_path}")
    return fraud_alerts