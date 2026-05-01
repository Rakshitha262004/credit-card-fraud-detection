

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import (
    load_data, check_data_quality, clean_data,
    engineer_features, scale_features, get_features_and_target
)
from train import (
    split_data, apply_smote,
    train_random_forest, train_xgboost,
    save_model, load_model
)
from evaluate import (
    evaluate_model, plot_confusion_matrix,
    plot_roc_curve, plot_feature_importance,
    plot_precision_recall_curve, save_classification_report
)
from simulate import run_simulation
from alerts import generate_alert_report


def main():
    print("\n" + "="*60)
    print("  CREDIT CARD FRAUD DETECTION SYSTEM")
    print("  Industry-Oriented ML Project")
    print("="*60 + "\n")

    # ─────────────────────────────────────────────
    # PHASE 1: LOAD & CLEAN DATA
    # ─────────────────────────────────────────────
    data_path = os.path.join('data', 'creditcard.csv')
    df = load_data(data_path)
    check_data_quality(df)
    df = clean_data(df)

    # ─────────────────────────────────────────────
    # PHASE 2: FEATURE ENGINEERING
    # ─────────────────────────────────────────────
    df = engineer_features(df)
    df, scaler = scale_features(df)

    # ─────────────────────────────────────────────
    # PHASE 3: SPLIT DATA
    # ─────────────────────────────────────────────
    X, y = get_features_and_target(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # ─────────────────────────────────────────────
    # PHASE 4: HANDLE IMBALANCE WITH SMOTE
    # ─────────────────────────────────────────────
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ─────────────────────────────────────────────
    # PHASE 5: TRAIN MODELS
    # ─────────────────────────────────────────────
    rf_model  = train_random_forest(X_train_sm, y_train_sm)
    xgb_model = train_xgboost(X_train_sm, y_train_sm)

    # Save models
    save_model(rf_model,  os.path.join('models', 'random_forest_model.pkl'))
    save_model(xgb_model, os.path.join('models', 'xgboost_model.pkl'))

    # ─────────────────────────────────────────────
    # PHASE 6: EVALUATE MODELS
    # ─────────────────────────────────────────────
    os.makedirs('images',  exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    rf_results  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Confusion matrices
    plot_confusion_matrix(
        y_test, rf_results['y_pred'], "Random Forest",
        save_path=os.path.join('images', 'confusion_matrix_rf.png')
    )
    plot_confusion_matrix(
        y_test, xgb_results['y_pred'], "XGBoost",
        save_path=os.path.join('images', 'confusion_matrix_xgb.png')
    )

    # ROC Curve (both models on same plot)
    all_results = {
        'Random Forest': rf_results,
        'XGBoost':       xgb_results
    }
    plot_roc_curve(
        y_test, all_results,
        save_path=os.path.join('images', 'roc_curve.png')
    )

    # Precision-Recall Curve
    plot_precision_recall_curve(
        y_test, all_results,
        save_path=os.path.join('images', 'precision_recall_curve.png')
    )

    # Feature Importance
    plot_feature_importance(
        rf_model, feature_columns, "Random Forest",
        save_path=os.path.join('images', 'feature_importance_rf.png')
    )
    plot_feature_importance(
        xgb_model, feature_columns, "XGBoost",
        save_path=os.path.join('images', 'feature_importance_xgb.png')
    )

    # Save classification reports
    save_classification_report(
        y_test, rf_results['y_pred'], "Random Forest",
        save_path=os.path.join('outputs', 'classification_report_rf.txt')
    )
    save_classification_report(
        y_test, xgb_results['y_pred'], "XGBoost",
        save_path=os.path.join('outputs', 'classification_report_xgb.txt')
    )

    # ─────────────────────────────────────────────
    # PHASE 7: VIRTUAL SIMULATION
    # ─────────────────────────────────────────────
    # Use the better performing model for simulation
    best_model = rf_model if rf_results['roc_auc'] >= xgb_results['roc_auc'] else xgb_model
    best_name  = "Random Forest" if rf_results['roc_auc'] >= xgb_results['roc_auc'] else "XGBoost"
    print(f"\n[INFO] Using {best_name} for simulation (higher ROC-AUC)")

    sim_results = run_simulation(
        model=best_model,
        feature_columns=feature_columns,
        n_legit=50,
        n_fraud=10
    )

    # ─────────────────────────────────────────────
    # PHASE 8: GENERATE ALERTS
    # ─────────────────────────────────────────────
    generate_alert_report(
        sim_results,
        save_path=os.path.join('outputs', 'fraud_alerts.csv')
    )

    # ─────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE — RESULTS SUMMARY")
    print("="*60)
    print(f"  Random Forest ROC-AUC : {rf_results['roc_auc']:.4f}")
    print(f"  XGBoost ROC-AUC       : {xgb_results['roc_auc']:.4f}")
    print(f"\n  Saved files:")
    print(f"    models/  → Trained model .pkl files")
    print(f"    images/  → All visualization plots")
    print(f"    outputs/ → Fraud alerts + reports")
    print("="*60 + "\n")
    print("[DONE] Project pipeline executed successfully.")
    print("[NEXT] Run the EDA notebook for in-depth analysis.")
    print("[NEXT] Upload the project to GitHub.")


if __name__ == "__main__":
    main()