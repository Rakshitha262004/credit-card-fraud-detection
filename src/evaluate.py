"""
evaluate.py
Model evaluation, metrics, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import os


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Generate full evaluation metrics for a model."""
    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT: {model_name}")
    print("="*50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)

    # Average Precision (better metric for imbalanced data)
    avg_precision = average_precision_score(y_test, y_prob)

    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print(f"Avg Precision    : {avg_precision:.4f}")
    print("="*50 + "\n")

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud']
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


def plot_roc_curve(y_test, results_dict: dict, save_path: str) -> None:
    """Plot and save ROC curves for multiple models."""
    plt.figure(figsize=(8, 6))

    for model_name, result in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        plt.plot(
            fpr, tpr,
            label=f"{model_name} (AUC = {result['roc_auc']:.4f})",
            linewidth=2
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curve saved to {save_path}")


def plot_feature_importance(model, feature_names: list, model_name: str,
                            save_path: str, top_n: int = 15) -> None:
    """Plot and save top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(
        range(top_n),
        importances[indices][::-1],
        color='steelblue',
        edgecolor='white'
    )
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - {model_name}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature importance plot saved to {save_path}")


def plot_precision_recall_curve(y_test, results_dict: dict, save_path: str) -> None:
    """Plot Precision-Recall curve — more informative than ROC for imbalanced data."""
    plt.figure(figsize=(8, 6))

    for model_name, result in results_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_prob'])
        plt.plot(
            recall, precision,
            label=f"{model_name} (AP = {result['avg_precision']:.4f})",
            linewidth=2
        )

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Precision-Recall curve saved to {save_path}")


def save_classification_report(y_test, y_pred, model_name: str, save_path: str) -> None:
    """Save the text classification report to a file."""
    report = classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Fraud']
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("="*50 + "\n")
        f.write(report)
    print(f"[INFO] Classification report saved to {save_path}")