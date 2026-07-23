"""
evaluate.py
Generates the evaluation artefacts required by the implementation strategy:
confusion matrix, ROC curve, accuracy graph, loss graph, and a text
classification report -- all saved to logs/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, required for saving figures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_and_save(
    model,
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    output_dir: Path = config.LOGS_DIR,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.argmax(y_test_onehot, axis=1)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    class_names = list(config.CLASS_NAMES)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    logger.info("Classification report:\n%s", report_text)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    drowsy_idx = class_names.index("Drowsy")
    fpr, tpr, _ = roc_curve(y_true == drowsy_idx, y_prob[:, drowsy_idx])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="CNN+CBAM").plot(ax=ax)
    ax.set_title("ROC Curve (Drowsy class)")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    metrics = {
        "test_accuracy": float(report_dict["accuracy"]),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved evaluation artefacts to %s", output_dir)
    return metrics


def plot_training_history(
    history_dict: Dict[str, Any], output_dir: Path = config.LOGS_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if "accuracy" in history_dict:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history_dict["accuracy"], label="train_accuracy")
        if "val_accuracy" in history_dict:
            ax.plot(history_dict["val_accuracy"], label="val_accuracy")
        ax.set_title("Accuracy over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_graph.png", dpi=150)
        plt.close(fig)

    if "loss" in history_dict:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history_dict["loss"], label="train_loss")
        if "val_loss" in history_dict:
            ax.plot(history_dict["val_loss"], label="val_loss")
        ax.set_title("Loss over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_graph.png", dpi=150)
        plt.close(fig)

    logger.info("Saved training history plots to %s", output_dir)
