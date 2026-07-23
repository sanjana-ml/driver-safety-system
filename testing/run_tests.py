"""
run_tests.py
Standalone CLI: loads the trained model and the dataset's held-out test
split, and (re)generates the confusion matrix / ROC curve / classification
report without re-training. Useful after swapping in a new dataset or
model checkpoint.

Usage:
    python -m testing.run_tests
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

import config
from cbam.cbam import CUSTOM_OBJECTS
from testing.evaluate import evaluate_and_save
from training.data_loader import load_dataset_arrays
from utils.exceptions import DriverSafetyError, ModelNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    if not config.MODEL_PATH.exists():
        logger.error(
            "No trained model found at %s. Run 'python train.py' first.",
            config.MODEL_PATH,
        )
        return 1

    try:
        logger.info("Loading dataset for evaluation...")
        _, _, (X_test, y_test) = load_dataset_arrays()

        logger.info("Loading trained model...")
        model = tf.keras.models.load_model(str(config.MODEL_PATH), custom_objects=CUSTOM_OBJECTS)

        logger.info("Evaluating on %d test samples...", len(X_test))
        metrics = evaluate_and_save(model, X_test, y_test)
        logger.info("Test accuracy: %.4f | ROC AUC: %.4f", metrics["test_accuracy"], metrics["roc_auc"])
        return 0
    except DriverSafetyError as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
