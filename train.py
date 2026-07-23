"""
train.py
Entry point: reads the uploaded dataset automatically, detects classes,
splits 80/10/10, builds the custom CNN+CBAM model, trains it with
augmentation/normalization/early-stopping/checkpointing/LR-scheduling/
TensorBoard, then saves the model, training history, and evaluation
artefacts (plots, confusion matrix, ROC curve).

Usage:
    python train.py
    python train.py --epochs 40 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

import config
from models.cnn_cbam import build_cnn_cbam_model
from testing.evaluate import evaluate_and_save, plot_training_history
from training.callbacks import get_callbacks
from training.data_loader import build_augmentation_generator, load_dataset_arrays
from utils.exceptions import DatasetError, DriverSafetyError
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Driver Safety CNN+CBAM model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--dataset-dir", type=str, default=str(config.DATASET_DIR),
        help="Override the dataset directory (default: dataset/).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.ensure_directories()

    try:
        logger.info("Loading dataset from %s ...", args.dataset_dir)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset_arrays(
            root=Path(args.dataset_dir)
        )
    except DriverSafetyError as exc:
        logger.error("Dataset error: %s", exc)
        logger.error(
            "Add your images under dataset/drowsy/ and dataset/not_drowsy/ "
            "(see dataset/README.md), then re-run 'python train.py'."
        )
        return 1

    logger.info(
        "Loaded dataset -> train: %d, val: %d, test: %d images",
        len(X_train), len(X_val), len(X_test),
    )

    model = build_cnn_cbam_model()
    if args.learning_rate != config.LEARNING_RATE:
        model.optimizer.learning_rate.assign(args.learning_rate)
    model.summary(print_fn=lambda line: logger.info(line))

    augmenter = build_augmentation_generator()
    augmenter.fit(X_train)
    train_generator = augmenter.flow(
        X_train, y_train, batch_size=args.batch_size, seed=config.RANDOM_SEED
    )

    callbacks = get_callbacks()

    start = time.time()
    try:
        history = model.fit(
            train_generator,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            steps_per_epoch=max(1, len(X_train) // args.batch_size),
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user -- saving current model state.")
        history = None

    elapsed = time.time() - start
    logger.info("Training finished in %.1f seconds.", elapsed)

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(config.MODEL_PATH))
    logger.info("Saved final model to %s", config.MODEL_PATH)

    if history is not None:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        config.TRAINING_HISTORY_PATH.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
        plot_training_history(history_dict)

    logger.info("Evaluating on held-out test split (%d images)...", len(X_test))
    try:
        metrics = evaluate_and_save(model, X_test, y_test)
        logger.info(
            "Final test accuracy: %.4f | ROC AUC: %.4f",
            metrics["test_accuracy"], metrics["roc_auc"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation step failed (model was still saved): %s", exc)

    logger.info("Done. You can now run: python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
