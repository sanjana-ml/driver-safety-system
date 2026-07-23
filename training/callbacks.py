"""callbacks.py -- Early stopping, checkpointing, LR scheduling, TensorBoard."""

from __future__ import annotations

from typing import List

import tensorflow as tf

import config


def get_callbacks() -> List[tf.keras.callbacks.Callback]:
    config.ensure_directories()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.BEST_CHECKPOINT_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LEARNING_RATE,
        verbose=1,
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(config.TENSORBOARD_DIR),
        histogram_freq=1,
    )

    return [early_stopping, checkpoint, reduce_lr, tensorboard]
