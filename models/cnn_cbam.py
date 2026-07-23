"""
cnn_cbam.py
Custom CNN backbone (no transfer learning) with CBAM attention blocks
inserted after each convolutional stage, for binary drowsiness
classification (Drowsy / Not Drowsy).
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from cbam.cbam import CBAM


def _conv_block(
    x: tf.Tensor,
    filters: int,
    block_name: str,
    l2_reg: float = 1e-4,
) -> tf.Tensor:
    """Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> CBAM -> Pool -> Dropout."""
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{block_name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = layers.ReLU(name=f"{block_name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{block_name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = layers.ReLU(name=f"{block_name}_relu2")(x)

    x = CBAM(
        reduction_ratio=config.CBAM_REDUCTION_RATIO, name=f"{block_name}_cbam"
    )(x)

    x = layers.MaxPooling2D(pool_size=2, name=f"{block_name}_pool")(x)
    x = layers.Dropout(0.25, name=f"{block_name}_dropout")(x)
    return x


def build_cnn_cbam_model(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
) -> tf.keras.Model:
    """Builds and returns the compiled CNN + CBAM model described in the
    implementation strategy: Conv -> BN -> ReLU -> Pool -> Dropout blocks,
    each augmented with a CBAM attention module, followed by dense layers
    and a softmax classification head."""

    inputs = layers.Input(shape=input_shape, name="face_input")

    x = _conv_block(inputs, filters=32, block_name="block1")
    x = _conv_block(x, filters=64, block_name="block2")
    x = _conv_block(x, filters=128, block_name="block3")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(
        128, activation=None, kernel_regularizer=regularizers.l2(1e-4), name="dense1"
    )(x)
    x = layers.BatchNormalization(name="dense1_bn")(x)
    x = layers.ReLU(name="dense1_relu")(x)
    x = layers.Dropout(dropout_rate, name="dense1_dropout")(x)

    x = layers.Dense(64, activation="relu", name="dense2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dense2_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DrowsinessCNN_CBAM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    m = build_cnn_cbam_model()
    m.summary()
