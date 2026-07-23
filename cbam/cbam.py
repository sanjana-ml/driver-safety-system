"""
cbam.py
Convolutional Block Attention Module (CBAM), implemented from scratch with
Keras layers only -- no third-party CBAM package is used.

Reference (for citation in the report, not code reuse):
Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

CBAM = Channel Attention -> Spatial Attention, applied sequentially and
multiplied element-wise with the input feature map.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    """Learns 'what' is meaningful in the feature map by squeezing the
    spatial dimensions with both average- and max-pooling, passing each
    through a shared MLP (bottleneck), summing, then applying a sigmoid
    gate over the channel axis."""

    def __init__(self, reduction_ratio: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape: Tuple[int, ...]) -> None:
        channels = int(input_shape[-1])
        hidden_units = max(1, channels // self.reduction_ratio)

        # Shared MLP (implemented as two 1x1-equivalent Dense layers so the
        # exact same weights are applied to both the avg- and max-pooled
        # descriptors, as specified in the original CBAM paper).
        self.shared_dense_one = layers.Dense(
            hidden_units, activation="relu", use_bias=True, name=f"{self.name}_mlp1"
        )
        self.shared_dense_two = layers.Dense(
            channels, activation=None, use_bias=True, name=f"{self.name}_mlp2"
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=False)

        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        channel_attention = tf.sigmoid(avg_out + max_out)
        channel_attention = tf.reshape(
            channel_attention, (-1, 1, 1, tf.shape(inputs)[-1])
        )
        return inputs * channel_attention

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    """Learns 'where' is meaningful by pooling across the channel axis
    (average and max), concatenating, and passing through a 7x7 conv to
    produce a single-channel spatial attention map."""

    def __init__(self, kernel_size: int = 7, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name=f"{self.name}_conv",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv(concat)
        return inputs * spatial_attention

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class CBAM(layers.Layer):
    """Sequential Channel Attention -> Spatial Attention block that can be
    dropped into any CNN after a convolutional feature map."""

    def __init__(
        self, reduction_ratio: int = 8, kernel_size: int = 7, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio, name=f"{self.name}_channel_att"
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size, name=f"{self.name}_spatial_att"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"reduction_ratio": self.reduction_ratio, "kernel_size": self.kernel_size}
        )
        return config


CUSTOM_OBJECTS = {
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
    "CBAM": CBAM,
}
