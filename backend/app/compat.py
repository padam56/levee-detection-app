from __future__ import annotations

import tensorflow as tf

from .metrics import (
    bce_dice_loss,
    bce_dice_loss_new,
    dice_coef,
    dice_loss,
    f1,
    focal_tversky_loss,
    jaccard,
    mcc_loss,
    mcc_metric,
    tversky,
    tversky_loss,
)
from .SandBoilNet import (
    PCALayer,
    attention_block,
    conv2d_bn,
    decoder_block,
    initial_conv2d_bn,
    iterLBlock,
    spatial_pooling_block,
)


try:
    import tensorflow_addons as tfa
except ImportError:
    class _TFAFallback:
        class layers:
            GroupNormalization = tf.keras.layers.GroupNormalization

    tfa = _TFAFallback()


class SpatialDropout2DCompat(tf.keras.layers.SpatialDropout2D):
    def __init__(self, rate, data_format=None, seed=None, **kwargs):
        trainable = kwargs.pop("trainable", True)
        kwargs.pop("noise_shape", None)
        super().__init__(rate=rate, data_format=data_format, seed=seed, **kwargs)
        self.trainable = trainable

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        cfg.pop("noise_shape", None)
        return cls(**cfg)


class DropoutCompat(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        trainable = kwargs.pop("trainable", True)
        super().__init__(rate=rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.trainable = trainable

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config))


class SeparableConv2DCompat(tf.keras.layers.SeparableConv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        kwargs.pop("groups", None)

        kernel_initializer = kwargs.pop("kernel_initializer", None)
        kernel_regularizer = kwargs.pop("kernel_regularizer", None)
        kernel_constraint = kwargs.pop("kernel_constraint", None)

        if "depthwise_initializer" not in kwargs and kernel_initializer is not None:
            kwargs["depthwise_initializer"] = kernel_initializer
        if "pointwise_initializer" not in kwargs and kernel_initializer is not None:
            kwargs["pointwise_initializer"] = kernel_initializer

        if "depthwise_regularizer" not in kwargs and kernel_regularizer is not None:
            kwargs["depthwise_regularizer"] = kernel_regularizer
        if "pointwise_regularizer" not in kwargs and kernel_regularizer is not None:
            kwargs["pointwise_regularizer"] = kernel_regularizer

        if "depthwise_constraint" not in kwargs and kernel_constraint is not None:
            kwargs["depthwise_constraint"] = kernel_constraint
        if "pointwise_constraint" not in kwargs and kernel_constraint is not None:
            kwargs["pointwise_constraint"] = kernel_constraint

        super().__init__(filters=filters, kernel_size=kernel_size, **kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config))


CUSTOM_OBJECTS = {
    "Addons>GroupNormalization": tfa.layers.GroupNormalization,
    "GroupNormalization": tfa.layers.GroupNormalization,
    "SpatialDropout2D": SpatialDropout2DCompat,
    "Dropout": DropoutCompat,
    "SeparableConv2D": SeparableConv2DCompat,
    "mcc_loss": mcc_loss,
    "mcc_metric": mcc_metric,
    "dice_coef": dice_coef,
    "dice_loss": dice_loss,
    "f1": f1,
    "tversky": tversky,
    "tversky_loss": tversky_loss,
    "focal_tversky_loss": focal_tversky_loss,
    "bce_dice_loss_new": bce_dice_loss_new,
    "jaccard": jaccard,
    "PCALayer": PCALayer,
    "bce_dice_loss": bce_dice_loss,
    "spatial_pooling_block": spatial_pooling_block,
    "attention_block": attention_block,
    "initial_conv2d_bn": initial_conv2d_bn,
    "conv2d_bn": conv2d_bn,
    "iterLBlock": iterLBlock,
    "decoder_block": decoder_block,
}
