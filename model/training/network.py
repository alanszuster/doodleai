"""The classifier network: a small residual CNN for 64x64 sketches."""

from __future__ import annotations

import keras
from keras import layers

# Keras' default of 0.99 needs a few hundred steps before inference-mode statistics are
# usable; 0.9 gets there in about fifty, which matters for short runs and early epochs.
BN_MOMENTUM = 0.9


def _conv_bn(x, filters, stride=1, activation=True):
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    return layers.ReLU()(x) if activation else x


def _residual_block(x, filters, stride):
    shortcut = x
    y = _conv_bn(x, filters, stride)
    y = _conv_bn(y, filters, 1, activation=False)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, use_bias=False, kernel_initializer="he_normal"
        )(x)
        shortcut = layers.BatchNormalization(momentum=BN_MOMENTUM)(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))


def build_model(image_size: int, n_classes: int, widths, blocks, dropout=0.3) -> keras.Model:
    """Image in ``[0, 1]`` (ink = 1), ``n_classes`` softmax probabilities out.

    The model takes ``uint8``-range input scaled by the caller, contains no augmentation and
    ends in a float32 softmax, so it can be exported and served as is.
    """
    inputs = keras.Input(shape=(image_size, image_size, 1), name="image")
    x = _conv_bn(inputs, widths[0])
    for stage, (filters, count) in enumerate(zip(widths, blocks, strict=True)):
        for block in range(count):
            x = _residual_block(x, filters, stride=2 if stage > 0 and block == 0 else 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name="doodle_resnet")


def build_augmentation(rotation: float, translation: float, zoom: float) -> keras.Sequential:
    """Random geometric augmentation, active only while training."""
    return keras.Sequential(
        [
            layers.RandomRotation(rotation, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(
                translation, translation, fill_mode="constant", fill_value=0.0
            ),
            layers.RandomZoom(zoom, fill_mode="constant", fill_value=0.0),
        ],
        name="augmentation",
    )
