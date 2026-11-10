"""Training data pipeline: reads the arrays written by prepare.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_meta(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def load_split(data_dir: Path, split: str):
    """Small splits (val/test) are loaded fully: ``(N, S, S, 1)`` uint8 and ``(N,)`` int labels."""
    x = np.load(Path(data_dir) / f"X_{split}.npy")[..., None]
    y = np.load(Path(data_dir) / f"y_{split}.npy").astype(np.int32)
    return x, y


def to_model_input(x):
    """uint8 ``0..255`` -> float32 ``0..1`` (the same scaling the API uses)."""
    return tf.cast(x, tf.float32) / 255.0


def train_dataset(
    data_dir: Path,
    batch_size,
    n_classes,
    label_smoothing_onehot=True,
    chunk=8192,
    buffer=20000,
    seed=0,
):
    """Endless shuffled stream of ``(image, one_hot_label)`` batches.

    X_train.npy is memory-mapped and was shuffled across classes by prepare.py, so reading
    contiguous chunks in random order, plus a modest shuffle buffer, mixes classes well
    without ever holding the whole array in memory.
    """
    data_dir = Path(data_dir)
    x_all = np.load(data_dir / "X_train.npy", mmap_mode="r")
    y_all = np.load(data_dir / "y_train.npy").astype(np.int32)
    size = x_all.shape[1]
    starts = np.arange(0, len(x_all), chunk)

    def generator():
        rng = np.random.default_rng(seed)
        while True:
            for start in rng.permutation(starts):
                stop = min(start + chunk, len(x_all))
                yield np.asarray(x_all[start:stop])[..., None], y_all[start:stop]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((None, size, size, 1), tf.uint8),
            tf.TensorSpec((None,), tf.int32),
        ),
    ).unbatch()
    ds = ds.shuffle(buffer, seed=seed).batch(batch_size, drop_remainder=True)
    ds = ds.map(
        lambda x, y: (to_model_input(x), tf.one_hot(y, n_classes) if label_smoothing_onehot else y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE), len(x_all) // batch_size


def eval_dataset(x, y, batch_size, n_classes):
    ds = tf.data.Dataset.from_tensor_slices((x, tf.one_hot(y, n_classes)))
    return ds.batch(batch_size).map(lambda a, b: (to_model_input(a), b)).prefetch(tf.data.AUTOTUNE)
