"""Find out what limits training speed, before optimising anything.

    python model/training/bench.py [--data-dir <work>/data] [--variants baseline,fp16,xla]

Times a few steps of the real model under variants that each change one thing: precision,
XLA, the augmentation layers, and (with ``--data-dir``) the real input pipeline instead of one
batch kept in memory. The one that moves ms/step most is what to work on. Run it on the target
GPUs (a Kaggle notebook cell is enough); it uses the same multi-GPU strategy as train.py and
reads the model size and batch size from config.toml unless overridden.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np

from network import build_augmentation, build_model
from prepare import DEFAULT_CONFIG, load_config

# name -> (precision, xla, augmentation, real data)
VARIANTS = {
    "baseline": ("fp32", False, True, False),
    "no-aug": ("fp32", False, False, False),
    "fp16": ("fp16", False, True, False),
    "xla": ("fp32", True, True, False),
    "fp16+xla": ("fp16", True, True, False),
    "real-data": ("fp32", False, True, True),
}


def build(size, classes, widths, blocks, *, precision, xla, augment):
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("mixed_float16" if precision == "fp16" else "float32")
    inner = build_model(size, classes, widths, blocks, 0.3)
    if augment:
        inputs = keras.Input(shape=(size, size, 1))
        model = keras.Model(inputs, inner(build_augmentation(0.04, 0.05, 0.1)(inputs)))
    else:
        model = inner
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        jit_compile=xla,
    )
    return model


def synthetic(size, classes, batch_size):
    import tensorflow as tf

    rng = np.random.default_rng(0)
    x = rng.random((batch_size, size, size, 1), dtype=np.float32)
    y = np.eye(classes, dtype=np.float32)[rng.integers(0, classes, batch_size)]
    return tf.data.Dataset.from_tensors((x, y)).repeat()


def seconds_per_step(model, dataset, steps, warmup):
    model.fit(dataset, steps_per_epoch=warmup, epochs=1, verbose=0)  # builds, compiles, warms up
    start = time.perf_counter()
    model.fit(dataset, steps_per_epoch=steps, epochs=1, verbose=0)
    return (time.perf_counter() - start) / steps


def run(
    variants, *, size, classes, batch_size, steps, warmup, widths, blocks, data_dir=None, seed=0
):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    print(f"GPUs: {len(gpus)}  replicas: {strategy.num_replicas_in_sync}  batch: {batch_size}")
    results = []
    for name in variants:
        precision, xla, augment, real = VARIANTS[name]
        if real and data_dir is None:
            print(f"{name}: skipped (needs --data-dir)")
            continue
        with strategy.scope():
            model = build(
                size, classes, widths, blocks, precision=precision, xla=xla, augment=augment
            )
        if real:
            from data import train_dataset

            dataset, _ = train_dataset(data_dir, batch_size, classes, seed=seed)
        else:
            dataset = synthetic(size, classes, batch_size)
        seconds = seconds_per_step(model, dataset, steps, warmup)
        row = {"variant": name, "ms_per_step": seconds * 1000, "img_per_s": batch_size / seconds}
        print(f"{name:10} {row['ms_per_step']:8.1f} ms/step  {row['img_per_s']:9.0f} img/s")
        results.append(row)
    keras.mixed_precision.set_global_policy("float32")
    return results


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="baseline,no-aug,fp16,xla,fp16+xla,real-data")
    parser.add_argument("--data-dir", type=Path, help="prepared data, for the real-data variant")
    parser.add_argument("--batch-size", type=int, default=cfg["train"]["batch_size"])
    parser.add_argument("--size", type=int, default=cfg["data"]["image_size"])
    parser.add_argument("--classes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--out", type=Path, help="also write the results as JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    results = run(
        [v.strip() for v in args.variants.split(",") if v.strip()],
        size=args.size,
        classes=args.classes,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        widths=cfg["train"]["widths"],
        blocks=cfg["train"]["blocks"],
        data_dir=args.data_dir,
    )
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
