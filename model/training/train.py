"""Train the classifier.

    python model/training/train.py --work-dir /path/to/work [--resume]

Reads ``<work-dir>/data`` (see prepare.py) and writes ``<runs-dir>/<name>/`` (default
``<work-dir>/runs``; point it at persistent storage on a platform that can lose the disk):
    best.keras     the model to ship (best validation accuracy, no augmentation inside)
    last.keras     full training state, used by --resume after an interrupted session
    history.csv, config.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import keras

from data import eval_dataset, load_meta, load_split, train_dataset
from network import build_augmentation, build_model
from prepare import DEFAULT_CONFIG, load_config


@keras.saving.register_keras_serializable(package="doodleai")
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up, then cosine decay down to ``floor``."""

    def __init__(self, peak, warmup_steps, total_steps, floor=1e-5):
        super().__init__()
        self.peak, self.warmup_steps = peak, warmup_steps
        self.total_steps, self.floor = total_steps, floor

    def __call__(self, step):
        import tensorflow as tf

        step = tf.cast(step, tf.float32)
        warmup = self.peak * (step + 1.0) / max(self.warmup_steps, 1)
        progress = tf.clip_by_value(
            (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1), 0.0, 1.0
        )
        cosine = self.floor + 0.5 * (self.peak - self.floor) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "peak": self.peak,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "floor": self.floor,
        }


class SaveBest(keras.callbacks.Callback):
    """Save the inner (augmentation-free) model whenever validation accuracy improves."""

    def __init__(self, inner, path):
        super().__init__()
        self.inner, self.path, self.best = inner, path, -1.0

    def on_epoch_end(self, epoch, logs=None):
        score = (logs or {}).get("val_accuracy", -1.0)
        if score > self.best:
            self.best = score
            self.inner.save(self.path)
            print(f"\n  new best val_accuracy {score:.4f} -> {self.path}")


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_training_model(cfg, meta):
    t, a = cfg["train"], cfg["augment"]
    n_classes, size = len(meta["classes"]), meta["image_size"]
    inner = build_model(size, n_classes, t["widths"], t["blocks"], t["dropout"])
    inputs = keras.Input(shape=(size, size, 1), name="image")
    outputs = inner(build_augmentation(a["rotation"], a["translation"], a["zoom"])(inputs))
    return keras.Model(inputs, outputs, name="training_wrapper"), inner


def train(
    cfg: dict,
    work_dir: Path,
    *,
    name=None,
    resume=False,
    epochs=None,
    max_steps=None,
    runs_dir=None,
) -> Path:
    t = cfg["train"]
    data_dir = work_dir / "data"
    meta = load_meta(data_dir)
    n_classes = len(meta["classes"])

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus and t["mixed_precision"]:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPUs: {len(gpus)}  mixed precision: {keras.mixed_precision.global_policy().name}")

    run_dir = (runs_dir or work_dir / "runs") / (name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"config": cfg, "git_commit": _git_commit()}, indent=2), encoding="utf-8"
    )

    ds, steps = train_dataset(
        data_dir,
        t["batch_size"],
        n_classes,
        chunk=t["shuffle_chunk"],
        buffer=t["shuffle_buffer"],
        seed=cfg["data"]["seed"],
    )
    steps = min(steps, max_steps) if max_steps else steps
    x_val, y_val = load_split(data_dir, "val")
    val_ds = eval_dataset(x_val, y_val, t["batch_size"], n_classes)
    total_epochs = epochs or t["epochs"]

    state_path, last_path, best_path = (
        run_dir / "state.json",
        run_dir / "last.keras",
        run_dir / "best.keras",
    )
    initial_epoch = 0
    if resume and last_path.exists():
        wrapper = keras.models.load_model(last_path)
        inner = wrapper.get_layer(index=-1)
        initial_epoch = json.loads(state_path.read_text())["epochs_done"]
        print(f"Resuming from epoch {initial_epoch}")
    else:
        wrapper, inner = build_training_model(cfg, meta)
        schedule = WarmupCosine(
            t["learning_rate"], t["warmup_epochs"] * steps, total_epochs * steps
        )
        wrapper.compile(
            optimizer=keras.optimizers.AdamW(schedule, weight_decay=t["weight_decay"]),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=t["label_smoothing"]),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            ],
        )
    inner.summary(print_fn=lambda line: None)
    print(f"Parameters: {inner.count_params():,}")

    class Checkpoint(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save(last_path)
            state_path.write_text(json.dumps({"epochs_done": epoch + 1}))

    callbacks = [
        SaveBest(inner, best_path),
        Checkpoint(),
        keras.callbacks.CSVLogger(run_dir / "history.csv", append=resume),
    ]
    wrapper.fit(
        ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps,
        callbacks=callbacks,
        verbose=2,
    )
    print(f"Best model: {best_path}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--runs-dir", type=Path, help="where runs are saved (default: <work-dir>/runs)"
    )
    parser.add_argument("--name", help="run name (default: timestamp); reuse it with --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train(
        load_config(args.config),
        args.work_dir,
        name=args.name,
        resume=args.resume,
        epochs=args.epochs,
        runs_dir=args.runs_dir,
    )


if __name__ == "__main__":
    main()
