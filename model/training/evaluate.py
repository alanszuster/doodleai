"""Evaluate a trained model on the held-out test split.

    python model/training/evaluate.py --model best.keras --data-dir <work>/data --out reports/

Writes ``metrics.json`` (top-1/top-3, macro F1, per-class accuracy, most confused pairs,
calibration) and ``confusion_matrix.png`` into ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import keras
import numpy as np

from data import eval_dataset, load_meta, load_split


def confusion_matrix(y_true, y_pred, n):
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def macro_f1(matrix):
    tp = np.diag(matrix).astype(float)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.mean())


def expected_calibration_error(probs, y_true, bins=15):
    confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def log_loss(probs, y_true):
    """Plain cross-entropy of the true class. Unlike the training loss it has no label smoothing,
    so it can approach 0 and means the same thing for every model."""
    true_prob = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.log(true_prob).mean())


def compute_metrics(probs, y_true, classes):
    n = len(classes)
    y_pred = probs.argmax(axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    matrix = confusion_matrix(y_true, y_pred, n)
    per_class = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    worst = np.dstack(
        np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1][:20], matrix.shape)
    )[0]
    metrics = {
        "samples": int(len(y_true)),
        "top1": float((y_pred == y_true).mean()),
        "top3": float((top3 == y_true[:, None]).any(axis=1).mean()),
        "macro_f1": macro_f1(matrix),
        "ece": expected_calibration_error(probs, y_true),
        "log_loss": log_loss(probs, y_true),
        "per_class_accuracy": {c: float(a) for c, a in zip(classes, per_class, strict=True)},
        "most_confused": [
            {"true": classes[i], "predicted": classes[j], "count": int(matrix[i, j])}
            for i, j in worst
            if matrix[i, j] > 0
        ],
    }
    return metrics, matrix


def save_confusion_plot(matrix, classes, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the confusion matrix plot")
        return
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(normalised, cmap="viridis")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=5)
    ax.set_yticks(range(len(classes)), classes, fontsize=5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def read_history(path):
    """``history.csv`` as ``{column: float array}``; NaN stays NaN (a diverged run)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]} if rows else {}


def save_history_plot(history_csv, path, title="Training history"):
    """Loss and accuracy per epoch, with the best validation epoch marked."""
    history = read_history(history_csv)
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the training history plot")
        return
    epochs = history["epoch"] + 1
    blue, amber, green, grey = "#2563eb", "#f59e0b", "#10b981", "#6b7280"
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_loss.plot(epochs, history["loss"], color=blue, lw=2, label="train")
    ax_loss.plot(epochs, history["val_loss"], color=amber, lw=2, label="validation")
    ax_loss.set_title("Loss", loc="left", fontweight="bold")
    ax_acc.plot(epochs, history["accuracy"] * 100, color=blue, lw=2, label="train top-1")
    ax_acc.plot(epochs, history["val_accuracy"] * 100, color=amber, lw=2, label="validation top-1")
    ax_acc.plot(epochs, history["val_top3"] * 100, color=green, lw=2, label="validation top-3")
    ax_acc.set_title("Accuracy (%)", loc="left", fontweight="bold")
    finite = np.isfinite(history["val_accuracy"])
    if finite.any():
        best = int(np.nanargmax(np.where(finite, history["val_accuracy"], -np.inf)))
        value = history["val_accuracy"][best] * 100
        ax_acc.scatter([epochs[best]], [value], color=amber, s=60, zorder=3)
        ax_acc.annotate(
            f"best {value:.1f}% (epoch {int(epochs[best])})",
            (epochs[best], value),
            textcoords="offset points",
            xytext=(-10, 10),
            ha="right",
            fontsize=9,
            color=grey,
        )
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=grey)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def evaluate(model_path, data_dir, out_dir, split="test", batch_size=512):
    meta = load_meta(data_dir)
    classes = meta["classes"]
    x, y = load_split(data_dir, split)
    model = keras.models.load_model(model_path)
    probs = model.predict(
        eval_dataset(x, y, batch_size, len(classes)).map(lambda a, b: a), verbose=0
    ).astype(np.float64)
    metrics, matrix = compute_metrics(probs, y, classes)
    metrics.update(
        run_id=os.environ.get("DOODLEAI_RUN_ID"),
        data_fingerprint=meta.get("fingerprint"),
        image_size=meta["image_size"],
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_plot(matrix, classes, out_dir / "confusion_matrix.png")
    history = Path(model_path).parent / "history.csv"  # written by train.py next to the model
    if history.exists():
        save_history_plot(
            history, out_dir / "training_history.png", f"Training history: {metrics['run_id']}"
        )
    print(
        f"{split}: top-1 {metrics['top1']:.2%}  top-3 {metrics['top3']:.2%}  "
        f"macro-F1 {metrics['macro_f1']:.3f}  ECE {metrics['ece']:.3f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.out, args.split)


if __name__ == "__main__":
    main()
