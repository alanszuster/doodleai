"""Select a binary detector threshold on validation data and report held-out test results."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from data import load_split


def choose_threshold(scores, labels, max_fpr):
    scores, labels = validate_data(scores, labels)
    if not 0 <= max_fpr < 1:
        raise ValueError("max-fpr must be at least 0 and below 1")
    negatives = np.sort(scores[labels == 0])
    if not len(negatives) or not np.any(labels == 1):
        raise ValueError("Validation must include both classes")
    allowed = int(max_fpr * len(negatives))
    cutoff = negatives[-allowed - 1]
    threshold = float(np.nextafter(cutoff, np.inf))
    if not 0 < threshold <= 1:
        raise ValueError("No usable threshold satisfies the requested false-positive rate")
    return threshold


def rates(scores, labels, threshold):
    scores, labels = validate_data(scores, labels)
    return (
        float(np.mean(scores[labels == 0] >= threshold)),
        float(np.mean(scores[labels == 1] >= threshold)),
    )


def validate_data(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        scores.ndim != 1
        or labels.shape != scores.shape
        or not np.all(np.isfinite(scores))
        or np.any((scores < 0) | (scores > 1))
        or not np.all(np.isin(labels, [0, 1]))
        or not np.any(labels == 0)
        or not np.any(labels == 1)
    ):
        raise ValueError("Scores must be finite probabilities with both binary labels")
    return scores, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-fpr", type=float, required=True)
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(split):
        images, labels = load_split(args.data_dir, split)
        scores = []
        for start in range(0, len(images), 512):
            batch = images[start : start + 512].astype(np.float32) / 255.0
            scores.append(session.run(None, {input_name: batch})[0][:, 1])
        return np.concatenate(scores), labels

    val_scores, val_labels = predict("val")
    threshold = choose_threshold(val_scores, val_labels, args.max_fpr)
    test_scores, test_labels = predict("test")
    print(f"threshold={threshold!r}")
    for name, scores, labels in (
        ("validation", val_scores, val_labels),
        ("test", test_scores, test_labels),
    ):
        fpr, recall = rates(scores, labels, threshold)
        print(f"{name}: false_positive_rate={fpr:.4%}, recall={recall:.4%}")


if __name__ == "__main__":
    main()
