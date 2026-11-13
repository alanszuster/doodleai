"""Export a trained Keras model to ONNX, the format the API serves.

    python model/training/export.py --model runs/main/best.keras --out best_model.onnx

Needs the ``training`` dependency group (TensorFlow and tf2onnx). The export is checked against
Keras on random input and fails loudly if the two disagree, so a bad conversion cannot ship.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Random augmentation layers do nothing at inference, and ONNX Runtime cannot run them.
AUGMENTATION = (
    "RandomRotation",
    "RandomTranslation",
    "RandomZoom",
    "RandomFlip",
    "RandomContrast",
    "RandomCrop",
    "RandomBrightness",
)


def strip_augmentation(model):
    """The model without its random augmentation layers (same layers, same weights).

    Models from this pipeline never contain them: train.py ships the inner model. The first
    model, trained before that, did. At inference the outputs are identical either way.
    """
    import keras

    layers = [layer for layer in model.layers if type(layer).__name__ not in AUGMENTATION]
    if len(layers) == len(model.layers):
        return model
    if not isinstance(model, keras.Sequential):
        raise ValueError("Augmentation can only be stripped from a Sequential model")
    return keras.Sequential([keras.Input(model.input_shape[1:]), *layers])


def export_onnx(keras_path, onnx_path, *, tolerance=1e-4, samples=16, seed=0) -> float:
    """Convert ``keras_path`` to ``onnx_path``; returns the largest output difference vs Keras."""
    import keras
    import onnxruntime

    model = strip_augmentation(keras.models.load_model(keras_path))
    batch = np.random.default_rng(seed).random((samples, *model.input_shape[1:]), dtype=np.float32)
    expected = model.predict(batch, verbose=0)  # Keras refuses to export a model never called

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(onnx_path), format="onnx")

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: batch})[0]
    diff = float(np.abs(expected - got).max())
    if diff > tolerance or not (expected.argmax(axis=1) == got.argmax(axis=1)).all():
        raise RuntimeError(f"The ONNX export differs from Keras by {diff:.2e} (limit {tolerance})")
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="the .keras file")
    parser.add_argument("--out", type=Path, required=True, help="the .onnx file to write")
    args = parser.parse_args()
    diff = export_onnx(args.model, args.out)
    print(f"Wrote {args.out} (max difference vs Keras {diff:.1e})")


if __name__ == "__main__":
    main()
