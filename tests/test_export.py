"""The Keras -> ONNX conversion that ships the model (needs the training group)."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")

import keras  # noqa: E402
import onnxruntime  # noqa: E402

import export  # noqa: E402
import registry  # noqa: E402
from network import build_model  # noqa: E402


def test_a_pipeline_model_exports_and_matches_keras(tmp_path):
    keras_path = tmp_path / "m.keras"
    build_model(32, 5, [4, 8], [1, 1]).save(keras_path)
    diff = export.export_onnx(keras_path, tmp_path / "m.onnx")
    assert diff < 1e-4
    session = onnxruntime.InferenceSession(str(tmp_path / "m.onnx"))
    assert session.get_inputs()[0].shape[1:] == [32, 32, 1]


def test_augmentation_layers_are_stripped_before_export(tmp_path):
    """The first model has random augmentation baked in; ONNX Runtime cannot run it."""
    model = keras.Sequential(
        [
            keras.Input((16, 16, 1)),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.save(tmp_path / "aug.keras")
    assert export.export_onnx(tmp_path / "aug.keras", tmp_path / "aug.onnx") < 1e-4


def test_a_model_that_cannot_be_stripped_is_refused():
    inputs = keras.Input((8, 8, 1))
    outputs = keras.layers.Dense(2)(keras.layers.RandomZoom(0.1)(inputs))
    with pytest.raises(ValueError, match="Sequential"):
        export.strip_augmentation(keras.Model(inputs, outputs))


def test_the_shipped_onnx_matches_the_shipped_keras_model():
    """The two committed files must describe the same network."""
    keras_model = export.strip_augmentation(
        keras.models.load_model(registry.ARTIFACTS_DIR / registry.MODEL_NAME)
    )
    session = onnxruntime.InferenceSession(str(registry.ARTIFACTS_DIR / registry.ONNX_NAME))
    batch = np.random.default_rng(1).random((8, *keras_model.input_shape[1:]), dtype=np.float32)
    got = session.run(None, {session.get_inputs()[0].name: batch})[0]
    assert np.abs(keras_model.predict(batch, verbose=0) - got).max() < 1e-4
