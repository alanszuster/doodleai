import numpy as np
import pytest
from PIL import Image

from classifier import DrawingClassifier
from render import draw_canvas, render_drawing


@pytest.fixture
def sample_image():
    return Image.new("L", (28, 28), 128)


@pytest.fixture
def loaded_classifier():
    clf = DrawingClassifier()
    clf.load_model()
    return clf


def test_classes_are_loaded():
    clf = DrawingClassifier()
    assert clf.model is None
    assert len(clf.classes) > 0


@pytest.mark.parametrize(
    "image",
    [
        Image.new("L", (28, 28), 128),
        Image.new("RGBA", (280, 280), (0, 0, 0, 0)),
        np.random.randint(0, 255, (28, 28), dtype=np.uint8),
    ],
)
def test_preprocess_gives_a_normalised_model_input(image):
    processed = DrawingClassifier().preprocess_image(image)
    assert processed.shape == (1, 28, 28, 1)
    assert processed.dtype == np.float32
    assert np.all(processed >= 0) and np.all(processed <= 1)


def test_predict_returns_top_three(loaded_classifier, sample_image):
    predictions = loaded_classifier.predict(sample_image)
    assert len(predictions) == 3
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0 <= c <= 100 for c in confidences)


def test_predict_without_a_model_raises(sample_image):
    with pytest.raises(RuntimeError):
        DrawingClassifier().predict(sample_image)


def test_missing_model_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError):
        DrawingClassifier().load_model()


def test_corrupt_model_file_raises_and_is_not_replaced(tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"this is not a model")
    monkeypatch.setenv("MODEL_PATH", str(corrupt))
    clf = DrawingClassifier()
    with pytest.raises(ValueError):
        clf.load_model()
    assert clf.model is None


def tiny_model(path, size, outputs, channels=1):
    """A throwaway ONNX model (flatten -> linear -> softmax) built without TensorFlow."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = np.random.default_rng(0).normal(size=(size * size * channels, outputs))
    nodes = [
        helper.make_node("Flatten", ["image"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "w"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [None, size, size, channels])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, outputs])],
        [numpy_helper.from_array(weights.astype(np.float32), "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)
    return path


def test_the_loaded_model_dictates_the_input_size(tmp_path, monkeypatch, sample_image):
    """A new model with another input size must work without touching the code."""
    clf = DrawingClassifier()
    monkeypatch.setenv("MODEL_PATH", str(tiny_model(tmp_path / "m.onnx", 64, len(clf.classes))))
    clf.load_model()
    assert clf.input_size == 64
    assert clf.preprocess_image(sample_image).shape == (1, 64, 64, 1)
    assert len(clf.predict(sample_image)) == 3


def test_a_model_that_does_not_fit_the_class_list_is_refused(tmp_path, monkeypatch):
    clf = DrawingClassifier()
    monkeypatch.setenv("MODEL_PATH", str(tiny_model(tmp_path / "m.onnx", 64, 7)))
    with pytest.raises(ValueError, match="classes"):
        clf.load_model()
    assert clf.model is None


def test_a_model_with_an_unsupported_input_is_refused(tmp_path, monkeypatch):
    clf = DrawingClassifier()
    path = tiny_model(tmp_path / "rgb.onnx", 64, len(clf.classes), channels=3)
    monkeypatch.setenv("MODEL_PATH", str(path))
    with pytest.raises(ValueError, match="input"):
        clf.load_model()


DRAWINGS = [
    [[[0, 255, 255, 0, 0], [0, 0, 255, 255, 0]]],  # a square
    [[[10, 120, 230], [240, 20, 240]], [[60, 190], [140, 140]]],  # a triangle and a bar
    [[[100], [100]]],  # a single tap
]


@pytest.mark.parametrize("size", [28, 64])
@pytest.mark.parametrize("drawing", DRAWINGS)
def test_the_api_sees_what_the_training_pipeline_rendered(drawing, size):
    """The website's picture, through the API's preprocessing, equals the training image.

    This pins the contract that makes the model behave in production as it did in evaluation:
    same shrink (LANCZOS), same polarity (ink = 1), same scaling.
    """
    clf = DrawingClassifier()
    clf.input_size = size
    from_api = clf.preprocess_image(draw_canvas(drawing))
    from_training = render_drawing(drawing, size).astype(np.float32) / 255.0
    assert np.array_equal(from_api[0, :, :, 0], from_training)


def test_loading_runs_one_throwaway_inference_so_the_first_request_is_fast(tmp_path, monkeypatch):
    import classifier as classifier_module

    runs = []

    class Counting(classifier_module.ort.InferenceSession):
        def run(self, *args, **kwargs):
            runs.append(1)
            return super().run(*args, **kwargs)

    clf = DrawingClassifier()
    monkeypatch.setattr(classifier_module.ort, "InferenceSession", Counting)
    monkeypatch.setenv("MODEL_PATH", str(tiny_model(tmp_path / "m.onnx", 32, len(clf.classes))))
    clf.load_model()
    assert runs == [1]


def test_predictions_are_calibrated_with_the_cards_temperature(tmp_path, monkeypatch, sample_image):
    import json

    import classifier as classifier_module

    clf = DrawingClassifier()
    monkeypatch.setenv("MODEL_PATH", str(tiny_model(tmp_path / "m.onnx", 28, len(clf.classes))))
    clf.load_model()
    plain = clf.predict(sample_image)

    card = tmp_path / "model_card.json"
    card.write_text(json.dumps({"temperature": 3.0}))
    monkeypatch.setattr(classifier_module, "CARD_PATH", str(card))
    calibrated = DrawingClassifier()
    calibrated.load_model()
    assert calibrated.temperature == 3.0
    after = calibrated.predict(sample_image)

    assert [p["class"] for p in after] == [p["class"] for p in plain]  # the ranking is unchanged
    assert after[0]["confidence"] < plain[0]["confidence"]  # T > 1 softens the confidence
