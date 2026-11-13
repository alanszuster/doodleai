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
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.keras"))
    with pytest.raises(FileNotFoundError):
        DrawingClassifier().load_model()


def test_corrupt_model_file_raises_and_is_not_replaced(tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.keras"
    corrupt.write_bytes(b"this is not a model")
    monkeypatch.setenv("MODEL_PATH", str(corrupt))
    clf = DrawingClassifier()
    with pytest.raises(ValueError):
        clf.load_model()
    assert clf.model is None


def tiny_model(path, size, outputs):
    """A throwaway model with the given square input and number of classes."""
    import keras

    model = keras.Sequential(
        [
            keras.Input((size, size, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(outputs, activation="softmax"),
        ]
    )
    model.save(path)
    return path


def test_the_loaded_model_dictates_the_input_size(tmp_path, monkeypatch, sample_image):
    """A new model with another input size must work without touching the code."""
    clf = DrawingClassifier()
    monkeypatch.setenv("MODEL_PATH", str(tiny_model(tmp_path / "m.keras", 64, len(clf.classes))))
    clf.load_model()
    assert clf.input_size == 64
    assert clf.preprocess_image(sample_image).shape == (1, 64, 64, 1)
    assert len(clf.predict(sample_image)) == 3


def test_a_model_that_does_not_fit_the_class_list_is_refused(tmp_path, monkeypatch):
    clf = DrawingClassifier()
    monkeypatch.setenv("MODEL_PATH", str(tiny_model(tmp_path / "m.keras", 64, 7)))
    with pytest.raises(ValueError, match="classes"):
        clf.load_model()
    assert clf.model is None


def test_a_model_with_an_unsupported_input_is_refused(tmp_path, monkeypatch):
    import keras

    clf = DrawingClassifier()
    path = tmp_path / "rgb.keras"
    keras.Sequential(
        [keras.Input((64, 64, 3)), keras.layers.Flatten(), keras.layers.Dense(len(clf.classes))]
    ).save(path)
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
