import numpy as np
import pytest
from PIL import Image

from classifier import DrawingClassifier


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
