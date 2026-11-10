
import numpy as np
import pytest
from PIL import Image

import classifier as classifier_module
from classifier import DrawingClassifier


@pytest.fixture
def sample_image():
    return Image.new('L', (28, 28), 128)


@pytest.fixture
def sample_numpy_image():
    return np.random.randint(0, 255, (28, 28), dtype=np.uint8)


def test_drawing_classifier_init():
    try:
        classifier = DrawingClassifier()
        assert classifier.model is None
        assert isinstance(classifier.classes, list)
        assert len(classifier.classes) > 0
    except FileNotFoundError:
        pytest.skip("classes.json not found")


def test_preprocess_image_pil(sample_image):  # pylint: disable=redefined-outer-name
    try:
        classifier = DrawingClassifier()
        processed = classifier.preprocess_image(sample_image)
        assert processed.shape == (1, 28, 28, 1)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)
    except FileNotFoundError:
        pytest.skip("classes.json not found")


def test_preprocess_image_numpy(sample_numpy_image):  # pylint: disable=redefined-outer-name
    try:
        classifier = DrawingClassifier()
        processed = classifier.preprocess_image(sample_numpy_image)
        assert processed.shape == (1, 28, 28, 1)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)
    except FileNotFoundError:
        pytest.skip("classes.json not found")


def test_create_simple_model():
    try:
        classifier = DrawingClassifier()
        model = classifier.create_simple_model()
        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, len(classifier.classes))
    except FileNotFoundError:
        pytest.skip("classes.json not found")


def test_predict_without_model(sample_image):  # pylint: disable=redefined-outer-name
    try:
        classifier = DrawingClassifier()
        classifier.model = None
        predictions = classifier.predict(sample_image)
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        assert predictions[0]['class'] == 'error'
        assert predictions[0]['confidence'] == 0.0
    except FileNotFoundError:
        pytest.skip("classes.json not found")


def test_predict_with_simple_model(sample_image):  # pylint: disable=redefined-outer-name
    try:
        classifier = DrawingClassifier()
        classifier.model = classifier.create_simple_model()
        predictions = classifier.predict(sample_image)
        assert isinstance(predictions, list)
        assert len(predictions) <= 3
        for pred in predictions:
            assert 'class' in pred
            assert 'confidence' in pred
            assert isinstance(pred['confidence'], (int, float))
            assert 0 <= pred['confidence'] <= 100
    except FileNotFoundError:
        pytest.skip("classes.json not found")


if __name__ == '__main__':
    pytest.main([__file__])


def test_new_classifier_is_not_pretrained():
    try:
        assert DrawingClassifier().pretrained is False
    except FileNotFoundError:
        pytest.skip("classes.json not found")


def test_model_loaded_from_file_is_pretrained():
    try:
        classifier = DrawingClassifier()
    except FileNotFoundError:
        pytest.skip("classes.json not found")
    classifier.load_model()
    assert classifier.model is not None
    assert classifier.pretrained is True


def test_missing_model_file_falls_back_to_an_untrained_model(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier_module, 'DEFAULT_MODEL_PATH', str(tmp_path / 'missing.keras'))
    monkeypatch.delenv('MODEL_PATH', raising=False)
    try:
        classifier = DrawingClassifier()
    except FileNotFoundError:
        pytest.skip("classes.json not found")
    classifier.load_model()
    assert classifier.model is not None
    assert classifier.pretrained is False


def test_corrupt_model_file_is_not_treated_as_trained(tmp_path, monkeypatch):
    corrupt = tmp_path / 'corrupt.keras'
    corrupt.write_bytes(b'this is not a model')
    monkeypatch.setattr(classifier_module, 'DEFAULT_MODEL_PATH', str(corrupt))
    monkeypatch.delenv('MODEL_PATH', raising=False)
    try:
        classifier = DrawingClassifier()
    except FileNotFoundError:
        pytest.skip("classes.json not found")
    classifier.load_model()
    assert classifier.pretrained is False
