import pytest
import numpy as np
import json
import os
import sys
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.drawing_classifier import DrawingClassifier

@pytest.fixture
def sample_image():
    """Create a sample 28x28 PIL image"""
    return Image.new('L', (28, 28), 128)  # Gray image

@pytest.fixture
def sample_numpy_image():
    """Create a sample 28x28 numpy array"""
    return np.random.randint(0, 255, (28, 28), dtype=np.uint8)

def test_drawing_classifier_init():
    """Test DrawingClassifier initialization"""
    # This might fail if classes.json doesn't exist
    try:
        classifier = DrawingClassifier()
        assert classifier.model is None  # Model not loaded yet
        assert isinstance(classifier.classes, list)
        assert len(classifier.classes) > 0
    except FileNotFoundError:
        # Expected if classes.json doesn't exist in test environment
        pytest.skip("classes.json not found - expected in test environment")

def test_preprocess_image_pil(sample_image):
    """Test image preprocessing with PIL Image"""
    try:
        classifier = DrawingClassifier()
        processed = classifier.preprocess_image(sample_image)

        assert processed.shape == (1, 28, 28, 1)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)
    except FileNotFoundError:
        pytest.skip("classes.json not found - expected in test environment")

def test_preprocess_image_numpy(sample_numpy_image):
    """Test image preprocessing with numpy array"""
    try:
        classifier = DrawingClassifier()
        processed = classifier.preprocess_image(sample_numpy_image)

        assert processed.shape == (1, 28, 28, 1)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)
    except FileNotFoundError:
        pytest.skip("classes.json not found - expected in test environment")

def test_create_simple_model():
    """Test model creation"""
    try:
        classifier = DrawingClassifier()
        model = classifier.create_simple_model()

        assert model is not None
        assert len(model.layers) > 0

        # Check input shape
        assert model.input_shape == (None, 28, 28, 1)

        # Check output shape
        assert model.output_shape == (None, len(classifier.classes))
    except FileNotFoundError:
        pytest.skip("classes.json not found - expected in test environment")

def test_predict_without_model(sample_image):
    """Test prediction when model is not loaded"""
    try:
        classifier = DrawingClassifier()
        classifier.model = None  # Ensure model is None

        predictions = classifier.predict(sample_image)

        assert isinstance(predictions, list)
        assert len(predictions) > 0
        assert predictions[0]['class'] == 'error'
        assert predictions[0]['confidence'] == 0.0
    except FileNotFoundError:
        pytest.skip("classes.json not found - expected in test environment")

def test_predict_with_simple_model(sample_image):
    """Test prediction with created model (untrained)"""
    try:
        classifier = DrawingClassifier()
        classifier.model = classifier.create_simple_model()

        predictions = classifier.predict(sample_image)

        assert isinstance(predictions, list)
        assert len(predictions) <= 3  # Top 3 predictions

        for pred in predictions:
            assert 'class' in pred
            assert 'confidence' in pred
            assert isinstance(pred['confidence'], (int, float))
            assert 0 <= pred['confidence'] <= 100
    except FileNotFoundError:
        pytest.skip("classes.json not found - expected in test environment")

if __name__ == '__main__':
    pytest.main([__file__])
