import json
import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "model", "artifacts")
CLASSES_PATH = os.path.join(ARTIFACTS_DIR, "classes.json")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.keras")

INPUT_SIZE = 28
TOP_K = 3


class DrawingClassifier:
    """Loads the trained model and classes; there is no untrained fallback."""

    def __init__(self):
        self.model = None
        self.classes = []
        self.load_classes()

    def load_classes(self):
        if not os.path.exists(CLASSES_PATH):
            raise FileNotFoundError(f"Required file {CLASSES_PATH} not found")
        with open(CLASSES_PATH, encoding="utf-8") as f:
            classes_dict = json.load(f)
        self.classes = [classes_dict[str(i)] for i in range(len(classes_dict))]
        logger.info("Loaded %d classes from %s", len(self.classes), CLASSES_PATH)

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
        image = image.convert("L")
        img_array = 255 - np.array(image)
        return (img_array.astype("float32") / 255.0).reshape(1, INPUT_SIZE, INPUT_SIZE, 1)

    def load_model(self):
        """Load the model file or raise. A missing/corrupt model is never replaced."""
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded from %s", model_path)

    def predict(self, image):
        """Return the top predictions. Raises if the model is not loaded or fails."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        scores = self.model.predict(self.preprocess_image(image), verbose=0)[0]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        return [
            {
                "class": self.classes[idx] if idx < len(self.classes) else f"class_{idx}",
                "confidence": round(float(scores[idx]) * 100, 1),
            }
            for idx in top_indices
        ]
