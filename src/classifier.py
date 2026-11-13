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

# Used until a model is loaded; a loaded model dictates its own input size.
DEFAULT_INPUT_SIZE = 28
TOP_K = 3


class DrawingClassifier:
    """Loads the trained model and classes; there is no untrained fallback."""

    def __init__(self):
        self.model = None
        self.classes = []
        self.input_size = DEFAULT_INPUT_SIZE
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

        size = self.input_size
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        image = image.convert("L")
        img_array = 255 - np.array(image)
        return (img_array.astype("float32") / 255.0).reshape(1, size, size, 1)

    def load_model(self):
        """Load the model file or raise. A missing/corrupt model is never replaced."""
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        model = tf.keras.models.load_model(model_path)
        self._adopt(model)
        self.model = model
        logger.info(
            "Model loaded from %s (input %dx%d)", model_path, self.input_size, self.input_size
        )

    def _adopt(self, model):
        """Take the input size from the model; refuse one that does not fit classes.json."""
        shape = model.input_shape
        if len(shape) != 4 or shape[1] != shape[2] or shape[3] != 1:
            raise ValueError(f"Unsupported model input {shape}: expected (None, N, N, 1)")
        outputs = model.output_shape[-1]
        if outputs != len(self.classes):
            raise ValueError(
                f"The model predicts {outputs} classes but classes.json lists {len(self.classes)}"
            )
        self.input_size = int(shape[1])

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
