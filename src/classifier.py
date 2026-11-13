import json
import logging
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "model", "artifacts")
CLASSES_PATH = os.path.join(ARTIFACTS_DIR, "classes.json")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.onnx")

# Used until a model is loaded; a loaded model dictates its own input size.
DEFAULT_INPUT_SIZE = 28
TOP_K = 3


class DrawingClassifier:
    """Loads the ONNX model and classes; there is no untrained fallback."""

    def __init__(self):
        self.model = None  # the onnxruntime session
        self.input_name = None
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
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as exc:  # onnxruntime raises its own types for a bad file
            raise ValueError(f"Cannot load the ONNX model {model_path}: {exc}") from exc
        self._adopt(session)
        self.model = session
        logger.info(
            "Model loaded from %s (input %dx%d)", model_path, self.input_size, self.input_size
        )

    def _adopt(self, session):
        """Take the input from the model; refuse one that does not fit classes.json."""
        inputs, outputs = session.get_inputs(), session.get_outputs()
        shape = inputs[0].shape if len(inputs) == 1 else None
        if not shape or len(shape) != 4 or shape[1] != shape[2] or shape[3] != 1:
            raise ValueError(f"Unsupported model input {shape}: expected [batch, N, N, 1]")
        if not isinstance(shape[1], int):
            raise ValueError(f"The model input size must be fixed, got {shape}")
        count = outputs[0].shape[-1] if len(outputs) == 1 else None
        if count != len(self.classes):
            raise ValueError(
                f"The model predicts {count} classes but classes.json lists {len(self.classes)}"
            )
        self.input_name = inputs[0].name
        self.input_size = int(shape[1])

    def predict(self, image):
        """Return the top predictions. Raises if the model is not loaded or fails."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        batch = self.preprocess_image(image)
        scores = self.model.run(None, {self.input_name: batch})[0][0]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        return [
            {
                "class": self.classes[idx] if idx < len(self.classes) else f"class_{idx}",
                "confidence": round(float(scores[idx]) * 100, 1),
            }
            for idx in top_indices
        ]
