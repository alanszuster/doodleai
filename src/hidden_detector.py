"""Optional binary detector, separate from the public classifier."""

import math
import os

import numpy as np
import onnxruntime as ort


class HiddenDetector:
    def __init__(self, path, threshold):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("HIDDEN_DETECTOR_THRESHOLD must be between 0 and 1")
        self.threshold = threshold
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(inputs[0].shape) != 4
            or inputs[0].shape[1:] != [64, 64, 1]
            or len(outputs) != 1
            or outputs[0].shape[-1] != 2
        ):
            raise ValueError(
                "Hidden detector must accept 64x64 grayscale images and output 2 scores"
            )
        self.input_name = inputs[0].name
        self.session.run(None, {self.input_name: np.zeros((1, 64, 64, 1), dtype=np.float32)})

    def matches(self, batch):
        scores = self.session.run(None, {self.input_name: batch})[0]
        if scores.shape != (1, 2):
            raise ValueError("Invalid hidden detector output shape")
        score = float(scores[0, 1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid hidden detector score")
        return score >= self.threshold


def load_hidden_detector():
    path = os.getenv("HIDDEN_DETECTOR_MODEL")
    if not path:
        return None
    threshold = os.getenv("HIDDEN_DETECTOR_THRESHOLD")
    if not threshold:
        raise ValueError("HIDDEN_DETECTOR_THRESHOLD is required when the detector is enabled")
    return HiddenDetector(path, float(threshold))
