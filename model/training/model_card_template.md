---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
  - sketch-recognition
  - quickdraw
  - image-classification
  - onnx
metrics:
  - accuracy
---

# doodleai: hand-drawn sketch classifier

A small residual CNN that classifies a hand-drawn sketch into one of {{CLASSES}} categories. Input
is a {{SIZE}}x{{SIZE}} grayscale image with ink = 1, output is a probability per class.

Live demo: https://www.alanszuster.dev/#ai-playground. Code and training pipeline:
https://github.com/alanszuster/doodleai. This revision is training run `{{RUN_ID}}`.

## Results

On a fixed held-out test set of {{TEST_SAMPLES}} drawings, identical for every model compared:

| top-1 | top-3 | macro F1 | calibration error (ECE) |
|---|---|---|---|
| {{TOP1}} | {{TOP3}} | {{F1}} | {{ECE}} |

Hardest classes: {{HARDEST}}.

## Files

- `best_model.onnx`: the model the API serves (input name `image`, shape `[batch, {{SIZE}}, {{SIZE}}, 1]`).
- `best_model.keras`: the Keras model, to continue training or evaluate.
- `model_card.json`: metrics per class, the test-set fingerprint, the git commit and the file hashes.
- `classes.json`: class index to name.

## Use

```python
import numpy as np, onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("best_model.onnx", providers=["CPUExecutionProvider"])
img = Image.open("sketch.png").convert("L").resize(({{SIZE}}, {{SIZE}}), Image.Resampling.LANCZOS)
x = (255 - np.asarray(img, dtype=np.float32)).reshape(1, {{SIZE}}, {{SIZE}}, 1) / 255.0
probs = session.run(None, {"image": x})[0][0]
```

Black strokes on a white background, one object, roughly centred and filling the frame.

## Training data

[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) simplified strokes, only
drawings the game recognised, rendered the way the demo website draws them (cropped, centred, 280 px
canvas, 22 px strokes) and then shrunk with LANCZOS. The data is CC BY 4.0: give credit to the
dataset when you redistribute the model. This project is not endorsed by Google.

## Limitations

The model knows only its {{CLASSES}} classes and always answers with one of them; it has no "none of
these" output. It was trained on drawings made for a game, so unusual styles and several objects in
one picture work worse.
