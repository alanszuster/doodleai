# AI Drawing Classifier API

[![CI and deploy](https://github.com/alanszuster/doodleai/actions/workflows/deploy-hf.yml/badge.svg)](https://github.com/alanszuster/doodleai/actions/workflows/deploy-hf.yml)
[![Release](https://img.shields.io/github/v/release/alanszuster/doodleai)](https://github.com/alanszuster/doodleai/releases)
[![License](https://img.shields.io/github/license/alanszuster/doodleai)](LICENSE)

**Try it live: [alanszuster.dev, AI Playground](https://www.alanszuster.dev/#ai-playground)**

REST API for recognising hand-drawn sketches. A convolutional neural network trained on Google's
[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset classifies a drawing
into one of 120 categories and returns the top three guesses. It powers the AI Playground on
[alanszuster.dev](https://www.alanszuster.dev).

<p align="center">
  <img src="https://raw.githubusercontent.com/alanszuster/doodleai/main/assets/ai-playground.png" alt="The AI Playground on alanszuster.dev classifying a drawn house" width="460">
</p>

## Stack

- Python 3.12, Flask, gunicorn
- ONNX Runtime (serving), TensorFlow / Keras (training)
- uv, Docker
- GitHub Actions, deployed to Hugging Face Spaces
- Training on a free Kaggle GPU, driven from a CLI

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Classify a drawing (PNG) |
| `GET /classes` | Supported categories |
| `GET /status` | Status, build version and served model |
| `GET /health` | Public liveness probe |

Everything except `/` and `/health` requires authentication.

## Run locally

```bash
uv sync
AUTH_ALLOW_API_KEY=true AI_API_KEY=dev-key uv run python src/app.py   # http://localhost:5000
```

Or with Docker:

```bash
docker build -t doodleai .
docker run -p 8080:8080 -e AUTH_ALLOW_API_KEY=true -e AI_API_KEY=dev-key doodleai
```

Tests and checks:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Training

`model/training/` holds the whole pipeline: data preparation, training, evaluation, experiment
records and the gate that decides which model reaches production. See
[model/training/README.md](model/training/README.md).

### Bundled model

The model included in this repository is run `20261116-132958-full`: a residual CNN trained for
15 epochs on Kaggle T4 x2 GPUs. It takes 64x64 grayscale images and predicts 120 classes. The best
validation checkpoint was saved at epoch 14.

Results on 120,000 held-out Quick Draw drawings (1,000 per class):

| Metric | Value |
|---|---:|
| Top-1 accuracy | 92.09% |
| Top-3 accuracy | 98.41% |
| Macro F1 | 0.921 |
| ECE before / after calibration | 0.0656 / 0.0194 |
| Log loss before / after calibration | 0.3472 / 0.2920 |

Calibration uses a temperature of 0.7871, fitted on the validation split. The API reads it from
the [model card](model/artifacts/model_card.json) and applies it to the model's probabilities.
The card records the source commit, dataset fingerprint, Kaggle version and model file hashes.

![Training and validation loss and accuracy over 15 epochs](model/reports/training_history.png)

### Confusion matrix

Rows are the true classes; columns are the predictions. Values are normalized per row.
Open the image at full size to read all 120 class labels.

![Row-normalized confusion matrix for 120 drawing classes](model/reports/confusion_matrix.png)

[Epoch-by-epoch data](model/reports/history.csv) ·
[Confusion matrix](model/reports/confusion_matrix.png) ·
[Full experiment metrics and per-class accuracy](model/experiments/20261116-132958-full.json)

## Repository layout

```
src/            API (Flask)
model/          served model (artifacts/) and training pipeline (training/)
tests/          API and pipeline tests
docker/         local docker-compose
Dockerfile      image for Hugging Face Spaces
```

## Delivery

Every push to `main` runs lint, tests, an image build and vulnerability scans, then deploys to
[Hugging Face Spaces](https://huggingface.co/spaces/alanoee/doodleai) and publishes a versioned
release (`v1.0.N`).

## Data and attribution

The model is trained on [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), a
dataset of drawings collected by Google and made available under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The drawings are rendered and
resized for training; this project is not endorsed by Google. The code is MIT licensed; the trained
model is derived from CC BY data, so give credit to the dataset when you redistribute it.

## License

MIT License - see LICENSE for details.
