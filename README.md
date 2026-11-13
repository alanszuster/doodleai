---
title: AI Drawing Classifier
sdk: docker
app_port: 8080
pinned: false
---

# AI Drawing Classifier API

REST API for real-time hand-drawn sketch recognition using a convolutional neural network trained on the Google Quick Draw! dataset.

## Features

- Custom CNN architecture for sketch classification
- 120 supported drawing categories
- Returns top-3 predictions with confidence scores
- Authentication with short-lived OIDC tokens (Vercel), no shared secret; static API key only as an optional fallback
- Rate limiting per endpoint
- Deployed on Hugging Face Spaces, Docker support included

## Supported Categories

120 categories across 8 groups:
**Animals**: bear, bee, butterfly, cat, cow, crab, camel, dog, dolphin, duck, elephant, fish, flamingo, frog, giraffe, hedgehog, horse, kangaroo, lion, monkey, octopus, owl, panda, penguin, pig, rabbit, shark, sheep, snake, spider, tiger, whale, zebra
**Food**: apple, banana, birthday cake, bread, carrot, cookie, donut, grapes, hamburger, hot dog, ice cream, broccoli, mushroom, pear, pineapple, pizza, strawberry, watermelon
**Vehicles**: airplane, bicycle, bus, car, firetruck, helicopter, motorbike, cruise ship, sailboat, submarine, train, truck
**Objects**: backpack, book, camera, chair, clock, computer, cup, drums, fork, guitar, hammer, hat, key, knife, lantern, microphone, pencil, piano, scissors, shoe, sword, umbrella
**Nature**: cloud, campfire, flower, leaf, lightning, moon, mountain, rainbow, snowflake, star, sun, tree
**Buildings**: bridge, castle, door, fence, house, lighthouse, windmill
**Body**: ear, eye, face, hand, nose, tooth
**Misc**: circle, crown, diamond, bowtie, hot air balloon, lollipop, skull, stop sign, tornado, cactus

## Repository Layout

```
src/                 API application (Flask): app.py, auth.py, classifier.py
model/
  artifacts/         trained model + class list served by the API (best_model.keras, classes.json)
  training/          data preparation, training and evaluation (run on a free GPU via train.ipynb)
  reports/           training plots
tests/               API, auth, rate-limit, classifier and training-pipeline tests
pyproject.toml, uv.lock  dependencies (uv): runtime, `dev` and `training` groups
Dockerfile           image for the API (only src/ and model/artifacts/ are copied in);
                     stays in the root because Hugging Face Spaces requires it there
docker/              docker-compose.yml for running the API locally
```

## Quick Start

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- TensorFlow CPU 2.x

### Installation

```bash
git clone <repository-url>
cd doodleai
uv sync
```

### Run the API

```bash
# the legacy static key is off by default; enable it explicitly for local use
AUTH_ALLOW_API_KEY=true AI_API_KEY=your-secret-key uv run python src/app.py
```

The API will be available at `http://localhost:5000`.

### Run with Docker

```bash
docker build -t doodleai .
# the legacy static key is off by default; enable it explicitly for local use
docker run -p 8080:8080 -e AUTH_ALLOW_API_KEY=true -e AI_API_KEY=your-secret-key doodleai
```

Or using Docker Compose (it enables the legacy key for local development):

```bash
AI_API_KEY=your-secret-key docker compose -f docker/docker-compose.yml up
```

## Authentication

Protected endpoints (`/predict`, `/classes`, `/get_random_word`) accept either:

1. **A Vercel OIDC token** (preferred): `Authorization: Bearer <jwt>`. The token is issued to the
   website's server functions, lives for a short time and is verified against Vercel's public keys
   (RS256 only, with `iss`, `aud`, `exp`, `iat`, `sub` required). Only the listed subjects are
   allowed, e.g. the production environment of one project. There is no secret to store or rotate.
2. **A static API key** (off by default): `x-api-key: <key>`, compared in constant time. Only accepted
   when `AUTH_ALLOW_API_KEY=true` and `AI_API_KEY` is set. Not needed when OIDC is configured.

`/` and `/health` are public. `/status` requires authentication (see below). OIDC is enabled only when issuer, audience and allowed subjects are
all set, so a half-finished configuration never lets anyone in.

| Variable | Example | Purpose |
|---|---|---|
| `OIDC_ISSUER` | `https://oidc.vercel.com/<team-slug>` | Token issuer (also where the public keys are fetched) |
| `OIDC_AUDIENCE` | `https://vercel.com/<team-slug>` | Expected `aud` claim |
| `OIDC_ALLOWED_SUBJECTS` | `owner:<team>:project:<project>:environment:production` | Comma-separated exact `sub` values |
| `AI_API_KEY` | – | Legacy static key (only used when allowed below) |
| `AUTH_ALLOW_API_KEY` | `true` | Re-enables the legacy key (default: off) |

Other protections:
- Request bodies are capped at 1 MB. Only PNG is accepted (checked by decoding, not by the declared
  type); animated images and images larger than 2048 px per side are rejected before decoding.
- Malformed input returns `400`; internal errors are logged with a request id and never returned to
  the client. Every response carries `X-Request-ID`, `X-Content-Type-Options`, `Cache-Control: no-store`
  and a restrictive `Content-Security-Policy`.
- Rate limits are per verified caller (the OIDC `sub`, or the legacy key), plus a per-IP limit in front
  of authentication (300 requests per minute) against floods of unauthenticated requests. The client
  IP comes from `X-Forwarded-For`, trusting `TRUSTED_PROXIES` proxies (default 1, as on Hugging Face
  Spaces).
- There is no CORS: the website calls this API from its server, not from the browser, and the token
  travels in the `Authorization` header, never in a cookie.
- The container runs as a non-root user (uid 1000) and the build is checked in CI (Trivy, Bandit,
  `pip-audit`).

## API Endpoints

Endpoints other than `/` and `/health` require authentication (see above).

### GET /

Returns API metadata and available endpoints.

```json
{
  "name": "AI Drawing Classifier API",
  "version": "1.2.0",
  "commit": "1a2b3c4",
  "model": {"run_id": "20261113-105549-full", "top1": 0.9, "sha256": "..."},
  "endpoints": {
    "POST /predict": "Classify a drawing image",
    "GET /classes": "Get list of supported classes",
    "GET /health": "Public liveness/readiness probe",
    "GET /status": "Authenticated check of auth and model"
  }
}
```

### POST /predict

Classifies a base64-encoded drawing. Rate limit: 10 requests/minute.

Request:
```json
{
  "image": "data:image/png;base64,<base64-encoded-image>"
}
```

Response:
```json
{
  "predictions": [
    {"class": "cat", "confidence": 92.1},
    {"class": "dog", "confidence": 5.3},
    {"class": "bird", "confidence": 1.8}
  ],
  "success": true
}
```

### GET /classes

Returns all supported drawing categories.

### GET /health

Public liveness/readiness probe (used by the Docker `HEALTHCHECK`, which runs `curl -f` against it,
so any non-2xx answer marks the container unhealthy). `200` with
`{"status": "healthy", "model": "loaded"}` only when the process is up **and** a *trained* model was
loaded from file, otherwise `503`. The random-weights fallback that `load_model()` creates when the
model file is missing or corrupt counts as not loaded, and the other endpoints refuse to serve it too.
The probe only observes: it never tries to load the model, so a public, unauthenticated endpoint
cannot trigger loading. It does not look at credentials, so it cannot tell you that authentication
works.

### GET /status

Authenticated check. Reaching it with a valid credential proves authentication works, and the
answer confirms the model is loaded: `200 {"status": "ok", "model": "loaded", "classes": 120}`, or
`503` if the model is missing. Use this, not `/health`, to check the whole path from the website.
Rate limit: 60 requests/minute.

### GET /get_random_word

Returns a random category for drawing challenges.

## Usage Example

```python
import requests
import base64

with open("drawing.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:5000/predict",
    headers={
        "Authorization": "Bearer <oidc-token>"
    },  # or {'x-api-key': ...} if you enabled the legacy key
    json={"image": f"data:image/png;base64,{image_data}"},
)

result = response.json()
print(result["predictions"][0])
```

## Training Your Own Model

Dataset preparation and training live in `model/training/`; see
[model/training/README.md](model/training/README.md). Training runs on a free GPU (Kaggle or Colab)
through [model/training/train.ipynb](model/training/train.ipynb), or locally:

```bash
uv sync --group training
uv run python model/training/prepare.py --work-dir <big disk>
uv run python model/training/train.py --work-dir <big disk> --name main
uv run python model/training/evaluate.py --model <big disk>/runs/main/best.keras --data-dir <big disk>/data --out report
```

## Training History

![Training History](model/reports/training_history.png)

## Technical Details

- **Architecture**: 4-layer CNN (32→64→128→256 filters) with BatchNormalization, GlobalAveragePooling2D, and online data augmentation (rotation, translation, zoom)
- **Input**: 28×28 grayscale images (resized from canvas, colors inverted to match training format)
- **Dataset**: Google Quick Draw! numpy bitmap format, 15,000 samples per class
- **Accuracy**: ~72% across 120 classes
- **Confidence**: top-3 predictions returned per query
- **Framework**: TensorFlow/Keras
- **Model size**: ~5.9 MB
- **Inference time**: <100ms
- **API**: Flask with per-caller rate limiting, served by gunicorn

## Running Tests

```bash
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
uv run bandit -q -r src
```

The suite covers authentication (valid, expired, forged and wrong-audience tokens, algorithm
confusion), input hardening, the health and status endpoints, and the model loader. It also loads
the real model and runs a prediction, because the endpoint tests accept `503` (model not loaded) and
would otherwise let a broken dependency update through CI.

## Deployment

### Hugging Face Spaces (primary)

The API is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/alanoee/doodleai) as a Docker space. Every push to `main` runs lint and tests, and only if they pass deploys via GitHub Actions (`.github/workflows/deploy-hf.yml`).

The deploy uses a Hugging Face [Trusted Publisher](https://huggingface.co/docs/hub/trusted-publishers): the workflow exchanges GitHub's short-lived OIDC token for a short-lived Hub token scoped to this Space, so **no `HF_TOKEN` secret is stored**. The publisher is configured on the Space (Settings → Trusted Publishers) with these claims, all of which must match:
- `repository` = `alanszuster/doodleai`
- `ref` = `refs/heads/main`
- `workflow_ref` = `alanszuster/doodleai/.github/workflows/deploy-hf.yml@refs/heads/main`

If you rename the workflow file or deploy from another branch, update the publisher too.

Required environment variables on the Space (Settings → Variables and secrets):
- `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_ALLOWED_SUBJECTS` - OIDC verification (see [Authentication](#authentication))
- `AI_API_KEY` and `AUTH_ALLOW_API_KEY=true` - legacy API key, only if you want it (off by default)
- `TRUSTED_PROXIES` - number of reverse proxies in front of the app (default `1`)
- `LOG_LEVEL` - `INFO` (default), `DEBUG`, ...

### CI/CD

- `.github/workflows/ci.yml` runs lint (pylint) and the tests for pull requests and for pushes to
  branches other than `main`.
- `.github/workflows/deploy-hf.yml` runs on every push to `main`: it calls `ci.yml` first and only
  deploys to the Space if lint and tests pass.
- GitHub Actions are pinned to commit SHAs. Dependabot (`.github/dependabot.yml`) opens weekly
  grouped pull requests for the actions and the `uv` dependencies; they are reviewed and merged by
  hand, nothing is merged automatically.

To deploy your own instance:
1. Fork this repository
2. Create a Hugging Face Space (Docker SDK)
3. On the Space, add a Trusted Publisher for your repository and workflow (see above)
4. Push to `main`

## Versioning

Every push to `main` is released. After the tests pass, the deploy workflow takes the next
version (`1.0.N`, one more than the last tag), stamps it into the build, deploys to Hugging Face
and, only after a successful deploy, tags the commit `v1.0.N` and publishes a GitHub Release
listing the commits since the previous one. One deploy is one version is one tag. To start a
new series (`1.1.0`, `2.0.0`), change `MAJOR_MINOR` in `.github/workflows/deploy-hf.yml`.

Production reports what it runs: `GET /health` returns `version`, and `/` and `/status` add the
commit and the served model (`run_id`, `top1`, `sha256` from `model_card.json`). A plain
checkout reports `dev`.

## Data and attribution

The model is trained on [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), a
dataset of drawings collected by Google and made available under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The drawings are rendered and
resized for training; this project is not endorsed by Google. The code is MIT licensed; the trained
model is derived from CC BY data, so give credit to the dataset when you redistribute it.

## License

MIT License - see LICENSE for details.
