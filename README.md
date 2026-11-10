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
  dataset/           Quick Draw! data, downloaded locally (git-ignored)
  training/          dataset preparation and training scripts
  reports/           training plots
tests/               API and classifier tests
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
AI_API_KEY=your-secret-key uv run python src/app.py
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

Other protections: request bodies are capped at 1 MB, images larger than 2048 px per side are
rejected before decoding, malformed input returns `400` and internal errors are logged, never
returned to the client.

## API Endpoints

Endpoints other than `/` and `/health` require authentication (see above).

### GET /

Returns API metadata and available endpoints.

```json
{
  "name": "AI Drawing Classifier API",
  "version": "1.0",
  "endpoints": {
    "POST /predict": "Classify a drawing image",
    "GET /classes": "Get list of supported classes",
    "GET /health": "Check API health"
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

### GET /get_random_word

Returns a random category for drawing challenges.

## Usage Example

```python
import requests
import base64

with open('drawing.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:5000/predict',
    headers={'Authorization': 'Bearer <oidc-token>'},  # or {'x-api-key': ...} during transition
    json={'image': f'data:image/png;base64,{image_data}'}
)

result = response.json()
print(result['predictions'][0])
```

## Training Your Own Model

Dataset preparation and training live in `model/`. See [model/training/README.md](model/training/README.md) for details.

```bash
uv sync --group training

# 1. Download and preprocess data
uv run python model/training/prepare_data.py

# 2. Train the model
uv run python model/training/train_model.py
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
- **API**: Flask with rate limiting and CORS

## Running Tests

```bash
AI_API_KEY=test-key uv run pytest -v
```

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
- `ADDITIONAL_ORIGINS` - comma-separated list of extra allowed CORS origins (the website's domains are built in)

To deploy your own instance:
1. Fork this repository
2. Create a Hugging Face Space (Docker SDK)
3. On the Space, add a Trusted Publisher for your repository and workflow (see above)
4. Push to `main`

## License

MIT License - see LICENSE for details.
