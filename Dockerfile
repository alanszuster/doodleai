# Base images are pinned by digest; Dependabot (docker ecosystem) keeps them current.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Pull in the fixed zlib (CVE-2026-76844, CVE-2026-85091) until the pinned base image
# digest catches up. Remove once Trivy no longer reports it without this line.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000; do the same everywhere.
RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY src/ src/
COPY VERSION VERSION
# model_card.json is optional (absent until the first model is promoted); the glob tolerates that.
COPY model/artifacts/best_model.keras model/artifacts/classes.json model/artifacts/model_card.jso[n] model/artifacts/

USER 1000
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)"]

CMD ["gunicorn", "-c", "src/gunicorn.conf.py", "--chdir", "src", "app:app"]
