import base64
import binascii
import io
import logging
import os
import re
import time
import uuid

from flask import Flask, g, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from auth import require_auth
from buildinfo import __version__, app_info, model_info
from classifier import DrawingClassifier

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("doodleai")

app = Flask(__name__)

# Number of reverse proxies in front of the app (Hugging Face Spaces: 1). It decides
# which X-Forwarded-For entry is trusted as the client IP. Set 0 to trust none.
TRUSTED_PROXIES = int(os.getenv("TRUSTED_PROXIES", "1"))
if TRUSTED_PROXIES > 0:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXIES)

# The website sends ~280x280 PNGs (a few KB). Cap everything well above that so a
# single request can never make the container read or decode something huge.
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
MAX_IMAGE_SIDE = 2048
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_SIDE * MAX_IMAGE_SIDE
PNG_DATA_URL_PREFIX = "data:image/png;base64,"


def caller_key():
    """Rate-limit key: the verified caller (set by require_auth), else the client IP."""
    return g.get("auth_identity") or get_remote_address()


# Two layers. The application limit runs before authentication, when no caller is
# known yet, so it is keyed by client IP: cheap protection against floods of
# unauthenticated requests. The per-route limits run after authentication and are
# keyed by the verified caller: a fair share for each identity. Behind the website's
# server functions every request shares a few IPs, so the per-caller layer is the one
# that matters for real traffic.
limiter = Limiter(
    caller_key,
    app=app,
    application_limits=["300 per minute"],
    storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
)


def route_limit(env_name, default):
    """Per-caller limit for a route, read per request so tests and operators can change it.

    All website traffic shares one verified identity, so these are circuit breakers for the
    whole site, not per-visitor quotas: fairness between visitors is enforced in front of the
    API (Vercel Firewall). Override with RATE_LIMIT_PREDICT / RATE_LIMIT_CLASSES.
    """
    return limiter.limit(lambda: os.getenv(env_name, default))


classifier = None


def init_model():
    """Load the model once, at startup. Failure leaves the app up but unhealthy."""
    global classifier
    logger.info("Starting doodleai version=%s", __version__)
    try:
        loaded = DrawingClassifier()
        loaded.load_model()
        classifier = loaded
        logger.info("Model loaded (%d classes)", len(loaded.classes))
        logger.info("Serving model=%s", model_info())
    except Exception:
        logger.exception("Model failed to load; /health will report 503")
        classifier = None


init_model()


def model_ready():
    """True when the model is loaded. Side-effect free: probes only observe."""
    return classifier is not None


def get_classifier():
    return classifier


REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def request_id_from(header):
    """A caller-supplied id if it is a short plain token, else a fresh one.

    Proxies may repeat the header ("abc,abc"), so only the first value is looked at.
    Anything else (spaces, control characters, huge values) never reaches logs or headers.
    """
    candidate = (header or "").split(",")[0].strip()
    return candidate if REQUEST_ID.fullmatch(candidate) else uuid.uuid4().hex[:16]


@app.before_request
def start_request():
    g.request_id = request_id_from(request.headers.get("X-Request-ID"))
    g.started = time.perf_counter()


@app.after_request
def secure_response(response):
    response.headers["X-Request-ID"] = g.get("request_id", "")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.path != "/health":
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.0f",
            g.get("request_id"),
            request.method,
            request.path,
            response.status_code,
            (time.perf_counter() - g.get("started", time.perf_counter())) * 1000,
        )
    return response


@app.errorhandler(Exception)
def handle_error(exc):
    """Every error answers JSON; internal details are logged, never returned."""
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.name}), exc.code
    logger.exception("Unhandled error request_id=%s", g.get("request_id"))
    return jsonify({"error": "Internal server error"}), 500


def confidence_bucket(confidence):
    """A coarse band for the logs (40 is the website's "not sure" line)."""
    if confidence < 40:
        return "<40"
    if confidence < 70:
        return "40-70"
    if confidence < 90:
        return "70-90"
    return "90+"


def decode_png(data_url):
    """Return a PIL image for a PNG data URL, or raise ValueError with a client message."""
    if not isinstance(data_url, str) or not data_url.startswith(PNG_DATA_URL_PREFIX):
        raise ValueError("Invalid image format")
    try:
        raw = base64.b64decode(data_url[len(PNG_DATA_URL_PREFIX) :], validate=True)
        # Image.open only reads the header, so the size checks happen before any pixel
        # data is decoded (protects against decompression bombs). Only PNG is accepted.
        image = Image.open(io.BytesIO(raw), formats=["PNG"])
    except Image.DecompressionBombError as exc:
        raise ValueError("Image too large") from exc
    except (binascii.Error, ValueError, OSError) as exc:
        raise ValueError("Invalid image data") from exc
    if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
        raise ValueError("Image too large")
    if getattr(image, "n_frames", 1) > 1:
        raise ValueError("Animated images are not supported")
    return image


@app.route("/")
def index():
    """Public and deliberately bare: no endpoint list, version or model details."""
    return jsonify({"name": "AI Drawing Classifier API"})


@app.route("/predict", methods=["POST"])
@require_auth
@route_limit("RATE_LIMIT_PREDICT", "120 per minute")
def predict_drawing():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "image" not in payload:
            return jsonify({"error": "Missing image data"}), 400

        try:
            image = decode_png(payload["image"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        clf = get_classifier()
        if clf is None:
            return jsonify({"error": "Model not loaded", "predictions": []}), 503

        started = time.perf_counter()
        predictions = clf.predict(image)
        logger.info(
            "request_id=%s inference_ms=%.0f top1=%s confidence=%s",
            g.request_id,
            (time.perf_counter() - started) * 1000,
            predictions[0]["class"],
            confidence_bucket(predictions[0]["confidence"]),
        )
        response = jsonify({"predictions": predictions, "success": True})
        run_id = (model_info() or {}).get("run_id")
        if run_id and REQUEST_ID.fullmatch(str(run_id)):
            response.headers["X-Model-Run"] = str(run_id)  # which model answered, for debugging
        return response

    except HTTPException:
        raise  # e.g. 413 Request Entity Too Large must reach the client unchanged
    except Exception:
        logger.exception("Prediction failed request_id=%s", g.request_id)
        return jsonify({"error": "Internal server error", "predictions": []}), 500


@app.route("/classes")
@require_auth
@route_limit("RATE_LIMIT_CLASSES", "120 per minute")
def get_classes():
    clf = get_classifier()
    if clf is None:
        return jsonify({"error": "Model not loaded", "classes": []}), 503
    return jsonify({"classes": clf.classes, "total_classes": len(clf.classes)})


@app.route("/health")
# Public, so keyed by client IP and never exempt: not a free flood target.
@limiter.limit("60 per minute")
def health_check():
    """Public probe (Docker HEALTHCHECK, CI smoke test, uptime monitors).

    "healthy" means the process is up AND the model is loaded. If it is missing
    this answers 503, so the container is not reported healthy while every
    prediction would fail. It only observes: it never tries to load the model, it
    does not check authentication, and it says nothing else (no version, no model
    details); use the authenticated /status for those.
    """
    if not model_ready():
        return jsonify({"status": "unhealthy"}), 503
    return jsonify({"status": "healthy"})


@app.route("/status")
@require_auth
@limiter.limit("60 per minute")
def status():
    """Authenticated check: reaching it proves the caller's credentials work."""
    if not model_ready():
        return jsonify({"status": "unavailable", "model": "not_loaded"}), 503
    return jsonify(
        {
            "status": "ok",
            "model": "loaded",
            "classes": len(classifier.classes),
            **app_info(),
            "model_info": model_info(),
        }
    )


if __name__ == "__main__":
    # Local development only; production runs under gunicorn (see Dockerfile).
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", host="0.0.0.0", port=5000)  # nosec B104
