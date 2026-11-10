import base64
import io
import os
import random

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
from werkzeug.exceptions import HTTPException

from auth import require_auth
from model.drawing_classifier import DrawingClassifier


app = Flask(__name__)

# The website sends ~280x280 PNGs (a few KB). Cap everything well above that so a
# single request can never make the container read or decode something huge.
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
MAX_IMAGE_SIDE = 2048


allowed_origins = [
    "https://www.alanszuster.dev",
    "https://alanszuster.dev",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8000",
]

additional = os.getenv('ADDITIONAL_ORIGINS', '')
if additional:
    allowed_origins.extend([o.strip() for o in additional.split(',') if o.strip()])

CORS(app, origins=allowed_origins)

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.getenv('RATELIMIT_STORAGE_URI', 'memory://')
)


classifier = None


def init_model():
    global classifier
    try:
        print("[DEBUG] Loading AI model...")
        classifier = DrawingClassifier()
        classifier.load_model()
        print("[DEBUG] AI model loaded successfully!")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] Error loading model: {e}")
        classifier = None


init_model()


def get_classifier():
    global classifier
    if classifier is None:
        init_model()
    return classifier


@app.route('/')
def index():
    try:
        return jsonify({
            'name': 'AI Drawing Classifier API',
            'version': '1.0',
            'endpoints': {
                'POST /predict': 'Classify a drawing image',
                'GET /classes': 'Get list of supported classes',
                'GET /health': 'Public liveness/readiness probe',
                'GET /status': 'Authenticated check of auth and model'
            }
        })
    except Exception:  # pylint: disable=broad-exception-caught
        app.logger.exception('Index failed')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@require_auth
def predict_drawing():
    try:
        payload = request.get_json(silent=True)
        if not payload or 'image' not in payload:
            return jsonify({'error': 'Missing image data'}), 400

        image_data = payload['image']

        if not isinstance(image_data, str) or not image_data.startswith('data:image/'):
            return jsonify({'error': 'Invalid image format'}), 400

        _, separator, encoded = image_data.partition(',')
        if not separator:
            return jsonify({'error': 'Invalid image format'}), 400

        try:
            # Image.open only reads the header, so the size check happens before
            # any pixel data is decoded (protects against decompression bombs).
            image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
            if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
                return jsonify({'error': 'Image too large'}), 400
        except (ValueError, OSError):
            return jsonify({'error': 'Invalid image data'}), 400

        clf = get_classifier()
        if clf is None:
            return jsonify({'error': 'Model not loaded', 'predictions': []}), 503

        predictions = clf.predict(image)
        return jsonify({'predictions': predictions, 'success': True})

    except HTTPException:
        raise  # e.g. 413 Request Entity Too Large must reach the client unchanged
    except Exception:  # pylint: disable=broad-exception-caught
        app.logger.exception('Prediction failed')
        return jsonify({'error': 'Internal server error', 'predictions': []}), 500


@app.route('/classes')
@limiter.limit("30 per minute")
@require_auth
def get_classes():
    try:
        clf = get_classifier()
        if clf is None:
            return jsonify({'error': 'Model not loaded', 'classes': []}), 503
        return jsonify({'classes': clf.classes, 'total_classes': len(clf.classes)})
    except Exception:  # pylint: disable=broad-exception-caught
        app.logger.exception('Listing classes failed')
        return jsonify({'error': 'Internal server error', 'classes': []}), 500


@app.route('/health')
def health_check():
    """Public probe (Docker HEALTHCHECK, uptime monitors).

    "healthy" means the process is up AND the model is loaded. If the model is
    missing this answers 503, so the container is not reported healthy while
    every prediction would fail. It does not check authentication; use
    /status for that.
    """
    if get_classifier() is None:
        return jsonify({'status': 'unhealthy', 'model': 'not_loaded', 'version': '1.0'}), 503
    return jsonify({'status': 'healthy', 'model': 'loaded', 'version': '1.0'})


@app.route('/status')
@limiter.limit("60 per minute")
@require_auth
def status():
    """Authenticated check: reaching it proves the caller's credentials work."""
    clf = get_classifier()
    if clf is None:
        return jsonify({'status': 'unavailable', 'model': 'not_loaded'}), 503
    return jsonify({'status': 'ok', 'model': 'loaded', 'classes': len(clf.classes)})


@app.route('/get_random_word')
@limiter.limit("60 per hour")
@require_auth
def get_random_word():
    clf = get_classifier()
    if clf is None:
        return jsonify({'error': 'Model not loaded', 'word': None}), 503
    return jsonify({'word': random.choice(clf.classes)})


if __name__ == '__main__':
    print("Starting AI Drawing Classifier API...")
    init_model()
    app.run(debug=os.getenv('FLASK_DEBUG') == '1', host='0.0.0.0', port=5000)
