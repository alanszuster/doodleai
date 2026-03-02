import base64
import io
import os
import random
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image

from model.drawing_classifier import DrawingClassifier


app = Flask(__name__)
AI_API_KEY = os.getenv('AI_API_KEY')


def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = request.headers.get('x-api-key')
        if not key or key != AI_API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return func(*args, **kwargs)
    return wrapper


allowed_origins = [
    "https://alanszuster.vercel.app",
    "https://alanszusterpage-alanszuster-alanszusters-projects.vercel.app",
    "https://alanszusterpage-alanszusters-projects.vercel.app",
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
                'GET /health': 'Check API health'
            }
        })
    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def predict_drawing():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'Missing image data'}), 400

        image_data = request.json['image']

        if not image_data.startswith('data:image/'):
            return jsonify({'error': 'Invalid image format'}), 400

        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        clf = get_classifier()
        if clf is None:
            return jsonify({'error': 'Model not loaded', 'predictions': []}), 503

        predictions = clf.predict(image)
        return jsonify({'predictions': predictions, 'success': True})

    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({'error': str(e), 'predictions': []}), 500


@app.route('/classes')
@limiter.limit("30 per minute")
@require_api_key
def get_classes():
    try:
        clf = get_classifier()
        if clf is None:
            return jsonify({'error': 'Model not loaded', 'classes': []}), 503
        return jsonify({'classes': clf.classes, 'total_classes': len(clf.classes)})
    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({'error': str(e), 'classes': []}), 500


@app.route('/health')
def health_check():
    model_status = 'loaded' if classifier is not None else 'not_loaded'
    return jsonify({'status': 'healthy', 'model': model_status, 'version': '1.0'})


@app.route('/get_random_word')
@limiter.limit("60 per hour")
@require_api_key
def get_random_word():
    clf = get_classifier()
    if clf is None:
        return jsonify({'error': 'Model not loaded', 'word': None}), 503
    return jsonify({'word': random.choice(clf.classes)})


if __name__ == '__main__':
    print("Starting AI Drawing Classifier API...")
    init_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
