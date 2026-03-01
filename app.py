from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import json
import base64
from PIL import Image
import io
import tensorflow as tf

import os
from functools import wraps
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

# Configure CORS for specific domains
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

# Rate limiting
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
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        classifier = None

init_model()
def get_classifier():
    """Lazy load classifier only when needed"""
    global classifier
    if classifier is None:
        init_model()
    return classifier

@app.route('/')
def index():
    try:
        print("[DEBUG] Index endpoint called")
        return jsonify({
            'name': 'AI Drawing Classifier API',
            'version': '1.0',
            'endpoints': {
                'POST /predict': 'Classify a drawing image',
                'GET /classes': 'Get list of supported classes',
                'GET /health': 'Check API health'
            }
        })
    except Exception as e:
        print(f"[ERROR] Index endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def predict_drawing():
    try:
        # Input validation
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'Missing image data'}), 400

        data = request.json
        image_data = data['image']

        # Validate base64 image format
        if not image_data.startswith('data:image/'):
            return jsonify({'error': 'Invalid image format'}), 400

        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        classifier = get_classifier()
        if classifier is None:
            return jsonify({
                'error': 'Model not loaded',
                'predictions': []
            }), 503

        predictions = classifier.predict(image)

        return jsonify({
            'predictions': predictions,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'predictions': []
        }), 500

@app.route('/classes')
@limiter.limit("30 per minute")
@require_api_key
def get_classes():
    """Get list of supported drawing classes"""
    try:
        print("[DEBUG] get_classes endpoint called")
        classifier = get_classifier()
        if classifier is None:
            print("[ERROR] Model not loaded in get_classes")
            return jsonify({'error': 'Model not loaded', 'classes': []}), 503
        return jsonify({
            'classes': classifier.classes,
            'total_classes': len(classifier.classes)
        })
    except Exception as e:
        print(f"[ERROR] get_classes endpoint error: {e}")
        return jsonify({'error': str(e), 'classes': []}), 500

@app.route('/health')
def health_check():
    """Check API health and model status"""
    global classifier
    model_status = 'loaded' if classifier is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'version': '1.0'
    })

@app.route('/get_random_word')
@limiter.limit("60 per hour")
@require_api_key
def get_random_word():
    """Get a random word from supported classes for drawing challenges"""
    classifier = get_classifier()
    if classifier is None:
        return jsonify({'error': 'Model not loaded', 'word': None}), 503

    import random
    word = random.choice(classifier.classes)
    return jsonify({'word': word})

# Don't initialize model on startup for Vercel - use lazy loading instead
# Initialize model for production (Vercel) - commented out for faster cold starts
# if os.getenv('VERCEL') or os.getenv('FLASK_ENV') == 'production':
#     print("Production environment detected - initializing model...")
#     init_model()

if __name__ == '__main__':
    print("Starting AI Drawing Classifier API...")
    print("Initializing AI model...")
    init_model()

    print("API will be available at: http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /           - API info")
    print("  POST /predict    - Classify drawing")
    print("  GET  /classes    - Get supported classes")
    print("  GET  /health     - Health check")
    print("  GET  /get_random_word - Get random class for challenges")

    app.run(debug=True, host='0.0.0.0', port=5000)
