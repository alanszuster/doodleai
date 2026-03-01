import os
os.environ['AI_API_KEY'] = 'test-key'

import pytest
import json
import base64
from io import BytesIO
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['RATELIMIT_ENABLED'] = False
    with app.test_client() as client:
        client.environ_base['HTTP_X_API_KEY'] = 'test-key'
        yield client


@pytest.fixture
def sample_image_base64():
    img = Image.new('L', (28, 28), 0)
    for x in range(28):
        for y in range(28):
            if ((x - 14) ** 2 + (y - 14) ** 2) < 64:
                img.putpixel((x, y), 255)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode()
    return f"data:image/png;base64,{img_base64}"


def test_index_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert data['name'] == 'AI Drawing Classifier API'


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'


def test_classes_endpoint(client):
    response = client.get('/classes')
    assert response.status_code in [200, 503]
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'classes' in data
        assert 'total_classes' in data
        assert isinstance(data['classes'], list)
    else:
        assert 'error' in data


def test_get_random_word_endpoint(client):
    response = client.get('/get_random_word')
    assert response.status_code in [200, 503]
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'word' in data
        assert isinstance(data['word'], str)
    else:
        assert 'error' in data


def test_predict_missing_data(client):
    response = client.post(
        '/predict',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Missing image data' in data['error']


def test_predict_invalid_image_format(client):
    response = client.post(
        '/predict',
        data=json.dumps({'image': 'invalid_format'}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Invalid image format' in data['error']


def test_predict_valid_image(client, sample_image_base64):
    response = client.post(
        '/predict',
        data=json.dumps({'image': sample_image_base64}),
        content_type='application/json'
    )
    assert response.status_code in [200, 503]
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'predictions' in data
        assert 'success' in data
        assert data['success'] is True
        assert isinstance(data['predictions'], list)
    else:
        assert 'error' in data
        assert 'Model not loaded' in data['error']


def test_rate_limiting_predict(client, sample_image_base64):
    for _ in range(3):
        response = client.post(
            '/predict',
            data=json.dumps({'image': sample_image_base64}),
            content_type='application/json'
        )
        assert response.status_code in [200, 429, 503]


if __name__ == '__main__':
    pytest.main([__file__])
