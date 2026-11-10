# pylint: disable=wrong-import-position,wrong-import-order,redefined-outer-name
import base64
import json
import os
import sys
from io import BytesIO

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, limiter

KEY = 'hardening-test-key'


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('AI_API_KEY', KEY)
    monkeypatch.setenv('AUTH_ALLOW_API_KEY', 'true')
    app.config['TESTING'] = True
    # The limiter is shared by every test in the process; keep these tests independent of it.
    previous = limiter.enabled
    limiter.enabled = False
    try:
        with app.test_client() as test_client:
            test_client.environ_base['HTTP_X_API_KEY'] = KEY
            yield test_client
    finally:
        limiter.enabled = previous


def png_data_url(size):
    buffer = BytesIO()
    Image.new('L', size, 0).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def post(client, body, raw=False):
    if raw:
        return client.post('/predict', data=body, content_type='application/json')
    return client.post('/predict', data=json.dumps(body), content_type='application/json')


def test_invalid_base64_is_a_client_error(client):
    response = post(client, {'image': 'data:image/png;base64,@@@not-base64@@@'})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Invalid image data'


def test_not_an_image_is_a_client_error(client):
    payload = base64.b64encode(b'this is not a png').decode()
    response = post(client, {'image': f'data:image/png;base64,{payload}'})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Invalid image data'


def test_missing_comma_is_a_client_error(client):
    assert post(client, {'image': 'data:image/png;base64'}).status_code == 400


def test_non_string_image_is_a_client_error(client):
    assert post(client, {'image': 123}).status_code == 400


def test_non_json_body_is_a_client_error(client):
    response = post(client, 'plain text', raw=True)
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing image data'


def test_huge_dimensions_are_rejected_before_decoding(client):
    # A blank 5000x5000 PNG is only a few KB but would need ~25 MB of pixels.
    response = post(client, {'image': png_data_url((5000, 5000))})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Image too large'


def test_normal_size_image_is_accepted(client):
    assert post(client, {'image': png_data_url((280, 280))}).status_code in (200, 503)


def test_oversized_request_body_is_rejected(client):
    response = post(client, 'x' * (1024 * 1024 + 1), raw=True)
    assert response.status_code == 413


def test_cors_allows_the_site_and_not_the_old_domain(client):
    allowed = client.get('/health', headers={'Origin': 'https://www.alanszuster.dev'})
    assert allowed.headers.get('Access-Control-Allow-Origin') == 'https://www.alanszuster.dev'
    old = client.get('/health', headers={'Origin': 'https://alanszuster.vercel.app'})
    assert 'Access-Control-Allow-Origin' not in old.headers
    other = client.get('/health', headers={'Origin': 'https://evil.example'})
    assert 'Access-Control-Allow-Origin' not in other.headers
