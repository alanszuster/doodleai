import base64
import json
from io import BytesIO

import pytest
from PIL import Image

from app import app, limiter

KEY = "hardening-test-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "true")
    app.config["TESTING"] = True
    # The limiter is shared by every test in the process; keep these tests independent of it.
    previous = limiter.enabled
    limiter.enabled = False
    try:
        with app.test_client() as test_client:
            test_client.environ_base["HTTP_X_API_KEY"] = KEY
            yield test_client
    finally:
        limiter.enabled = previous


def png_data_url(size):
    buffer = BytesIO()
    Image.new("L", size, 0).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def post(client, body, raw=False):
    if raw:
        return client.post("/predict", data=body, content_type="application/json")
    return client.post("/predict", data=json.dumps(body), content_type="application/json")


def test_invalid_base64_is_a_client_error(client):
    response = post(client, {"image": "data:image/png;base64,@@@not-base64@@@"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid image data"


def test_not_an_image_is_a_client_error(client):
    payload = base64.b64encode(b"this is not a png").decode()
    response = post(client, {"image": f"data:image/png;base64,{payload}"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid image data"


def test_missing_comma_is_a_client_error(client):
    assert post(client, {"image": "data:image/png;base64"}).status_code == 400


def test_non_string_image_is_a_client_error(client):
    assert post(client, {"image": 123}).status_code == 400


def test_non_json_body_is_a_client_error(client):
    response = post(client, "plain text", raw=True)
    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing image data"


def test_huge_dimensions_are_rejected_before_decoding(client):
    # A blank 5000x5000 PNG is only a few KB but would need ~25 MB of pixels.
    response = post(client, {"image": png_data_url((5000, 5000))})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Image too large"


def test_normal_size_image_is_accepted(client):
    assert post(client, {"image": png_data_url((280, 280))}).status_code in (200, 503)


def test_oversized_request_body_is_rejected(client):
    response = post(client, "x" * (1024 * 1024 + 1), raw=True)
    assert response.status_code == 413


def test_cors_is_not_enabled(client):
    response = client.get("/health", headers={"Origin": "https://www.alanszuster.dev"})
    assert "Access-Control-Allow-Origin" not in response.headers


def image_data_url(fmt, **save_kwargs):
    buffer = BytesIO()
    Image.new("RGB", (28, 28), "white").save(buffer, format=fmt, **save_kwargs)
    mime = fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("fmt", ["GIF", "JPEG", "BMP"])
def test_only_png_is_accepted(client, fmt):
    assert post(client, {"image": image_data_url(fmt)}).status_code == 400


def test_png_mislabelled_as_another_type_is_rejected(client):
    other = image_data_url("PNG").replace("data:image/png", "data:image/gif")
    assert post(client, {"image": other}).status_code == 400


def test_gif_hidden_behind_a_png_header_is_rejected(client):
    gif = image_data_url("GIF").replace("data:image/gif", "data:image/png")
    response = post(client, {"image": gif})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid image data"


def test_animated_png_is_rejected(client):
    frames = [Image.new("RGB", (28, 28), colour) for colour in ("white", "black")]
    buffer = BytesIO()
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:])
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    response = post(client, {"image": url})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Animated images are not supported"


def test_responses_carry_security_headers(client):
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Request-ID"]


def test_request_id_from_the_caller_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_errors_are_json(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Not Found"}


def test_request_id_is_sanitised(client):
    ok = client.get("/health", headers={"X-Request-ID": "abc.DEF_123-x"})
    assert ok.headers["X-Request-ID"] == "abc.DEF_123-x"
    repeated = client.get("/health", headers={"X-Request-ID": "abc-123,abc-123"})
    assert repeated.headers["X-Request-ID"] == "abc-123"  # a proxy repeated the header
    for bad in ("has space", "x" * 65, "semi;colon", 'quo"te'):
        got = client.get("/health", headers={"X-Request-ID": bad}).headers["X-Request-ID"]
        assert got != bad and len(got) == 16
