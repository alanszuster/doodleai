import base64
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

import app as app_module
from app import app, limiter

KEY = "health-test-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "true")
    app.config["TESTING"] = True
    previous = limiter.enabled
    limiter.enabled = False
    try:
        with app.test_client() as test_client:
            yield test_client
    finally:
        limiter.enabled = previous


def auth_headers():
    return {"x-api-key": KEY}


def drawing_payload():
    image = Image.new("RGB", (280, 280), "white")
    ImageDraw.Draw(image).ellipse((40, 40, 240, 240), outline="black", width=22)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def reload_attempts(monkeypatch):
    """Records every attempt to (re)load the model; the loading itself is stubbed out."""
    calls = []
    monkeypatch.setattr(app_module, "init_model", lambda: calls.append(1))
    return calls


def test_health_is_public_and_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}  # nothing else is disclosed


def test_health_is_unhealthy_when_the_model_cannot_load(client, monkeypatch):
    monkeypatch.setattr(app_module, "classifier", None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "unhealthy"


def test_status_requires_authentication(client):
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"x-api-key": "wrong"}).status_code == 401


def test_status_with_credentials_checks_the_model(client):
    response = client.get("/status", headers=auth_headers())
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model"] == "loaded"
    assert data["classes"] > 0


def test_status_is_unavailable_when_the_model_cannot_load(client, monkeypatch):
    monkeypatch.setattr(app_module, "classifier", None)
    response = client.get("/status", headers=auth_headers())
    assert response.status_code == 503
    assert response.get_json()["status"] == "unavailable"


def test_model_really_predicts(client):
    """Smoke test with the real model, so a broken dependency bump cannot pass CI.

    The other endpoint tests accept 503 ("model not loaded"), which would hide
    exactly that kind of failure.
    """
    image = Image.new("RGB", (280, 280), "white")
    ImageDraw.Draw(image).ellipse((40, 40, 240, 240), outline="black", width=22)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    response = client.post("/predict", json={"image": payload}, headers=auth_headers())
    assert response.status_code == 200
    predictions = response.get_json()["predictions"]
    assert len(predictions) == 3
    assert predictions[0]["class"] == "circle"
    assert 0 < predictions[0]["confidence"] <= 100


def test_probes_never_try_to_load_the_model(client, monkeypatch, reload_attempts):
    """/health is public: it must only observe, not trigger loading."""
    monkeypatch.setattr(app_module, "classifier", None)
    for _ in range(3):
        assert client.get("/health").status_code == 503
        assert client.get("/status", headers=auth_headers()).status_code == 503
    assert reload_attempts == []


def test_predict_does_not_try_to_reload_a_model_that_failed_to_load(
    client, monkeypatch, reload_attempts
):
    """The model is loaded once, at startup. A failed start is fixed by a restart."""
    monkeypatch.setattr(app_module, "classifier", None)
    response = client.post("/predict", json={"image": drawing_payload()}, headers=auth_headers())
    assert response.status_code == 503
    assert response.get_json()["error"] == "Model not loaded"
    assert client.get("/classes", headers=auth_headers()).status_code == 503
    assert reload_attempts == []
