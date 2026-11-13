# os.environ must be set before importing app to override AI_API_KEY at module load time
import os

os.environ["AI_API_KEY"] = "test-key"
os.environ["AUTH_ALLOW_API_KEY"] = "true"  # these tests authenticate with the legacy key

import base64
import json
from io import BytesIO

import pytest
from PIL import Image

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["RATELIMIT_ENABLED"] = False
    with app.test_client() as client:
        client.environ_base["HTTP_X_API_KEY"] = "test-key"
        yield client


@pytest.fixture
def sample_image_base64():
    img = Image.new("L", (28, 28), 0)
    for x in range(28):
        for y in range(28):
            if ((x - 14) ** 2 + (y - 14) ** 2) < 64:
                img.putpixel((x, y), 255)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_data = buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode()
    return f"data:image/png;base64,{img_base64}"


def test_index_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == {"name": "AI Drawing Classifier API"}  # no endpoint list, version or model


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
    assert data["status"] == "healthy"


def test_classes_endpoint(client):
    response = client.get("/classes")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["total_classes"] == len(data["classes"]) > 0


def test_predict_missing_data(client):
    response = client.post("/predict", data=json.dumps({}), content_type="application/json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Missing image data" in data["error"]


def test_predict_invalid_image_format(client):
    response = client.post(
        "/predict", data=json.dumps({"image": "invalid_format"}), content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Invalid image format" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__])


def test_predict_valid_image(client, sample_image_base64):
    response = client.post(
        "/predict", data=json.dumps({"image": sample_image_base64}), content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True and isinstance(data["predictions"], list)


def test_version_and_model_are_only_shown_to_authenticated_callers(client, monkeypatch):
    import buildinfo

    monkeypatch.setattr(buildinfo, "CARD_PATH", "/nonexistent")  # no model card
    buildinfo.model_info.cache_clear()
    try:
        for public in ("/", "/health"):
            assert "version" not in json.loads(client.get(public).data)
        status = json.loads(client.get("/status").data)
        assert status["version"] == buildinfo.__version__ and status["model_info"] is None
    finally:
        buildinfo.model_info.cache_clear()


def test_predictions_are_logged_by_class_and_confidence_band_never_the_image(
    client, sample_image_base64, caplog
):
    import logging

    with caplog.at_level(logging.INFO, logger="doodleai"):
        response = client.post(
            "/predict",
            data=json.dumps({"image": sample_image_base64}),
            content_type="application/json",
        )
    assert response.status_code == 200
    line = next(r.getMessage() for r in caplog.records if "inference_ms" in r.getMessage())
    assert "top1=" in line and "confidence=" in line
    assert "base64" not in caplog.text and sample_image_base64[30:60] not in caplog.text


def test_confidence_bands():
    from app import confidence_bucket

    assert [confidence_bucket(c) for c in (0, 39.9, 40, 69.9, 70, 89.9, 90, 100)] == [
        "<40",
        "<40",
        "40-70",
        "40-70",
        "70-90",
        "70-90",
        "90+",
        "90+",
    ]
