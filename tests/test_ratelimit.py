import pytest

import auth
from app import app, limiter
from oidc import AUDIENCE, ISSUER, PREVIEW_SUBJECT, SUBJECT, FakeJwksClient, make_token

PROTECTED = "/classes"  # limited to 60 per hour per caller in these tests


@pytest.fixture
def client(monkeypatch, private_key):
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("OIDC_ALLOWED_SUBJECTS", f"{SUBJECT},{PREVIEW_SUBJECT}")
    monkeypatch.setenv("AI_API_KEY", "legacy-test-key")
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "true")
    monkeypatch.setattr(
        auth, "_get_jwks_client", lambda _issuer: FakeJwksClient(private_key.public_key())
    )
    monkeypatch.setenv("RATE_LIMIT_CLASSES", "60 per hour")  # production default is 60 per minute
    app.config["TESTING"] = True
    limiter.enabled = True
    limiter.reset()
    try:
        with app.test_client() as test_client:
            yield test_client
    finally:
        limiter.reset()
        limiter.enabled = False


def bearer(key, **claims):
    return {"Authorization": f"Bearer {make_token(key, **claims)}"}


def test_each_verified_caller_has_its_own_budget(client, private_key):
    """All traffic comes from the website's servers, yet callers do not share a limit."""
    production = bearer(private_key, sub=SUBJECT)
    preview = bearer(private_key, sub=PREVIEW_SUBJECT)

    for _ in range(60):
        assert client.get(PROTECTED, headers=production).status_code in (200, 503)
    assert client.get(PROTECTED, headers=production).status_code == 429
    assert client.get(PROTECTED, headers=preview).status_code in (200, 503)


def test_legacy_key_is_a_separate_caller(client, private_key):
    production = bearer(private_key, sub=SUBJECT)
    for _ in range(60):
        client.get(PROTECTED, headers=production)
    assert client.get(PROTECTED, headers=production).status_code == 429
    assert client.get(PROTECTED, headers={"x-api-key": "legacy-test-key"}).status_code in (200, 503)


def test_unauthenticated_requests_do_not_use_a_callers_budget(client, private_key):
    production = bearer(private_key, sub=SUBJECT)
    for _ in range(5):
        assert client.get(PROTECTED).status_code == 401
    for _ in range(60):
        assert client.get(PROTECTED, headers=production).status_code in (200, 503)


def test_ip_shield_throttles_unauthenticated_floods(client):
    statuses = {client.get(PROTECTED).status_code for _ in range(305)}
    assert statuses == {401, 429}


def test_ip_shield_throttles_unknown_paths_across_urls_and_methods(client):
    responses = [
        client.open(f"/api/scan-{i % 5}", method="GET" if i % 2 else "POST") for i in range(301)
    ]
    assert all(response.status_code == 404 for response in responses[:300])
    assert responses[300].status_code == 429
    assert client.get(PROTECTED).status_code == 429


def test_unknown_paths_do_not_consume_another_ips_budget(client):
    for _ in range(300):
        assert client.get("/api/.env", headers={"X-Forwarded-For": "192.0.2.1"}).status_code == 404
    assert client.get("/api/.env", headers={"X-Forwarded-For": "192.0.2.1"}).status_code == 429
    assert client.get("/api/.env", headers={"X-Forwarded-For": "192.0.2.2"}).status_code == 404


def test_wrong_method_requests_share_the_ip_shield(client):
    for _ in range(300):
        assert client.get("/predict").status_code == 405
    assert client.get("/predict").status_code == 429
    assert client.get("/api/.env").status_code == 429


def test_health_is_throttled_per_ip(client):
    statuses = [client.get("/health").status_code for _ in range(65)]
    assert statuses[:60] == [200] * 60 and set(statuses[60:]) == {429}


def test_default_route_limits_do_not_throttle_normal_site_traffic(client, private_key, monkeypatch):
    """All site traffic is one identity, so the defaults are circuit breakers, not quotas."""
    monkeypatch.delenv("RATE_LIMIT_CLASSES")  # the fixture sets a low one for the tests above
    production = bearer(private_key, sub=SUBJECT)
    statuses = {client.get(PROTECTED, headers=production).status_code for _ in range(50)}
    assert 429 not in statuses
